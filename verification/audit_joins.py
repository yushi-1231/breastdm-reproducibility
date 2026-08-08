#!/usr/bin/env python3
"""
Close out the remaining alignment questions before editing the rebuttal once.

CHECK 1 (high) — is decision_level's vit_prob column itself correctly ordered?
  We verified senet_prob matches the 2D loader element-wise. If vit_prob is
  misaligned RELATIVE to senet_prob inside the same file, then
  decision_fused_prob is wrong too — and decision-level fusion is a reported row
  in the paper. Test: attach 2D loader paths to decision_level, merge with the
  path-aligned val0755 export by (case, fname), and correlate.
  RULE  r > 0.999 and max abs diff < 1e-4  -> vit_prob is correctly ordered
        anything less                       -> decision_fused_prob is also affected

CHECK 2 (medium) — was the 3D ensemble built by path or by position?
  prob_3d_ensemble is the one fusion input still standing. Its members are the
  three *_test_probs.csv files, which DO carry paths. Test: are those three files
  in the same path order, and does the validation-AUC-weighted average of their
  probabilities reproduce prob_3d_ensemble?
  RULE  max abs diff < 1e-6 -> the 3D side is verified end to end

Run from the Classification task directory:
    cd "/data/lmy_tp/YS_tp/LG-CAFN/Classification task"
    export PYTHONPATH="/data/lmy_tp/YS_tp/LG-CAFN/Classification task:$PYTHONPATH"
    python /data/lmy_tp/YS_tp/LG-CAFN/rebuttal_2026/audit_remaining_joins.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

CLS = "/data/lmy_tp/YS_tp/LG-CAFN/Classification task"
CONTAM = "/data/lmy_tp/YS_tp/LG-CAFN/data/BreaDM"
R3D = "/data/lmy_tp/YS_tp/LG-CAFN/classification3D_reproduction"
HYB = "/data/lmy_tp/YS_tp/hybrid-position/data_out"
DECISION = os.path.join(CLS, "fusion_results/img9Se/decision_level_test_probs.csv")
FUSION = os.path.join(
    R3D, "results/3d_vit_fusion_img9Se_val_auc_weighted_val_auc_weighted_"
         "3dresnet50_3dmobilenet_3dshufflenetv2_test_predictions.csv")
sys.path.insert(0, CLS)

base = os.path.basename


def _midrank(x):
    J = np.argsort(x, kind="mergesort"); z = x[J]; N = len(x)
    t = np.empty(N, float); i = 0
    while i < N:
        j = i
        while j < N and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1); i = j
    o = np.empty(N, float); o[J] = t + 1
    return o


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p = int(y.sum()); n = len(y) - p
    return (_midrank(s)[y == 1].sum() - p * (p + 1) / 2.0) / (p * n)


from data_loader import CorrectBreastDataset

ds = CorrectBreastDataset(root_path=CONTAM, exp_type="img9Se", mode="test", img_size=96)
order_2d = [s["path"] for s in ds.samples]

print("=" * 72)
print("CHECK 1 - is decision_level's vit_prob column correctly ordered?")
print("=" * 72)

dec = pd.read_csv(DECISION)
if len(dec) != len(order_2d):
    raise SystemExit("row mismatch: decision_level %d vs 2D loader %d"
                     % (len(dec), len(order_2d)))
dec = dec.assign(path=order_2d)
dec["case"] = dec["path"].map(lambda p: base(os.path.dirname(p)))
dec["fname"] = dec["path"].map(base)

ex = os.path.join(HYB, "vit7_val0755_test_probs.csv")
if not os.path.exists(ex):
    print("  export not found: %s" % ex)
    print("  -> cannot run CHECK 1; skipping")
else:
    e = pd.read_csv(ex)
    pcol = [c for c in e.columns if c.startswith("prob_")][0]
    m = dec[["case", "fname", "label", "vit_prob"]].merge(
        e[["case", "fname", pcol]], on=["case", "fname"])
    print("  merged %d of %d export rows (export is the de-duplicated 403 set)"
          % (len(m), len(e)))
    a = m["vit_prob"].to_numpy(float)
    b = m[pcol].to_numpy(float)
    r = stats.pearsonr(a, b)[0]
    mx = float(np.max(np.abs(a - b)))
    print("  Pearson r = %.6f   max abs diff = %.3e" % (r, mx))
    if r > 0.999 and mx < 1e-4:
        print("  -> VERIFIED: vit_prob is correctly ordered inside decision_level.")
        print("     decision_fused_prob and the decision-level fusion row are SAFE.")
    else:
        print("  -> PROBLEM: vit_prob is misaligned inside decision_level itself,")
        print("     so decision_fused_prob and the reported decision-level fusion")
        print("     row are also affected. Widen the correction accordingly.")

print()
print("=" * 72)
print("CHECK 2 - was the 3D ensemble built by path or by position?")
print("=" * 72)

members = sorted(glob.glob(os.path.join(R3D, "probs/img9Se/*_test_probs.csv")))
print("  member files: %s" % ", ".join(base(f) for f in members))
if len(members) < 2:
    print("  -> too few member files to check")
else:
    orders = {}
    for f in members:
        d = pd.read_csv(f)
        orders[base(f)] = d["path"].tolist()
    ref = orders[base(members[0])]
    all_same = all(v == ref for v in orders.values())
    print("  all member files in the SAME path order: %s" % all_same)
    if not all_same:
        for k, v in orders.items():
            diff = sum(1 for i, p in enumerate(v) if p != ref[i])
            print("    %-34s differs from the first in %d positions" % (k, diff))
        print("  -> the 3D ensemble average may itself be misaligned")
    else:
        # reproduce prob_3d_ensemble from the members
        fu = pd.read_csv(FUSION)
        fu["k"] = fu["path"]
        dfs = []
        for f in members:
            d = pd.read_csv(f)[["path", "prob_malignant"]]
            d = d.rename(columns={"prob_malignant": base(f).replace("_test_probs.csv", "")})
            dfs.append(d.set_index("path"))
        mem = pd.concat(dfs, axis=1)
        merged = fu.set_index("path").join(mem, how="inner")
        cols = [c for c in mem.columns]
        print("  member columns: %s" % ", ".join(cols))
        print("  rows joined by path: %d of %d" % (len(merged), len(fu)))

        simple = merged[cols].mean(axis=1).to_numpy()
        target = merged["prob_3d_ensemble"].to_numpy(float)
        print("  simple mean of members vs prob_3d_ensemble: max abs diff %.3e"
              % float(np.max(np.abs(simple - target))))

        # least-squares weights, to see whether SOME convex weighting reproduces it
        A = merged[cols].to_numpy(float)
        w, *_ = np.linalg.lstsq(A, target, rcond=None)
        resid = float(np.max(np.abs(A @ w - target)))
        print("  best-fit weights %s  -> max abs residual %.3e"
              % (np.round(w, 4).tolist(), resid))
        if resid < 1e-6:
            print("  -> VERIFIED: prob_3d_ensemble is a weighted average of the")
            print("     members joined BY PATH. The 3D side is sound.")
        else:
            print("  -> prob_3d_ensemble is not reproducible from the members as")
            print("     path-joined; the 3D ensemble may have been built positionally.")
            print("     Investigate ensemble_3d_probs.py before trusting 0.8611.")

print()
print("=" * 72)
print("Also worth eyeballing, cheaply, in the same sitting:")
print("  wc -l 'Classification task/features_fusion/img9Se/y_test_all.txt'")
print("  and confirm it is 446 lines in the same class order as the 2D loader,")
print("  since feature-level fusion is another reported row built from text dumps.")
print("=" * 72)
