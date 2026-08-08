#!/usr/bin/env python3
"""
EXP-2 (img17Se) version of the join check.

img17Se is one of the CLEAN released trees, so there is no de-duplication step
and the full test set is used. The reported Exp-2 extension figure is 0.8929.

Final step on the ViT join: compare the two loader orders, and if they differ,
repair the merge by path and recompute everything that depended on it.

Already established:
  - decision_level_test_probs.csv rows ARE in CorrectBreastDataset order
    (senet_prob matched a path-aligned SENet50 export element-wise)
  - the used prob_vit column has eta^2 = 0.2757, matching the documented
    val0755 checkpoint's 0.2772 and far from the failed runs (0.039-0.105)

So the only open question is whether the 3D loader enumerated the test set in
the same order as CorrectBreastDataset.

PRE-REGISTERED RULE for the recomputed fusion
  |corrected - 0.8929| < 0.02  -> robust; report the documented version
  |corrected - 0.8929| > 0.05  -> the reported figure depended on the join
  in between                    -> report both and explain

Run from the Classification task directory:
    cd "/data/lmy_tp/YS_tp/LG-CAFN/Classification task"
    export PYTHONPATH="/data/lmy_tp/YS_tp/LG-CAFN/Classification task:$PYTHONPATH"
    python /data/lmy_tp/YS_tp/LG-CAFN/rebuttal_2026/fix_vit_join.py
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
OUT = "/data/lmy_tp/YS_tp/LG-CAFN/rebuttal_2026"
DECISION = os.path.join(CLS, "fusion_results/img17Se/decision_level_test_probs.csv")
_c = sorted(glob.glob(os.path.join(
    R3D, "results/3d_vit_fusion_img17Se*_test_predictions.csv")))
if not _c:
    raise SystemExit("no img17Se fusion prediction csv found under results/ — "
                     "list that directory and point FUSION at the right file")
FUSION = _c[-1]
# img17Se is one of the clean released trees — no duplicated cases to drop
LEAKED = set()
sys.path.insert(0, CLS)


def _midrank(x):
    J = np.argsort(x, kind="mergesort")
    z = x[J]
    N = len(x)
    t = np.empty(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    o = np.empty(N, float)
    o[J] = t + 1
    return o


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    p = int(y.sum())
    n = len(y) - p
    if p == 0 or n == 0:
        return float("nan")
    return (_midrank(s)[y == 1].sum() - p * (p + 1) / 2.0) / (p * n)


def within_class_r(a, b, y):
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    r0 = stats.pearsonr(a[y == 0], b[y == 0])[0]
    r1 = stats.pearsonr(a[y == 1], b[y == 1])[0]
    pooled = (r0 * (n0 - 1) + r1 * (n1 - 1)) / (n0 + n1 - 2)
    return pooled, r0, r1


# ---------------------------------------------------------------- STEP 1
from data_loader import CorrectBreastDataset

ds = CorrectBreastDataset(root_path=CONTAM, exp_type="img17Se", mode="test", img_size=96)
order_2d = [s["path"] for s in ds.samples]

p3 = sorted(glob.glob(os.path.join(R3D, "probs/img17Se/*_test_probs.csv")))
if not p3:
    raise SystemExit("no 3D probs csv found under probs/img17Se/")
order_3d = pd.read_csv(p3[0])["path"].tolist()

print("=" * 72)
print("STEP 1 - do the two loaders enumerate the test set identically?")
print("=" * 72)
print("  2D loader (CorrectBreastDataset): %d paths" % len(order_2d))
print("  3D loader (%s): %d paths" % (os.path.basename(p3[0]), len(order_3d)))
same_set = set(order_2d) == set(order_3d)
identical = order_2d == order_3d
print("  same SET of paths:   %s" % same_set)
print("  same ORDER of paths: %s" % identical)

if (not identical) and same_set:
    pos2d = {p: i for i, p in enumerate(order_2d)}
    moved = sum(1 for i, p in enumerate(order_3d) if pos2d[p] != i)
    lab2d = {s["path"]: s["label"] for s in ds.samples}
    cross = sum(1 for i, p in enumerate(order_3d) if lab2d[p] != lab2d[order_2d[i]])
    print("  positions that differ: %d of %d" % (moved, len(order_3d)))
    print("  positions where the LABEL also differs: %d" % cross)
    if cross == 0:
        print("  -> a within-class permutation; the label guard could not see it")
    else:
        print("  -> crosses class boundaries; the label guard should have fired")

if identical:
    print()
    print("  The positional join was VALID - prob_vit is correctly attached.")
    print("  The 0.8020 vs 0.7948 gap must then come from decisionFusion.py's own")
    print("  evaluation path (transform, img_size, normalisation). Stop here and")
    print("  read that script; nothing needs recomputing.")
    raise SystemExit(0)

# ---------------------------------------------------------------- STEP 2
print()
print("=" * 72)
print("STEP 2 - repair: attach 2D paths to decision_level, then merge by path")
print("=" * 72)

dec = pd.read_csv(DECISION)
if len(dec) != len(order_2d):
    raise SystemExit("row mismatch: decision_level %d vs 2D loader %d"
                     % (len(dec), len(order_2d)))
dec = dec.assign(path=order_2d)
dec["case"] = dec["path"].map(lambda p: os.path.basename(os.path.dirname(p)))
dec["fname"] = dec["path"].map(os.path.basename)

fu = pd.read_csv(FUSION)
fu["case"] = fu["path"].map(lambda p: os.path.basename(os.path.dirname(p)))
fu["fname"] = fu["path"].map(os.path.basename)

right = dec[["case", "fname", "label", "vit_prob"]].rename(
    columns={"vit_prob": "prob_vit_fixed", "label": "label_dec"})
m = fu[["case", "fname", "label", "prob_3d_ensemble", "prob_vit"]].merge(
    right, on=["case", "fname"])
if not (m["label"] == m["label_dec"]).all():
    raise SystemExit("label mismatch after the path merge - stop and inspect")
m = m[~m["case"].isin(LEAKED)].reset_index(drop=True)
print("  merged: %d samples, %d cases   (img17Se is clean, nothing dropped)"
      % (len(m), m["case"].nunique()))

y = m["label"].to_numpy()
p3d = m["prob_3d_ensemble"].to_numpy(float)
old = m["prob_vit"].to_numpy(float)
new = m["prob_vit_fixed"].to_numpy(float)

print()
print("  Table 2 Exp-2 reference: ViT-7 img17Se reproduced AUC is the value to")
print("  expect from the path-joined column; if it matches, the repair is sound.")
print("  ViT AUC   as joined %.4f   path-joined %.4f" % (auc(y, old), auc(y, new)))
print("  raw r between the two ViT columns: %+.4f" % stats.pearsonr(old, new)[0])
print("  values identical: %s" % np.allclose(old, new))

# ---------------------------------------------------------------- STEP 3
print()
print("=" * 72)
print("STEP 3 - recompute everything that depended on the join")
print("=" * 72)

for tag, pv in (("as joined  ", old), ("path-joined", new)):
    fused = 0.5 * p3d + 0.5 * pv
    r, r0, r1 = within_class_r(p3d, pv, y)
    g = m.assign(pv=pv).groupby("case").agg(
        label=("label", "first"),
        a=("prob_3d_ensemble", "mean"),
        b=("pv", "mean"))
    w3 = (g["a"] >= 0.5).astype(int) != g["label"]
    wv = (g["b"] >= 0.5).astype(int) != g["label"]
    best_single = max(auc(y, p3d), auc(y, pv))
    span = []
    for w in np.arange(0.0, 1.001, 0.05):
        if auc(y, (1 - w) * p3d + w * pv) > best_single:
            span.append(w)

    print()
    print("  %s" % tag)
    print("    3D ens %.4f   ViT %.4f   50/50 fused %.4f"
          % (auc(y, p3d), auc(y, pv), auc(y, fused)))
    print("    within-class r(3D, ViT) = %+.4f   (benign %+.3f, malignant %+.3f)"
          % (r, r0, r1))
    print("    case-level: both correct %d, only 3D wrong %d, only ViT wrong %d, both wrong %d"
          % (int((~w3 & ~wv).sum()), int((w3 & ~wv).sum()),
             int((~w3 & wv).sum()), int((w3 & wv).sum())))
    if span:
        print("    beats the better branch for w in [%.2f, %.2f]  (%d/21 grid points)"
              % (min(span), max(span), len(span)))
    else:
        print("    never beats the better single branch")

corrected = auc(y, 0.5 * p3d + 0.5 * new)
diff = abs(corrected - 0.8929)
print()
print("=" * 72)
print("corrected 50/50 fusion AUC = %.4f   vs reported 0.8929   diff %.4f"
      % (corrected, diff))
if diff < 0.02:
    print("VERDICT: robust - numbers barely move; report the documented version")
elif diff > 0.05:
    print("VERDICT: the reported figure depended on the faulty join.")
    print("         Tables 3-4, the r = +0.036 claim, the 6-and-6 split and the")
    print("         weight sweep all need replacing - in the rebuttal as well")
    print("         as the camera-ready.")
else:
    print("VERDICT: in between - report both and explain")
print("=" * 72)

path_out = os.path.join(OUT, "fusion_pathjoined_corrected_exp2.csv")
m.to_csv(path_out, index=False)
print()
print("wrote %s" % path_out)
