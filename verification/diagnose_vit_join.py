#!/usr/bin/env python3
"""
Is the prob_vit column in the 3D+ViT fusion output joined to the right samples?

TEST A — pure arithmetic, no join needed.
  If prob_vit is a within-class permutation of the documented checkpoint's
  output, then r(exported, used) must equal the exported column's between-class
  variance fraction (eta-squared):
      X  = class mean + within-class deviation
      X' = class mean + independent deviation from the same distribution
      Cov(X, X') = Var(class mean),  Var(X) = Var(X') = Var(class mean) + Var(dev)
      => r(X, X') = eta^2
  Observed r = 0.3353.
  DECISION, FIXED BEFORE RUNNING
    eta^2 within 0.30-0.37   -> quantitatively consistent with a within-class
                                permutation; treat misalignment as established
    eta^2 above 0.45         -> observed r is too LOW for a pure within-class
                                permutation; something else is wrong (different
                                checkpoint, or cross-class scrambling)
    eta^2 below 0.25         -> observed r is too HIGH; the two columns share
                                more than class membership

TEST B — recover the permutation using SENet50 as a key.
  decision_level_test_probs.csv carries senet_prob but no path. Export SENet50
  with paths and match values row by row. Unique matches give an exact
  row-index -> path map, which both verifies the positional join AND allows
  prob_vit to be re-joined correctly without retraining anything.

Run from the Classification task directory:
    cd "/data/lmy_tp/YS_tp/LG-CAFN/Classification task"
    export PYTHONPATH="/data/lmy_tp/YS_tp/LG-CAFN/Classification task:$PYTHONPATH"
    CUDA_VISIBLE_DEVICES=0 python /data/lmy_tp/YS_tp/LG-CAFN/rebuttal_2026/diagnose_vit_join.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

CLS = "/data/lmy_tp/YS_tp/LG-CAFN/Classification task"
CONTAM_ROOT = "/data/lmy_tp/YS_tp/LG-CAFN/data/BreaDM"          # 446 rows / 53 cases
DECISION_CSV = os.path.join(CLS, "fusion_results/img9Se/decision_level_test_probs.csv")
FUSION_CSV = ("/data/lmy_tp/YS_tp/LG-CAFN/classification3D_reproduction/results/"
              "3d_vit_fusion_img9Se_val_auc_weighted_val_auc_weighted_"
              "3dresnet50_3dmobilenet_3dshufflenetv2_test_predictions.csv")
VIT7_EXPORT_GLOB = "/data/lmy_tp/YS_tp/hybrid-position/data_out/vit7_*_test_probs.csv"
SENET_CKPT = os.path.join(
    CLS, "final_classification_results/checkpoints/current_tested_old_checkpoints/"
         "senet50_img9Se_best_auc0.9234.pth")
sys.path.insert(0, CLS)


def eta_squared(values, labels):
    """Fraction of total variance explained by class membership."""
    v = np.asarray(values, float); y = np.asarray(labels)
    grand = v.mean()
    ss_total = ((v - grand) ** 2).sum()
    ss_between = sum(len(v[y == c]) * (v[y == c].mean() - grand) ** 2 for c in np.unique(y))
    return ss_between / ss_total


print("=" * 74)
print("TEST A — is r = 0.3353 what a within-class permutation would give?")
print("=" * 74)

cands = sorted(glob.glob(VIT7_EXPORT_GLOB))
if not cands:
    print(f"  no exports found at {VIT7_EXPORT_GLOB}")
    print("  (run the export from the other window first, or point this at your own)")
else:
    for f in cands:
        d = pd.read_csv(f)
        pcol = [c for c in d.columns if c.startswith("prob_")]
        if not pcol:
            print(f"  {os.path.basename(f)}: no prob_ column")
            continue
        e2 = eta_squared(d[pcol[0]], d["label"])
        verdict = ("consistent with within-class permutation" if 0.30 <= e2 <= 0.37
                   else "observed r too LOW for a pure within-class permutation" if e2 > 0.45
                   else "observed r too HIGH — columns share more than class" if e2 < 0.25
                   else "borderline")
        print(f"  {os.path.basename(f):42s} n={len(d):4d}  eta^2 = {e2:.4f}   {verdict}")

# same statistic on the used column, for reference
fu = pd.read_csv(FUSION_CSV)
print(f"\n  for reference, eta^2 of the USED prob_vit column "
      f"({len(fu)} rows) = {eta_squared(fu['prob_vit'], fu['label']):.4f}")
print("  (a within-class permutation preserves eta^2, so these two should agree)")


print()
print("=" * 74)
print("TEST B — recover the permutation with SENet50 as a key")
print("=" * 74)

dec = pd.read_csv(DECISION_CSV)
print(f"  decision_level_test_probs.csv: {len(dec)} rows, columns {list(dec.columns)}")
lab = dec["label"].to_numpy()
runs = 1 + int((np.diff(lab) != 0).sum())
print(f"  label sequence has {runs} run(s) — {'sorted by class, so the label guard checked almost nothing' if runs <= 2 else 'interleaved'}")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import CorrectBreastDataset
import eval_one_checkpoint_youden as E

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = CorrectBreastDataset(root_path=CONTAM_ROOT, exp_type="img9Se",
                          mode="test", img_size=96)
print(f"  CorrectBreastDataset on the contaminated tree: {len(ds.samples)} samples")

loader = DataLoader(ds, batch_size=16, shuffle=False, drop_last=False, num_workers=2)
model = E.create_model("senet50", E.infer_in_channels("img9Se"), num_classes=2)
ck = torch.load(SENET_CKPT, map_location="cpu")
sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
model.load_state_dict(E.clean_state_dict(sd))
model = model.to(device).eval()

probs = []
with torch.no_grad():
    for b in loader:
        o = model(b[0].to(device))
        if isinstance(o, (list, tuple)):
            o = o[0]
        probs.extend(F.softmax(o, dim=1)[:, 1].cpu().numpy().tolist())

ex = pd.DataFrame({
    "idx_2dloader": np.arange(len(probs)),
    "path": [s["path"] for s in ds.samples],
    "case": [s["case_id"] for s in ds.samples],
    "fname": [os.path.basename(s["path"]) for s in ds.samples],
    "label": [s["label"] for s in ds.samples],
    "senet": probs,
})
out = "/data/lmy_tp/YS_tp/LG-CAFN/rebuttal_2026/senet50_contaminated_test_probs.csv"
ex.to_csv(out, index=False)
print(f"  exported SENet50 with paths -> {out}")

if len(ex) != len(dec):
    print(f"\n  ROW COUNT MISMATCH: export {len(ex)} vs decision_level {len(dec)}")
    print("  -> the decision-level file was not built on this tree; stop and locate the writer")
else:
    same_order = np.allclose(ex["senet"].to_numpy(), dec["senet_prob"].to_numpy(), atol=1e-5)
    print(f"\n  senet_prob matches the 2D loader's order element-wise: {same_order}")
    if same_order:
        print("  -> decision_level rows ARE in CorrectBreastDataset order; the")
        print("     positional join is only valid if the 3D loader used that same order")
    else:
        # try to recover the permutation by value matching
        a = ex["senet"].to_numpy(); b = dec["senet_prob"].to_numpy()
        mapping, ambiguous, unmatched = {}, 0, 0
        for j, val in enumerate(b):
            hits = np.where(np.abs(a - val) < 1e-6)[0]
            if len(hits) == 1:
                mapping[j] = int(hits[0])
            elif len(hits) == 0:
                unmatched += 1
            else:
                ambiguous += 1
        print(f"  value matching: {len(mapping)} unique, {ambiguous} ambiguous, {unmatched} unmatched")
        if len(mapping) > 0.9 * len(b):
            perm = pd.DataFrame({"decision_row": list(mapping),
                                 "loader_idx": [mapping[k] for k in mapping]})
            perm = perm.merge(ex[["idx_2dloader", "path", "case", "fname", "label"]],
                             left_on="loader_idx", right_on="idx_2dloader")
            pout = "/data/lmy_tp/YS_tp/LG-CAFN/rebuttal_2026/decision_level_row_to_path.csv"
            perm.to_csv(pout, index=False)
            print(f"  -> permutation recovered, written to {pout}")
            print("     prob_vit can now be re-joined by path without retraining")
            within = (perm["label"].to_numpy() == dec["label"].to_numpy()[perm["decision_row"]]).all()
            print(f"     recovered mapping stays within class: {within}")
        elif unmatched > 0.5 * len(b):
            print("  -> senet_prob does not come from this checkpoint or this preprocessing;")
            print("     the problem is inside the decision-level file, not the positional join")

print()
print("=" * 74)
print("NEXT, only after the above: recompute the fusion with a path-joined merge")
print("using the documented ViT (test AUC 0.8020) and compare against 0.9101.")
print("Rule: difference under 0.02 -> robust; over 0.05 -> the reported figure")
print("depended on the faulty join; in between -> report both.")
print("=" * 74)
