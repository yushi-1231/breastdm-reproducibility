#!/usr/bin/env python3
"""
Recompute Exp-1 (img9Se) test metrics with the 6 leaked cases excluded.

Uses only already-saved per-sample probability CSVs — no model inference needed.
Validation-selected Youden thresholds are unaffected by the leak, so they are
reused as-is.

Run from anywhere:
    python recompute_clean_exp1.py
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = "/data/lmy_tp/YS_tp/LG-CAFN"
PROBS_3D = f"{ROOT}/classification3D_reproduction/probs/img9Se"
RESULTS_3D = f"{ROOT}/classification3D_reproduction/results"

LEAKED = {
    "BreaDM-Ma-1802", "BreaDM-Ma-1803", "BreaDM-Ma-1804",
    "BreaDM-Ma-1806", "BreaDM-Ma-1807", "BreaDM-Ma-1808",
}


def case_of(path):
    """Extract case ID from a sample path .../<Class>/<CASE>/p-NNN.npy"""
    return os.path.basename(os.path.dirname(path))


def metrics(y, p, thr):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return dict(
        AUC=roc_auc_score(y, p),
        Sens=100 * tp / (tp + fn) if tp + fn else float("nan"),
        Spec=100 * tn / (tn + fp) if tn + fp else float("nan"),
        Acc=100 * (tp + tn) / len(y),
        Prec=100 * tp / (tp + fp) if tp + fp else float("nan"),
        n=len(y),
    )


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]


def show(name, before, after):
    print(f"\n{name}")
    print(f"  {'':10s} {'AUC':>8s} {'Sens':>7s} {'Spec':>7s} {'Acc':>7s} {'Prec':>7s} {'n':>5s}")
    for tag, m in (("contaminated", before), ("clean", after)):
        print(f"  {tag:10s} {m['AUC']:8.4f} {m['Sens']:7.2f} {m['Spec']:7.2f} "
              f"{m['Acc']:7.2f} {m['Prec']:7.2f} {m['n']:5d}")
    print(f"  {'Δ AUC':10s} {after['AUC'] - before['AUC']:+8.4f}")


print("=" * 78)
print("PART 1 — individual 3D models (Table 2, Exp-1 rows with saved probs)")
print("=" * 78)

for f in sorted(glob.glob(f"{PROBS_3D}/*_test_probs.csv")):
    model = os.path.basename(f).replace("_test_probs.csv", "")
    df = pd.read_csv(f)

    # validation threshold from the matching val file
    vf = f.replace("_test_probs.csv", "_val_probs.csv")
    if os.path.exists(vf):
        vdf = pd.read_csv(vf)
        thr = youden_threshold(vdf["label"], vdf["prob_malignant"])
    else:
        thr = 0.5

    keep = ~df["path"].map(case_of).isin(LEAKED)
    before = metrics(df["label"], df["prob_malignant"], thr)
    after = metrics(df.loc[keep, "label"], df.loc[keep, "prob_malignant"], thr)
    show(f"{model}  (val-Youden thr = {thr:.4f})", before, after)


print()
print("=" * 78)
print("PART 2 — 3D ensemble and 3D+ViT fusion (Table 4, Exp-1)")
print("=" * 78)

for f in sorted(glob.glob(f"{RESULTS_3D}/3d_vit_fusion_img9Se*_test_predictions.csv")):
    tag = os.path.basename(f).replace("_test_predictions.csv", "")
    df = pd.read_csv(f)
    keep = ~df["path"].map(case_of).isin(LEAKED)

    for col, label in (("prob_3d_ensemble", "3D ensemble alone"),
                       ("prob_vit", "ViT branch alone"),
                       ("prob_fused", "3D ensemble + ViT")):
        if col not in df.columns:
            continue
        # threshold: recover from the summary CSV if present, else Youden on full test
        summary = f.replace("_test_predictions.csv", ".csv")
        thr = None
        if os.path.exists(summary):
            s = pd.read_csv(summary)
            if "threshold_from_val_youden" in s.columns:
                thr = float(s["threshold_from_val_youden"].iloc[0])
        if thr is None:
            thr = 0.5

        before = metrics(df["label"], df[col], thr)
        after = metrics(df.loc[keep, "label"], df.loc[keep, col], thr)
        show(f"{tag}\n    [{label}]  (thr = {thr:.4f})", before, after)


print()
print("=" * 78)
print("SUMMARY OF LEAK")
print("=" * 78)
f = sorted(glob.glob(f"{RESULTS_3D}/3d_vit_fusion_img9Se*_test_predictions.csv"))
if f:
    df = pd.read_csv(f[0])
    cases = df["path"].map(case_of)
    leaked_rows = cases.isin(LEAKED)
    print(f"  total test samples          : {len(df)}")
    print(f"  leaked test samples         : {int(leaked_rows.sum())} "
          f"({100*leaked_rows.mean():.1f}%)")
    print(f"  clean test samples          : {int((~leaked_rows).sum())}")
    print(f"  total test cases            : {cases.nunique()}")
    print(f"  clean test cases            : {cases[~leaked_rows].nunique()}")
    print(f"  class balance (clean)       : "
          f"{int((df.loc[~leaked_rows,'label']==0).sum())} benign / "
          f"{int((df.loc[~leaked_rows,'label']==1).sum())} malignant")
