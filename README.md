# Reproducibility Analysis of the BreastDM Benchmark

Code, verification scripts and full result tables for:

> **Replication and Reproducibility Analysis of Breast Tumor Classification and
> Segmentation Benchmarks on the BreastDM Dataset**
> Shi Yu, University of Houston
> MICCAI 2026 · MSB EMERGE Workshop (accepted, poster)

Reproduction target: Zhao et al., *BreastDM: A DCE-MRI dataset for breast tumor image
segmentation and classification*, **Computers in Biology and Medicine 164 (2023) 107255**.
Released data and code: `github.com/smallboy-code/Breast-cancer-dataset`
(branch `master`, commit `592561c9fc617a345fd6e881eb326fcc521263a2`, 2023-09-04).

---

## What this repository is for

This is not a re-release of BreastDM. It holds the scripts needed to **check the claims
in the paper**, the corrected evaluation code, and the full result tables the paper refers
to but has no room to print.

Every number the paper reports can be traced to a named checkpoint, a named preprocessing
path, and a sample identifier. No file in this pipeline is joined to another by row
position. That rule exists because breaking it is what produced Finding 2 below.

---

## Finding 1 — the released archive duplicates training cases into two test folders

`BreaDM.zip` as distributed places byte-identical copies of training cases inside two of
its eight derived test folders.

| Tree | Test cases as released | Duplicated from train | Affected test samples |
|---|---|---|---|
| `cls/img9Se` | 53 | 6 malignant (`BreaDM-Ma-1802/1803/1804/1806/1807/1808`) | 43 / 446 = **9.6 %** |
| `seg` | 50 | 3 benign (`BreaDM-Be-1801/1803/1804`) | 238 / 7089 = **3.4 %** |

Clean: `cls/img17Se`, `cls/GLCM` (both), `cls/LBP` (both), `seg3D`.

**Removing the duplicated cases recovers the split the paper describes, exactly:**
47 test cases = 17 benign + 30 malignant, and exactly 6851 segmentation test images.
In both trees the duplicated cases are the first *N* entries of the sorted training list.

```bash
python verification/detect_duplication.py --root /path/to/BreaDM
```

### Archive identity

Two downloads seven months apart agree:

```
size    1,303,964,654 bytes
sha256  64d8e621b98aa3d88ba66d283e06b6592d87d122e1faceb90904f1bfd97ce6e5
md5     9123ac5c880f9dd756efdd64bf57bf22
```

### Scope

The defect is in **distribution**. Work that consumed `cls/img9Se` or `seg` as released,
as this reproduction initially did, is affected.

The published results appear consistent with the intended de-duplicated split and show no
evidence of being affected by it — the percentages in the original Table 3 are only
representable as integer ratios over 403 samples, and the segmentation counts match the
de-duplicated tree exactly. The authors' internal evaluation files cannot be established
from the publication alone, so no stronger claim is made here. Studies that used these two
folders may wish to verify their own splits; no claim is made about any particular study
without its data loading being checked.

Note that the de-duplicated trees here are built with **hardlinks, not symlinks** — the
segmentation loader uses `os.walk` without `followlinks`, so a symlink tree reads as empty.

---

## Finding 2 — an error in *this* reproduction, and what it cost

The exploratory 3D-ensemble + ViT fusion reported in the submitted version **does not hold
and has been withdrawn.**

`fusion_3d_vit.py` read ViT probabilities from a CSV with no sample identifier and merged
them with the 3D ensemble **by row position**. The only guard was a label-consistency
check — and because the test set is ordered benign-then-malignant, that check passes under
*any* within-class permutation. It was passing while being wrong.

| | Positions differing | Crossing a class boundary |
|---|---|---|
| Exp-1 | 443 / 446 | 0 |
| Exp-2 | 400 / 403 | 0 |

Re-merging by sample path:

| | 3D ensemble | ViT | Fusion **as reported** | Fusion **corrected** |
|---|---|---|---|---|
| Exp-1 | 0.8611 | 0.8020 | 0.9101 | **0.8550** |
| Exp-2 | 0.8687 | 0.7919 | 0.8929 | **0.8583** |

The fusion falls **below the better single branch in both experiments**. Not a weakened
effect; an absent one. A weight sweep never beats the better branch at any mixing ratio.
The error was found and reported before the rebuttal deadline.

### The audit around it

The damage is bounded to one merge, and both sides of it were verified independently:

- `decision_level_test_probs.csv`'s own columns are correctly ordered — `vit_prob` matches
  its source checkpoint's export at r = 1.000000, max difference 5.96e-08
- `prob_3d_ensemble` is exactly reproducible from its three members joined by path
  (residual 5.55e-16), with weights matching the documented validation-AUC scheme

```bash
python verification/audit_joins.py      # re-runs both checks
python verification/fix_vit_join.py     # produces the corrected merge
```

No retraining was required to repair it.

---

## Corrected results

De-duplicated test set: **47 cases / 403 classification samples / 6851 segmentation
images.** Checkpoints and validation-selected Youden thresholds are unaffected; only test
evaluation was recomputed.

### Segmentation (Table 1)

| Model | DSC | mIoU | PPV |
|---|---|---|---|
| FCN-50 | 68.06 | 78.11 | 68.95 |
| FCN-101 | 67.58 | 77.78 | 71.73 |
| DeepLabV3-50 | 68.72 | 78.06 | 65.09 |
| DeepLabV3-101 | 68.03 | 77.99 | 71.02 |
| Unet | 72.55 | 82.11 | 75.65 |
| Unet-VGG16 | **73.61** | **82.13** | 74.10 |
| UNeXt | 70.67 | 79.92 | 71.53 |
| 3D-Unet | 68.07 | 79.15 | 66.76 |
| 3D-VNet | 64.90 | 77.91 | 64.55 |
| 3D-DenseSeg | 62.61 | 76.77 | 68.96 |

The three 3D rows are evaluated on `seg3D`, which is clean, and need no correction —
independently confirmed by the sample count, 141 = 47 cases × 3 sequences, where 47 is the
de-duplicated case count (the released `seg` tree would give 50).

> **mIoU exceeds DSC by definition here.** mIoU averages IoU over both classes and
> background IoU is very high on these slices, while DSC and PPV are foreground-only.
> Foreground-only IoU is in `results/foreground_iou_summary.csv`.

DSC also has a large per-image standard deviation (23–26 points, against 11–12 for mIoU on
the same images). This is structural, not noise: BreastDM ROIs are small, so slices at the
edge of a lesion hold a few dozen foreground pixels and score near zero while central
slices exceed 0.85. Per-image distributions are in `results/per_image_*.csv`.

### Classification, Experiment 1 (AUC)

| Model | As released | Corrected |
|---|---|---|
| VGG16 | 0.7423 | 0.7346 |
| VGG19 | 0.8103 | 0.8004 |
| ResNet50 | 0.8560 | 0.8411 |
| ResNet101 | 0.7621 | 0.7564 |
| DenseNet121 | 0.8372 | 0.8344 |
| DenseNet169 | 0.7947 | 0.7848 |
| SENet50 | 0.8177 | 0.8025 |
| SENet101 | 0.7415 | 0.7258 |
| ViT-6 | 0.8602 | **0.8525** |
| ViT-7 | 0.8000 | 0.8020 |
| 3DResNet18 | 0.8113 | 0.7961 |
| 3DResNet50 | 0.8582 | **0.8464** |
| 3DResNet101 | 0.7983 | 0.7819 |
| 3DResNeXt101 | 0.8219 | 0.8028 |
| 3DShuffleNet | 0.7976 | 0.7745 |
| 3DShuffleNetV2 | 0.8346 | 0.8152 |
| 3DMobileNet | 0.8507 | 0.8325 |
| 3DMobileNetV2 | 0.8388 | 0.8215 |
| Feature-level fusion | 0.8313 | 0.8219 |
| Decision-level fusion | 0.8272 | 0.8151 |
| LG-CAFN | 0.8847 | **0.8523 ± 0.0109** † |

† LG-CAFN is retrained at the input size the original paper specifies and reported as the
mean over three seeds; see below.

**Experiment 2 uses `cls/img17Se`, which is clean, and needs no correction at all.**

**Specificity is identical before and after in every classification row**, because all
duplicated classification cases are malignant — an independent confirmation that the
correction is localised correctly. Sensitivity falls 2–3 points, the signature of memorised
training cases.

Operating points at the validation-selected Youden threshold for every model are in
`results/operating_points_dedup.csv`.

---

## What actually moves these numbers

| Factor | Effect | Documented in the original paper? |
|---|---|---|
| Which sequences are evaluated (3D segmentation) | up to **10.9 DSC points** | No |
| Unreported hyperparameter (batch size) | 0.110 validation AUC | No |
| Random seed | 0.022–0.042 test AUC | No |
| The split duplication above | 0.003–0.023 test AUC | No |

Scoring the 3D segmentation models on the second post-contrast sequence alone rather than
on all three moves DSC from 62.61 to 73.50 for 3D-DenseSeg. The original paper does not
state which convention it uses; the all-sequence figure is reported here.

Test variance runs 2.5–4.2× validation variance in every model measured, and the
best-validating seed is not the best-testing seed.

### The validation set is 19 cases

Three of the observations above share one cause, and it is worth stating separately.

LG-CAFN retrained at the input size the paper specifies reaches **99.8 % training
accuracy** against 86.9 % in the original configuration — the corrected model fits the
training set completely — while **validation AUC rises** from 0.8835 to 0.9878 ± 0.0026
and test AUC falls to 0.8523 ± 0.0109. The validation–test gap goes from 0.0095 to
0.1355 ± 0.0135 with a single configuration variable changed.

The 19-case validation set cannot see the memorisation. It also selects Youden thresholds
that do not transfer: those thresholds span 0.31 to 0.999, and 3DResNeXt101 has a *higher*
AUC than 3DResNet101 (0.8028 vs. 0.7819) but 16 points *lower* accuracy (56.58 % vs.
72.21 %) because its threshold sits at 0.999.

Since checkpoint selection, threshold selection and early stopping are all keyed to that
validation set, a reported test figure is in part a draw over which run was made. This is a
property of the benchmark protocol, not of any model in it.

Per-seed figures are in `results/lgcafn_true96_seeds.csv`.

---

## Provenance

"Reproduction" is not one thing. Each component here came from somewhere different.

| Component | Source |
|---|---|
| 2D segmentation architectures | Released code (`Segmentation task/unet/src/`) |
| UNeXt | The official UNeXt implementation, as bundled in the release |
| 2D segmentation training / data / evaluation | Independently reimplemented |
| 2D classification and SENet–ViT fusion | Substantially modified from released code |
| Feature-level fusion I/O format | Released code |
| **Entire 3D classification pipeline** | **Independently reconstructed** |
| **All 3D segmentation models** | **Independently reconstructed** |

The release contains **no 3D code of any kind**, for either task, while the paper reports
3D segmentation results and eight 3D classification models. Verifiable in one command:

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/smallboy-code/Breast-cancer-dataset && \
  cd Breast-cancer-dataset && git ls-tree -r --name-only HEAD
```

### Where the paper and the released code disagree

| Setting | Paper | Released code |
|---|---|---|
| Classification input size | 96 × 96 | 224 × 224, RGB `ImageFolder` |
| Segmentation optimizer | Adam | SGD with momentum |
| Split | 7 : 1 : 2 | `--split_train_ratio 0.8` (README) |
| ViT epochs | ≤ 100 | `--epochs 20` (README) |
| Models covered | 18 classification models | README lists 5 |

Where these conflict, this reproduction follows the **paper**, and says so per setting in
`results/per_model_config.csv`.

A pattern worth noting: fidelity here was high on every **numerically specified** setting
and diverged only on the one described in **prose** ("flipping, scaling, and clipping").
Numeric specifications transmit; prose specifications get reinterpreted.

### Preprocessing

The segmentation augmentation branch is gated on `if ALBUMENTATIONS_AVAILABLE:` — an import
try/except, not a flag — and the two paths differ in input normalisation by 127.5×. Nothing
in the saved configuration records which one ran, so it was settled empirically: forcing the
import to fail collapses DSC to 2.27 % against 70.81 % with it available, so training went
through the albumentations path and the segmentation models were trained on inputs in
[-0.0039, 0.0039].

Note that `val.py` hard-codes `transform=None` and therefore uses a *different*
preprocessing. Table 1 came from `inference.py`. Do not check Table 1 with `val.py`.

### Configuration is not uniform

Read from the reported checkpoints: FCN-50, FCN-101, DeepLabV3-50/-101 and UNeXt use
lr 0.01 / 100 epochs / patience 10; **Unet** uses lr 0.0005 / 200 / 10; **Unet-VGG16** uses
lr 0.01 / 300 / patience 15. The epoch budgets never bind — early stopping fired in every
run — so what differs in effect is Unet's learning rate and Unet-VGG16's patience.

Segmentation checkpoints serialise their full training configuration; classification
checkpoints are bare `state_dict`s and carry none, so that half of
`results/per_model_config.csv` is reconstructed from run directories and script defaults
rather than read from the artifacts.

---

## Repository layout

```
├── verification/          the scripts the paper's claims rest on
│   ├── detect_duplication.py
│   ├── audit_joins.py
│   └── fix_vit_join.py
├── results/
│   ├── per_model_config.csv           per-model configuration
│   ├── operating_points_dedup.csv     Youden thresholds, sens/spec/acc, all models
│   ├── lgcafn_true96_seeds.csv        the three-seed LG-CAFN retrain
│   ├── per_image_*.csv                per-image DSC / IoU distributions
│   └── foreground_iou_summary.csv     foreground-only IoU alongside mIoU
├── figures/               qualitative segmentation and Grad-CAM outputs
├── classification/
└── segmentation/
```

---

## Known limitations, stated rather than omitted

- **Single dataset.** No external validation cohort. Every conclusion here is about
  BreastDM.
- **Metrics are sample-level.** The test split is patient-disjoint, but each case
  contributes multiple ROI samples and AUC is computed over samples. Bootstrapping over
  patients accounts for within-patient dependence; it does **not** convert the metric to a
  patient-level AUC. The sample-level metric is inherited from the original benchmark for
  comparability, not chosen.
- **Two decision rules appear in the submitted version.** Reported sample-level metrics use
  validation-selected Youden thresholds, while a case-level analysis used a plain 0.5. That
  analysis has been withdrawn along with the fusion, but the inconsistency is recorded here.
- **ViT checkpoint selection was not a rule.** Across the four ViT-7 checkpoints, Exp-1 uses
  one from the default configuration and Exp-2 one from a later re-run; the de facto rule
  was "use the newer batch unless it is much worse", which was never written down. Training
  histories exist only for the later runs.
- **Training code is being prepared** and will be added here. The verification and
  evaluation scripts, which the paper's claims depend on, are complete.

---

## Citation

```bibtex
@inproceedings{yu2026breastdm,
  title     = {Replication and Reproducibility Analysis of Breast Tumor
               Classification and Segmentation Benchmarks on the BreastDM Dataset},
  author    = {Yu, Shi},
  booktitle = {MICCAI 2026 Workshops (MSB EMERGE)},
  year      = {2026}
}
```

<!-- Update booktitle, pages and DOI once the proceedings are published. -->

## Contact

Shi Yu — `syu25@cougarnet.uh.edu`

## Acknowledgements

This work was carried out under the supervision of Prof. Lei Li (Rice University).
Experiments were run on a shared GPU server.
