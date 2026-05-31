# UC Experiment — Synthetic Data Augmentation for Ulcerative Colitis Severity Prediction

This project investigates how **missing-value imputation** and **synthetic data augmentation** affect the classification of Ulcerative Colitis severity (Mayo endoscopic score 0–3) from routine laboratory blood tests.

## Dataset

- **252 patients**, **55 clinical lab features** (blood morphology, inflammatory markers, liver/kidney panels, lipid panels)
- Target: `mayo` score (0 = remission, 1 = mild, 2 = moderate, 3 = severe)
- Source: `data/raw/uc_diagnostic_tests.csv` (European decimal format)

## Pipeline Overview

```
Raw Data (N=252, many NaNs)
    │
    ▼
Target Variants ──► Feature Variants ──► Imputation ──► Synthesis ──► Classifiers
(4-class Mayo,       (all, missingness,    (mean, MICE,    (none, random,   (Random Forest,
 binary 0-1 vs 2-3)   missing-drop,         KNN, SoftImpute, SMOTE, ADASYN,   CatBoost,
                      fold-selected)       GAIN, PMM, raw) CTGAN/TVAE*)     Stacking)
    │
    ▼
Strict leakage-free CV + exploratory optimistic split-after-augmentation lane
    │
    ▼
Shortlist selection CV → final 5×2 CV → Dietterich paired tests
```

\* GAN synthesis and nested hyperparameter tuning are available through flags because they are expensive.

## Project Structure

```
uc_experiment/
├── pipeline/                 # Refactored experiment code
│   ├── config.py             # Constants, hyperparameters, method registries
│   ├── imputers.py           # 6 imputation methods (fit/transform API)
│   ├── preprocessing.py      # Target/feature variants and fold-local transforms
│   ├── synthesizers.py       # Synthesis strategies (single + mixed)
│   ├── classifiers.py        # Classifier factory
│   ├── stats.py              # 5×2 CV paired t-test, Holm-Bonferroni
│   ├── visualize.py          # All plotting functions
│   └── run.py                # Main experiment runner
├── notebooks/                # Legacy Jupyter notebooks (reference only)
├── data/raw/                 # Raw dataset
├── results/                  # Generated outputs (CSVs + plots)
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

### 1. Create a conda environment

```bash
conda create -n uc_experiment python=3.13 -y
conda activate uc_experiment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "from pipeline.config import ALL_SYNTH; print('OK:', ALL_SYNTH)"
```

## Running the Experiment

```bash
conda activate uc_experiment
python -m pipeline.run
```

The default run evaluates both strict and exploratory lanes for both targets using the focused profile. Useful alternatives:

```bash
# Strict 4-class only, focused profile
python -m pipeline.run --lane strict --target mayo_4class

# Binary severity target only
python -m pipeline.run --lane strict --target severity_binary

# Reproduce the older broad grid shape without the new raw-CatBoost lane
python -m pipeline.run --profile legacy --lane strict --target mayo_4class

# Include CTGAN/TVAE/mixed GAN synthesis in the focused profile
python -m pipeline.run --include-gans

# Include nested RF/CatBoost randomized-search models
python -m pipeline.run --include-tuned

# Fast GAN screening: direct TVAE only, fewer epochs, targeted grid
python -u -m pipeline.run --lane strict --target both --profile focused \
  --synth-methods none,tvae \
  --feature-variants all,drop_gt70_missing \
  --imputation-methods mean,pmm,gain \
  --models rf,catboost,stacking \
  --tvae-epochs 30 \
  2>&1 | tee run_gan_tvae_screen.log
```

> **Note:** `--include-gans` and `--include-tuned` can be very slow. Use them after the focused strict run identifies promising target/feature/imputation regions.

The normal model grid contains three classifier families: Random Forest, CatBoost, and a stacking ensemble combining both.

The grid can also be narrowed with `--feature-variants`, `--imputation-methods`, `--synth-methods`, and `--models`. GAN epoch counts can be reduced for screening with `--ctgan-epochs` and `--tvae-epochs`; final reported runs should state the epoch setting used.

### Outputs

All results are saved to the `results/` directory:

| File | Description |
|---|---|
| `strict_selection_results.csv` | Leakage-free model-selection CV fold scores |
| `strict_selection_summary.csv` | Selection CV aggregates used for shortlisting |
| `strict_final_results.csv` | Final 5×2 CV fold scores for shortlisted configs |
| `strict_final_summary.csv` | Final strict mean ± std per configuration |
| `exploratory_results.csv` | Optimistic notebook-style split-after-augmentation scores |
| `exploratory_summary.csv` | Optimistic lane aggregate scores |
| `all_fold_results.csv` | Compatibility alias for `strict_final_results.csv` |
| `summary.csv` | Compatibility alias for `strict_final_summary.csv` |
| `statistical_tests.csv` | Paired t-test results with Holm-Bonferroni correction |
| `imputation_impact_selection_*.png` | Target-specific selection-CV imputation impact charts |
| `synthesis_impact_selection_*.png` | Target-specific selection-CV synthesis impact charts |
| `top_strict_final_configs_*.png` | Compact target-specific charts for shortlisted final configs |
| `best_strict_final_confusion_matrix.png` | Confusion matrix for the top strict final configuration |
| `statistical_tests.png` | Significance plot for targeted comparisons |

## Methodology

### Imputation Methods
| Method | Description |
|---|---|
| **MICE** | Iterative multivariate imputation (20 iterations) |
| **KNN** | K-nearest neighbors (k=5) |
| **SoftImpute** | Low-rank matrix completion via SVD |
| **GAIN** | Generative Adversarial Imputation Nets (PyTorch) |
| **PMM** | Predictive Mean Matching |

### Synthesis Strategies
| Method | Type | Description |
|---|---|---|
| **none** | Baseline | No augmentation |
| **RandomOverSampler** | Oversampling | Duplicates minority-class rows as a robust class-balance baseline |
| **SMOTE** | Oversampling | Synthetic Minority Over-sampling Technique |
| **ADASYN** | Oversampling | Adaptive Synthetic Sampling (borderline-focused) |
| **CTGAN** | Generative | Conditional Tabular GAN (150 epochs) |
| **TVAE** | Generative | Tabular Variational Autoencoder (150 epochs) |
| **SMOTE→CTGAN** | Mixed | Balance classes first, then generate from balanced distribution |
| **SMOTE→TVAE** | Mixed | Balance classes first, then generate from balanced distribution |
| **ADASYN→CTGAN** | Mixed | Adaptive balance, then generate |
| **ADASYN→TVAE** | Mixed | Adaptive balance, then generate |

### Evaluation
- **Strict lane:** all preprocessing, imputation, synthesis, feature selection, and model fitting happen inside training folds; held-out folds contain real patients only.
- **Exploratory lane:** split-after-augmentation benchmark that mirrors the optimistic notebook protocol and is labeled separately.
- **Target variants:** 4-class Mayo `0/1/2/3` and grouped binary severity `0–1` vs `2–3`.
- **Cross-validation:** selection CV for shortlisting, then final 5×2 CV (Dietterich, 1998).
- **Primary metric:** Balanced Accuracy, with macro/weighted F1, per-class recall, ordinal error metrics, and binary ROC-AUC/PR-AUC where applicable.
- **Statistical testing:** 5×2 CV paired t-test with Holm-Bonferroni correction for family-wise error rate control
