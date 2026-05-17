# UC Experiment — Synthetic Data Augmentation for Ulcerative Colitis Severity Prediction

This project investigates how **missing-value imputation** and **synthetic data augmentation** affect the classification of Ulcerative Colitis severity (Mayo endoscopic score 0–3) from routine laboratory blood tests.

## Dataset

- **251 patients**, **55 clinical lab features** (blood morphology, inflammatory markers, liver/kidney panels, lipid panels)
- Target: `mayo` score (0 = remission, 1 = mild, 2 = moderate, 3 = severe)
- Source: `data/raw/uc_diagnostic_tests.csv` (European decimal format)

## Pipeline Overview

```
Raw Data (N=251, many NaNs)
    │
    ▼
5 Imputation Methods ──► 9 Synthesis Strategies ──► 3 Classifiers
(MICE, KNN, SoftImpute,   (none, SMOTE, ADASYN,     (Random Forest,
 GAIN, PMM)                CTGAN, TVAE,               CatBoost,
                           SMOTE→CTGAN, SMOTE→TVAE,   Stacking)
                           ADASYN→CTGAN, ADASYN→TVAE)
    │
    ▼
5×2 Cross-Validation (10 folds)
    │
    ▼
Statistical Testing (Dietterich's 5×2 CV paired t-test + Holm-Bonferroni)
```

**Total configurations evaluated:** 138 (5 imputation × 9 synthesis × 3 classifiers + 3 baseline) × 10 folds = 1,380 model trainings.

## Project Structure

```
uc_experiment/
├── pipeline/                 # Refactored experiment code
│   ├── config.py             # Constants, hyperparameters, method registries
│   ├── imputers.py           # 6 imputation methods (fit/transform API)
│   ├── synthesizers.py       # 9 synthesis strategies (single + mixed)
│   ├── classifiers.py        # Classifier factory (RF, CatBoost, Stacking)
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

> **Note:** The full experiment takes approximately **5–8 hours** due to CTGAN/TVAE training (150 epochs × ~300 GAN runs across all folds). Plan to run overnight.

### Outputs

All results are saved to the `results/` directory:

| File | Description |
|---|---|
| `all_fold_results.csv` | Raw per-fold scores (1,380 rows) |
| `summary.csv` | Mean ± std per configuration (138 rows) |
| `statistical_tests.csv` | Paired t-test results with Holm-Bonferroni correction |
| `imputation_impact.png` | Bar chart: imputation methods vs balanced accuracy |
| `synthesis_impact.png` | Bar chart: synthesis methods vs balanced accuracy |
| `heatmap_*.png` | Imputation × synthesis heatmap per classifier |
| `best_confusion_matrix.png` | Confusion matrix for the top configuration |
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
| **SMOTE** | Oversampling | Synthetic Minority Over-sampling Technique |
| **ADASYN** | Oversampling | Adaptive Synthetic Sampling (borderline-focused) |
| **CTGAN** | Generative | Conditional Tabular GAN (150 epochs) |
| **TVAE** | Generative | Tabular Variational Autoencoder (150 epochs) |
| **SMOTE→CTGAN** | Mixed | Balance classes first, then generate from balanced distribution |
| **SMOTE→TVAE** | Mixed | Balance classes first, then generate from balanced distribution |
| **ADASYN→CTGAN** | Mixed | Adaptive balance, then generate |
| **ADASYN→TVAE** | Mixed | Adaptive balance, then generate |

### Evaluation
- **Cross-validation:** 5×2 CV (Dietterich, 1998) — 5 repetitions of 2-fold stratified CV
- **Primary metric:** Balanced Accuracy (handles class imbalance)
- **Statistical testing:** 5×2 CV paired t-test with Holm-Bonferroni correction for family-wise error rate control
