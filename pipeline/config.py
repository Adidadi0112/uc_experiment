"""Central configuration — all constants live here."""

TARGET = "mayo"
RAW_PATH = "data/raw/uc_diagnostic_tests.csv"
RESULTS_DIR = "results"

# --- Target variants ---
TARGET_4CLASS = "mayo_4class"
TARGET_BINARY = "severity_binary"
TARGET_VARIANTS = [TARGET_4CLASS, TARGET_BINARY]
BINARY_LOW_CLASSES = [0, 1]
BINARY_HIGH_CLASSES = [2, 3]

# --- Feature/data variants ---
FEATURE_VARIANTS = [
    "all",
    "missing_indicators",
    "drop_gt50_missing",
    "drop_gt70_missing",
    "selected",
]
FOCUSED_FEATURE_VARIANTS = [
    "all",
    "missing_indicators",
    "drop_gt50_missing",
    "selected",
]
SELECTED_FEATURE_COUNT = 12
MISSING_DROP_THRESHOLDS = {
    "drop_gt50_missing": 0.50,
    "drop_gt70_missing": 0.70,
}
OUTLIER_POLICIES = ["none", "winsorize"]
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99

# --- Method registries ---
IMPUTATION_METHODS = ["mean", "mice", "knn", "softimpute", "gain", "pmm"]
FOCUSED_IMPUTATION_METHODS = ["mean", "mice", "gain"]
BASELINE_IMPUTATION = "mean"
RAW_IMPUTATION = "raw"

SINGLE_SYNTH = ["random_over", "ctgan", "tvae", "smote", "adasyn"]
MIXED_SYNTH = ["smote_ctgan", "smote_tvae", "adasyn_ctgan", "adasyn_tvae"]
ALL_SYNTH = ["none"] + SINGLE_SYNTH + MIXED_SYNTH
FOCUSED_SYNTH = ["none", "random_over", "smote", "adasyn"]
GAN_SYNTH = ["ctgan", "tvae", "smote_ctgan", "smote_tvae", "adasyn_ctgan", "adasyn_tvae"]

CLASSIFIERS = [
    "rf",
    "catboost",
    "stacking",
]
FOCUSED_CLASSIFIERS = [
    "rf",
    "catboost",
    "stacking",
]
TUNED_CLASSIFIERS = ["rf_tuned", "catboost_tuned"]
RAW_CLASSIFIERS = ["catboost", "catboost_tuned"]

# --- Cross-validation ---
CV_REPEATS = 5
CV_FOLDS = 2
MODEL_SELECTION_CV_REPEATS = 2
MODEL_SELECTION_CV_FOLDS = 5
RANDOM_SEED = 42
SHORTLIST_TOP_N = 12
EXPLORATORY_TEST_SIZE = 0.2

# --- Hyperparameters ---
CTGAN_EPOCHS = 150
TVAE_EPOCHS = 150
GAN_FAST_MODE = False
GAIN_ITERATIONS = 5000
GAIN_BATCH_SIZE = 32
GAIN_HINT_RATE = 0.9
GAIN_ALPHA = 100
RF_ESTIMATORS = 100
CB_ITERATIONS = 500
TUNING_CV_FOLDS = 3
TUNING_N_ITER = 8
