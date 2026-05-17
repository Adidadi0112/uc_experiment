"""Central configuration — all constants live here."""

TARGET = "mayo"
RAW_PATH = "data/raw/uc_diagnostic_tests.csv"
RESULTS_DIR = "results"

# --- Method registries ---
IMPUTATION_METHODS = ["mice", "knn", "softimpute", "gain", "pmm"]
BASELINE_IMPUTATION = "mean"

SINGLE_SYNTH = ["ctgan", "tvae", "smote", "adasyn"]
MIXED_SYNTH = ["smote_ctgan", "smote_tvae", "adasyn_ctgan", "adasyn_tvae"]
ALL_SYNTH = ["none"] + SINGLE_SYNTH + MIXED_SYNTH  # 9 total

CLASSIFIERS = ["rf", "catboost", "stacking"]

# --- Cross-validation ---
CV_REPEATS = 5
CV_FOLDS = 2
RANDOM_SEED = 42

# --- Hyperparameters ---
CTGAN_EPOCHS = 150
TVAE_EPOCHS = 150
GAIN_ITERATIONS = 5000
GAIN_BATCH_SIZE = 32
GAIN_HINT_RATE = 0.9
GAIN_ALPHA = 100
RF_ESTIMATORS = 100
CB_ITERATIONS = 500
