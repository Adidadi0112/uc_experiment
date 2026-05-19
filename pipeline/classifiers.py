"""Classifier factory."""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from catboost import CatBoostClassifier
try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
except Exception:  # pragma: no cover - optional dependency detail
    BalancedRandomForestClassifier = None
    EasyEnsembleClassifier = None

from .config import (
    CB_ITERATIONS,
    RF_ESTIMATORS,
    RANDOM_SEED,
    TUNING_CV_FOLDS,
    TUNING_N_ITER,
)


def get_classifier(name: str):
    """Return a fresh (unfitted) classifier instance."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, random_state=RANDOM_SEED
        )
    if name == "rf_balanced":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if name == "balanced_rf":
        if BalancedRandomForestClassifier is None:
            return get_classifier("rf_balanced")
        return BalancedRandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if name == "easy_ensemble":
        if EasyEnsembleClassifier is None:
            return get_classifier("rf_balanced")
        return EasyEnsembleClassifier(
            n_estimators=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if name == "catboost":
        return _catboost()
    if name in {"catboost_balanced", "catboost_raw"}:
        return _catboost(auto_class_weights="Balanced")
    if name == "rf_tuned":
        return _tuned_random_forest()
    if name == "catboost_tuned":
        return _tuned_catboost()
    if name == "stacking":
        return StackingClassifier(
            estimators=[
                ("rf", get_classifier("rf_balanced")),
                ("cb", get_classifier("catboost_balanced")),
            ],
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
            cv=3,
        )
    raise ValueError(f"Unknown classifier: {name}")


def _catboost(**kwargs):
    params = {
        "iterations": CB_ITERATIONS,
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }
    params.update(kwargs)
    return CatBoostClassifier(**params)


def _tuning_cv():
    return StratifiedKFold(
        n_splits=TUNING_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def _tuned_random_forest():
    base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    params = {
        "n_estimators": [200, 400, 700],
        "max_depth": [None, 3, 5, 8, 12],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, None],
    }
    return RandomizedSearchCV(
        base,
        params,
        n_iter=TUNING_N_ITER,
        scoring="balanced_accuracy",
        cv=_tuning_cv(),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
    )


def _tuned_catboost():
    base = _catboost(auto_class_weights="Balanced")
    params = {
        "depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 10, 20],
        "iterations": [200, 400, 700],
    }
    return RandomizedSearchCV(
        base,
        params,
        n_iter=TUNING_N_ITER,
        scoring="balanced_accuracy",
        cv=_tuning_cv(),
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=True,
    )
