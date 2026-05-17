"""Classifier factory."""

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

from .config import RF_ESTIMATORS, CB_ITERATIONS, RANDOM_SEED


def get_classifier(name: str):
    """Return a fresh (unfitted) classifier instance."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, random_state=RANDOM_SEED
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=CB_ITERATIONS, random_seed=RANDOM_SEED, verbose=0
        )
    if name == "stacking":
        return StackingClassifier(
            estimators=[
                ("rf", get_classifier("rf")),
                ("cb", get_classifier("catboost")),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
        )
    raise ValueError(f"Unknown classifier: {name}")
