"""Leakage-aware target and feature preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from .config import (
    BINARY_HIGH_CLASSES,
    BINARY_LOW_CLASSES,
    MISSING_DROP_THRESHOLDS,
    SELECTED_FEATURE_COUNT,
    TARGET_BINARY,
    TARGET_4CLASS,
    WINSOR_LOWER_Q,
    WINSOR_UPPER_Q,
)


def make_target(y: pd.Series, variant: str) -> pd.Series:
    """Return the requested target representation."""
    y_int = y.astype(int)
    if variant == TARGET_4CLASS:
        return y_int
    if variant == TARGET_BINARY:
        low = set(BINARY_LOW_CLASSES)
        high = set(BINARY_HIGH_CLASSES)
        unknown = sorted(set(y_int.unique()) - low - high)
        if unknown:
            raise ValueError(f"Cannot map target values to binary classes: {unknown}")
        return y_int.map(lambda value: 0 if value in low else 1).astype(int)
    raise ValueError(f"Unknown target variant: {variant}")


def target_labels(variant: str) -> list[int]:
    """Stable label order for metrics and confusion matrices."""
    if variant == TARGET_4CLASS:
        return [0, 1, 2, 3]
    if variant == TARGET_BINARY:
        return [0, 1]
    raise ValueError(f"Unknown target variant: {variant}")


class FeaturePreprocessor:
    """Apply fold-local feature policies before imputation.

    The fitted object stores only training-fold decisions, so test-fold
    transform cannot peek at held-out missingness rates.
    """

    def __init__(self, variant: str):
        self.variant = variant

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePreprocessor":
        del y
        if self.variant in MISSING_DROP_THRESHOLDS:
            threshold = MISSING_DROP_THRESHOLDS[self.variant]
            missing_rate = X.isna().mean()
            self.columns_ = missing_rate[missing_rate <= threshold].index.tolist()
        else:
            self.columns_ = X.columns.tolist()

        if self.variant == "missing_indicators":
            self.indicator_columns_ = X.columns[X.isna().any()].tolist()
        else:
            self.indicator_columns_ = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.loc[:, self.columns_].copy()
        for col in self.indicator_columns_:
            out[f"{col}__missing"] = X[col].isna().astype(int)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class FoldFeatureSelector:
    """Select a compact numeric feature set inside each training fold."""

    def __init__(self, variant: str, k: int = SELECTED_FEATURE_COUNT):
        self.variant = variant
        self.k = k

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FoldFeatureSelector":
        self.input_columns_ = X.columns.tolist()
        if self.variant != "selected":
            self.selected_columns_ = self.input_columns_
            self.selector_ = None
            return self

        variances = X.var(axis=0, skipna=True)
        usable_columns = variances[variances.fillna(0) > 0].index.tolist()
        if not usable_columns:
            self.selected_columns_ = self.input_columns_
            self.selector_ = None
            return self

        k = min(self.k, len(usable_columns))
        self.selector_ = SelectKBest(score_func=f_classif, k=k)
        self.selector_.fit(X.loc[:, usable_columns], y)
        mask = self.selector_.get_support()
        self.selected_columns_ = [col for col, keep in zip(usable_columns, mask) if keep]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_columns_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class Winsorizer:
    """Clip numeric features using training-fold quantiles."""

    def __init__(self, lower_q: float = WINSOR_LOWER_Q, upper_q: float = WINSOR_UPPER_Q):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: pd.DataFrame) -> "Winsorizer":
        self.lower_ = X.quantile(self.lower_q, numeric_only=False)
        self.upper_ = X.quantile(self.upper_q, numeric_only=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def coerce_predictions(y_pred) -> np.ndarray:
    """Flatten sklearn/CatBoost prediction outputs into a 1-D integer array."""
    return np.asarray(y_pred).reshape(-1).astype(int)
