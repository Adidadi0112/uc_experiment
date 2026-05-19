"""Synthesis / augmentation strategies (single + mixed)."""

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
try:
    from sdv.sampling import Condition
except Exception:  # pragma: no cover - depends on installed SDV version
    Condition = None

from .config import TARGET, CTGAN_EPOCHS, TVAE_EPOCHS

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_series(y, name=TARGET):
    return pd.Series(y, name=name).reset_index(drop=True)


def _safe_k_neighbors(y, default=5):
    """Choose a neighbor count valid for the smallest class in this fold."""
    counts = pd.Series(y).value_counts()
    if counts.empty:
        return default
    min_count = int(counts.min())
    return max(1, min(default, min_count - 1))


def _resample_scaled(X, y, sampler_factory, random_state, fallback="random_over"):
    """Scale for neighbor geometry, resample, then inverse-transform features."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    try:
        sampler = sampler_factory(_safe_k_neighbors(y), random_state)
        X_res, y_res = sampler.fit_resample(X_scaled, y)
    except Exception:
        if fallback == "smote":
            sampler = SMOTE(
                sampling_strategy="not majority",
                k_neighbors=_safe_k_neighbors(y),
                random_state=random_state,
            )
        else:
            sampler = RandomOverSampler(
                sampling_strategy="not majority",
                random_state=random_state,
            )
        X_res, y_res = sampler.fit_resample(X_scaled, y)

    X_res = pd.DataFrame(
        scaler.inverse_transform(X_res),
        columns=X.columns,
    ).reset_index(drop=True)
    return (
        X_res,
        _as_series(y_res, name=y.name if hasattr(y, "name") else TARGET),
    )


def _generate_with_gan(X, y, synthesizer_cls, epochs, random_state):
    """
    Scale → train GAN → class-conditioned sample N rows → inverse-scale → concat.

    The target is explicitly categorical. For SDV versions that support
    conditions, synthetic rows are sampled per class using the training-fold
    class distribution. If the installed SDV version cannot condition, it falls
    back to unconditioned sampling and coerces target values to known classes.

    Returns (X_augmented, y_augmented).
    """
    np.random.seed(random_state)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    df_scaled = pd.concat(
        [X_scaled.reset_index(drop=True), _as_series(y).reset_index(drop=True)], axis=1
    )

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df_scaled)
    try:
        meta.update_column(column_name=TARGET, sdtype="categorical")
    except Exception:
        pass

    synth = synthesizer_cls(meta, epochs=epochs, verbose=False)
    synth.fit(df_scaled)
    synthetic = _sample_conditioned(synth, y, len(X))

    # Inverse-transform features only
    X_syn = pd.DataFrame(
        scaler.inverse_transform(synthetic.drop(columns=[TARGET])),
        columns=X.columns,
    )
    y_syn = _coerce_generated_target(synthetic[TARGET], y)

    X_aug = pd.concat([X.reset_index(drop=True), X_syn], ignore_index=True)
    y_aug = pd.concat([_as_series(y), y_syn], ignore_index=True)
    return X_aug, y_aug


def _sample_conditioned(synth, y, total_rows):
    counts = _as_series(y).value_counts().sort_index()
    if Condition is not None and hasattr(synth, "sample_from_conditions"):
        try:
            conditions = [
                Condition(num_rows=int(count), column_values={TARGET: label})
                for label, count in counts.items()
            ]
            sampled = synth.sample_from_conditions(conditions)
            if len(sampled) >= total_rows:
                return sampled.head(total_rows).reset_index(drop=True)
        except Exception:
            pass
    return synth.sample(num_rows=total_rows).reset_index(drop=True)


def _coerce_generated_target(values, y_reference):
    known = np.array(sorted(pd.Series(y_reference).astype(int).unique()))
    coerced = []
    for value in values:
        try:
            numeric = int(round(float(value)))
            coerced.append(int(known[np.argmin(np.abs(known - numeric))]))
        except Exception:
            if value in known:
                coerced.append(int(value))
            else:
                coerced.append(int(np.random.choice(known)))
    return pd.Series(coerced, name=TARGET).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Single methods
# ---------------------------------------------------------------------------

def _smote(X, y, rs):
    return _resample_scaled(
        X,
        y,
        lambda k, seed: SMOTE(
            sampling_strategy="not majority",
            k_neighbors=k,
            random_state=seed,
        ),
        rs,
    )


def _adasyn(X, y, rs):
    return _resample_scaled(
        X,
        y,
        lambda k, seed: ADASYN(
            sampling_strategy="not majority",
            n_neighbors=k,
            random_state=seed,
        ),
        rs,
        fallback="smote",
    )


def _random_over(X, y, rs):
    sampler = RandomOverSampler(sampling_strategy="not majority", random_state=rs)
    X_res, y_res = sampler.fit_resample(X, y)
    return (
        pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True),
        _as_series(y_res, name=y.name if hasattr(y, "name") else TARGET),
    )


def _ctgan(X, y, rs):
    return _generate_with_gan(X, y, CTGANSynthesizer, CTGAN_EPOCHS, rs)


def _tvae(X, y, rs):
    return _generate_with_gan(X, y, TVAESynthesizer, TVAE_EPOCHS, rs)


# ---------------------------------------------------------------------------
# Mixed pipelines: resampler → GAN
# ---------------------------------------------------------------------------

def _mixed(X, y, resample_fn, gan_fn, rs):
    """Chain: balance classes first, then generate from balanced distribution."""
    X_bal, y_bal = resample_fn(X, y, rs)
    X_aug, y_aug = gan_fn(X_bal, y_bal, rs)
    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    "none":         lambda X, y, rs: (X.copy(), y.copy()),
    "random_over":  _random_over,
    "smote":        _smote,
    "adasyn":       _adasyn,
    "ctgan":        _ctgan,
    "tvae":         _tvae,
    "smote_ctgan":  lambda X, y, rs: _mixed(X, y, _smote, _ctgan, rs),
    "smote_tvae":   lambda X, y, rs: _mixed(X, y, _smote, _tvae, rs),
    "adasyn_ctgan": lambda X, y, rs: _mixed(X, y, _adasyn, _ctgan, rs),
    "adasyn_tvae":  lambda X, y, rs: _mixed(X, y, _adasyn, _tvae, rs),
}


def augment(X_train: pd.DataFrame, y_train: pd.Series,
            method: str, random_state: int = 42):
    """
    Augment training data with the given synthesis method.

    Returns (X_augmented, y_augmented).
    """
    if method not in _DISPATCH:
        raise ValueError(f"Unknown synthesis method: {method}")
    return _DISPATCH[method](X_train, y_train, random_state)
