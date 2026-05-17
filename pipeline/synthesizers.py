"""Synthesis / augmentation strategies (single + mixed)."""

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from .config import TARGET, CTGAN_EPOCHS, TVAE_EPOCHS

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(X, y, sampler):
    """Run an imblearn resampler, return (X_resampled, y_resampled) as DataFrames."""
    X_res, y_res = sampler.fit_resample(X, y)
    return (
        pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True),
        y_res.reset_index(drop=True),
    )


def _generate_with_gan(X, y, synthesizer_cls, epochs, random_state):
    """
    Scale → train GAN → sample N rows → inverse-scale → concat with original.
    Returns (X_augmented, y_augmented).
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    df_scaled = pd.concat(
        [X_scaled.reset_index(drop=True), y.reset_index(drop=True)], axis=1
    )

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df_scaled)

    synth = synthesizer_cls(meta, epochs=epochs, verbose=False)
    synth.fit(df_scaled)
    synthetic = synth.sample(num_rows=len(X))

    # Inverse-transform features only
    X_syn = pd.DataFrame(
        scaler.inverse_transform(synthetic.drop(columns=[TARGET])),
        columns=X.columns,
    )
    y_syn = synthetic[TARGET].reset_index(drop=True)

    X_aug = pd.concat([X.reset_index(drop=True), X_syn], ignore_index=True)
    y_aug = pd.concat([y.reset_index(drop=True), y_syn], ignore_index=True)
    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Single methods
# ---------------------------------------------------------------------------

def _smote(X, y, rs):
    return _resample(X, y, SMOTE(sampling_strategy="not majority", random_state=rs))


def _adasyn(X, y, rs):
    return _resample(X, y, ADASYN(sampling_strategy="not majority", random_state=rs))


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
