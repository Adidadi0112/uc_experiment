"""Imputation methods with fit/transform interface for proper CV."""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd
import torch
import torch.nn as nn
import torch.optim as optim

from .config import GAIN_ITERATIONS, GAIN_BATCH_SIZE, GAIN_HINT_RATE, GAIN_ALPHA


# ---------------------------------------------------------------------------
# Sklearn-based imputers (already have fit/transform)
# ---------------------------------------------------------------------------

class _SklearnImputer:
    """Thin wrapper to return DataFrames from sklearn imputers."""

    def __init__(self, imputer):
        self._imp = imputer

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._imp.fit_transform(X), columns=X.columns, index=X.index
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._imp.transform(X), columns=X.columns, index=X.index
        )


# ---------------------------------------------------------------------------
# SoftImpute — self-contained numpy/scipy implementation, no fancyimpute dep
# ---------------------------------------------------------------------------

def _soft_impute(X_nan: np.ndarray, max_rank: int = 10, max_iter: int = 100,
                 tol: float = 1e-4, lambda_: float = 0.0) -> np.ndarray:
    """
    SoftImpute via iterative SVD soft-thresholding.
    Fills NaNs with column means initially, then iterates.
    lambda_=0 gives truncated SVD imputation (rank-constrained).
    """
    mask = np.isnan(X_nan)
    Z = X_nan.copy()
    # Initialise missing values with column means
    col_means = np.nanmean(Z, axis=0)
    for j in range(Z.shape[1]):
        Z[mask[:, j], j] = col_means[j]

    for _ in range(max_iter):
        Z_old = Z.copy()
        U, s, Vt = svd(Z, full_matrices=False)
        # Soft-threshold singular values; with lambda_=0 just truncate to max_rank
        s_thresh = np.maximum(s - lambda_, 0)[:max_rank]
        Z_new = (U[:, :max_rank] * s_thresh) @ Vt[:max_rank, :]
        # Only update previously-missing positions
        Z[mask] = Z_new[mask]
        if np.linalg.norm(Z - Z_old) / (np.linalg.norm(Z_old) + 1e-8) < tol:
            break
    return Z


class _SoftImputeWrapper:
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._cols = X.columns
        self._n_train = len(X)
        self._train_means = X.mean()
        self._train_completed = _soft_impute(X.values.astype(float))
        return pd.DataFrame(self._train_completed.copy(), columns=self._cols, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # SoftImpute is transductive by design. For held-out folds we avoid
        # peeking at test-feature distributions and use train-fold means.
        return X.fillna(self._train_means).loc[:, self._cols]


# ---------------------------------------------------------------------------
# GAIN — train GAN on train, forward-pass on test (Option 1)
# ---------------------------------------------------------------------------

class _GAINImputer:
    """GAIN (Generative Adversarial Imputation Nets) with fit/transform."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    class _G(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d * 2, d), nn.ReLU(),
                nn.Linear(d, d), nn.ReLU(),
                nn.Linear(d, d), nn.Sigmoid(),
            )

        def forward(self, x, m):
            return self.net(torch.cat([x, m], dim=1))

    class _D(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d * 2, d), nn.ReLU(),
                nn.Linear(d, d), nn.ReLU(),
                nn.Linear(d, d), nn.Sigmoid(),
            )

        def forward(self, x, h):
            return self.net(torch.cat([x, h], dim=1))

    # -- normalisation helpers --
    def _norm(self, data):
        out = data.copy()
        for i in range(out.shape[1]):
            out[:, i] = (out[:, i] - self._lo[i]) / (self._hi[i] + 1e-6)
        return out

    def _denorm(self, data):
        out = data.copy()
        for i in range(out.shape[1]):
            out[:, i] = out[:, i] * (self._hi[i] + 1e-6) + self._lo[i]
        return out

    def _impute_array(self, data_x):
        """Run trained Generator on a matrix with NaNs, return imputed matrix."""
        N, D = data_x.shape
        mask = 1.0 - np.isnan(data_x).astype(float)
        normed = self._norm(np.nan_to_num(data_x, nan=0.0))

        x_t = torch.tensor(normed, dtype=torch.float32)
        m_t = torch.tensor(mask, dtype=torch.float32)
        z = torch.rand(N, D) * 0.01
        x_in = m_t * x_t + (1 - m_t) * z

        with torch.no_grad():
            g_out = self._gen(x_in, m_t).numpy()

        imputed_norm = mask * normed + (1 - mask) * g_out
        return self._denorm(imputed_norm)

    # -- public API --
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        data_x = X.values.astype(float)
        N, D = data_x.shape
        mask = 1.0 - np.isnan(data_x).astype(float)

        # Per-column min/max (ignoring NaN)
        self._lo = np.nanmin(data_x, axis=0)
        rng = np.nanmax(data_x, axis=0) - self._lo
        self._hi = rng
        self._cols = X.columns

        normed = np.nan_to_num(self._norm(data_x), nan=0.0)
        x_t = torch.tensor(normed, dtype=torch.float32)
        m_t = torch.tensor(mask, dtype=torch.float32)

        gen = self._G(D)
        disc = self._D(D)
        opt_g = optim.Adam(gen.parameters())
        opt_d = optim.Adam(disc.parameters())

        bs = min(GAIN_BATCH_SIZE, N)
        for _ in range(GAIN_ITERATIONS):
            idx = np.random.choice(N, bs, replace=False)
            xb, mb = x_t[idx], m_t[idx]
            zb = torch.rand(bs, D) * 0.01
            xn = mb * xb + (1 - mb) * zb
            hb = (torch.rand(bs, D) > (1 - GAIN_HINT_RATE)).float()
            hb = mb * hb + 0.5 * (1 - hb)

            # Discriminator step
            opt_d.zero_grad()
            g_sample = gen(xn, mb)
            hat = mb * xb + (1 - mb) * g_sample
            d_prob = disc(hat.detach(), hb)
            d_loss = -torch.mean(
                mb * torch.log(d_prob + 1e-8) + (1 - mb) * torch.log(1 - d_prob + 1e-8)
            )
            d_loss.backward()
            opt_d.step()

            # Generator step
            opt_g.zero_grad()
            g_sample = gen(xn, mb)
            hat = mb * xb + (1 - mb) * g_sample
            d_prob = disc(hat, hb)
            g_loss = -torch.mean((1 - mb) * torch.log(d_prob + 1e-8))
            mse = torch.mean((mb * xb - mb * g_sample) ** 2) / torch.mean(mb)
            (g_loss + GAIN_ALPHA * mse).backward()
            opt_g.step()

        self._gen = gen.eval()
        imputed = self._impute_array(data_x)
        return pd.DataFrame(imputed, columns=self._cols, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        imputed = self._impute_array(X.values.astype(float))
        return pd.DataFrame(imputed, columns=self._cols, index=X.index)


# ---------------------------------------------------------------------------
# PMM — store regression models and observed pools
# ---------------------------------------------------------------------------

class _PMMImputer:
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._cols = X.columns
        self._means = X.mean()
        df = X.copy()
        filled = df.fillna(self._means)

        self._models, self._obs_vals, self._obs_preds = {}, {}, {}

        for col in self._cols:
            if X[col].isna().sum() == 0:
                continue
            nan_mask = X[col].isna()
            feats = filled.drop(columns=[col])
            target = X[col]

            model = LinearRegression()
            model.fit(feats[~nan_mask], target[~nan_mask])

            pred_obs = model.predict(feats[~nan_mask])
            pred_mis = model.predict(feats[nan_mask])

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(pred_obs.reshape(-1, 1))
            _, indices = nn.kneighbors(pred_mis.reshape(-1, 1))

            df.loc[nan_mask, col] = target[~nan_mask].iloc[indices.flatten()].values

            self._models[col] = model
            self._obs_vals[col] = target[~nan_mask].values
            self._obs_preds[col] = pred_obs

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        filled = df.fillna(self._means)

        for col in self._cols:
            if col not in self._models or X[col].isna().sum() == 0:
                continue
            nan_mask = X[col].isna()
            feats = filled.drop(columns=[col])
            pred_mis = self._models[col].predict(feats[nan_mask])

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self._obs_preds[col].reshape(-1, 1))
            _, indices = nn.kneighbors(pred_mis.reshape(-1, 1))

            df.loc[nan_mask, col] = self._obs_vals[col][indices.flatten()]
        return df


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_imputer(method: str, random_state: int = 42):
    """Return a fresh imputer with .fit_transform(X) / .transform(X) API."""
    if method == "mean":
        return _SklearnImputer(_with_keep_empty(SimpleImputer, strategy="mean"))
    if method == "mice":
        return _SklearnImputer(
            _with_keep_empty(IterativeImputer, max_iter=20, random_state=random_state)
        )
    if method == "knn":
        return _SklearnImputer(_with_keep_empty(KNNImputer, n_neighbors=5))
    if method == "softimpute":
        return _SoftImputeWrapper()
    if method == "gain":
        return _GAINImputer(random_state=random_state)
    if method == "pmm":
        return _PMMImputer()
    raise ValueError(f"Unknown imputation method: {method}")


def _with_keep_empty(cls, **kwargs):
    try:
        return cls(**kwargs, keep_empty_features=True)
    except TypeError:
        return cls(**kwargs)
