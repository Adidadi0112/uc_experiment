"""Visualisation — all plots consume the results DataFrame."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pathlib import Path

from .config import RESULTS_DIR

sns.set_theme(style="whitegrid")
_OUT = Path(RESULTS_DIR)


def _save(fig, name):
    fig.savefig(_OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_imputation_impact(
    df: pd.DataFrame,
    baseline: dict | None = None,
    suffix: str = "",
):
    """Bar chart: imputation × model with optional baseline lines and error bars."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=df, x="imputation", y="bal_acc_mean", hue="model",
        ci=None, ax=ax,
    )
    # Error bars from std
    _add_errorbars(ax, df, "imputation", "bal_acc_mean", "bal_acc_std", "model")

    if baseline:
        colors = {"rf": "C0", "catboost": "C1", "stacking": "C2"}
        for clf, score in baseline.items():
            ax.axhline(y=score, color=colors.get(clf, "gray"),
                       linestyle="--", linewidth=2, alpha=0.7, label=f"{clf} baseline")
    ax.set_title("Impact of Imputation on Balanced Accuracy (mean ± std)")
    ax.set_ylabel("Balanced Accuracy")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, f"imputation_impact{suffix}")


def plot_synthesis_impact(
    df: pd.DataFrame,
    baseline: dict | None = None,
    suffix: str = "",
):
    """Bar chart: synthesis × model with optional baseline lines and error bars."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(
        data=df, x="synthesis", y="bal_acc_mean", hue="model",
        ci=None, ax=ax,
    )
    _add_errorbars(ax, df, "synthesis", "bal_acc_mean", "bal_acc_std", "model")

    if baseline:
        colors = {"rf": "C0", "catboost": "C1", "stacking": "C2"}
        for clf, score in baseline.items():
            ax.axhline(y=score, color=colors.get(clf, "gray"),
                       linestyle="--", linewidth=2, alpha=0.7, label=f"{clf} baseline")
    ax.set_title("Impact of Synthesis Method on Balanced Accuracy (mean ± std)")
    ax.set_ylabel("Balanced Accuracy")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, f"synthesis_impact{suffix}")


def plot_heatmap(summary: pd.DataFrame, model_name: str, suffix: str = ""):
    """Imputation × synthesis heatmap for one classifier."""
    sub = summary[summary["model"] == model_name]
    if sub.empty:
        return
    # Build annotation matrix: "mean\n±std"
    mean_m = sub.pivot(index="imputation", columns="synthesis", values="bal_acc_mean")
    std_m = sub.pivot(index="imputation", columns="synthesis", values="bal_acc_std")
    annot = mean_m.round(3).astype(str) + "\n±" + std_m.round(3).astype(str)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(mean_m, annot=annot, fmt="", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{model_name} — Balanced Accuracy (mean ± std)")
    fig.tight_layout()
    _save(fig, f"heatmap_{model_name}{suffix}")


def plot_confusion_matrix(y_true, y_pred, title: str, name: str = "best_confusion_matrix"):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=sorted(set(y_true))).plot(
        cmap="Blues", ax=ax, values_format=".2f"
    )
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, name)


def plot_statistical_tests(test_results: list[dict]):
    """Horizontal bar chart of p-values with significance threshold."""
    if not test_results:
        return
    df = pd.DataFrame(test_results)
    df = df.sort_values("p_value")

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    colors = ["green" if s else "salmon" for s in df["significant"]]
    ax.barh(df["comparison"], -np.log10(df["p_value"] + 1e-15), color=colors)
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", label="α=0.05")
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Statistical Significance of Comparisons (Holm-Bonferroni)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "statistical_tests")


def plot_top_configs(
    summary: pd.DataFrame,
    target_variant: str,
    suffix: str = "",
    top_n: int = 15,
):
    """Horizontal bar chart for shortlisted final configs."""
    if summary.empty:
        return
    sub = summary[summary["target_variant"] == target_variant].copy()
    if sub.empty:
        return

    sub = sub.sort_values("balanced_accuracy_mean", ascending=False).head(top_n)
    sub = sub.sort_values("balanced_accuracy_mean", ascending=True)
    sub["label"] = (
        sub["feature_variant"].astype(str)
        + " | "
        + sub["imputation"].astype(str)
        + " | "
        + sub["synthesis"].astype(str)
        + " | "
        + sub["model"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(12, max(5, len(sub) * 0.45)))
    ax.barh(
        sub["label"],
        sub["balanced_accuracy_mean"],
        xerr=sub.get("balanced_accuracy_std"),
        color="C0",
        alpha=0.85,
        capsize=3,
    )
    ax.set_xlabel("Balanced Accuracy")
    ax.set_title(f"Top Strict Final Configurations — {target_variant}")
    ax.set_xlim(left=0, right=min(1.0, max(0.55, sub["balanced_accuracy_mean"].max() + 0.08)))
    fig.tight_layout()
    _save(fig, f"top_strict_final_configs{suffix}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _add_errorbars(ax, df, x_col, y_col, err_col, hue_col):
    """Overlay error bars onto a seaborn barplot (best-effort)."""
    try:
        hue_vals = df[hue_col].unique()
        x_vals = df[x_col].unique()
        n_hue = len(hue_vals)
        bar_width = 0.8 / n_hue  # seaborn default
        for i, hue in enumerate(hue_vals):
            sub = df[df[hue_col] == hue].set_index(x_col).reindex(x_vals)
            x_pos = np.arange(len(x_vals)) + (i - (n_hue - 1) / 2) * bar_width
            ax.errorbar(x_pos, sub[y_col], yerr=sub[err_col],
                        fmt="none", c="black", capsize=3, linewidth=1)
    except Exception:
        pass  # Non-critical; skip if layout mismatch
