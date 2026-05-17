"""Main experiment runner — single entry point.

Usage:  python -m pipeline.run
"""

import os, time, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score,
    precision_recall_fscore_support,
)
from pathlib import Path

from .config import (
    TARGET, RAW_PATH, RESULTS_DIR,
    IMPUTATION_METHODS, BASELINE_IMPUTATION, ALL_SYNTH,
    CLASSIFIERS, CV_REPEATS, CV_FOLDS, RANDOM_SEED,
)
from .imputers import get_imputer
from .synthesizers import augment
from .classifiers import get_classifier
from .stats import paired_ttest_5x2cv, holm_bonferroni
from . import visualize as viz

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _score(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1,
        "precision_weighted": prec,
        "recall_weighted": rec,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out = Path(RESULTS_DIR)
    out.mkdir(exist_ok=True)

    # 1. Load raw data
    df = pd.read_csv(RAW_PATH, decimal=",")
    X_raw = df.drop(columns=[TARGET])
    y_raw = df[TARGET]
    print(f"Loaded {RAW_PATH}: {X_raw.shape[0]} rows, {X_raw.shape[1]} features")

    # 2. Set up 5×2 CV
    cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED
    )
    n_folds = CV_FOLDS * CV_REPEATS
    splits = list(cv.split(X_raw, y_raw))

    # Build config list: (imputation, synth_methods)
    configs = [
        (BASELINE_IMPUTATION, ["none"]),        # baseline: mean + none
    ] + [
        (imp, ALL_SYNTH) for imp in IMPUTATION_METHODS  # 5 × 9
    ]
    total_runs = sum(len(s) for _, s in configs) * len(CLASSIFIERS) * n_folds
    print(f"Total evaluations: {total_runs}")

    all_results = []
    run_count = 0
    t0 = time.time()

    # 3. Main loop
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_tr_raw = X_raw.iloc[train_idx].reset_index(drop=True)
        X_te_raw = X_raw.iloc[test_idx].reset_index(drop=True)
        y_tr = y_raw.iloc[train_idx].reset_index(drop=True)
        y_te = y_raw.iloc[test_idx].reset_index(drop=True)

        for imp_method, synth_list in configs:
            # Fit imputer on train, transform both
            imputer = get_imputer(imp_method)
            X_tr_imp = imputer.fit_transform(X_tr_raw)
            X_te_imp = imputer.transform(X_te_raw)

            for synth_method in synth_list:
                # Augment train only
                try:
                    X_tr_aug, y_tr_aug = augment(
                        X_tr_imp, y_tr, synth_method,
                        random_state=RANDOM_SEED + fold_idx,
                    )
                except Exception as e:
                    print(f"  [SKIP] {imp_method}+{synth_method} fold {fold_idx}: {e}")
                    continue

                for clf_name in CLASSIFIERS:
                    clf = get_classifier(clf_name)
                    try:
                        clf.fit(X_tr_aug, y_tr_aug)
                        y_pred = clf.predict(X_te_imp)
                    except Exception as e:
                        print(f"  [SKIP] {clf_name} on {imp_method}+{synth_method} fold {fold_idx}: {e}")
                        continue

                    row = {
                        "fold": fold_idx,
                        "imputation": imp_method,
                        "synthesis": synth_method,
                        "model": clf_name,
                        "n_train": len(X_tr_aug),
                        **_score(y_te, y_pred),
                    }
                    all_results.append(row)
                    run_count += 1

                    if run_count % 30 == 0:
                        elapsed = time.time() - t0
                        pct = run_count / total_runs * 100
                        print(f"  [{pct:5.1f}%] {run_count}/{total_runs}  "
                              f"({elapsed:.0f}s elapsed)")

    # 4. Save raw fold-level results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out / "all_fold_results.csv", index=False)
    print(f"\nSaved {len(results_df)} fold-level results → {out / 'all_fold_results.csv'}")

    # 5. Aggregate: mean ± std per configuration
    group_cols = ["imputation", "synthesis", "model"]
    agg = (
        results_df.groupby(group_cols)
        .agg(
            bal_acc_mean=("balanced_accuracy", "mean"),
            bal_acc_std=("balanced_accuracy", "std"),
            f1_mean=("f1_weighted", "mean"),
            f1_std=("f1_weighted", "std"),
            n_train_mean=("n_train", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(out / "summary.csv", index=False)
    print(f"Saved summary → {out / 'summary.csv'}")

    # 6. Print top 10
    top = agg.sort_values("bal_acc_mean", ascending=False).head(10)
    print("\n=== Top 10 Configurations ===")
    print(top.to_string(index=False))

    # 7. Statistical tests
    test_results = _run_statistical_tests(results_df, agg)
    if test_results:
        pd.DataFrame(test_results).to_csv(out / "statistical_tests.csv", index=False)
        print(f"\nSaved statistical tests → {out / 'statistical_tests.csv'}")

    # 8. Baseline scores for plot lines
    baseline_agg = agg[
        (agg["imputation"] == BASELINE_IMPUTATION) & (agg["synthesis"] == "none")
    ]
    baseline_dict = dict(zip(baseline_agg["model"], baseline_agg["bal_acc_mean"]))

    # 9. Plots (exclude baseline imputation from grouped charts)
    plot_df = agg[agg["imputation"] != BASELINE_IMPUTATION]
    viz.plot_imputation_impact(plot_df, baseline_dict)
    viz.plot_synthesis_impact(plot_df, baseline_dict)
    for clf in CLASSIFIERS:
        viz.plot_heatmap(plot_df, clf)
    viz.plot_statistical_tests(test_results)
    print(f"All plots saved to {out}/")

    elapsed_total = time.time() - t0
    print(f"\n✓ Done in {elapsed_total / 60:.1f} minutes")


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def _run_statistical_tests(results_df, summary):
    """Run targeted comparisons with Holm-Bonferroni correction."""
    n_folds = CV_FOLDS * CV_REPEATS

    def _get_scores(imp, syn, clf):
        """Extract the 10 fold-level balanced_accuracy scores for a config."""
        sub = results_df[
            (results_df["imputation"] == imp)
            & (results_df["synthesis"] == syn)
            & (results_df["model"] == clf)
        ].sort_values("fold")
        return sub["balanced_accuracy"].values

    # Find best configs
    top = summary.sort_values("bal_acc_mean", ascending=False)
    single_synths = top[top["synthesis"].isin(["ctgan", "tvae", "smote", "adasyn"])]
    mixed_synths = top[top["synthesis"].str.contains("_", na=False)]
    baseline = top[(top["imputation"] == BASELINE_IMPUTATION) & (top["synthesis"] == "none")]

    if single_synths.empty or mixed_synths.empty:
        print("Not enough results for statistical tests.")
        return []

    best_single = single_synths.iloc[0]
    best_mixed = mixed_synths.iloc[0]
    best_overall = top.iloc[0]

    comparisons = []

    # 1. Best single vs best mixed
    _add_comparison(comparisons, results_df, "Best single vs best mixed",
                    best_single, best_mixed, _get_scores, n_folds)

    # 2. smote_ctgan vs adasyn_ctgan (for best imputation+model combo)
    for clf in CLASSIFIERS:
        for imp in IMPUTATION_METHODS:
            s1 = _get_scores(imp, "smote_ctgan", clf)
            s2 = _get_scores(imp, "adasyn_ctgan", clf)
            if len(s1) == n_folds and len(s2) == n_folds:
                _add_pair(comparisons, f"smote_ctgan vs adasyn_ctgan ({imp}/{clf})",
                          s1, s2)
                break
        if comparisons and len(comparisons) >= 2:
            break

    # 3. Best overall vs baseline
    for _, b_row in baseline.iterrows():
        s_best = _get_scores(best_overall["imputation"], best_overall["synthesis"],
                             best_overall["model"])
        s_base = _get_scores(b_row["imputation"], b_row["synthesis"], b_row["model"])
        if len(s_best) == n_folds and len(s_base) == n_folds:
            _add_pair(comparisons, f"Best overall vs baseline ({b_row['model']})",
                      s_best, s_base)

    # 4. Top-4 vs baseline (mean across classifiers)
    top4 = top.head(4)
    for _, t_row in top4.iterrows():
        for _, b_row in baseline.iterrows():
            if t_row["model"] == b_row["model"]:
                s_t = _get_scores(t_row["imputation"], t_row["synthesis"], t_row["model"])
                s_b = _get_scores(b_row["imputation"], b_row["synthesis"], b_row["model"])
                if len(s_t) == n_folds and len(s_b) == n_folds:
                    lbl = (f"{t_row['imputation']}+{t_row['synthesis']}+{t_row['model']}"
                           f" vs baseline")
                    _add_pair(comparisons, lbl, s_t, s_b)

    if not comparisons:
        return []

    return holm_bonferroni(comparisons)


def _add_comparison(comps, results_df, label, row_a, row_b, get_fn, n_folds):
    s_a = get_fn(row_a["imputation"], row_a["synthesis"], row_a["model"])
    s_b = get_fn(row_b["imputation"], row_b["synthesis"], row_b["model"])
    if len(s_a) == n_folds and len(s_b) == n_folds:
        _add_pair(comps, label, s_a, s_b)


def _add_pair(comps, label, scores_a, scores_b):
    t, p = paired_ttest_5x2cv(scores_a, scores_b)
    comps.append({"comparison": label, "t_stat": round(t, 4), "p_value": round(p, 6)})


if __name__ == "__main__":
    main()
