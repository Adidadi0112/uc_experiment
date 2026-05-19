"""Experiment runner.

Usage examples:
    python -m pipeline.run
    python -m pipeline.run --lane strict --target mayo_4class --stage both
    python -m pipeline.run --profile full --include-gans --include-tuned
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from . import visualize as viz
from .classifiers import get_classifier
from .config import (
    ALL_SYNTH,
    BASELINE_IMPUTATION,
    CLASSIFIERS,
    CV_FOLDS,
    CV_REPEATS,
    EXPLORATORY_TEST_SIZE,
    FEATURE_VARIANTS,
    FOCUSED_CLASSIFIERS,
    FOCUSED_FEATURE_VARIANTS,
    FOCUSED_IMPUTATION_METHODS,
    FOCUSED_SYNTH,
    GAN_SYNTH,
    IMPUTATION_METHODS,
    MODEL_SELECTION_CV_FOLDS,
    MODEL_SELECTION_CV_REPEATS,
    RANDOM_SEED,
    RAW_CLASSIFIERS,
    RAW_IMPUTATION,
    RAW_PATH,
    RESULTS_DIR,
    SHORTLIST_TOP_N,
    TARGET,
    TARGET_4CLASS,
    TARGET_BINARY,
    TARGET_VARIANTS,
    TUNED_CLASSIFIERS,
)
from .imputers import get_imputer
from .preprocessing import (
    FeaturePreprocessor,
    FoldFeatureSelector,
    Winsorizer,
    coerce_predictions,
    make_target,
    target_labels,
)
from .stats import holm_bonferroni, paired_ttest_5x2cv
from .synthesizers import augment

warnings.filterwarnings("ignore")


CONFIG_COLUMNS = [
    "target_variant",
    "feature_variant",
    "outlier_policy",
    "imputation",
    "synthesis",
    "model",
]


@dataclass(frozen=True)
class ExperimentConfig:
    target_variant: str
    feature_variant: str
    outlier_policy: str
    imputation: str
    synthesis: str
    model: str

    @property
    def key(self) -> str:
        return "|".join(str(getattr(self, col)) for col in CONFIG_COLUMNS)


def main():
    args = _parse_args()
    out = Path(RESULTS_DIR)
    out.mkdir(exist_ok=True)
    _set_global_seed(RANDOM_SEED)

    df = pd.read_csv(RAW_PATH, decimal=",")
    print(f"Loaded {RAW_PATH}: {df.shape[0]} rows, {df.shape[1] - 1} features")

    target_variants = _select_target_variants(args.target)
    configs = build_configs(args, target_variants)
    print(
        f"Prepared {len(configs)} configs "
        f"({args.profile} profile, target={args.target}, lane={args.lane})"
    )

    strict_final_results = pd.DataFrame()
    strict_final_summary = pd.DataFrame()
    strict_final_predictions = pd.DataFrame()

    if args.lane in {"strict", "both"}:
        strict_final_results, strict_final_summary, strict_final_predictions = run_strict(
            df, configs, args, out
        )

    if args.lane in {"exploratory", "both"}:
        run_exploratory(df, configs, args, out)

    if not strict_final_summary.empty:
        _write_legacy_aliases(strict_final_results, strict_final_summary)
        test_results = _run_statistical_tests(strict_final_results, strict_final_summary)
        if test_results:
            pd.DataFrame(test_results).to_csv(out / "statistical_tests.csv", index=False)
            viz.plot_statistical_tests(test_results)
        _plot_final_outputs(strict_final_summary, strict_final_predictions)

    print("Done.")


# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="Run UC Mayo classification experiments.")
    parser.add_argument(
        "--lane",
        choices=["strict", "exploratory", "both"],
        default="both",
        help="strict is leakage-free CV; exploratory reproduces optimistic split-after-augmentation.",
    )
    parser.add_argument(
        "--target",
        choices=[TARGET_4CLASS, TARGET_BINARY, "both"],
        default="both",
        help="Target representation to evaluate.",
    )
    parser.add_argument(
        "--stage",
        choices=["selection", "final", "both"],
        default="both",
        help="Run model-selection CV, final 5x2 CV, or both.",
    )
    parser.add_argument(
        "--profile",
        choices=["focused", "full", "legacy"],
        default="focused",
        help="focused is the practical default; full restores the broad method grid.",
    )
    parser.add_argument(
        "--include-gans",
        action="store_true",
        help="Add CTGAN/TVAE and mixed GAN synthesis to focused profile.",
    )
    parser.add_argument(
        "--include-tuned",
        action="store_true",
        help="Add nested RF/CatBoost randomized-search estimators.",
    )
    parser.add_argument(
        "--winsorize",
        action="store_true",
        help="Add 1st/99th percentile clipping as the outlier policy.",
    )
    parser.add_argument(
        "--shortlist-n",
        type=int,
        default=SHORTLIST_TOP_N,
        help="Top configs per target carried from selection CV into final 5x2 CV.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Debug helper: cap configs after deterministic grid construction.",
    )
    return parser.parse_args()


def _select_target_variants(value: str) -> list[str]:
    if value == "both":
        return TARGET_VARIANTS
    return [value]


def build_configs(args, target_variants: list[str]) -> list[ExperimentConfig]:
    if args.profile == "legacy":
        feature_variants = ["all"]
        imputations = IMPUTATION_METHODS
        synths = [method for method in ALL_SYNTH if method != "random_over"]
        models = ["rf", "catboost", "stacking"]
    elif args.profile == "full":
        feature_variants = FEATURE_VARIANTS
        imputations = IMPUTATION_METHODS
        synths = ALL_SYNTH
        models = CLASSIFIERS.copy()
    else:
        feature_variants = FOCUSED_FEATURE_VARIANTS
        imputations = FOCUSED_IMPUTATION_METHODS
        synths = FOCUSED_SYNTH.copy()
        if args.include_gans:
            synths = _unique(synths + GAN_SYNTH)
        models = FOCUSED_CLASSIFIERS.copy()

    if args.include_tuned:
        models = _unique(models + TUNED_CLASSIFIERS)

    outlier_policies = ["none", "winsorize"] if args.winsorize else ["none"]
    configs: list[ExperimentConfig] = []

    for target_variant in target_variants:
        for feature_variant in feature_variants:
            for outlier_policy in outlier_policies:
                for imputation in imputations:
                    imputation_synths = ["none"] if imputation == BASELINE_IMPUTATION and args.profile == "legacy" else synths
                    for synthesis in imputation_synths:
                        for model in models:
                            configs.append(
                                ExperimentConfig(
                                    target_variant=target_variant,
                                    feature_variant=feature_variant,
                                    outlier_policy=outlier_policy,
                                    imputation=imputation,
                                    synthesis=synthesis,
                                    model=model,
                                )
                            )

                raw_models = RAW_CLASSIFIERS if args.include_tuned else RAW_CLASSIFIERS[:2]
                if args.profile != "legacy" and feature_variant != "selected":
                    for model in raw_models:
                        configs.append(
                            ExperimentConfig(
                                target_variant=target_variant,
                                feature_variant=feature_variant,
                                outlier_policy=outlier_policy,
                                imputation=RAW_IMPUTATION,
                                synthesis="none",
                                model=model,
                            )
                        )

    configs = _dedupe_configs(configs)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    return configs


def _unique(values):
    return list(dict.fromkeys(values))


def _dedupe_configs(configs):
    return list({cfg.key: cfg for cfg in configs}.values())


# ---------------------------------------------------------------------------
# Strict leakage-free lane
# ---------------------------------------------------------------------------


def run_strict(df, configs, args, out: Path):
    selection_summary = pd.DataFrame()
    final_configs = configs

    if args.stage in {"selection", "both"}:
        print("\n=== Strict lane: model-selection CV ===")
        selection_results, selection_predictions = _run_cv(
            df,
            configs,
            stage="selection",
            n_splits=MODEL_SELECTION_CV_FOLDS,
            n_repeats=MODEL_SELECTION_CV_REPEATS,
        )
        selection_results.to_csv(out / "strict_selection_results.csv", index=False)
        selection_predictions.to_csv(out / "strict_selection_predictions.csv", index=False)
        selection_summary = _summarize(selection_results)
        selection_summary.to_csv(out / "strict_selection_summary.csv", index=False)
        if args.stage == "both":
            final_configs = _shortlist(selection_summary, args.shortlist_n)
            print(f"Shortlisted {len(final_configs)} configs for final 5x2 CV")

    final_results = pd.DataFrame()
    final_summary = pd.DataFrame()
    final_predictions = pd.DataFrame()
    if args.stage in {"final", "both"}:
        print("\n=== Strict lane: final 5x2 CV ===")
        final_results, final_predictions = _run_cv(
            df,
            final_configs,
            stage="final_5x2",
            n_splits=CV_FOLDS,
            n_repeats=CV_REPEATS,
        )
        final_results.to_csv(out / "strict_final_results.csv", index=False)
        final_predictions.to_csv(out / "strict_final_predictions.csv", index=False)
        final_summary = _summarize(final_results)
        final_summary.to_csv(out / "strict_final_summary.csv", index=False)
        if not final_summary.empty:
            print("\nTop strict final configs:")
            print(
                final_summary.sort_values("balanced_accuracy_mean", ascending=False)
                .head(12)
                .to_string(index=False)
            )

    return final_results, final_summary, final_predictions


def _run_cv(df, configs, stage: str, n_splits: int, n_repeats: int):
    X_raw = df.drop(columns=[TARGET])
    results = []
    predictions = []
    t0 = time.time()
    run_count = 0
    total_runs = len(configs) * n_splits * n_repeats

    for target_variant in sorted({cfg.target_variant for cfg in configs}):
        y_raw = make_target(df[TARGET], target_variant)
        target_configs = [cfg for cfg in configs if cfg.target_variant == target_variant]
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=RANDOM_SEED,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_raw, y_raw)):
            X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
            X_test_raw = X_raw.iloc[test_idx].reset_index(drop=True)
            y_train = y_raw.iloc[train_idx].reset_index(drop=True)
            y_test = y_raw.iloc[test_idx].reset_index(drop=True)
            fold_cache = {"base": {}, "synth": {}}

            for cfg in target_configs:
                seed = RANDOM_SEED + fold_idx
                row, pred_rows = _evaluate_fold(
                    cfg,
                    X_train_raw,
                    X_test_raw,
                    y_train,
                    y_test,
                    test_idx,
                    fold_idx,
                    stage,
                    seed,
                    fold_cache,
                )
                results.append(row)
                predictions.extend(pred_rows)
                run_count += 1
                if run_count % 50 == 0:
                    elapsed = time.time() - t0
                    pct = run_count / max(total_runs, 1) * 100
                    print(f"  [{pct:5.1f}%] {run_count}/{total_runs} ({elapsed:.0f}s)")

    return pd.DataFrame(results), pd.DataFrame(predictions)


def _evaluate_fold(
    cfg: ExperimentConfig,
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    test_idx,
    fold_idx,
    stage,
    seed,
    fold_cache,
):
    _set_global_seed(seed)
    row = {
        "lane": "strict",
        "stage": stage,
        "fold": fold_idx,
        **asdict(cfg),
        "config_key": cfg.key,
        "status": "ok",
        "error": "",
    }
    pred_rows = []
    try:
        X_train, X_test, y_train_aug = _prepare_fold_data(
            cfg, X_train_raw, X_test_raw, y_train, seed, fold_cache
        )
        clf = get_classifier(cfg.model)
        clf.fit(X_train, y_train_aug)
        y_pred = coerce_predictions(clf.predict(X_test))
        y_score = _positive_class_score(clf, X_test)
        metrics = _score(y_test, y_pred, y_score, cfg.target_variant)
        row.update(
            {
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
                **metrics,
            }
        )
        for source_idx, true_value, pred_value, score in zip(
            test_idx, y_test, y_pred, _score_iter(y_score, len(y_pred))
        ):
            pred_rows.append(
                {
                    "lane": "strict",
                    "stage": stage,
                    "fold": fold_idx,
                    **asdict(cfg),
                    "config_key": cfg.key,
                    "source_index": int(source_idx),
                    "y_true": int(true_value),
                    "y_pred": int(pred_value),
                    "score_positive": score,
                }
            )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": repr(exc),
                "n_train": np.nan,
                "n_test": len(X_test_raw),
                "n_features": np.nan,
            }
        )
        row.update(_empty_metrics(cfg.target_variant))
    return row, pred_rows


def _prepare_fold_data(cfg, X_train_raw, X_test_raw, y_train, seed, fold_cache):
    base_key = (
        cfg.target_variant,
        cfg.feature_variant,
        cfg.outlier_policy,
        cfg.imputation,
    )
    synth_key = (*base_key, cfg.synthesis)

    if synth_key in fold_cache["synth"]:
        X_train, X_test, y_train_aug = fold_cache["synth"][synth_key]
        return X_train.copy(), X_test.copy(), y_train_aug.copy()

    if base_key in fold_cache["base"]:
        X_train, X_test, y_base = fold_cache["base"][base_key]
        X_train = X_train.copy()
        X_test = X_test.copy()
        y_base = y_base.copy()
    else:
        X_train, X_test, y_base = _prepare_base_fold_data(
            cfg, X_train_raw, X_test_raw, y_train, seed
        )
        fold_cache["base"][base_key] = (X_train.copy(), X_test.copy(), y_base.copy())

    if cfg.synthesis != "none":
        X_train, y_train_aug = augment(X_train, y_base, cfg.synthesis, random_state=seed)
    else:
        X_train = X_train.reset_index(drop=True)
        y_train_aug = y_base.reset_index(drop=True)

    fold_cache["synth"][synth_key] = (X_train.copy(), X_test.copy(), y_train_aug.copy())
    return X_train, X_test, y_train_aug


def _prepare_base_fold_data(cfg, X_train_raw, X_test_raw, y_train, seed):
    feature_preprocessor = FeaturePreprocessor(cfg.feature_variant)
    X_train = feature_preprocessor.fit_transform(X_train_raw, y_train)
    X_test = feature_preprocessor.transform(X_test_raw)

    if cfg.imputation != RAW_IMPUTATION:
        imputer = get_imputer(cfg.imputation, random_state=seed)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    selector = FoldFeatureSelector(cfg.feature_variant)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    if cfg.outlier_policy == "winsorize":
        winsorizer = Winsorizer()
        X_train = winsorizer.fit_transform(X_train)
        X_test = winsorizer.transform(X_test)

    return X_train, X_test, y_train


# ---------------------------------------------------------------------------
# Exploratory optimistic lane
# ---------------------------------------------------------------------------


def run_exploratory(df, configs, args, out: Path):
    print("\n=== Exploratory lane: split after preprocessing/augmentation (optimistic) ===")
    X_raw = df.drop(columns=[TARGET])
    rows = []
    predictions = []
    full_cache = {"base": {}, "synth": {}}

    for idx, cfg in enumerate(configs, start=1):
        _set_global_seed(RANDOM_SEED)
        y_raw = make_target(df[TARGET], cfg.target_variant)
        row = {
            "lane": "exploratory_optimistic",
            "stage": "single_split_after_augmentation",
            "fold": 0,
            **asdict(cfg),
            "config_key": cfg.key,
            "status": "ok",
            "error": "",
        }
        try:
            X_all, y_all = _prepare_full_data_optimistic(
                cfg, X_raw, y_raw, RANDOM_SEED, full_cache
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=EXPLORATORY_TEST_SIZE,
                random_state=RANDOM_SEED,
                stratify=y_all,
            )
            clf = get_classifier(cfg.model)
            clf.fit(X_train, y_train)
            y_pred = coerce_predictions(clf.predict(X_test))
            y_score = _positive_class_score(clf, X_test)
            row.update(
                {
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_features": X_train.shape[1],
                    **_score(y_test, y_pred, y_score, cfg.target_variant),
                }
            )
            for local_idx, true_value, pred_value, score in zip(
                X_test.index, y_test, y_pred, _score_iter(y_score, len(y_pred))
            ):
                predictions.append(
                    {
                        "lane": "exploratory_optimistic",
                        "stage": "single_split_after_augmentation",
                        "fold": 0,
                        **asdict(cfg),
                        "config_key": cfg.key,
                        "source_index": int(local_idx),
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                        "score_positive": score,
                    }
                )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "n_train": np.nan,
                    "n_test": np.nan,
                    "n_features": np.nan,
                    **_empty_metrics(cfg.target_variant),
                }
            )
        rows.append(row)
        if idx % 50 == 0:
            print(f"  exploratory {idx}/{len(configs)}")

    results = pd.DataFrame(rows)
    preds = pd.DataFrame(predictions)
    summary = _summarize(results)
    results.to_csv(out / "exploratory_results.csv", index=False)
    preds.to_csv(out / "exploratory_predictions.csv", index=False)
    summary.to_csv(out / "exploratory_summary.csv", index=False)
    if not summary.empty:
        print("\nTop exploratory optimistic configs:")
        print(
            summary.sort_values("balanced_accuracy_mean", ascending=False)
            .head(12)
            .to_string(index=False)
        )


def _prepare_full_data_optimistic(cfg, X_raw, y_raw, seed, full_cache):
    base_key = (
        cfg.target_variant,
        cfg.feature_variant,
        cfg.outlier_policy,
        cfg.imputation,
    )
    synth_key = (*base_key, cfg.synthesis)
    if synth_key in full_cache["synth"]:
        X_all, y_all = full_cache["synth"][synth_key]
        return X_all.copy(), y_all.copy()

    if base_key in full_cache["base"]:
        X_all, y_base = full_cache["base"][base_key]
        X_all = X_all.copy()
        y_base = y_base.copy()
    else:
        X_all, y_base = _prepare_base_full_data_optimistic(cfg, X_raw, y_raw, seed)
        full_cache["base"][base_key] = (X_all.copy(), y_base.copy())

    if cfg.synthesis != "none":
        X_all, y_all = augment(X_all, y_base, cfg.synthesis, random_state=seed)
    else:
        X_all = X_all.reset_index(drop=True)
        y_all = y_base.reset_index(drop=True)

    full_cache["synth"][synth_key] = (X_all.copy(), y_all.copy())
    return X_all, y_all


def _prepare_base_full_data_optimistic(cfg, X_raw, y_raw, seed):
    feature_preprocessor = FeaturePreprocessor(cfg.feature_variant)
    X_all = feature_preprocessor.fit_transform(X_raw, y_raw)

    if cfg.imputation != RAW_IMPUTATION:
        imputer = get_imputer(cfg.imputation, random_state=seed)
        X_all = imputer.fit_transform(X_all)

    selector = FoldFeatureSelector(cfg.feature_variant)
    X_all = selector.fit_transform(X_all, y_raw)

    if cfg.outlier_policy == "winsorize":
        X_all = Winsorizer().fit_transform(X_all)

    return X_all, y_raw


# ---------------------------------------------------------------------------
# Metrics, summaries, plots
# ---------------------------------------------------------------------------


def _score(y_true, y_pred, y_score, target_variant):
    labels = target_labels(target_variant)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, recalls, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "precision_weighted": prec_w,
        "precision_macro": prec_m,
        "recall_weighted": rec_w,
        "recall_macro": rec_m,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "ordinal_mae": np.nan,
        "adjacent_error_rate": np.nan,
        "distant_error_rate": np.nan,
    }
    for label, recall in zip(labels, recalls):
        metrics[f"recall_class_{label}"] = recall

    if target_variant == TARGET_BINARY:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = _safe_div(tp, tp + fn)
        metrics["specificity"] = _safe_div(tn, tn + fp)
        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            except ValueError:
                pass
            try:
                metrics["pr_auc"] = average_precision_score(y_true, y_score)
            except ValueError:
                pass
    else:
        diff = np.abs(np.asarray(y_true, dtype=int) - np.asarray(y_pred, dtype=int))
        metrics["ordinal_mae"] = float(diff.mean())
        metrics["adjacent_error_rate"] = float(np.mean(diff == 1))
        metrics["distant_error_rate"] = float(np.mean(diff > 1))

    return metrics


def _empty_metrics(target_variant):
    return {key: np.nan for key in _score(
        pd.Series(target_labels(target_variant)),
        np.asarray(target_labels(target_variant)),
        None,
        target_variant,
    )}


def _positive_class_score(clf, X):
    if not hasattr(clf, "predict_proba"):
        return None
    try:
        proba = np.asarray(clf.predict_proba(X))
    except Exception:
        return None
    if proba.ndim == 1:
        return proba
    if proba.shape[1] < 2:
        return None
    classes = getattr(clf, "classes_", None)
    if classes is not None and 1 in list(classes):
        return proba[:, list(classes).index(1)]
    return proba[:, -1]


def _score_iter(y_score, n):
    if y_score is None:
        return [np.nan] * n
    return [float(value) for value in np.asarray(y_score).reshape(-1)]


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    metric_cols = [
        "balanced_accuracy",
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "precision_weighted",
        "precision_macro",
        "recall_weighted",
        "recall_macro",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "ordinal_mae",
        "adjacent_error_rate",
        "distant_error_rate",
        "n_train",
        "n_features",
    ]
    metric_cols += [col for col in results.columns if col.startswith("recall_class_")]
    metric_cols = [col for col in _unique(metric_cols) if col in results.columns]

    agg_spec = {
        "n_rows": ("status", "size"),
        "n_success": ("status", lambda s: int((s == "ok").sum())),
        "n_folds": ("fold", "nunique"),
    }
    for col in metric_cols:
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_std"] = (col, "std")

    return (
        results.groupby(["lane", "stage", *CONFIG_COLUMNS], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values("balanced_accuracy_mean", ascending=False)
    )


def _shortlist(summary: pd.DataFrame, top_n: int) -> list[ExperimentConfig]:
    if summary.empty:
        return []
    eligible = summary[summary["n_success"] > 0].copy()
    selected = []
    for target_variant, sub in eligible.groupby("target_variant"):
        top = sub.sort_values("balanced_accuracy_mean", ascending=False).head(top_n)
        baseline = sub[
            (sub["imputation"] == BASELINE_IMPUTATION)
            & (sub["synthesis"] == "none")
            & (sub["feature_variant"] == "all")
            & (sub["outlier_policy"] == "none")
        ]
        selected.append(pd.concat([top, baseline], ignore_index=True))
    if not selected:
        return []
    selected_df = pd.concat(selected, ignore_index=True).drop_duplicates(CONFIG_COLUMNS)
    return _configs_from_frame(selected_df)


def _configs_from_frame(df: pd.DataFrame) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(**{col: row[col] for col in CONFIG_COLUMNS})
        for _, row in df.iterrows()
    ]


def _run_statistical_tests(results_df, summary):
    n_folds = CV_FOLDS * CV_REPEATS
    comparisons = []
    if results_df.empty or summary.empty:
        return comparisons

    for target_variant, sub in summary.groupby("target_variant"):
        sub = sub[sub["n_success"] == n_folds]
        if sub.empty:
            continue
        best = sub.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
        baselines = sub[
            (sub["imputation"] == BASELINE_IMPUTATION)
            & (sub["synthesis"] == "none")
            & (sub["feature_variant"] == "all")
            & (sub["outlier_policy"] == "none")
        ].sort_values("balanced_accuracy_mean", ascending=False)
        if baselines.empty:
            continue

        best_scores = _scores_for_config(results_df, best)
        base_scores = _scores_for_config(results_df, baselines.iloc[0])
        if len(best_scores) == n_folds and len(base_scores) == n_folds:
            _add_pair(
                comparisons,
                f"{target_variant}: best overall vs best baseline",
                best_scores,
                base_scores,
            )

        same_model = baselines[baselines["model"] == best["model"]]
        if not same_model.empty:
            same_model_scores = _scores_for_config(results_df, same_model.iloc[0])
            if len(same_model_scores) == n_folds:
                _add_pair(
                    comparisons,
                    f"{target_variant}: best vs same-model baseline",
                    best_scores,
                    same_model_scores,
                )

    return holm_bonferroni(comparisons) if comparisons else []


def _scores_for_config(results_df, config_row):
    mask = pd.Series(True, index=results_df.index)
    for col in CONFIG_COLUMNS:
        mask &= results_df[col] == config_row[col]
    mask &= results_df["status"] == "ok"
    return results_df.loc[mask].sort_values("fold")["balanced_accuracy"].values


def _add_pair(comparisons, label, scores_a, scores_b):
    t_stat, p_value = paired_ttest_5x2cv(scores_a, scores_b)
    comparisons.append(
        {"comparison": label, "t_stat": round(t_stat, 4), "p_value": round(p_value, 6)}
    )


def _plot_final_outputs(summary, predictions):
    if summary.empty:
        return

    plot_ready = summary.rename(
        columns={
            "balanced_accuracy_mean": "bal_acc_mean",
            "balanced_accuracy_std": "bal_acc_std",
            "f1_weighted_mean": "f1_mean",
            "f1_weighted_std": "f1_std",
            "n_train_mean": "n_train_mean",
        }
    )

    for target_variant in plot_ready["target_variant"].unique():
        sub = plot_ready[
            (plot_ready["target_variant"] == target_variant)
            & (plot_ready["feature_variant"] == "all")
            & (plot_ready["outlier_policy"] == "none")
        ].copy()
        if sub.empty:
            continue
        baseline = sub[
            (sub["imputation"] == BASELINE_IMPUTATION)
            & (sub["synthesis"] == "none")
        ]
        baseline_dict = dict(zip(baseline["model"], baseline["bal_acc_mean"]))
        viz.plot_imputation_impact(
            sub[sub["imputation"] != RAW_IMPUTATION],
            baseline_dict,
            suffix=f"_{target_variant}",
        )
        viz.plot_synthesis_impact(
            sub[sub["imputation"] != RAW_IMPUTATION],
            baseline_dict,
            suffix=f"_{target_variant}",
        )

    best = summary.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
    if predictions.empty:
        return
    pred_mask = pd.Series(True, index=predictions.index)
    for col in CONFIG_COLUMNS:
        pred_mask &= predictions[col] == best[col]
    best_preds = predictions[pred_mask]
    if not best_preds.empty:
        viz.plot_confusion_matrix(
            best_preds["y_true"],
            best_preds["y_pred"],
            title=f"Best strict final: {best['target_variant']} {best['model']}",
            name="best_strict_final_confusion_matrix",
        )


def _write_legacy_aliases(results_df, summary_df):
    out = Path(RESULTS_DIR)
    results_df.to_csv(out / "all_fold_results.csv", index=False)
    summary_df.to_csv(out / "summary.csv", index=False)


def _safe_div(num, den):
    return float(num / den) if den else np.nan


def _set_global_seed(seed: int):
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


if __name__ == "__main__":
    main()
