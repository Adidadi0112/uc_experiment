"""Statistical testing: 5×2 CV paired t-test and Holm-Bonferroni correction."""

import numpy as np
from scipy import stats


def paired_ttest_5x2cv(scores_a: np.ndarray, scores_b: np.ndarray):
    """
    Dietterich's 5×2 CV paired t-test.

    Parameters
    ----------
    scores_a, scores_b : arrays of shape (10,) — 5 repeats × 2 folds.

    Returns
    -------
    t_stat, p_value
    """
    diffs = scores_a - scores_b
    # Estimate variance per repeat (pair of 2 folds)
    s_sq = []
    for i in range(5):
        d1, d2 = diffs[2 * i], diffs[2 * i + 1]
        mean_d = (d1 + d2) / 2.0
        s_sq.append((d1 - mean_d) ** 2 + (d2 - mean_d) ** 2)

    denom = np.sqrt((1.0 / 5.0) * sum(s_sq))
    if denom == 0:
        return 0.0, 1.0

    t_stat = diffs[0] / denom
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=5)
    return float(t_stat), float(p_value)


def holm_bonferroni(results: list[dict], alpha: float = 0.05) -> list[dict]:
    """
    Apply Holm-Bonferroni correction to a list of test results.

    Parameters
    ----------
    results : list of dicts, each must have 'comparison' and 'p_value' keys.
    alpha   : family-wise error rate.

    Returns
    -------
    Same list with added 'p_adjusted', 'significant', and 'rank' keys.
    """
    n = len(results)
    sorted_res = sorted(results, key=lambda r: r["p_value"])

    for rank, r in enumerate(sorted_res, start=1):
        adjusted_alpha = alpha / (n - rank + 1)
        r["rank"] = rank
        r["adjusted_alpha"] = round(adjusted_alpha, 6)
        r["significant"] = r["p_value"] < adjusted_alpha

    return sorted_res
