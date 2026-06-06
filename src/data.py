"""
Loader for the Chen–Zimmermann Open Source Cross-Sectional Asset Pricing dataset.

Provides 212 published cross-sectional return predictors (predictability = "clear"
or "likely") with their original published t-statistics.

Install: pip install openassetpricing
Docs   : https://www.openassetpricing.com
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Predictability tiers in the Chen–Zimmermann doc table.
# "1_clear" + "2_likely" = the 212 published factors in the headline count.
_PRED_ORDER = ["1_clear", "2_likely", "4_not", "indirect", "9_drop"]

_LEVEL_MAP = {
    "clear":     ["1_clear"],
    "likely":    ["1_clear", "2_likely"],
    "not_clear": ["4_not"],
    "all":       _PRED_ORDER,
}


def load_published_factors() -> pd.DataFrame:
    """
    Load the Chen–Zimmermann predictor table and return a clean DataFrame.

    Fetches the signal documentation CSV from openassetpricing, keeps only rows
    with a reported t-stat, renames columns, and takes |t_stat| (a handful of
    factors are reported with a negative sign convention).

    Returns
    -------
    df : pd.DataFrame with columns:
         - signal_name    : str, predictor identifier (Acronym)
         - t_stat         : float, |published t-statistic|
         - predictability : str, one of "1_clear" / "2_likely" / "4_not" / ...
         - year           : int, publication year
    """
    import openassetpricing as oap

    ap = oap.OpenAP()
    raw = ap.dl_signal_doc("pandas")

    df = pd.DataFrame({
        "signal_name":    raw["Acronym"],
        "t_stat":         raw["T-Stat"].abs(),
        "predictability": raw["Predictability in OP"],
        "year":           raw["Year"],
    })

    df = df.dropna(subset=["t_stat"]).reset_index(drop=True)
    df = df.set_index("signal_name")
    return df


def filter_by_predictability(df: pd.DataFrame, level: str = "clear") -> pd.DataFrame:
    """
    Filter the factor table to a predictability-flag tier.

    Parameters
    ----------
    df    : DataFrame from load_published_factors()
    level : "clear"     → only the 165 "1_clear" factors
            "likely"    → "1_clear" + "2_likely" (the 212 published factors)
            "not_clear" → the "4_not" tier
            "all"       → everything (including indirect / drop)

    Returns
    -------
    filtered : pd.DataFrame
    """
    if level not in _LEVEL_MAP:
        raise ValueError(f"level must be one of {list(_LEVEL_MAP)}, got {level!r}")
    keep = _LEVEL_MAP[level]
    return df[df["predictability"].isin(keep)].copy()


def compute_pvals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a two-sided p-value column derived from each factor's published t-stat.
    p = 2 * (1 - Phi(|t_stat|))

    Parameters
    ----------
    df : DataFrame with a t_stat column

    Returns
    -------
    df : same DataFrame with a new 'p_value' column added (in-place copy)
    """
    df = df.copy()
    df["p_value"] = 2 * (1 - norm.cdf(df["t_stat"].abs()))
    return df


def apply_corrections_sweep(
    df: pd.DataFrame,
    m_assumed_list: list[int],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each assumed total number of tests m, compute the BY-adjusted implied
    threshold and flag which published factors survive, then append the haircut.

    The number of *published* tests is fixed (len(df)), but the true number of
    tests m includes all the unpublished searches — this function lets you sweep
    over plausible values to see how sensitive survival is to that assumption.

    For each m the Bonferroni implied threshold is:
        t* = Phi^{-1}(1 - alpha / (2m))

    And BY's threshold for the k-th ranked p-value is tighter by c(m) = H_m.
    Here we report the Bonferroni t* as the conservative headline bar, and run
    BY directly on the observed p-values for the survive/reject column.

    Parameters
    ----------
    df              : DataFrame from load_published_factors() (with p_value col
                      added via compute_pvals, or we add it internally)
    m_assumed_list  : list of m values to sweep (e.g. [300, 600, 1000])
    alpha           : significance level

    Returns
    -------
    results : pd.DataFrame indexed by signal_name, with columns:
              t_stat, predictability, year, p_value,
              and for each m: survive_bonf_{m}, survive_by_{m},
                              threshold_bonf_{m}, haircut_bonf_{m}
    """
    from src.corrections import benjamini_yekutieli, implied_threshold

    if "p_value" not in df.columns:
        df = compute_pvals(df)

    results = df.copy()
    pvals = df["p_value"].values

    # BY is applied to only the observed factors (controls FDR within this set).
    # Bonferroni uses the external m to account for all unpublished tests too.
    reject_by = benjamini_yekutieli(pvals, alpha=alpha)

    for m in m_assumed_list:
        t_star = implied_threshold(m, alpha=alpha, method="bonferroni")

        # Bonferroni with external m: reject if p_i <= alpha / m_assumed
        reject_bonf = pvals <= alpha / m

        results[f"threshold_bonf_{m}"] = t_star
        results[f"survive_bonf_{m}"]   = reject_bonf
        results[f"survive_by_{m}"]     = reject_by
        results[f"haircut_bonf_{m}"]   = (1 - t_star / df["t_stat"]).clip(lower=0)

    return results
