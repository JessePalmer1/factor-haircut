"""
Loader for the Chen–Zimmermann Open Source Cross-Sectional Asset Pricing dataset.

Provides 212 published cross-sectional return predictors with their original
published t-statistics and documentation/predictability flags.

Install: pip install openassetpricing
Docs   : https://www.openassetpricing.com
"""

import pandas as pd


def load_published_factors() -> pd.DataFrame:
    """
    Load the Chen–Zimmermann predictor table and return a clean DataFrame.

    Columns returned (at minimum):
      - signal_name   : str, predictor identifier
      - t_stat        : float, original published t-statistic (absolute value)
      - predictability: str, documentation flag ("clear" / "likely" / etc.)

    Returns
    -------
    df : pd.DataFrame indexed by signal_name
    """
    # TODO: implement using openassetpricing
    raise NotImplementedError


def filter_by_predictability(df: pd.DataFrame, level: str = "clear") -> pd.DataFrame:
    """
    Filter the factor table to a predictability-flag tier.

    Parameters
    ----------
    df    : DataFrame from load_published_factors()
    level : one of {"clear", "likely", "not_clear", "all"}

    Returns
    -------
    filtered : pd.DataFrame
    """
    # TODO: implement
    raise NotImplementedError


def compute_pvals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a two-sided p-value column derived from each factor's published t-stat.
    p = 2 * (1 - Phi(|t|))

    Parameters
    ----------
    df : DataFrame with a t_stat column

    Returns
    -------
    df : same DataFrame with a new 'p_value' column
    """
    # TODO: implement
    raise NotImplementedError
