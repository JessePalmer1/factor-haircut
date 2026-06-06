"""
Multiple-testing corrections for the factor-haircut project.

Each function takes an array of p-values and a target level alpha, and returns
a boolean reject vector aligned with the input order (NOT the sorted order).

Corrections implemented:
  - bonferroni   : FWER control, valid under any dependence
  - holm         : FWER control, step-down, uniformly more powerful than Bonferroni
  - benjamini_hochberg   : FDR control under independence / positive dependence (PRDS)
  - benjamini_yekutieli  : FDR control under arbitrary dependence (uses harmonic factor c(m))

Utilities:
  - implied_threshold : t-stat needed for a Bonferroni-adjusted p to hit alpha
  - haircut           : proportional reduction in SR implied by adjusting t
"""

import numpy as np
from scipy.stats import norm


def bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Bonferroni correction (FWER).
    Reject H_i if p_i <= alpha / m.

    Parameters
    ----------
    pvals : array of p-values
    alpha : family-wise significance level

    Returns
    -------
    reject : bool array, same order as pvals
    """
    p = np.asarray(pvals)
    return p <= alpha / p.size


def holm(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Holm step-down correction (FWER).
    Sort ascending; reject p_(i) while p_(i) <= alpha / (m - i + 1); stop at first failure.

    Parameters
    ----------
    pvals : array of p-values
    alpha : family-wise significance level

    Returns
    -------
    reject : bool array, same order as pvals
    """
    p = np.asarray(pvals)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha / (m - np.arange(m))   # alpha/m, alpha/(m-1), ..., alpha/1
    fails = ranked > thresholds
    # stop at first failure; if none, reject everything
    cutoff = int(np.argmax(fails)) if fails.any() else m
    reject = np.zeros(m, dtype=bool)
    reject[order[:cutoff]] = True
    return reject


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini–Hochberg step-up correction (FDR).
    Find largest k such that p_(k) <= (k / m) * alpha; reject all up to rank k.
    Valid under independence and positive regression dependence (PRDS).

    Parameters
    ----------
    pvals : array of p-values
    alpha : false discovery rate level

    Returns
    -------
    reject : bool array, same order as pvals
    """
    p = np.asarray(pvals)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= thresholds
    # step-up: largest k where p_(k) passes; reject everything up to and including k
    kmax = int(np.max(np.where(below)[0])) + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:kmax]] = True
    return reject


def benjamini_yekutieli(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini–Yekutieli step-up correction (FDR under arbitrary dependence).
    Identical to BH but threshold is (k / (m * c(m))) * alpha,
    where c(m) = sum(1/i for i in 1..m) is the harmonic factor.

    Parameters
    ----------
    pvals : array of p-values
    alpha : false discovery rate level

    Returns
    -------
    reject : bool array, same order as pvals
    """
    p = np.asarray(pvals)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    thresholds = (np.arange(1, m + 1) / (m * c_m)) * alpha
    below = ranked <= thresholds
    kmax = int(np.max(np.where(below)[0])) + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:kmax]] = True
    return reject


def implied_threshold(m: int, alpha: float = 0.05, method: str = "bonferroni") -> float:
    """
    The t-statistic required for an adjusted p-value to reach alpha.
    For Bonferroni (two-sided): t* = Phi^{-1}(1 - alpha / (2m)).

    Parameters
    ----------
    m      : number of tests
    alpha  : significance level
    method : one of {"bonferroni"}  (extend for other methods in Phase 3)

    Returns
    -------
    t_star : float
    """
    if method == "bonferroni":
        return float(norm.ppf(1 - alpha / (2 * m)))
    raise ValueError(f"Unknown method: {method!r}")


def haircut(sr_original: float, sr_adjusted: float) -> float:
    """
    Proportional reduction in Sharpe ratio after multiplicity adjustment.
    haircut = 1 - sr_adjusted / sr_original

    Parameters
    ----------
    sr_original : reported Sharpe ratio
    sr_adjusted : Sharpe after the multiplicity correction is applied

    Returns
    -------
    haircut : float in [0, 1]
    """
    return 1.0 - sr_adjusted / sr_original
