"""
Data-generating processes for the factor-haircut project.

Phase 2 — independent Gaussian:
  generate_tstats   : m independent t-stats, pi1 fraction truly non-null

Phase 3 — realistic DGP:
  factor_covariance : valid PSD correlation matrix via latent-factor structure (BB' + D)
  mvt_draws         : multivariate Student-t draws via Gaussian / chi-squared scaling

Simulation harness:
  run_simulation    : Monte Carlo loop returning empirical FWER, FDR, and power
"""

import numpy as np
from scipy.stats import norm


def generate_tstats(
    m: int,
    pi1: float,
    delta: float,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate m independent t-statistics under Gaussian assumptions.

    A fraction pi1 are truly non-null (signal), the rest are pure noise.
    Non-null stats are drawn from N(delta, 1); null stats from N(0, 1).

    Parameters
    ----------
    m     : total number of factors / tests
    pi1   : fraction that are truly non-null (signal), in [0, 1]
    delta : mean shift for non-null factors (effect size)
    rng   : numpy random Generator (optional; created if None)

    Returns
    -------
    tstats     : (m,) float array of t-statistics
    is_signal  : (m,) bool array — True where the factor is truly non-null
    """
    rng = rng or np.random.default_rng()
    m1 = int(round(pi1 * m))
    m0 = m - m1
    is_signal = np.zeros(m, dtype=bool)
    is_signal[:m1] = True
    tstats = np.empty(m)
    if m1 > 0:
        tstats[:m1] = rng.normal(loc=delta, scale=1.0, size=m1)
    if m0 > 0:
        tstats[m1:] = rng.normal(loc=0.0, scale=1.0, size=m0)
    # shuffle so signal/null positions aren't always at the front
    idx = rng.permutation(m)
    return tstats[idx], is_signal[idx]


def factor_covariance(
    m: int,
    k: int = 3,
    loading_scale: float = 0.6,
    idio_low: float = 0.5,
    idio_high: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Build a valid PSD correlation matrix via a latent-factor structure.

    Sigma = B @ B.T + D,  where B is (m x k) loadings and D is diagonal idiosyncratic.
    Normalized to a correlation matrix (unit diagonal) so it integrates cleanly
    with standardized test statistics.

    Parameters
    ----------
    m            : number of factors
    k            : number of latent macro drivers
    loading_scale: scale of random loadings B
    idio_low     : lower bound of uniform idiosyncratic variance
    idio_high    : upper bound of uniform idiosyncratic variance
    rng          : numpy random Generator

    Returns
    -------
    corr : (m, m) correlation matrix, valid PSD by construction
    """
    rng = rng or np.random.default_rng()
    B = rng.normal(size=(m, k)) * loading_scale
    D = np.diag(rng.uniform(idio_low, idio_high, size=m))
    Sigma = B @ B.T + D
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d)


def mvt_draws(
    mu: np.ndarray,
    L: np.ndarray,
    nu: float,
    n_reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw from a multivariate Student-t distribution.

    X = mu + (L @ z) / sqrt(w / nu),  z ~ N(0, I_m),  w ~ chi2(nu)
    As nu -> inf this recovers the multivariate Gaussian.

    Parameters
    ----------
    mu    : (m,) mean vector
    L     : (m, m) lower Cholesky factor of the scale matrix
    nu    : degrees of freedom (nu = 3-5 is realistically fat-tailed)
    n_reps: number of independent replications (columns)
    rng   : numpy random Generator

    Returns
    -------
    draws : (m, n_reps) array
    """
    m = L.shape[0]
    z = rng.normal(size=(m, n_reps))
    w = rng.chisquare(nu, size=n_reps)
    return mu[:, None] + (L @ z) / np.sqrt(w / nu)


def run_simulation(
    m: int,
    pi1: float,
    delta: float,
    n_reps: int,
    corrections: dict,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> dict:
    """
    Monte Carlo loop to estimate empirical FWER, FDR, and power for each correction.

    Parameters
    ----------
    m           : number of tests per trial
    pi1         : fraction truly non-null
    delta       : effect size for non-null factors
    n_reps      : number of Monte Carlo replications
    corrections : dict mapping name -> callable(pvals, alpha) -> bool reject vector
    alpha       : significance level
    rng         : numpy random Generator

    Returns
    -------
    results : dict mapping correction name -> {"fwer": float, "fdr": float, "power": float}
    """
    rng = rng or np.random.default_rng()
    accum = {name: {"fwer_hits": 0, "fdr_sum": 0.0, "power_sum": 0.0}
             for name in corrections}

    for _ in range(n_reps):
        tstats, is_signal = generate_tstats(m, pi1, delta, rng=rng)
        pvals = 2 * (1 - norm.cdf(np.abs(tstats)))
        n_signal = is_signal.sum()

        for name, fn in corrections.items():
            reject = fn(pvals, alpha)
            false_rej = (reject & ~is_signal).sum()
            true_rej = (reject & is_signal).sum()
            total_rej = reject.sum()

            accum[name]["fwer_hits"] += int(false_rej > 0)
            accum[name]["fdr_sum"] += false_rej / total_rej if total_rej > 0 else 0.0
            accum[name]["power_sum"] += true_rej / n_signal if n_signal > 0 else 0.0

    return {
        name: {
            "fwer": v["fwer_hits"] / n_reps,
            "fdr": v["fdr_sum"] / n_reps,
            "power": v["power_sum"] / n_reps,
        }
        for name, v in accum.items()
    }
