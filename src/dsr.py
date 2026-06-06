"""
Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

PSR answers: what is the probability the true Sharpe beats a benchmark SR*?
DSR sets that benchmark to the expected maximum Sharpe across N trials —
the order-statistic / Gumbel extreme-value result that mirrors the max-t story
in the cross-sectional multiple-testing analysis.

References
----------
Bailey & López de Prado (2014). The Deflated Sharpe Ratio. JPM.
"""

import math

import numpy as np
from scipy.stats import norm

# Euler–Mascheroni constant
EULER_GAMMA = 0.5772156649015328


def psr(
    sr_hat: float,
    T: int,
    skew: float,
    kurt: float,
    sr_benchmark: float,
) -> float:
    """
    Probabilistic Sharpe Ratio.

    PSR(SR*) = Phi[ (SR_hat - SR*) * sqrt(T - 1)
                    / sqrt(1 - skew * SR_hat + (kurt - 1) / 4 * SR_hat^2) ]

    Note: kurt here is the *non-excess* kurtosis (normal = 3).
    Note: SR_hat should be the per-period (not annualized) Sharpe.

    Parameters
    ----------
    sr_hat       : estimated per-period Sharpe ratio
    T            : number of return observations
    skew         : skewness of the return series (gamma_3)
    kurt         : non-excess kurtosis (gamma_4); use 3 for normal returns
    sr_benchmark : the threshold SR* to test against

    Returns
    -------
    psr_value : float in (0, 1)
    """
    se = math.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
    z = (sr_hat - sr_benchmark) * math.sqrt(T - 1) / se
    return float(norm.cdf(z))


def dsr_benchmark(
    n_trials: int,
    sr_variance: float,
) -> float:
    """
    Expected maximum Sharpe ratio across n_trials, via the Gumbel / extreme-value approximation.

    SR* = sqrt(V_hat) * [ (1 - gamma) * Phi^{-1}(1 - 1/N)
                         + gamma      * Phi^{-1}(1 - 1/(N*e)) ]

    where gamma is the Euler–Mascheroni constant and V_hat is the variance of
    the N trial Sharpe ratios. This is the same order-statistic result as
    E[max_i t_i] ~ sqrt(2 ln m) applied to the Sharpe axis.

    Parameters
    ----------
    n_trials    : N, the number of strategy trials / backtests
    sr_variance : V_hat, variance of the Sharpe ratios across the N trials

    Returns
    -------
    sr_star : float, the benchmark Sharpe
    """
    scale = math.sqrt(sr_variance)
    term1 = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
    term2 = EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))
    return float(scale * (term1 + term2))


def deflated_sharpe(
    sr_hat: float,
    T: int,
    skew: float,
    kurt: float,
    n_trials: int,
    sr_variance: float,
) -> float:
    """
    Deflated Sharpe Ratio.

    Computes PSR evaluated at the expected-maximum benchmark SR* from dsr_benchmark().
    A DSR close to 0.5 means the strategy is indistinguishable from the best of
    n_trials noise strategies; DSR >> 0.5 means genuine skill.

    Parameters
    ----------
    sr_hat      : estimated per-period Sharpe ratio
    T           : number of return observations
    skew        : skewness of returns
    kurt        : non-excess kurtosis of returns (3 = normal)
    n_trials    : number of backtests / trials (N)
    sr_variance : variance of Sharpe ratios across the N trials

    Returns
    -------
    dsr_value : float in (0, 1)
    """
    sr_star = dsr_benchmark(n_trials, sr_variance)
    return psr(sr_hat, T, skew, kurt, sr_star)
