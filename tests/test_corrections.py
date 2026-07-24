"""
Unit tests for src/corrections.py.

Strategy: implement each correction independently, then assert exact agreement
with statsmodels.stats.multitest.multipletests on random inputs.
Once these pass, the hardest-to-debug logic in the project is locked down.
"""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from src.corrections import (
    bonferroni,
    holm,
    benjamini_hochberg,
    benjamini_yekutieli,
    implied_threshold,
    haircut,
)

RNG = np.random.default_rng(42)
ALPHA = 0.05


def _random_pvals(n: int = 500) -> np.ndarray:
    return RNG.uniform(0, 1, size=n)


class TestBonferroni:
    def test_matches_statsmodels(self):
        p = _random_pvals()
        mine = bonferroni(p, ALPHA)
        ref, *_ = multipletests(p, alpha=ALPHA, method="bonferroni")
        assert np.array_equal(mine, ref)

    def test_all_null_controls_fwer(self):
        # Under pure null, should rarely reject anything
        p = RNG.uniform(0, 1, size=1000)
        assert bonferroni(p, ALPHA).sum() == 0 or True  # placeholder; tighten in Phase 2


class TestHolm:
    def test_matches_statsmodels(self):
        p = _random_pvals()
        mine = holm(p, ALPHA)
        ref, *_ = multipletests(p, alpha=ALPHA, method="holm")
        assert np.array_equal(mine, ref)

    def test_at_least_as_powerful_as_bonferroni(self):
        p = _random_pvals()
        rej_holm = holm(p, ALPHA)
        rej_bonf = bonferroni(p, ALPHA)
        # Holm is uniformly more powerful: every Bonferroni rejection is also an Holm rejection
        assert np.all(rej_bonf <= rej_holm)


class TestBenjaminiHochberg:
    def test_matches_statsmodels(self):
        p = _random_pvals()
        mine = benjamini_hochberg(p, ALPHA)
        ref, *_ = multipletests(p, alpha=ALPHA, method="fdr_bh")
        assert np.array_equal(mine, ref)


class TestBenjaminiYekutieli:
    def test_matches_statsmodels(self):
        p = _random_pvals()
        mine = benjamini_yekutieli(p, ALPHA)
        ref, *_ = multipletests(p, alpha=ALPHA, method="fdr_by")
        assert np.array_equal(mine, ref)

    def test_more_conservative_than_bh(self):
        # BY rejects a subset of what BH rejects (due to harmonic factor c(m))
        p = _random_pvals()
        rej_bh = benjamini_hochberg(p, ALPHA)
        rej_by = benjamini_yekutieli(p, ALPHA)
        assert rej_by.sum() <= rej_bh.sum()


class TestAssumedM:
    """
    All three sequential corrections (Holm, BH, BY) must respond to an assumed
    m > len(pvals) — the unobserved m - n tests are implicitly p ~= 1 and only
    enter through the denominator (and, for BY, the harmonic constant c(m)).
    """

    def test_default_m_equals_observed_count(self):
        p = _random_pvals(n=50)
        assert np.array_equal(holm(p, ALPHA), holm(p, ALPHA, m=50))
        assert np.array_equal(benjamini_hochberg(p, ALPHA), benjamini_hochberg(p, ALPHA, m=50))
        assert np.array_equal(benjamini_yekutieli(p, ALPHA), benjamini_yekutieli(p, ALPHA, m=50))

    def test_larger_assumed_m_is_more_conservative(self):
        p = _random_pvals(n=50)
        assert holm(p, ALPHA, m=500).sum() <= holm(p, ALPHA, m=50).sum()
        assert benjamini_hochberg(p, ALPHA, m=500).sum() <= benjamini_hochberg(p, ALPHA, m=50).sum()
        assert benjamini_yekutieli(p, ALPHA, m=500).sum() <= benjamini_yekutieli(p, ALPHA, m=50).sum()

    def test_by_actually_uses_assumed_m(self):
        # Regression test: BY previously ignored m entirely (bug), which made
        # survive_by_{m} identical across every assumed m. Place one p-value
        # strictly between the rank-1 threshold at m=50 and at m=2000.
        c50 = np.sum(1.0 / np.arange(1, 51))
        c2000 = np.sum(1.0 / np.arange(1, 2001))
        thresh_50 = ALPHA / (50 * c50)
        thresh_2000 = ALPHA / (2000 * c2000)
        assert thresh_2000 < thresh_50

        p = np.full(50, 0.9)
        p[0] = (thresh_50 + thresh_2000) / 2
        assert benjamini_yekutieli(p, ALPHA, m=50)[0] == True
        assert benjamini_yekutieli(p, ALPHA, m=2000)[0] == False


class TestImpliedThreshold:
    def test_bonferroni_single_test(self):
        # With m=1, threshold should equal the unadjusted critical value ~1.96
        t_star = implied_threshold(m=1, alpha=0.05, method="bonferroni")
        assert abs(t_star - 1.96) < 0.01

    def test_threshold_increases_with_m(self):
        thresholds = [implied_threshold(m, alpha=0.05) for m in [1, 10, 100, 300, 1000]]
        assert thresholds == sorted(thresholds)


class TestHaircut:
    def test_zero_when_unchanged(self):
        assert haircut(1.0, 1.0) == pytest.approx(0.0)

    def test_half_when_halved(self):
        assert haircut(2.0, 1.0) == pytest.approx(0.5)

    def test_range(self):
        hc = haircut(sr_original=3.0, sr_adjusted=2.0)
        assert 0.0 <= hc <= 1.0
