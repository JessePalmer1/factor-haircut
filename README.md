# The Factor Zoo Haircut

## Question

Given that hundreds of equity factors have been tested, what t-statistic does a newly reported factor *actually* need to be believable — and how many of the 212 published factors in the Chen–Zimmermann dataset survive that bar?

## Inputs / Outputs

**Inputs:**
- A reported t-statistic (or annualized Sharpe ratio + sample length T)
- The estimated number of trials m (how many factors were tested, published or not)
- An assumed cross-test correlation (controlled via latent-factor structure)
- Tail-thickness parameter ν (degrees of freedom for the return distribution)

**Outputs:**
- The multiplicity-adjusted significance threshold
- The haircut — the proportional reduction in Sharpe implied by the correction
- The Deflated Sharpe Ratio (DSR) — probability the true Sharpe beats an order-statistic benchmark
- A verdict: *survives* or *does not survive* the adjusted bar

## Method

1. **Multiple-testing corrections** — Bonferroni, Holm (FWER), Benjamini–Hochberg, Benjamini–Yekutieli (FDR) applied to the cross-section of factors.
2. **Realistic simulation** — correlated factors via a latent-factor covariance (Cholesky-induced, valid PSD by construction) and fat-tailed returns via multivariate Student-t draws.
3. **Deflated Sharpe Ratio** — the DSR benchmark is the expected maximum Sharpe across N trials (an order-statistic / Gumbel extreme-value result), bridging the cross-sectional and time-series multiplicity stories.
4. **Real-data application** — the Chen–Zimmermann *Open Source Cross-Sectional Asset Pricing* dataset (212 published predictors with original t-stats).

## Key finding

~`[Z]`% of the 212 published factors fail to clear a multiple-testing–adjusted significance threshold at plausible values of m. *(Fill in Z after Phase 5.)*

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Structure

```
factor-haircut/
├── README.md
├── requirements.txt
├── notes/theory.md          # Phase 1 derivations
├── src/
│   ├── corrections.py       # Bonferroni, Holm, BH, BY, haircut
│   ├── simulate.py          # DGP: correlated + fat-tailed test stats
│   ├── dsr.py               # Probabilistic & Deflated Sharpe Ratio
│   └── data.py              # Chen–Zimmermann loader
├── notebooks/
│   ├── 01_theory_sanity.ipynb
│   ├── 02_mvs_independent.ipynb
│   ├── 03_realistic_dgp.ipynb
│   ├── 04_dsr_validation.ipynb
│   └── 05_realdata_application.ipynb
├── tests/test_corrections.py
└── app.py                   # Streamlit dashboard
```

## References

- Harvey, Liu & Zhu (2016) — multiple testing in finance, haircut framing
- Benjamini & Hochberg (1995) — FDR
- Benjamini & Yekutieli (2001) — FDR under arbitrary dependence
- Bailey & López de Prado (2014) — Deflated Sharpe Ratio
- Chen & Zimmermann (2022) — Open Source Cross-Sectional Asset Pricing
