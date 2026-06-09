# The Factor Zoo Haircut

**Given that hundreds of equity factors have been tested, what t-statistic does a factor *actually* need to be believable — and how many published factors survive that bar?**

Built an interactive factor "haircut" and Deflated Sharpe Ratio calculator (Python, NumPy/pandas, Streamlit) using Monte Carlo simulation with latent-factor correlation (Cholesky-induced) and fat-tailed returns plus Benjamini–Yekutieli false-discovery control, that showed **~48% of the 188 published equity factors** in the Chen–Zimmermann dataset fail to clear a Bonferroni-adjusted significance threshold (m = 600 assumed trials, t* = 3.93). Under the less conservative BY false-discovery rate control, **~21% fail**.

---

## The core idea

Under the null, a factor's t-statistic is a draw from N(0, 1). A single test rejects at the 5% level when |t| > 1.96. But if you run *m* independent tests on pure noise, the expected maximum t-stat grows like √(2 ln m). For m = 300 that's ≈ 3.38 — so the conventional t > 2 bar is meaningless in a world where hundreds of factors have been tried.

Every technique in this project is a principled way of raising the bar to account for m. The Deflated Sharpe Ratio is the same idea applied to backtested Sharpe ratios.

---

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501**.

---

## What the app does

**Tab 1 — Calculator.** Enter a reported t-stat, the assumed number of trials m, and significance level α. The app returns:
- The Bonferroni-adjusted threshold t* and whether the factor clears it
- The haircut — the proportional Sharpe reduction implied by the correction
- The Deflated Sharpe Ratio (DSR) and a skill verdict

A plot shows the null max-statistic distribution (the expected best of m noise factors) with the factor and threshold marked.

**Tab 2 — Published factors.** Loads the Chen–Zimmermann 188-factor dataset and shows survival under Bonferroni (sweeping m = 300 / 600 / 1000) and BY (FDR 5%), with a sortable table and sensitivity chart.

---

## Key findings

| Assumed m | Bonferroni t* | Survivors (Bonf) | Survivors (BY) |
|----------:|:-------------:|:----------------:|:--------------:|
| 300       | 3.76          | 103 / 188 (55%)  | 148 / 188 (79%) |
| 600       | 3.93          | 97 / 188 (52%)   | 148 / 188 (79%) |
| 1000      | 4.06          | 92 / 188 (49%)   | 148 / 188 (79%) |

BY survival is stable across m because it controls FDR *within* the observed factors; Bonferroni's bar rises as m does. Classic factors like Momentum (t = 3.74) and Leverage (t = 3.93) sit right at the edge and fall out under Bonferroni at m ≥ 600.

---

## Structure

```
factor-haircut/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── corrections.py      # Bonferroni, Holm, BH, BY, implied_threshold, haircut
│   ├── simulate.py         # DGP: correlated (Cholesky) + fat-tailed (multivariate-t)
│   ├── dsr.py              # Probabilistic & Deflated Sharpe Ratio
│   └── data.py             # Chen–Zimmermann loader and corrections sweep
└── tests/
    └── test_corrections.py # Diffs all corrections against statsmodels
```

---

## Method

1. **Multiple-testing corrections** — Bonferroni and Holm control the family-wise error rate (FWER); Benjamini–Hochberg and Benjamini–Yekutieli control the false discovery rate (FDR). BY is valid under arbitrary test dependence, which makes it the right choice for correlated factors.

2. **Realistic simulation** — factors are correlated via a latent-factor covariance Σ = BB′ + D (valid PSD by construction), with fat-tailed returns via multivariate Student-t draws (X = μ + Lz / √(w/ν)).

3. **Deflated Sharpe Ratio** — the DSR benchmark is the expected maximum Sharpe across N trials, approximated via the Gumbel extreme-value formula. A DSR > 0.5 means the strategy's Sharpe likely exceeds the best noise strategy; DSR ≈ 0.5 means indistinguishable from luck.

4. **Real data** — Chen–Zimmermann *Open Source Cross-Sectional Asset Pricing* dataset (188 predictors with reported t-stats, installable via `pip install openassetpricing`).

---

## References

- Harvey, Liu & Zhu (2016). *…and the Cross-Section of Expected Returns.* RFS. — haircut framing
- Benjamini & Hochberg (1995). *Controlling the False Discovery Rate.* JRSS-B.
- Benjamini & Yekutieli (2001). *The Control of the FDR Under Dependency.* Ann. Stat.
- Bailey & López de Prado (2014). *The Deflated Sharpe Ratio.* JPM.
- Chen & Zimmermann (2022). *Open Source Cross-Sectional Asset Pricing.* CFR.
