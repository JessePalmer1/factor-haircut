# The Factor Zoo Haircut: A Multiple-Testing & Deflated-Sharpe Toolkit

> A ~40-hour quant project that builds a working tool to answer one question:
> **Given that hundreds of factors have been tested, what t-statistic does a factor *actually* need to be believable — and how many "published" factors survive that bar?**

*Tech stack: Python, NumPy, pandas, SciPy, statsmodels, matplotlib, Streamlit.*
*Math focus: linear regression inference, t-statistics, p-values, the false discovery rate, order statistics / extreme-value theory, and the Deflated Sharpe Ratio.*

---

## Table of contents

1. [Why this project](#1-why-this-project)
2. [The single idea everything hangs on](#2-the-single-idea-everything-hangs-on)
3. [Learning objectives](#3-learning-objectives)
4. [Prerequisites & setup](#4-prerequisites--setup)
5. [Repository structure](#5-repository-structure)
6. [The iterative plan (Phases 0–6)](#6-the-iterative-plan)
   - [Phase 0 — Setup & scoping](#phase-0--setup--scoping-2h)
   - [Phase 1 — Theory & written notes](#phase-1--theory--written-notes-8h)
   - [Phase 2 — Minimal viable simulation](#phase-2--minimal-viable-simulation-6h)
   - [Phase 3 — Realistic data-generating process](#phase-3--realistic-data-generating-process-9h)
   - [Phase 4 — Deflated Sharpe Ratio](#phase-4--deflated-sharpe-ratio-5h)
   - [Phase 5 — Real-data application](#phase-5--real-data-application-5h)
   - [Phase 6 — Streamlit dashboard & writeup](#phase-6--streamlit-dashboard--writeup-5h)
7. [Time budget](#7-time-budget)
8. [Resume line](#8-resume-line)
9. [Stretch goals](#9-stretch-goals)
10. [Formula reference card](#10-formula-reference-card)
11. [References & reading list](#11-references--reading-list)
12. [Glossary](#12-glossary)

---

## 1. Why this project

In empirical finance, a factor (a stock characteristic that predicts returns) is usually "discovered" by sorting stocks on the characteristic, forming a long–short portfolio, and running a regression whose intercept or mean has a t-statistic. The convention is that **t > 2 means real**. But hundreds of researchers have tested thousands of characteristics. If you try enough noise, *some* of it crosses t = 2 by pure chance. This is the **multiple-testing** (a.k.a. data-snooping) problem, and it is arguably the central methodological issue in modern factor investing.

This project builds a tool that corrects for it. You will:

- **Learn the inference math cold** — what a t-stat actually guarantees, why it breaks under repeated testing, and the family of corrections (Bonferroni, Holm, Benjamini–Hochberg, Benjamini–Yekutieli).
- **Simulate the problem honestly** — with realistic correlation between factors and fat-tailed returns, so the demonstration reflects real markets rather than a textbook idealization.
- **Bridge to modern quant** — implement Marcos López de Prado's **Deflated Sharpe Ratio (DSR)**, which applies the same multiplicity logic to backtested Sharpe ratios.
- **Apply it to real published factors** — using the Chen–Zimmermann *Open Source Cross-Sectional Asset Pricing* dataset (212 published predictors with their original t-stats, installable via `pip install openassetpricing`).
- **Ship an interactive product** — a Streamlit dashboard where anyone can enter a t-stat (or Sharpe), the number of trials, and an assumed test correlation, and instantly see whether the factor survives.

**What makes it resume-worthy:** it is niche, it produces a real and reusable artifact rather than re-running a regression on prices, it demonstrates statistical maturity (the difference between a junior who reports t-stats and a senior who *interrogates* them), and the interactive front end invites engagement during a screen.

---

## 2. The single idea everything hangs on

Under the null hypothesis that a factor has no real predictive power, its t-statistic is approximately a draw from a standard normal, $t \sim \mathcal{N}(0, 1)$. A single such test rejects the null at the 5% level when $|t| > 1.96$.

Now suppose you run $m$ *independent* such tests on pure noise. The largest t-statistic you observe is the **maximum of $m$ standard-normal draws**, and its expected value grows like:

$$
\mathbb{E}\!\left[\max_{1 \le i \le m} t_i\right] \approx \sqrt{2 \ln m}.
$$

For $m = 300$ this is $\sqrt{2 \ln 300} \approx 3.38$. In other words, **if you test ~300 worthless factors, you should expect your best one to post a t-stat around 3.4 — with zero true signal.** The threshold $t > 2$ is meaningless in that world.

Every technique in this project is a principled way of raising the bar to account for $m$. The Deflated Sharpe Ratio is the *same idea* applied to Sharpe ratios instead of regression t-stats: its benchmark is the expected maximum Sharpe across $N$ trials. Keep this picture in your head — the maximum of many draws is large by chance — and the whole project becomes one coherent story told on two axes (the cross-section of factors, and the time series of a backtest).

---

## 3. Learning objectives

By the end you should be able to explain each of the following in two sentences, from memory:

- The exact meaning of a t-statistic and a p-value in a regression, and the assumptions behind them.
- Why repeated testing inflates false positives, quantified via the expected maximum of $m$ draws.
- The difference between the **family-wise error rate (FWER)** and the **false discovery rate (FDR)**, and when you'd target each.
- How Bonferroni, Holm, Benjamini–Hochberg (BH), and Benjamini–Yekutieli (BY) work, step by step, and exactly which dependence assumptions each requires.
- Why the harmonic factor $c(m) = \sum_{i=1}^{m} 1/i$ in BY is the price of validity under arbitrary dependence.
- How to construct a valid (positive semi-definite) covariance matrix with a realistic latent-factor structure, and how to simulate from it with Cholesky.
- How fat tails change the behavior of t-statistics and error rates.
- The Probabilistic Sharpe Ratio and the Deflated Sharpe Ratio, and why the DSR benchmark is an order-statistic.
- What a "haircut" is and how to compute it.

---

## 4. Prerequisites & setup

**Math background assumed:** basic probability, the normal distribution, ordinary least squares at the level of "I know what a regression coefficient and its standard error are." Everything else is built up in Phase 1.

**Environment:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scipy statsmodels matplotlib streamlit openassetpricing
```

**Recommended tooling:** a notebook (Jupyter or VS Code interactive) for exploration in Phases 1–5, with finalized logic refactored into plain `.py` modules so the Streamlit app and your tests can import it.

---

## 5. Repository structure

Set this up in Phase 0 and grow into it. Keeping logic in importable modules (not buried in a notebook) is what lets Phase 6's app reuse Phases 2–5 without copy-paste.

```
factor-haircut/
├── README.md                 # the public face: question, method, key charts, findings
├── requirements.txt
├── notes/
│   └── theory.md             # your Phase 1 derivations (the "learn it cold" doc)
├── src/
│   ├── corrections.py        # Bonferroni, Holm, BH, BY, implied threshold, haircut
│   ├── simulate.py           # data-generating process: correlation + fat tails
│   ├── dsr.py                # Probabilistic & Deflated Sharpe Ratio
│   └── data.py               # load & clean the Chen–Zimmermann published t-stats
├── notebooks/
│   ├── 01_theory_sanity.ipynb
│   ├── 02_mvs_independent.ipynb
│   ├── 03_realistic_dgp.ipynb
│   ├── 04_dsr_validation.ipynb
│   └── 05_realdata_application.ipynb
├── tests/
│   └── test_corrections.py   # diff your implementations against statsmodels
└── app.py                    # Streamlit dashboard
```

---

## 6. The iterative plan

**The governing principle:** each phase ends with something that *runs and is verified correct* before the next phase adds realism or features. You build the simplest honest version first (independent, Gaussian, one correction), confirm it matches theory, then layer in correlation, fat tails, more corrections, the DSR, real data, and finally the UI. If you run out of time, you stop at a phase boundary and still have a complete, defensible deliverable.

---

### Phase 0 — Setup & scoping (2h)

**Objective.** Stand up the repo and commit to a precise question so scope doesn't creep.

**Steps.**
1. Create the directory structure above; initialize git; write `requirements.txt`.
2. Write a one-paragraph problem statement at the top of `README.md`: state the question, the inputs (a t-stat / Sharpe, a number of trials $m$, an assumed correlation), and the output (an adjusted threshold, a verdict, a haircut).
3. Create empty stub modules with function signatures (even just `def benjamini_yekutieli(pvals, alpha): ...  # TODO`). Writing the signatures first forces you to decide your interfaces early.

**Deliverable / Definition of Done.** A repo that imports cleanly and a README with a crisp problem statement. You can articulate, in one sentence, what the finished tool takes in and spits out.

---

### Phase 1 — Theory & written notes (8h)

**Objective.** Understand the statistics deeply enough to *teach* it. No production coding yet — only a worked-examples notebook and a written `notes/theory.md`. This is the phase that makes the project a genuine learning experience rather than a wrapper around a library.

**Why it matters.** If you can derive these results, every later debugging session becomes "is my code wrong or my understanding wrong?" — and you'll know it's the code, because the understanding is solid.

**Topics, in order:**

**(a) What a t-stat and p-value are.** For a regression coefficient $\hat{\beta}$ with standard error $\mathrm{se}(\hat{\beta})$, the t-statistic is $t = \hat{\beta} / \mathrm{se}(\hat{\beta})$. Under the null $\beta = 0$ and OLS assumptions, $t$ follows a Student-t distribution that is approximately $\mathcal{N}(0,1)$ in large samples. The two-sided p-value is

$$
p = 2\left(1 - \Phi(|t|)\right),
$$

where $\Phi$ is the standard normal CDF. **Write out in your own words what "$p = 0.05$" actually promises and — crucially — what it does not.**

**(b) The maximum-of-$m$-draws problem.** Derive (or carefully reproduce) $\mathbb{E}[\max_i t_i] \approx \sqrt{2 \ln m}$ for $m$ i.i.d. standard normals, and tabulate it for $m \in \{10, 50, 100, 300, 1000\}$. This single table is the motivation for the entire project.

**(c) FWER vs FDR.** Let $V$ = number of true nulls you falsely reject, $R$ = total rejections.
- **Family-wise error rate:** $\mathrm{FWER} = \Pr(V \ge 1)$ — the probability of *even one* false discovery. Use it when a single false factor is expensive.
- **False discovery rate:** $\mathrm{FDR} = \mathbb{E}[V / R]$ (with $0/0 \equiv 0$) — the expected *fraction* of your discoveries that are false. Use it when you'll trade a basket and can tolerate some duds.

**(d) The four corrections.** Work each one by hand on a toy set of five p-values before coding anything.

- **Bonferroni (controls FWER).** Reject $H_i$ if $p_i \le \alpha / m$. Dead simple, valid under any dependence, but conservative (low power).
- **Holm (controls FWER, uniformly more powerful than Bonferroni).** Sort ascending $p_{(1)} \le \dots \le p_{(m)}$. Step *down*: for $i = 1, 2, \dots$ reject $H_{(i)}$ as long as $p_{(i)} \le \alpha/(m - i + 1)$; at the first failure, stop and reject nothing further.
- **Benjamini–Hochberg (controls FDR under independence/positive dependence).** Sort ascending. Step *up*: find the **largest** $k$ such that $p_{(k)} \le \frac{k}{m}\alpha$, then reject $H_{(1)}, \dots, H_{(k)}$.
- **Benjamini–Yekutieli (controls FDR under *arbitrary* dependence).** Identical to BH but with threshold $\frac{k}{m \cdot c(m)}\alpha$, where

$$
c(m) = \sum_{i=1}^{m} \frac{1}{i} \approx \ln m + 0.5772.
$$

**Why BY is the one HLZ use:** factor tests are correlated in unknown ways, and BH's guarantee can fail under negative dependence. The harmonic factor $c(m)$ is exactly the toll BY pays to be valid no matter how the tests are correlated — which is why it's more conservative than BH. Understanding *why* $c(m)$ appears is the conceptual high point of this phase.

**(e) The haircut and the implied threshold.** A **haircut** is the proportional reduction in a strategy's Sharpe ratio (or return) implied by adjusting its significance for multiplicity:

$$
\text{haircut} = 1 - \frac{t_{\text{adjusted}}}{t_{\text{original}}} = 1 - \frac{\mathrm{SR}_{\text{adjusted}}}{\mathrm{SR}_{\text{original}}}.
$$

The **implied threshold** is the t-stat needed for an adjusted p-value to hit $\alpha$. For Bonferroni, two-sided:

$$
t^\* = \Phi^{-1}\!\left(1 - \frac{\alpha}{2m}\right).
$$

**(f) The DSR bridge (preview).** Read enough of López de Prado's Deflated Sharpe Ratio to see that its benchmark Sharpe is the *expected maximum* of $N$ trial Sharpe ratios — the order-statistic again. You'll implement it in Phase 4; here just connect it conceptually to (b).

**Deliverable / Definition of Done.** `notes/theory.md` containing every derivation above plus the five-p-value worked examples for all four corrections, and the $\mathbb{E}[\max]$ table. A notebook that numerically confirms $\mathbb{E}[\max_i t_i] \approx \sqrt{2 \ln m}$ by simulation.

---

### Phase 2 — Minimal viable simulation (6h)

**Objective.** Build and *verify* the simulation harness in its simplest honest form: $m$ **independent**, **Gaussian** test statistics, of which a known fraction are truly non-null. Implement Bonferroni and Holm. This locks in a correct skeleton before you add anything that could obscure a bug.

**Why simplest-first.** With independence and normality, you know the exact theoretical error rates, so any deviation is a bug you can catch immediately. Add realism only once the plumbing is proven.

**Steps.**
1. In `simulate.py`, write a function that generates $m$ t-stats: a fraction $\pi_1$ are "true" (mean shifted by an effect size $\delta$, e.g. drawn $\sim \mathcal{N}(\delta, 1)$) and the rest are null ($\sim \mathcal{N}(0,1)$). Return the t-stats *and* a boolean ground-truth mask.
2. Convert to two-sided p-values.
3. In `corrections.py`, implement `bonferroni` and `holm`, each returning a boolean "reject" vector.
4. Write a Monte Carlo loop over many reps that records, per correction: empirical FWER, power (fraction of true factors correctly flagged), and the realized FDR.

**Code guidance — the BH/BY step logic and how to debug it (your stated roadblock).** Implement BH/BY yourself in vectorized NumPy, then assert agreement with `statsmodels` so the tricky step-up indexing is unit-tested rather than guessed:

```python
import numpy as np
from statsmodels.stats.multitest import multipletests

def benjamini_yekutieli(pvals, alpha=0.05):
    p = np.asarray(pvals)
    m = p.size
    order = np.argsort(p)                  # ascending
    ranked = p[order]
    c_m = np.sum(1.0 / np.arange(1, m + 1)) # harmonic factor
    thresh = (np.arange(1, m + 1) / (m * c_m)) * alpha
    below = ranked <= thresh
    kmax = np.max(np.where(below)[0]) + 1 if below.any() else 0  # largest passing rank
    reject = np.zeros(m, dtype=bool)
    if kmax > 0:
        reject[order[:kmax]] = True        # reject everything up to kmax (step-up)
    return reject

# Sanity check — must match the reference on random inputs:
p = np.random.rand(500)
mine = benjamini_yekutieli(p, 0.05)
ref  = multipletests(p, alpha=0.05, method="fdr_by")[0]
assert np.array_equal(mine, ref), "BY logic mismatch"
```

Do the same diff for BH (`method="fdr_bh"`) and Holm (`method="holm"`). Once these asserts pass, the hardest-to-debug part of the project is behind you.

**Deliverable / Definition of Done.** A chart showing uncorrected testing's FWER blowing well past $\alpha = 0.05$ as $m$ grows, while Bonferroni and Holm pin it at (or below) $\alpha$. Passing tests in `tests/test_corrections.py`. **If the uncorrected FWER isn't roughly $1 - (1-\alpha)^m$ here, stop and fix it before moving on.**

---

### Phase 3 — Realistic data-generating process (9h)

**Objective.** Replace the toy DGP with one that reflects real markets: **correlated** factors and **fat-tailed** returns. Add BH and BY. Produce the project's headline result — the widening gap between "naive t > 2" and "what's actually safe."

**Why it matters.** Independence and normality make the corrections look easy. Reality is harder, and showing you handled it is what separates this from a homework exercise.

**(a) Correlated factors via a latent-factor covariance (your roadblock #1).** Do **not** hand-build a covariance matrix and hope it's positive semi-definite. Construct one that is valid *by design* and financially meaningful — a few latent macro drivers plus idiosyncratic noise:

$$
\Sigma = B B^{\top} + D,
$$

where $B$ is $m \times k$ factor loadings (small $k$, e.g. 3–5 latent drivers) and $D = \mathrm{diag}(d_1, \dots, d_m)$ with $d_i > 0$. This is automatically positive (semi-)definite, and the loadings give you block/clustered structure you can dial up or down. To draw a correlated vector with mean $\mu$:

$$
x = \mu + L z, \qquad L = \mathrm{chol}(\Sigma), \qquad z \sim \mathcal{N}(0, I_m),
$$

so that $\mathrm{Cov}(x) = L L^{\top} = \Sigma$.

```python
def factor_covariance(m, k=3, idio=0.5, rng=None):
    rng = rng or np.random.default_rng()
    B = rng.normal(size=(m, k)) * 0.6        # loadings on k latent drivers
    D = np.diag(rng.uniform(idio, 1.0, size=m))
    Sigma = B @ B.T + D
    # normalize to a correlation matrix so diagonals are 1
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d)

L = np.linalg.cholesky(Sigma)
draws = mu + (L @ rng.normal(size=(m, n_reps)))   # each column is one correlated trial
```

If you ever build $\Sigma$ from a hand-tuned target that isn't PSD, repair it by **eigenvalue clipping** (eigendecompose, set negative eigenvalues to a small $\varepsilon$, reconstruct, rescale the diagonal to 1); the rigorous version is Higham's nearest-correlation-matrix algorithm. Mention you know the distinction even if you only use clipping.

**(b) Fat tails via the multivariate Student-t (your roadblock #2).** Equity returns have heavier tails than the normal, which produces more extreme t-stats by chance. Generate multivariate-t draws by scaling the Gaussian:

$$
X = \mu + \frac{L z}{\sqrt{w / \nu}}, \qquad z \sim \mathcal{N}(0, I_m), \quad w \sim \chi^2_\nu,
$$

independent. The marginals are Student-t with $\nu$ degrees of freedom; small $\nu$ (3–5) is realistically fat-tailed, and $\nu \to \infty$ recovers the Gaussian — giving you a clean dial to study.

```python
def mvt_draws(mu, L, nu, n_reps, rng):
    m = L.shape[0]
    z = rng.normal(size=(m, n_reps))
    w = rng.chisquare(nu, size=n_reps)
    return mu[:, None] + (L @ z) / np.sqrt(w / nu)
```

**(c) The experiment.** Re-run the size/power analysis across a grid of $m$, correlation strength (via $k$ and loading scale), and tail thickness $\nu$. The findings to surface:
- Uncorrected testing over-rejects *even more* as tails fatten and correlation rises.
- BY still controls error under this dependence, but at a visible cost in power versus BH.
- The implied safe t-stat threshold drifts well above 2 — quantify how far.

**Deliverable / Definition of Done.** The "money chart": empirical FWER/FDR (corrected and uncorrected) as a function of $m$, correlation, and $\nu$, plus the implied-threshold curve. `simulate.py` exposes a clean API used by later phases. All four corrections pass their reference-diff tests.

---

### Phase 4 — Deflated Sharpe Ratio (5h)

**Objective.** Implement the Probabilistic Sharpe Ratio (PSR) and then the Deflated Sharpe Ratio (DSR), connecting the cross-sectional multiple-testing story to the time-series backtest-overfitting story.

**Math.** Let $\hat{\mathrm{SR}}$ be the per-period (not annualized) Sharpe of a return stream of length $T$, with skewness $\gamma_3$ and kurtosis $\gamma_4$ (non-excess; $\gamma_4 = 3$ for a normal). The PSR — the probability the true Sharpe exceeds a benchmark $\mathrm{SR}^\*$ — is:

$$
\widehat{\mathrm{PSR}}(\mathrm{SR}^\*) = \Phi\!\left[
\frac{\left(\hat{\mathrm{SR}} - \mathrm{SR}^\*\right)\sqrt{T - 1}}
{\sqrt{1 - \gamma_3\,\hat{\mathrm{SR}} + \dfrac{\gamma_4 - 1}{4}\,\hat{\mathrm{SR}}^2}}
\right].
$$

(Note the denominator is the standard error of the Sharpe estimate; for normal returns it reduces to $\sqrt{(1 + \tfrac12 \hat{\mathrm{SR}}^2)/(T-1)}$ — this is where non-normality enters honestly.)

The **DSR** is the PSR evaluated at the benchmark equal to the *expected maximum* Sharpe across $N$ trials:

$$
\mathrm{SR}^\* = \sqrt{\hat{V}}\left[(1 - \gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right],
$$

where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant, $\hat{V}$ is the variance of the Sharpe ratios across the $N$ trials, and $e$ is Euler's number. **This $\mathrm{SR}^\*$ is the extreme-value (Gumbel) approximation to the expected maximum of $N$ draws — exactly the order-statistic from Section 2, now on the Sharpe axis.** Call that connection out explicitly; it's the intellectual core of the whole project.

**Steps.**
1. Implement `psr(sr, T, skew, kurt, sr_benchmark)` and `deflated_sharpe(sr, T, skew, kurt, n_trials, sr_variance)`.
2. **Validate against your own simulation:** generate $N$ strategies with *zero* true skill, confirm the realized maximum Sharpe tracks $\mathrm{SR}^\*$, and confirm a genuinely borderline strategy returns DSR $\approx 0.5$.

**Deliverable / Definition of Done.** `dsr.py` with tested `psr` and `deflated_sharpe`, plus a validation notebook showing the simulated max-Sharpe matches the analytic $\mathrm{SR}^\*$.

---

### Phase 5 — Real-data application (5h)

**Objective.** Point the tool at real published factors and report how many survive. This is the phase that turns a simulation into a finding.

**Data (your roadblock #3 — solved).** Use the Chen–Zimmermann *Open Source Cross-Sectional Asset Pricing* dataset: `pip install openassetpricing`. It provides 212 published cross-sectional return predictors, each with the **original published t-stat** and a documentation flag rating its predictability as "clear" or "likely." That removes the 10+ hours you'd otherwise lose hand-compiling a list — you load a table and go.

**Steps.**
1. In `data.py`, load the published-predictor table and extract each factor's reported t-stat. Keep the documentation/predictability flags for slicing later.
2. The number of trials $m$ is genuinely unknown (it includes all the unpublished tests nobody reports). Run the analysis across **several plausible $m$** (e.g. 300, 600, 1000+) and treat that sensitivity as a result in itself — *how robust is "survival" to your assumption about how much searching happened?*
3. For each factor, compute the BY-adjusted significance, the implied threshold, and the haircut; flag survivors vs. casualties under both FWER and FDR control.
4. Report the headline figure: the fraction of the 212 factors that clear a multiplicity-adjusted bar (e.g. an HLZ-style threshold around 3.0).

**Deliverable / Definition of Done.** A results table (factor, published t, adjusted significance, haircut, survives?) and a "survivors vs. casualties" chart. One clearly stated headline number you can defend.

---

### Phase 6 — Streamlit dashboard & writeup (5h)

**Objective.** Wrap everything in an interactive web app and finish the documentation so a recruiter can engage with it in two minutes.

**App spec.**
- **Inputs:** a reported t-stat (or a Sharpe + sample length $T$), the estimated number of trials $m$, an assumed test correlation, and the tail parameter $\nu$.
- **Outputs:** the multiplicity-adjusted threshold, the haircut %, the DSR, and a live verdict ("survives / does not survive"), alongside a plot of the null max-statistic distribution with the user's factor marked on it.
- **Second tab:** render the Phase 5 results on the real Chen–Zimmermann factors.

Because Phases 2–5 live in importable modules, `app.py` is mostly wiring inputs to functions and functions to charts — no logic duplication.

**Writeup.** Finish `README.md`: the question, the method, the money chart, the headline finding, and a short "what I learned about t-statistics" paragraph. Include a screenshot or GIF of the app. Optionally deploy to Streamlit Community Cloud and put the live link at the top.

**Deliverable / Definition of Done.** A running, shareable app and a README that stands on its own.

---

## 7. Time budget

| Phase | Focus | Hours |
|------:|-------|------:|
| 0 | Setup & scoping | 2 |
| 1 | Theory & written notes | 8 |
| 2 | Minimal viable simulation | 6 |
| 3 | Realistic DGP (correlation + fat tails) | 9 |
| 4 | Deflated Sharpe Ratio | 5 |
| 5 | Real-data application | 5 |
| 6 | Streamlit dashboard & writeup | 5 |
| | **Total** | **40** |

There is no slack here. If you fall behind, cut from the *stretch goals* below, not from Phases 1–3 — the theory and the verified simulation are the parts that make this real.

---

## 8. Resume line

Fill in `Z` with the number *you* measure in Phase 5 (don't borrow one from a paper — it has to be yours to defend in an interview):

> **Built** an interactive factor "haircut" and Deflated Sharpe Ratio calculator (Python, NumPy/pandas, Streamlit) **using** Monte Carlo simulation with latent-factor correlation (Cholesky-induced) and fat-tailed returns plus Benjamini–Yekutieli false-discovery control, **that** showed ~`[Z]`% of the 212 published equity factors in the Chen–Zimmermann dataset fail to clear a multiple-testing–adjusted significance threshold.

---

## 9. Stretch goals

Pursue these only after Phase 6 is complete and shippable.

- **Probability of Backtest Overfitting (PBO)** via Combinatorially-Symmetric Cross-Validation (Bailey et al., 2017). High impact and the natural companion to DSR — but the single heaviest item, easily 8+ hours, so it's deliberately out of the core 40.
- **Unknown-$m$ inference.** Treat the number of trials as a latent quantity and put a range/prior on it, reporting survival as a function of that belief.
- **Power analysis for the corrections.** Quantify how much true signal each method sacrifices, to argue *which* correction a desk should actually use.
- **Bayesian shrinkage of t-stats** (an empirical-Bayes alternative to frequentist correction) as a comparison method.

---

## 10. Formula reference card

**t-stat → p-value (two-sided):** $\;p = 2(1 - \Phi(|t|))$

**Expected max of $m$ standard normals:** $\;\mathbb{E}[\max_i t_i] \approx \sqrt{2 \ln m}$

**FWER:** $\;\Pr(V \ge 1)$  **FDR:** $\;\mathbb{E}[V/R]$

**Bonferroni:** reject if $p_i \le \alpha/m$

**Holm (step-down):** reject $p_{(i)}$ while $p_{(i)} \le \alpha/(m-i+1)$

**Benjamini–Hochberg (step-up):** largest $k$ with $p_{(k)} \le \tfrac{k}{m}\alpha$

**Benjamini–Yekutieli:** largest $k$ with $p_{(k)} \le \tfrac{k}{m\,c(m)}\alpha$, $\;c(m)=\sum_{i=1}^m \tfrac1i$

**Implied threshold (Bonferroni, two-sided):** $\;t^\* = \Phi^{-1}(1 - \alpha/(2m))$

**Haircut:** $\;1 - \mathrm{SR}_{\text{adj}}/\mathrm{SR}$

**Valid covariance:** $\;\Sigma = BB^\top + D$, simulate $x = \mu + Lz$, $L=\mathrm{chol}(\Sigma)$

**Multivariate-t draw:** $\;X = \mu + Lz/\sqrt{w/\nu}$, $z\sim\mathcal N(0,I)$, $w\sim\chi^2_\nu$

**PSR:** $\;\Phi\!\left[(\hat{\mathrm{SR}}-\mathrm{SR}^\*)\sqrt{T-1}\big/\sqrt{1-\gamma_3\hat{\mathrm{SR}}+\tfrac{\gamma_4-1}{4}\hat{\mathrm{SR}}^2}\right]$

**DSR benchmark:** $\;\mathrm{SR}^\*=\sqrt{\hat V}\big[(1-\gamma)\Phi^{-1}(1-\tfrac1N)+\gamma\Phi^{-1}(1-\tfrac{1}{Ne})\big]$, $\gamma\approx0.5772$

---

## 11. References & reading list

- Harvey, C., Liu, Y., & Zhu, H. (2016). *…and the Cross-Section of Expected Returns.* Review of Financial Studies. — the multiple-testing-in-finance paper; the source of the "haircut" framing.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate.* JRSS-B. — the original FDR method.
- Benjamini, Y., & Yekutieli, D. (2001). *The Control of the False Discovery Rate Under Dependency.* Annals of Statistics. — the dependence-robust version and the $c(m)$ factor.
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure.* — the step-down FWER method.
- Bailey, D., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. (2017). *The Probability of Backtest Overfitting.* Journal of Computational Finance. — for the PBO stretch goal.
- Chen, A., & Zimmermann, T. (2022). *Open Source Cross-Sectional Asset Pricing.* Critical Finance Review. — the 212-factor dataset; `pip install openassetpricing`, docs at openassetpricing.com.
- Higham, N. (2002). *Computing the Nearest Correlation Matrix.* — the rigorous PSD-repair method.

---

## 12. Glossary

- **Factor / anomaly:** a stock characteristic claimed to predict cross-sectional returns.
- **Multiple testing / data snooping:** running many hypothesis tests, which inflates the chance of false positives.
- **FWER:** probability of at least one false discovery across all tests.
- **FDR:** expected fraction of discoveries that are false.
- **Haircut:** the proportional reduction in a strategy's Sharpe/return once you correct its significance for multiplicity.
- **PSR / DSR:** Probabilistic / Deflated Sharpe Ratio — the probability a Sharpe beats a benchmark, where the DSR's benchmark accounts for the number of trials.
- **Cholesky decomposition:** factorization $\Sigma = LL^\top$ used to turn independent noise into correlated draws.
- **Positive semi-definite (PSD):** the property a valid covariance matrix must have (no negative variances in any direction).
- **Euler–Mascheroni constant ($\gamma \approx 0.5772$):** appears in the expected-maximum / extreme-value formula behind the DSR benchmark.
