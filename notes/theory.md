# Theory notes — The Factor Zoo Haircut

Phase 1 deliverable. Work through each section before writing any production code.
Each section should contain: the result, a derivation or careful reproduction, and
a numerical sanity check (either by hand or pointing to notebook 01).

---

## 1. t-statistics and p-values

**Setup.** For a regression coefficient β̂ with standard error se(β̂), the t-stat is
t = β̂ / se(β̂). Under H₀: β = 0 and OLS assumptions, t ~ N(0,1) asymptotically.

**Two-sided p-value:** p = 2(1 − Φ(|t|))

**What p = 0.05 promises:**
<!-- TODO: write this in your own words — what it guarantees and what it does NOT -->

**What it does NOT promise:**
<!-- TODO -->

---

## 2. The maximum-of-m-draws problem

**Claim:** E[max_{i=1..m} t_i] ≈ sqrt(2 ln m) for m i.i.d. N(0,1).

**Derivation:**
<!-- TODO: reproduce via order-statistic / extreme-value argument -->

**Table: expected maximum vs m**

| m    | sqrt(2 ln m) | Simulated E[max] |
|------|-------------|-----------------|
| 10   |             |                 |
| 50   |             |                 |
| 100  |             |                 |
| 300  |             |                 |
| 1000 |             |                 |

*(Fill in from notebook 01.)*

**Interpretation:** if you test ~300 worthless factors, you should expect the best
to post t ≈ 3.38 by pure chance. The threshold t > 2 is meaningless in that world.

---

## 3. FWER vs FDR

Let V = false rejections, R = total rejections.

**FWER = Pr(V ≥ 1)** — probability of even one false discovery.
Use when: a single false factor is expensive (e.g. a concentrated long-only bet).

**FDR = E[V/R]** (0/0 ≡ 0) — expected fraction of discoveries that are false.
Use when: you're trading a diversified basket and can tolerate some duds.

---

## 4. The four corrections

### 4a. Bonferroni (FWER)

Reject H_i if p_i ≤ α/m.

**Why it works:** by a union bound, Pr(any false rejection) ≤ m · (α/m) = α.
**Limitation:** conservative (low power), especially under positive correlation.

**Worked example (five p-values):**
<!-- TODO: p = [0.001, 0.008, 0.039, 0.041, 0.210], alpha = 0.05, m = 5 -->

### 4b. Holm (FWER, step-down)

Sort ascending p_(1) ≤ ... ≤ p_(m). For i = 1, 2, ...:
  reject H_(i) as long as p_(i) ≤ α/(m − i + 1); stop at the first failure.

**Why it's uniformly more powerful than Bonferroni:**
<!-- TODO -->

**Worked example:**
<!-- TODO -->

### 4c. Benjamini–Hochberg (FDR, step-up)

Sort ascending. Find the largest k such that p_(k) ≤ (k/m)·α.
Reject H_(1), ..., H_(k).

**Valid under:** independence and positive regression dependence (PRDS).

**Worked example:**
<!-- TODO -->

### 4d. Benjamini–Yekutieli (FDR, arbitrary dependence)

Same as BH but threshold is (k / (m·c(m)))·α, where c(m) = Σ_{i=1}^{m} 1/i.

**Why c(m) appears:**
<!-- TODO: this is the conceptual high point — derive or carefully explain -->

**c(m) for key values:**
- m = 100: c(100) ≈ 5.19
- m = 300: c(300) ≈ 6.28
- m = 1000: c(1000) ≈ 7.49

**Why HLZ use BY:** factor tests are correlated in unknown ways. BH's guarantee
can fail under negative dependence. BY pays c(m) to be valid regardless.

**Worked example:**
<!-- TODO -->

---

## 5. The haircut and implied threshold

**Haircut:** proportional reduction in SR after adjusting for multiplicity.
  haircut = 1 − t_adjusted / t_original = 1 − SR_adjusted / SR_original

**Implied threshold (Bonferroni, two-sided):**
  t* = Φ⁻¹(1 − α/(2m))

**Table: implied threshold vs m (Bonferroni, α = 0.05)**

| m    | t*  |
|------|-----|
| 1    | 1.96|
| 10   |     |
| 100  |     |
| 300  |     |
| 1000 |     |

*(Fill in from src/corrections.py.)*

---

## 6. DSR bridge (preview)

The DSR benchmark SR* is the expected maximum Sharpe across N trial strategies —
the same order-statistic logic as E[max t_i] ~ sqrt(2 ln m), but on the Sharpe axis.

SR* = sqrt(V̂) · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(Ne))]

where γ ≈ 0.5772 is the Euler–Mascheroni constant.

Connection to Section 2: <!-- TODO: write out the parallel explicitly -->
