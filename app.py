"""
Streamlit dashboard — The Factor Zoo Haircut

Tab 1: Interactive calculator
  Inputs : reported t-stat, number of trials m, tail parameter nu, alpha
           + Sharpe / T / sr_variance for DSR
  Outputs: Bonferroni threshold, haircut %, DSR, verdict,
           + plot of null max-statistic distribution with factor marked

Tab 2: Chen–Zimmermann real-data results
  Shows the 188 published factors with survival flags under several m assumptions.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, ".")
from src.corrections import implied_threshold, haircut
from src.dsr import deflated_sharpe, dsr_benchmark
from src.data import (
    load_published_factors,
    filter_by_predictability,
    compute_pvals,
    apply_corrections_sweep,
)

st.set_page_config(page_title="Factor Zoo Haircut", layout="wide")
st.title("The Factor Zoo Haircut")
st.caption(
    "Given that hundreds of factors have been tested, what t-statistic does a factor "
    "*actually* need to be believable?"
)

tab1, tab2 = st.tabs(["Calculator", "Published Factors (Chen–Zimmermann)"])


# ---------------------------------------------------------------------------
# Tab 1 — Interactive calculator
# ---------------------------------------------------------------------------
with tab1:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Inputs")

        t_reported = st.number_input(
            "Reported t-statistic", value=3.0, min_value=0.0, step=0.1,
            help="The published (or observed) absolute t-stat for the factor.",
        )
        m_trials = st.slider(
            "Assumed number of trials tested (m)", 10, 2000, 300, 10,
            help="Includes unpublished tests. Try 300–1000 to see sensitivity.",
        )
        alpha = st.select_slider(
            "Significance level α", options=[0.01, 0.05, 0.10], value=0.05,
        )

        st.divider()
        st.subheader("Deflated Sharpe Ratio")

        sr_hat = st.number_input(
            "Estimated Sharpe ratio (per-period)", value=0.10, step=0.01,
            help="Per-period (not annualised). Monthly SR ≈ annual SR / √12.",
        )
        T_obs = st.number_input(
            "Return observations T", value=252, min_value=10, step=12,
        )
        skew_input = st.number_input("Return skewness", value=0.0, step=0.1)
        kurt_input = st.number_input(
            "Return kurtosis (non-excess; normal = 3)", value=3.0, min_value=1.0, step=0.1,
        )
        sr_variance = st.number_input(
            "Variance of Sharpe ratios across trials", value=0.01, min_value=0.0, step=0.001,
            format="%.4f",
        )

    with col_out:
        st.subheader("Results")

        # --- Multiple-testing correction ---
        t_star = implied_threshold(m_trials, alpha=alpha, method="bonferroni")
        survives = t_reported >= t_star
        hc = max(0.0, 1.0 - t_star / t_reported) if t_reported > 0 else 1.0

        if survives:
            st.success(f"**SURVIVES** the Bonferroni bar (m = {m_trials})")
        else:
            st.error(f"**DOES NOT SURVIVE** the Bonferroni bar (m = {m_trials})")

        c1, c2, c3 = st.columns(3)
        c1.metric("Required t*", f"{t_star:.2f}", help="Bonferroni threshold")
        c2.metric("Your t", f"{t_reported:.2f}")
        c3.metric("Haircut", f"{hc:.1%}", help="1 − t* / t_reported")

        st.divider()

        # --- DSR ---
        st.markdown("**Deflated Sharpe Ratio**")
        if m_trials >= 2 and sr_variance > 0:
            sr_star = dsr_benchmark(m_trials, sr_variance)
            dsr_val = deflated_sharpe(sr_hat, int(T_obs), skew_input, kurt_input, m_trials, sr_variance)
            d1, d2, d3 = st.columns(3)
            d1.metric("SR benchmark (SR*)", f"{sr_star:.4f}", help="Expected max SR across trials")
            d2.metric("DSR", f"{dsr_val:.3f}", help="P(true SR > SR*). >0.5 = genuine skill")
            if dsr_val > 0.95:
                d3.markdown("**Verdict:** :green[Strong skill signal]")
            elif dsr_val > 0.5:
                d3.markdown("**Verdict:** :orange[Marginal — above noise floor]")
            else:
                d3.markdown("**Verdict:** :red[Indistinguishable from noise]")
        else:
            st.info("Set m ≥ 2 and sr_variance > 0 to compute DSR.")

        st.divider()

        # --- Null distribution plot ---
        st.markdown("**Null max-statistic distribution**")
        st.caption(
            f"Distribution of the *maximum* t-stat across {m_trials} pure-noise tests. "
            "Your factor is the vertical line."
        )

        rng = np.random.default_rng(42)
        n_sim = 20_000
        max_tstats = np.array([
            rng.standard_normal(m_trials).max() for _ in range(n_sim)
        ])
        expected_max = np.sqrt(2 * np.log(m_trials))

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(max_tstats, bins=80, density=True, color="#4C72B0", alpha=0.75, label="Null max t-stat")
        ax.axvline(t_reported, color="#DD4444", linewidth=2, label=f"Your factor  t={t_reported:.2f}")
        ax.axvline(t_star, color="#228B22", linewidth=1.5, linestyle="--", label=f"Bonferroni t*={t_star:.2f}")
        ax.axvline(expected_max, color="#FF8C00", linewidth=1, linestyle=":", label=f"E[max] ≈{expected_max:.2f}")
        ax.set_xlabel("Max t-statistic across m noise tests")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 2 — Real-data results
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Chen–Zimmermann published factors — multiple-testing survival")
    st.caption(
        "188 of the 212 published cross-sectional predictors have a reported t-stat. "
        "Bonferroni uses an *assumed* total trial count m (including unpublished tests). "
        "BY controls the false discovery rate within the 188 observed factors."
    )

    @st.cache_data(show_spinner="Loading Chen–Zimmermann data…")
    def get_results(m_list):
        df = load_published_factors()
        df = filter_by_predictability(df, level="likely")
        df = compute_pvals(df)
        return apply_corrections_sweep(df, m_assumed_list=m_list)

    M_VALUES = [300, 600, 1000]
    results = get_results(M_VALUES)

    # --- Summary metrics ---
    m_sel = st.select_slider(
        "Assumed m for Bonferroni column",
        options=M_VALUES,
        value=600,
    )
    t_bar = results[f"threshold_bonf_{m_sel}"].iloc[0]
    n_bonf = int(results[f"survive_bonf_{m_sel}"].sum())
    n_by   = int(results[f"survive_by_{m_sel}"].sum())
    n_total = len(results)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Factors analysed", n_total)
    m2.metric(f"Bonferroni t* (m={m_sel})", f"{t_bar:.2f}")
    m3.metric("Bonferroni survivors", f"{n_bonf} ({100*n_bonf//n_total}%)")
    m4.metric("BY survivors (FDR 5%)", f"{n_by} ({100*n_by//n_total}%)")

    # --- Survival sensitivity chart ---
    st.markdown("**Survival rate vs assumed m**")
    bonf_counts = [int(results[f"survive_bonf_{m}"].sum()) for m in M_VALUES]
    by_counts   = [int(results[f"survive_by_{m}"].sum())   for m in M_VALUES]

    fig2, ax2 = plt.subplots(figsize=(5, 2.8))
    ax2.plot(M_VALUES, [100*c/n_total for c in bonf_counts], "o-", color="#DD4444", label="Bonferroni")
    ax2.plot(M_VALUES, [100*c/n_total for c in by_counts],   "s--", color="#4C72B0", label="BY (FDR)")
    ax2.set_xlabel("Assumed total trials m")
    ax2.set_ylabel("% factors surviving")
    ax2.set_ylim(0, 100)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # --- Full factor table ---
    st.markdown("**Full factor table**")

    survive_col = f"survive_bonf_{m_sel}"
    by_col      = f"survive_by_{m_sel}"
    hc_col      = f"haircut_bonf_{m_sel}"

    display = results[[
        "t_stat", "p_value", "predictability", "year",
        survive_col, by_col, hc_col,
    ]].copy()
    display.columns = [
        "t-stat", "p-value", "predictability", "year",
        f"Bonferroni (m={m_sel})", "BY (FDR 5%)", "Haircut (Bonf)",
    ]
    display = display.sort_values("t-stat", ascending=False)
    display["p-value"]       = display["p-value"].map("{:.4f}".format)
    display["Haircut (Bonf)"] = display["Haircut (Bonf)"].map("{:.1%}".format)

    filter_surv = st.checkbox("Show survivors only", value=False)
    if filter_surv:
        display = display[display[f"Bonferroni (m={m_sel})"] == True]

    st.dataframe(display, use_container_width=True, height=500)
