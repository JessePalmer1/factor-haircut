"""
Streamlit dashboard — The Factor Zoo Haircut

Tab 1: Interactive calculator
  Inputs : reported t-stat (or Sharpe + T), number of trials m,
           correlation strength (k latent drivers), tail parameter nu
  Outputs: adjusted threshold, haircut %, DSR, verdict,
           + plot of null max-statistic distribution with factor marked

Tab 2: Chen–Zimmermann real-data results (Phase 5 output)
  Shows the full 212-factor table with survival flags under several m assumptions.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Phase 2–5 modules — wiring only; all logic lives in src/
# from src.corrections import implied_threshold, haircut, benjamini_yekutieli
# from src.simulate import factor_covariance, mvt_draws
# from src.dsr import deflated_sharpe
# from src.data import load_published_factors, compute_pvals

st.set_page_config(page_title="Factor Zoo Haircut", layout="wide")
st.title("The Factor Zoo Haircut")
st.caption("Does your factor survive multiple-testing adjustment?")

tab1, tab2 = st.tabs(["Calculator", "Published Factors (Chen–Zimmermann)"])

# ---------------------------------------------------------------------------
# Tab 1 — Interactive calculator
# ---------------------------------------------------------------------------
with tab1:
    st.header("Factor significance calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Inputs")
        t_reported = st.number_input("Reported t-statistic", value=3.0, step=0.1)
        m_trials = st.slider("Number of trials tested (m)", min_value=10, max_value=2000, value=300, step=10)
        nu = st.slider("Tail thickness ν (degrees of freedom; ∞ = Gaussian)", min_value=3, max_value=100, value=5)
        alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05)

        st.divider()
        st.subheader("Deflated Sharpe Ratio inputs")
        sr_hat = st.number_input("Estimated Sharpe ratio (per period)", value=0.5, step=0.05)
        T_obs = st.number_input("Number of return observations T", value=252, step=12)
        sr_variance = st.number_input("Variance of Sharpe ratios across trials", value=0.1, step=0.01)

    with col2:
        st.subheader("Results")
        st.info("Implement Phase 2–4 modules, then wire outputs here.")
        # TODO: call implied_threshold, haircut, deflated_sharpe and display results

# ---------------------------------------------------------------------------
# Tab 2 — Real-data results
# ---------------------------------------------------------------------------
with tab2:
    st.header("Chen–Zimmermann published factors")
    st.info("Load results from Phase 5 (src/data.py) and display survival table here.")
    # TODO: load_published_factors() -> compute_pvals() -> apply BY -> render table
