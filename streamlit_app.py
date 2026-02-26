#!/usr/bin/env python3
"""
Welfare Evaluation App — Streamlit Web Version
================================================
Evaluates lifetime welfare across climate-work scenarios using
additive log + CRRA-in-leisure, CES, and ecological specifications.

Deploy on Streamlit Community Cloud for a shareable URL.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Welfare Evaluation",
    page_icon="\U0001F30D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour constants ────────────────────────────────────────────
COLOUR_SC = "#1B7A8A"
COLOUR_PC = "#D84727"
COLOUR_SC_LIGHT = "#7EC8CF"
COLOUR_PC_LIGHT = "#F0A08C"

# ── Region abbreviations ────────────────────────────────────────
REGION_ABBREV = {
    "East Asia": "EASA",
    "Europe": "EURO",
    "Latin America": "LATAM",
    "Middle East/North Africa": "MENA",
    "North America/ Oceania": "NAOC",
    "Russia/ Central Asia": "RUCA",
    "South & South-East Asia": "SSEA",
    "Subsaharan Africa": "SSAF",
    "World": "WORLD",
}


def abbrev(name):
    return REGION_ABBREV.get(name, name[:6])


# ── Default parameters ──────────────────────────────────────────
FIXED_T = 4000.0  # fixed time endowment (hrs/yr) — not user-adjustable

DEFAULT_PARAMS = {
    "T": FIXED_T,
    "sigma": 1.0,
    "r": 2.0,
    "phi": 12.0,
    "gamma_raw": 13.33,
    "epsilon": 0.8,
}


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(file_bytes):
    """Load and parse the welfare input Excel file from uploaded bytes."""
    wb = pd.ExcelFile(io.BytesIO(file_bytes))
    result = {}
    for sheet in ["SC", "PC"]:
        raw = pd.read_excel(wb, sheet_name=sheet, header=None)
        first_header = raw.index[raw.iloc[:, 1] == "Variable"].tolist()[0]
        region_names = raw.iloc[first_header, 2:].dropna().tolist()
        data_rows = raw[raw.iloc[:, 1] != "Variable"].copy()
        data_rows = data_rows.dropna(subset=[0, 1])
        data_rows = data_rows[pd.to_numeric(data_rows.iloc[:, 0], errors="coerce").notna()]
        records = []
        for _, row in data_rows.iterrows():
            year = int(row.iloc[0])
            var_name = str(row.iloc[1])
            for j, region in enumerate(region_names):
                records.append({
                    "Year": year, "Region": region,
                    "Variable": var_name, "Value": float(row.iloc[2 + j]),
                })
        df_long = pd.DataFrame(records)
        df_wide = df_long.pivot_table(
            index=["Year", "Region"], columns="Variable", values="Value"
        ).reset_index()
        col_map = {}
        for c in df_wide.columns:
            cl = c.lower() if isinstance(c, str) else c
            if "income" in str(cl) or "consumption" in str(cl) or "gdp" in str(cl):
                col_map[c] = "Income"
            elif "labour" in str(cl) or "labor" in str(cl):
                col_map[c] = "LabourHours"
            elif "temperature" in str(cl) or "temp" in str(cl):
                col_map[c] = "TempChange"
            elif "population" in str(cl):
                col_map[c] = "Population"
        df_wide.rename(columns=col_map, inplace=True)
        result[sheet] = df_wide
    return result


# ══════════════════════════════════════════════════════════════════
# WELFARE COMPUTATION  (identical to desktop app)
# ══════════════════════════════════════════════════════════════════

def param_to_coeff(pct_per_degC):
    return pct_per_degC / 100.0


def calibrate_chi(y0, h0, T, sigma):
    ell0 = T - h0
    if ell0 <= 0 or h0 <= 0:
        return 0.0
    return (ell0 ** sigma) / h0


def effective_income(y, delta_T, phi, gamma, damage_type="exponential"):
    coeff = phi + gamma
    if damage_type == "quadratic":
        # DICE/Nordhaus-style: convex in ΔT, never hits zero
        return y / (1.0 + coeff * delta_T ** 2)
    return y * np.exp(-coeff * delta_T)


def period_utility_additive(y_hat, h, T, sigma, chi):
    ell = T - h
    ell = np.maximum(ell, 1e-6)
    y_hat = np.maximum(y_hat, 1e-6)
    u_inc = np.log(y_hat)
    if abs(sigma - 1.0) < 1e-10:
        u_leisure = chi * np.log(ell)
    else:
        u_leisure = chi * (ell ** (1.0 - sigma)) / (1.0 - sigma)
    return u_inc + u_leisure


def period_utility_ces(y_hat, h, T, epsilon, alpha):
    ell = T - h
    ell = np.maximum(ell, 1e-6)
    y_hat = np.maximum(y_hat, 1e-6)
    rho = (epsilon - 1.0) / epsilon
    if abs(rho) < 1e-10:
        return alpha * np.log(y_hat) + (1 - alpha) * np.log(ell)
    agg = alpha * (y_hat ** rho) + (1 - alpha) * (ell ** rho)
    agg = np.maximum(agg, 1e-30)
    return np.log(agg) / rho


def calibrate_alpha_ces(y0, h0, T, epsilon):
    ell0 = T - h0
    if ell0 <= 0 or h0 <= 0:
        return 0.5
    vot = y0 / h0
    mrs_ratio = vot * (ell0 / y0) ** (1.0 / epsilon)
    return 1.0 / (1.0 + mrs_ratio)


def env_quality(delta_T, kappa):
    """Environmental quality index: E = 1 / (1 + kappa * delta_T).
    E = 1 at zero warming (pristine), declines toward 0 as warming rises."""
    return 1.0 / (1.0 + kappa * np.maximum(delta_T, 0.0))


def effective_income_output_only(y, delta_T, phi, damage_type="exponential"):
    """Effective income with OUTPUT damage only (no well-being/amenity damage).
    Used in the ecological specification where non-market damages enter via E."""
    if damage_type == "quadratic":
        return y / (1.0 + phi * delta_T ** 2)
    return y * np.exp(-phi * delta_T)


def period_utility_ecological(y_eff, h, E, T, sigma, chi, alpha_cE, epsilon_cE, y_ref=1.0):
    """Ecological utility: CES composite of (normalized income, environment) + CRRA leisure.

    u = (1/rho) * ln[alpha * y_norm^rho + (1-alpha) * E^rho] + chi * ell^(1-sigma)/(1-sigma)

    where y_norm = y_eff / y_ref normalises income to be on a comparable scale to E (both ~1),
    and rho = (epsilon_cE - 1) / epsilon_cE controls substitutability.

    Without normalisation, y ~ 50,000 vs E ~ 0.7-1.0, and the CES composite
    breaks down for rho < 0 (the y^rho term becomes negligible).
    """
    ell = T - h
    ell = np.maximum(ell, 1e-6)
    y_eff = np.maximum(y_eff, 1e-6)
    E = np.maximum(E, 1e-6)

    # Normalise income to be on comparable scale to E
    y_norm = y_eff / y_ref

    # CES composite of normalised income and environment
    rho = (epsilon_cE - 1.0) / epsilon_cE
    if abs(rho) < 1e-10:
        # Cobb-Douglas limit: rho -> 0
        u_composite = alpha_cE * np.log(y_norm) + (1.0 - alpha_cE) * np.log(E)
    elif epsilon_cE < 0.02:
        # Near-Leontief: use min
        u_composite = np.log(np.minimum(y_norm ** alpha_cE, E ** (1.0 - alpha_cE)))
    else:
        agg = alpha_cE * (y_norm ** rho) + (1.0 - alpha_cE) * (E ** rho)
        agg = np.maximum(agg, 1e-30)
        u_composite = np.log(agg) / rho

    # Leisure component (same as additive)
    if abs(sigma - 1.0) < 1e-10:
        u_leisure = chi * np.log(ell)
    else:
        u_leisure = chi * (ell ** (1.0 - sigma)) / (1.0 - sigma)

    return u_composite + u_leisure


def period_utility_augmented_gdp(y_hat, h, T):
    """Augmented GDP utility: u = y + (y/h) * ell.

    Linear in income and leisure, with leisure valued at the
    implied wage rate y/h.  Equivalent to u = y * T / h.
    No diminishing returns — this is the implicit utility of the
    augmented-GDP approach used in GJP.
    """
    ell = T - h
    ell = max(ell, 1e-6)
    y_hat = max(y_hat, 1e-6)
    h = max(h, 1e-6)
    return y_hat + (y_hat / h) * ell


def income_equivalent_linear(W_sc, W_pc):
    """Income equivalent for linear (Augmented GDP) utility.

    Since u is linear in y, a uniform % increase in income
    scales welfare proportionally: lambda = W_sc / W_pc - 1.
    """
    if abs(W_pc) < 1e-30:
        return 0.0
    return (W_sc / W_pc - 1.0) * 100.0


def calibrate_kappa(gamma_pct):
    """Calibrate kappa so that the marginal welfare loss from E at 1°C warming
    roughly matches the non-market damage evidence (gamma %/°C).

    For small ΔT, E ≈ 1 - kappa*ΔT, so the proportional loss is kappa*ΔT.
    At 1°C we want this to be gamma/100.  Hence kappa ≈ gamma/100.
    """
    return gamma_pct / 100.0


def calibrate_alpha_ecological(y0, E0, gamma_pct):
    """Calibrate alpha_cE so that at the baseline the share of welfare
    attributed to environment matches the empirical damage evidence.

    We anchor so that the expenditure share on the environment composite
    equals gamma / (phi + gamma) — i.e. the non-market share of total damages.
    """
    # Default: equal weight, user can override
    return 0.5


def compute_welfare(df, params, spec="additive"):
    T = params["T"]
    sigma = params["sigma"]
    r = params["r"] / 100.0
    phi = param_to_coeff(params["phi"])
    gamma = param_to_coeff(params["gamma_raw"])
    epsilon = params["epsilon"]
    beta = 1.0 / (1.0 + r)
    damage_type = params.get("damage_type", "exponential")

    years = df["Year"].values
    y = df["Income"].values
    h = df["LabourHours"].values
    dT_increments = df["TempChange"].values
    dT = np.cumsum(dT_increments)
    n = len(years)
    t_idx = np.arange(n)

    y_no_damage = y.copy()
    y0, h0 = y[0], h[0]

    if spec == "augmented_gdp":
        # Augmented GDP: u = y + (y/h)*ell — linear, no curvature params
        y_hat = effective_income(y, dT, phi, gamma, damage_type)
        E = None
        u = np.array([period_utility_augmented_gdp(y_hat[i], h[i], T)
                       for i in range(n)])
        u_no_damage = np.array([period_utility_augmented_gdp(y_no_damage[i], h[i], T)
                                 for i in range(n)])
    elif spec == "ecological":
        # Ecological: output damage on income only; environment enters utility directly
        kappa = params.get("kappa", calibrate_kappa(params["gamma_raw"]))
        alpha_cE = params.get("alpha_cE", 0.5)
        epsilon_cE = params.get("epsilon_cE", 0.5)
        chi_override = params.get("chi_override", None)
        chi = chi_override if chi_override is not None else calibrate_chi(y0, h0, T, sigma)

        # Only output damage (phi) hits income; gamma goes into E
        y_hat = effective_income_output_only(y, dT, phi, damage_type)
        E = env_quality(dT, kappa)
        E_pristine = np.ones(n)  # no damage counterfactual

        # Use baseline income as reference for normalisation (y and E on same scale)
        y_ref = y0

        u = np.array([period_utility_ecological(y_hat[i], h[i], E[i], T, sigma, chi,
                                                 alpha_cE, epsilon_cE, y_ref)
                       for i in range(n)])
        u_no_damage = np.array([period_utility_ecological(y_no_damage[i], h[i], E_pristine[i],
                                                           T, sigma, chi, alpha_cE, epsilon_cE, y_ref)
                                 for i in range(n)])
    else:
        # Additive and CES: all damage (phi+gamma) hits income
        y_hat = effective_income(y, dT, phi, gamma, damage_type)
        E = None

        if spec == "additive":
            chi_override = params.get("chi_override", None)
            chi = chi_override if chi_override is not None else calibrate_chi(y0, h0, T, sigma)
            u = np.array([period_utility_additive(y_hat[i], h[i], T, sigma, chi)
                           for i in range(n)])
            u_no_damage = np.array([period_utility_additive(y_no_damage[i], h[i], T, sigma, chi)
                                     for i in range(n)])
        else:
            alpha = calibrate_alpha_ces(y0, h0, T, epsilon)
            u = np.array([period_utility_ces(y_hat[i], h[i], T, epsilon, alpha)
                           for i in range(n)])
            u_no_damage = np.array([period_utility_ces(y_no_damage[i], h[i], T, epsilon, alpha)
                                     for i in range(n)])

    discount = beta ** t_idx
    W = np.sum(discount * u)
    W_no_damage = np.sum(discount * u_no_damage)

    result = {
        "welfare": W,
        "welfare_no_damage": W_no_damage,
        "damage_cost": W_no_damage - W,
        "years": years,
        "period_utility": u,
        "period_utility_no_damage": u_no_damage,
        "effective_income": y_hat,
        "raw_income": y,
        "leisure": T - h,
        "temp_change": dT,
        "discount_factors": discount,
        "discounted_utility": discount * u,
    }
    if E is not None:
        result["env_quality"] = E
    return result


def income_equivalent(W_sc, W_pc, n_periods, r):
    beta = 1.0 / (1.0 + r / 100.0)
    annuity = np.sum(beta ** np.arange(n_periods))
    if annuity == 0:
        return 0.0
    lam = np.exp((W_sc - W_pc) / annuity) - 1.0
    return lam * 100.0


def compute_world_welfare_popweighted(data, params, spec, method="utilitarian",
                                       inequality_aversion=1.0):
    regions = sorted(data["SC"]["Region"].unique())
    regions = [r for r in regions if r != "World"]
    scenario_results = {}
    for scenario in ["SC", "PC"]:
        regional = []
        for region in regions:
            df_r = data[scenario][data[scenario]["Region"] == region].sort_values("Year").reset_index(drop=True)
            res = compute_welfare(df_r, params, spec=spec)
            pop = df_r["Population"].values if "Population" in df_r.columns else np.ones(len(df_r))
            avg_pop = np.mean(pop)
            regional.append({"region": region, "welfare": res["welfare"],
                             "avg_pop": avg_pop, "full_result": res})
        total_pop = sum(r["avg_pop"] for r in regional)
        for r in regional:
            r["weight"] = r["avg_pop"] / total_pop
        welfares = np.array([r["welfare"] for r in regional])
        weights = np.array([r["weight"] for r in regional])
        if method == "utilitarian":
            W_world = np.sum(weights * welfares)
        else:
            e = inequality_aversion
            if abs(e - 1.0) < 1e-10:
                W_world = np.exp(np.sum(weights * np.log(welfares)))
            else:
                W_world = np.sum(weights * (welfares ** (1 - e))) ** (1.0 / (1 - e))
        scenario_results[scenario] = {"welfare": W_world, "regional": regional, "weights": weights}

    n = len(data["SC"][data["SC"]["Region"] == regions[0]]["Year"].unique())
    if spec == "augmented_gdp":
        ce = income_equivalent_linear(scenario_results["SC"]["welfare"],
                                            scenario_results["PC"]["welfare"])
    else:
        ce = income_equivalent(scenario_results["SC"]["welfare"],
                                     scenario_results["PC"]["welfare"], n, params["r"])
    return {"SC": scenario_results["SC"], "PC": scenario_results["PC"],
            "income_equivalent": ce, "regions": regions}


# ══════════════════════════════════════════════════════════════════
# PLOTTING HELPERS (Plotly)
# ══════════════════════════════════════════════════════════════════

def _plotly_layout(title="", height=450):
    return dict(
        template="plotly_white",
        title=dict(text=title, font=dict(size=15, family="Helvetica Neue")),
        font=dict(family="Helvetica Neue", size=11),
        margin=dict(l=60, r=30, t=50, b=50),
        height=height,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#DDDDDD",
                    borderwidth=1, font=dict(size=10)),
    )


# ══════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════

def main():
    st.title("\U0001F30D Welfare Evaluation")
    st.caption("Compare lifetime welfare under Sustainable Convergence (SC) vs "
               "Productivist Convergence (PC) across world regions.")

    # ── Sidebar: File upload + parameters ──
    with st.sidebar:
        st.header("Data & Parameters")

        uploaded = st.file_uploader("Upload DataInput_Welfare.xlsx", type=["xlsx", "xls"])
        if uploaded is None:
            st.info("Upload the Excel file to begin.")
            st.stop()

        data = load_data(uploaded.read())
        regions = sorted(data["SC"]["Region"].unique())

        st.success(f"Loaded {len(regions)} regions, "
                   f"{data['SC']['Year'].nunique()} years")

        st.divider()
        st.subheader("Utility Specification")
        spec = st.radio("Specification",
                        ["Augmented GDP", "Additive (Log + CRRA)", "CES", "Ecological"],
                        horizontal=True,
                        help="Augmented GDP: linear u = y + (y/h)·ℓ — no diminishing returns "
                             "(GJP baseline). "
                             "Additive: log(income) + CRRA leisure, separable. "
                             "CES: constant-elasticity-of-substitution composite of "
                             "income and leisure. "
                             "Ecological: CES composite of income and environmental "
                             "quality, plus separate CRRA leisure term.")
        if "Augmented" in spec:
            spec_key = "augmented_gdp"
        elif "Additive" in spec:
            spec_key = "additive"
        elif "CES" in spec:
            spec_key = "ces"
        else:
            spec_key = "ecological"

        st.divider()
        st.subheader("Parameters")

        # Initialise session state with defaults (on first run)
        _slider_keys = {
            "sl_r": DEFAULT_PARAMS["r"],
            "sl_sigma": DEFAULT_PARAMS["sigma"],
            "sl_epsilon": DEFAULT_PARAMS["epsilon"],
            "sl_phi": DEFAULT_PARAMS["phi"],
            "sl_gamma": DEFAULT_PARAMS["gamma_raw"],
            "sl_chi": 4.0,
            "sl_epsilon_cE": 0.5,
            "sl_alpha_cE": 0.5,
            "sl_kappa": 0.133,
        }
        for k, v in _slider_keys.items():
            if k not in st.session_state:
                st.session_state[k] = v
        if "chi_mode" not in st.session_state:
            st.session_state["chi_mode"] = "Calibrate from 2025 data"
        if "kappa_mode" not in st.session_state:
            st.session_state["kappa_mode"] = "Calibrate from \u03B3"
        if "damage_type" not in st.session_state:
            st.session_state["damage_type"] = "Exponential"

        # Reset spec-specific sliders to their defaults when switching specs.
        # Streamlit can show min-of-range for sliders that haven't rendered yet
        # if the key was only set programmatically. Force correct defaults on
        # spec change so ecological params don't appear as 0.01/−1.0.
        _prev_spec = st.session_state.get("_prev_spec", None)
        if _prev_spec != spec_key:
            st.session_state["_prev_spec"] = spec_key
            if spec_key == "ecological":
                st.session_state["sl_sigma"] = DEFAULT_PARAMS["sigma"]     # 1.0
                st.session_state["sl_epsilon_cE"] = 0.5
                st.session_state["sl_alpha_cE"] = 0.5
                st.session_state["sl_kappa"] = 0.133
                st.session_state["chi_mode"] = "Calibrate from 2025 data"
                st.session_state["kappa_mode"] = "Calibrate from \u03B3"
            elif spec_key == "additive":
                st.session_state["sl_sigma"] = DEFAULT_PARAMS["sigma"]     # 1.0
                st.session_state["chi_mode"] = "Calibrate from 2025 data"
            elif spec_key == "ces":
                st.session_state["sl_epsilon"] = DEFAULT_PARAMS["epsilon"] # 0.8

        def _reset_params():
            for k, v in _slider_keys.items():
                st.session_state[k] = v
            st.session_state["chi_mode"] = "Calibrate from 2025 data"
            st.session_state["kappa_mode"] = "Calibrate from \u03B3"
            st.session_state["damage_type"] = "Exponential"

        r_val = st.slider("Discount rate r (%)", 0.5, 6.0, step=0.1, key="sl_r",
                          help="Pure rate of time preference. Higher values discount "
                               "future welfare more heavily. Stern (2006) uses 0.1%; "
                               "Nordhaus (2017) uses ~1.5%. Our default follows Nordhaus.")

        if spec_key == "augmented_gdp":
            # No curvature parameters — linear utility
            sigma_val = DEFAULT_PARAMS["sigma"]
            epsilon_val = DEFAULT_PARAMS["epsilon"]
            chi_override = None
            st.caption("Augmented GDP: u = y + (y/h) \u00B7 \u2113.  "
                       "No curvature parameters \u2014 only discount rate and damage function apply.")
        elif spec_key == "additive":
            sigma_val = st.slider("Leisure curvature \u03C3", -1.0, 3.0,
                                  step=0.1, key="sl_sigma",
                                  help="CRRA curvature on leisure. "
                                       "\u03C3 = 1 gives log utility; \u03C3 > 1 means sharper "
                                       "diminishing returns to leisure; \u03C3 < 1 means flatter.")
            epsilon_val = DEFAULT_PARAMS["epsilon"]

            chi_mode = st.radio("Leisure weight \u03C7",
                                ["Calibrate from 2025 data", "Set manually"],
                                horizontal=True, key="chi_mode",
                                help="How much utility comes from leisure relative to income. "
                                     "'Calibrate' pins \u03C7 so that the marginal value of "
                                     "leisure equals the marginal value of income at "
                                     "baseline hours and income.")
            chi_override = None
            if chi_mode == "Set manually":
                chi_override = st.slider("Leisure weight \u03C7", 0.5, 10.0,
                                         step=0.1, key="sl_chi",
                                         help="Manual leisure weight. Higher values mean "
                                              "leisure contributes more to welfare relative "
                                              "to income.")
            else:
                st.caption("\u03C7 = \u2113\u2080\u1D9E / h\u2080 (calibrated per region)")
        elif spec_key == "ces":
            sigma_val = DEFAULT_PARAMS["sigma"]
            epsilon_val = st.slider("CES elasticity \u03B5", 0.2, 2.0,
                                    step=0.1, key="sl_epsilon",
                                    help="Elasticity of substitution between income and "
                                         "leisure. \u03B5 < 1: complements (hard to trade off); "
                                         "\u03B5 = 1: Cobb-Douglas; \u03B5 > 1: substitutes "
                                         "(easy to trade off).")
            chi_override = None
        else:
            # Ecological specification
            sigma_val = st.slider("Leisure curvature \u03C3", -1.0, 3.0,
                                  step=0.1, key="sl_sigma",
                                  help="CRRA curvature on leisure. "
                                       "\u03C3 = 1 gives log utility; \u03C3 > 1 means sharper "
                                       "diminishing returns to leisure; \u03C3 < 1 means flatter.")
            epsilon_val = DEFAULT_PARAMS["epsilon"]

            chi_mode = st.radio("Leisure weight \u03C7",
                                ["Calibrate from 2025 data", "Set manually"],
                                horizontal=True, key="chi_mode",
                                help="How much utility comes from leisure relative to income. "
                                     "'Calibrate' pins \u03C7 so that the marginal value of "
                                     "leisure equals the marginal value of income at "
                                     "baseline hours and income.")
            chi_override = None
            if chi_mode == "Set manually":
                chi_override = st.slider("Leisure weight \u03C7", 0.5, 10.0,
                                         step=0.1, key="sl_chi",
                                         help="Manual leisure weight. Higher values mean "
                                              "leisure contributes more to welfare relative "
                                              "to income.")
            else:
                st.caption("\u03C7 = \u2113\u2080\u1D9E / h\u2080 (calibrated per region)")

            st.divider()
            st.subheader("Ecological Parameters")
            epsilon_cE_val = st.slider(
                "Substitutability \u03B5(y,E)",
                0.01, 2.0, step=0.01, key="sl_epsilon_cE",
                help="Elasticity of substitution between income and environment. "
                     "0 = Leontief (no substitution), 1 = Cobb-Douglas, >1 = substitutes.")
            alpha_cE_val = st.slider(
                "Income weight \u03B1(y,E)",
                0.1, 0.9, step=0.05, key="sl_alpha_cE",
                help="Weight on income in the CES composite. "
                     "Higher = income matters more; lower = environment matters more.")

            # Show interpretation of epsilon_cE
            if epsilon_cE_val < 0.1:
                st.caption("\u2248 Leontief: welfare \u2248 min(income, environment)")
            elif abs(epsilon_cE_val - 1.0) < 0.05:
                st.caption("\u2248 Cobb-Douglas: moderate substitutability")
            elif epsilon_cE_val > 1.5:
                st.caption("High substitutability: income can offset env. loss")
            else:
                st.caption(f"\u03B5(y,E) = {epsilon_cE_val:.2f}: limited substitutability")

            # Kappa control
            kappa_mode = st.radio("Env. sensitivity \u03BA",
                                  ["Calibrate from \u03B3", "Set manually"],
                                  horizontal=True, key="kappa_mode",
                                  help="Controls how fast environmental quality E degrades with warming. "
                                       "E = 1/(1 + \u03BA\u00b7\u0394T). "
                                       "'Calibrate from \u03B3' sets \u03BA = \u03B3/100 to match "
                                       "Dietrich & Nichols (2025). 'Set manually' lets you explore "
                                       "alternative calibrations (e.g., from bottom-up damage estimates).")
            if kappa_mode == "Set manually":
                kappa_val = st.slider("Env. sensitivity \u03BA",
                                      0.01, 1.0, step=0.01, key="sl_kappa",
                                      help="Environmental sensitivity parameter. "
                                           "At 1\u00b0C: E = 1/(1+\u03BA). "
                                           "Low (~0.03): health/mortality only. "
                                           "Medium (~0.13): Dietrich & Nichols. "
                                           "High (~0.25+): includes ecosystem collapse.")
                # Show what E looks like at key temperatures
                E_at_2 = 1.0 / (1.0 + kappa_val * 2.0)
                E_at_4 = 1.0 / (1.0 + kappa_val * 4.0)
                st.caption(f"E at 2\u00b0C = {E_at_2:.2f}  |  E at 4\u00b0C = {E_at_4:.2f}")
            else:
                kappa_val = None  # will be computed from gamma below
                _gamma_preview = st.session_state.get("sl_gamma", DEFAULT_PARAMS["gamma_raw"])
                st.caption(f"\u03BA = \u03B3/100 = {_gamma_preview/100:.4f}")

        st.divider()
        st.subheader("Damage Function")
        damage_type_label = st.radio("Damage specification",
                                     ["Exponential", "Quadratic"],
                                     horizontal=True, key="damage_type",
                                     help="Functional form for how temperature maps to damage. "
                                          "Exponential: damages grow multiplicatively with \u0394T "
                                          "(Bilal-K\u00e4nzig 2024). Quadratic: damages grow with "
                                          "\u0394T\u00B2 (DICE-style, Nordhaus).")
        damage_type = damage_type_label.lower()
        if spec_key == "ecological":
            _kappa_source = "\u03BA = \u03B3/100" if kappa_val is None else f"\u03BA = {kappa_val:.3f} (manual)"
            st.caption("**Ecological mode:** output damage (\u03C6) hits income; "
                       "\u03BA sets environmental quality degradation.")
            if damage_type == "exponential":
                st.caption(f"\u0177 = y \u00B7 exp[\u2013\u03C6\u00B7\u0394T]  |  "
                           f"E = 1 / (1 + \u03BA\u00B7\u0394T),  {_kappa_source}")
            else:
                st.caption(f"\u0177 = y / [1 + \u03C6\u00B7\u0394T\u00B2]  |  "
                           f"E = 1 / (1 + \u03BA\u00B7\u0394T),  {_kappa_source}")
        else:
            if damage_type == "exponential":
                st.caption("\u0177 = y \u00B7 exp[\u2013(\u03C6+\u03B3)\u00B7\u0394T]")
            else:
                st.caption("\u0177 = y / [1 + (\u03C6+\u03B3)\u00B7\u0394T\u00B2]  (DICE-style)")

        phi_val = st.slider("Output damage (%/\u00B0C)", 0.0, 30.0,
                            step=1.0, key="sl_phi",
                            help="GDP/output loss per degree of warming. "
                                 "Default 12% based on Bilal & K\u00e4nzig (2024). "
                                 "This reduces effective income.")
        gamma_val = st.slider("Well-being damage (%/\u00B0C)", 0.0, 30.0,
                              step=1.0, key="sl_gamma",
                              help="Non-market welfare loss per degree of warming "
                                   "(health, amenity, ecosystem services). Default 13.3% "
                                   "based on Dietrich & Nichols (2025). In Additive/CES specs, "
                                   "this adds to \u03C6 in the damage function. In Ecological spec, "
                                   "it calibrates \u03BA for environmental quality.")

        st.button("Reset Parameters", on_click=_reset_params,
                  use_container_width=True)

        params = {
            "T": FIXED_T, "sigma": sigma_val, "r": r_val,
            "phi": phi_val, "gamma_raw": gamma_val, "epsilon": epsilon_val,
            "damage_type": damage_type, "chi_override": chi_override,
        }
        if spec_key == "ecological":
            params["epsilon_cE"] = epsilon_cE_val
            params["alpha_cE"] = alpha_cE_val
            params["kappa"] = kappa_val if kappa_val is not None else calibrate_kappa(gamma_val)

        st.divider()
        st.subheader("Region")
        region = st.selectbox("Select region", regions,
                              index=regions.index("World") if "World" in regions else 0)

        agg_method = "raw"
        aversion_val = 1.0
        if region == "World":
            st.subheader("World Aggregation")
            agg_method = st.radio("Method",
                                  ["raw", "utilitarian", "atkinson"],
                                  format_func=lambda x: {
                                      "raw": "Raw World row",
                                      "utilitarian": "Utilitarian (pop-weighted)",
                                      "atkinson": "Atkinson (inequality-averse)"
                                  }[x],
                                  help="How to aggregate regional welfare into a world figure. "
                                       "'Raw' uses the World row from the data directly. "
                                       "'Utilitarian' sums population-weighted regional welfare. "
                                       "'Atkinson' penalises inequality across regions.")
            if agg_method == "atkinson":
                aversion_val = st.slider("Inequality aversion (e)", 0.0, 3.0, 1.0, 0.1,
                                         help="Atkinson inequality aversion parameter. "
                                              "e = 0: no aversion (same as utilitarian). "
                                              "e = 1: moderate aversion. Higher values "
                                              "place more weight on worse-off regions.")

    # ── Compute welfare for selected region ──
    results = {}
    if region == "World" and agg_method != "raw":
        pw = compute_world_welfare_popweighted(
            data, params, spec_key, method=agg_method,
            inequality_aversion=aversion_val)
        for scenario in ["SC", "PC"]:
            df_r = data[scenario][data[scenario]["Region"] == "World"].sort_values("Year").reset_index(drop=True)
            results[scenario] = compute_welfare(df_r, params, spec=spec_key)
            results[scenario]["welfare"] = pw[scenario]["welfare"]
        pw_nd = compute_world_welfare_popweighted(
            data, {**params, "phi": 0.0, "gamma_raw": 0.0},
            spec_key, method=agg_method, inequality_aversion=aversion_val)
        for scenario in ["SC", "PC"]:
            results[scenario]["welfare_no_damage"] = pw_nd[scenario]["welfare"]
            results[scenario]["damage_cost"] = pw_nd[scenario]["welfare"] - pw[scenario]["welfare"]
        ce = pw["income_equivalent"]
    else:
        for scenario in ["SC", "PC"]:
            df_r = data[scenario][data[scenario]["Region"] == region].sort_values("Year").reset_index(drop=True)
            results[scenario] = compute_welfare(df_r, params, spec=spec_key)
        n = len(results["SC"]["years"])
        if spec_key == "augmented_gdp":
            ce = income_equivalent_linear(results["SC"]["welfare"],
                                               results["PC"]["welfare"])
        else:
            ce = income_equivalent(results["SC"]["welfare"], results["PC"]["welfare"],
                                         n, params["r"])
    results["income_equivalent"] = ce

    # ── Compute all regions (for Regional, Sensitivity, Progress tabs) ──
    all_region_results = []
    for reg in regions:
        row = {"Region": reg}
        for scenario in ["SC", "PC"]:
            df_r = data[scenario][data[scenario]["Region"] == reg].sort_values("Year").reset_index(drop=True)
            if df_r.empty:
                continue
            res = compute_welfare(df_r, params, spec=spec_key)
            row[f"{scenario}_welfare"] = res["welfare"]
            row[f"{scenario}_welfare_no_damage"] = res["welfare_no_damage"]
            row[f"{scenario}_damage_cost"] = res["damage_cost"]
            row[f"{scenario}_avg_inc"] = np.mean(res["raw_income"])
            row[f"{scenario}_avg_leisure"] = np.mean(res["leisure"])
            row[f"{scenario}_terminal_temp"] = res["temp_change"][-1]
            row[f"{scenario}_utility_2025"] = res["period_utility"][0]
            row[f"{scenario}_utility_2100"] = res["period_utility"][-1]
            row[f"{scenario}_effinc_2025"] = res["effective_income"][0]
            row[f"{scenario}_effinc_2100"] = res["effective_income"][-1]
            row[f"{scenario}_leisure_2025"] = res["leisure"][0]
            row[f"{scenario}_leisure_2100"] = res["leisure"][-1]
            row[f"{scenario}_temp_2025"] = res["temp_change"][0]
            row[f"{scenario}_temp_2100"] = res["temp_change"][-1]
        n_yr = len(data["SC"][data["SC"]["Region"] == reg]["Year"].unique())
        if spec_key == "augmented_gdp":
            row["CE_pct"] = income_equivalent_linear(
                row.get("SC_welfare", 0), row.get("PC_welfare", 0))
        else:
            row["CE_pct"] = income_equivalent(
                row.get("SC_welfare", 0), row.get("PC_welfare", 0), n_yr, params["r"])
        all_region_results.append(row)

    # ══════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════
    tab_summary, tab_charts, tab_regional, tab_sensitivity, tab_progress, tab_model = \
        st.tabs(["Summary", "Charts", "Regional", "Sensitivity", "Progress", "Model"])

    # ────────────────────────────────────────────────────────────
    # TAB 1: SUMMARY
    # ────────────────────────────────────────────────────────────
    with tab_summary:
        st.subheader(f"Welfare Results — {region}")
        spec_labels = {"augmented_gdp": "Augmented GDP", "additive": "Additive (Log + CRRA)",
                       "ces": "CES", "ecological": "Ecological"}
        spec_text = f"**{spec_labels.get(spec_key, spec_key)}**"
        if region == "World" and agg_method != "raw":
            spec_text += f"  |  Aggregation: {agg_method.title()}"
            if agg_method == "atkinson":
                spec_text += f" (e={aversion_val:.1f})"
        st.markdown(spec_text)

        W_sc = results["SC"]["welfare"]
        W_pc = results["PC"]["welfare"]
        W_sc_nd = results["SC"]["welfare_no_damage"]
        W_pc_nd = results["PC"]["welfare_no_damage"]

        col1, col2, col3 = st.columns(3)
        col1.metric("SC Welfare", f"{W_sc:.2f}")
        col2.metric("PC Welfare", f"{W_pc:.2f}")
        col3.metric("Income Equivalent", f"{ce:+.2f}%",
                     delta=f"{'SC' if ce > 0 else 'PC'} preferred")

        summary_metrics = [
            "Lifetime Welfare (with damages)",
            "Lifetime Welfare (no damages)",
            "Climate Damage Cost",
            "Avg Raw Income",
            "Avg Effective Income",
            "Avg Leisure (hrs/yr)",
            "Terminal Temp Change (\u00B0C)",
        ]
        summary_sc = [
            f"{W_sc:.2f}", f"{W_sc_nd:.2f}",
            f"{results['SC']['damage_cost']:.2f}",
            f"{np.mean(results['SC']['raw_income']):,.0f}",
            f"{np.mean(results['SC']['effective_income']):,.0f}",
            f"{np.mean(results['SC']['leisure']):,.0f}",
            f"{results['SC']['temp_change'][-1]:.2f}",
        ]
        summary_pc = [
            f"{W_pc:.2f}", f"{W_pc_nd:.2f}",
            f"{results['PC']['damage_cost']:.2f}",
            f"{np.mean(results['PC']['raw_income']):,.0f}",
            f"{np.mean(results['PC']['effective_income']):,.0f}",
            f"{np.mean(results['PC']['leisure']):,.0f}",
            f"{results['PC']['temp_change'][-1]:.2f}",
        ]
        if spec_key == "ecological":
            summary_metrics.append("Terminal Env. Quality (E)")
            summary_sc.append(f"{results['SC']['env_quality'][-1]:.3f}")
            summary_pc.append(f"{results['PC']['env_quality'][-1]:.3f}")

        summary_data = {"Metric": summary_metrics, "SC": summary_sc, "PC": summary_pc}
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Interpretation
        if ce > 0:
            st.success(f"SC yields **{abs(ce):.1f}%** higher lifetime welfare than PC "
                       f"for {region}.")
        elif ce < 0:
            st.error(f"PC yields **{abs(ce):.1f}%** higher lifetime welfare than SC "
                     f"for {region}.")
        else:
            st.info(f"SC and PC yield approximately equal welfare for {region}.")

        param_str = (f"r={params['r']:.1f}%,  "
                     f"T={params['T']:.0f}h,  output damage={params['phi']:.1f}%/\u00B0C,  "
                     f"well-being damage={params['gamma_raw']:.1f}%/\u00B0C,  "
                     f"damage fn={params.get('damage_type', 'exponential')}")
        if spec_key != "augmented_gdp":
            param_str = f"\u03C3={params['sigma']:.1f},  " + param_str
        if spec_key in ("additive", "ecological"):
            if params.get("chi_override") is not None:
                param_str += f",  \u03C7={params['chi_override']:.1f} (manual)"
            else:
                param_str += ",  \u03C7=calibrated"
        if spec_key == "ces":
            param_str += f",  \u03B5={params['epsilon']:.1f}"
        if spec_key == "ecological":
            _kappa_label = f"{params['kappa']:.4f}"
            if kappa_val is not None:
                _kappa_label += " (manual)"
            else:
                _kappa_label += " (=\u03B3/100)"
            param_str += (f",  \u03B5(y,E)={params['epsilon_cE']:.2f}"
                          f",  \u03B1(y,E)={params['alpha_cE']:.2f}"
                          f",  \u03BA={_kappa_label}")
        st.caption(f"Parameters: {param_str}")

    # ────────────────────────────────────────────────────────────
    # TAB 2: CHARTS
    # ────────────────────────────────────────────────────────────
    with tab_charts:
        st.subheader(f"Time Series — {region}")
        years = results["SC"]["years"]

        # For ecological spec, show 3x2 grid with env quality; otherwise 2x2
        if spec_key == "ecological":
            fig = make_subplots(rows=3, cols=2,
                                subplot_titles=("Per-Capita Income", "Leisure (hrs/yr)",
                                                "Environmental Quality (E)", "Temperature Change",
                                                "Period Utility", "Discounted Utility"),
                                vertical_spacing=0.08, horizontal_spacing=0.08)
            chart_height = 900
        else:
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=("Per-Capita Income", "Leisure (hrs/yr)",
                                                "Period Utility", "Temperature Change"),
                                vertical_spacing=0.12, horizontal_spacing=0.08)
            chart_height = 650

        # Income
        fig.add_trace(go.Scatter(x=years, y=results["SC"]["raw_income"],
                                  name="SC (raw)", line=dict(color=COLOUR_SC, width=2.2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=results["PC"]["raw_income"],
                                  name="PC (raw)", line=dict(color=COLOUR_PC, width=2.2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=results["SC"]["effective_income"],
                                  name="SC (eff.)", line=dict(color=COLOUR_SC_LIGHT, width=1.6, dash="dash"),
                                  showlegend=True),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=results["PC"]["effective_income"],
                                  name="PC (eff.)", line=dict(color=COLOUR_PC_LIGHT, width=1.6, dash="dash"),
                                  showlegend=True),
                      row=1, col=1)

        # Leisure
        fig.add_trace(go.Scatter(x=years, y=results["SC"]["leisure"],
                                  name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                  showlegend=False),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=years, y=results["PC"]["leisure"],
                                  name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                  showlegend=False),
                      row=1, col=2)

        if spec_key == "ecological":
            # Environmental Quality (row 2, col 1)
            fig.add_trace(go.Scatter(x=years, y=results["SC"]["env_quality"],
                                      name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                      showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=years, y=results["PC"]["env_quality"],
                                      name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                      showlegend=False),
                          row=2, col=1)

            # Temperature (row 2, col 2)
            fig.add_trace(go.Scatter(x=years, y=results["SC"]["temp_change"],
                                      name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                      showlegend=False),
                          row=2, col=2)
            fig.add_trace(go.Scatter(x=years, y=results["PC"]["temp_change"],
                                      name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                      showlegend=False),
                          row=2, col=2)

            # Period Utility (row 3, col 1)
            fig.add_trace(go.Scatter(x=years, y=results["SC"]["period_utility"],
                                      name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                      showlegend=False),
                          row=3, col=1)
            fig.add_trace(go.Scatter(x=years, y=results["PC"]["period_utility"],
                                      name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                      showlegend=False),
                          row=3, col=1)

            # Discounted Utility (row 3, col 2)
            fig.add_trace(go.Scatter(x=years, y=results["SC"]["discounted_utility"],
                                      name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                      showlegend=False),
                          row=3, col=2)
            fig.add_trace(go.Scatter(x=years, y=results["PC"]["discounted_utility"],
                                      name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                      showlegend=False),
                          row=3, col=2)

            fig.update_yaxes(title_text="E (0\u20131)", row=2, col=1)
            fig.update_yaxes(title_text="\u0394T (\u00B0C)", row=2, col=2)
            fig.update_yaxes(title_text="u(t)", row=3, col=1)
            fig.update_yaxes(title_text="\u03B2\u1D57 \u00B7 u(t)", row=3, col=2)
        else:
            # Period Utility (row 2, col 1)
            fig.add_trace(go.Scatter(x=years, y=results["SC"]["period_utility"],
                                      name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                      showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=years, y=results["PC"]["period_utility"],
                                      name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                      showlegend=False),
                          row=2, col=1)

            # Temperature (row 2, col 2)
            fig.add_trace(go.Scatter(x=years, y=results["SC"]["temp_change"],
                                      name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                      showlegend=False),
                          row=2, col=2)
            fig.add_trace(go.Scatter(x=years, y=results["PC"]["temp_change"],
                                      name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                      showlegend=False),
                          row=2, col=2)

            fig.update_yaxes(title_text="u(t)", row=2, col=1)
            fig.update_yaxes(title_text="\u0394T (\u00B0C)", row=2, col=2)

        layout_opts = _plotly_layout(f"Welfare Analysis — {region}", height=chart_height)
        layout_opts["legend"] = dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                     bgcolor="rgba(255,255,255,0.9)",
                                     bordercolor="#DDDDDD", borderwidth=1, font=dict(size=10))
        fig.update_layout(**layout_opts)
        fig.update_yaxes(title_text="Income (2025 Euro PPP)", row=1, col=1)
        fig.update_yaxes(title_text="Hours", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────────────────────
    # TAB 3: REGIONAL
    # ────────────────────────────────────────────────────────────
    with tab_regional:
        st.subheader("All Regions — Welfare Comparison")

        reg_df = pd.DataFrame([{
            "Region": abbrev(r["Region"]),
            "W(SC)": f"{r.get('SC_welfare', 0):.2f}",
            "W(PC)": f"{r.get('PC_welfare', 0):.2f}",
            "Damage SC": f"{r.get('SC_damage_cost', 0):.2f}",
            "Damage PC": f"{r.get('PC_damage_cost', 0):.2f}",
            "CE (%)": f"{r.get('CE_pct', 0):+.2f}%",
        } for r in all_region_results])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        # Bar chart — Income Equivalent (%)
        ce_values = [r.get("CE_pct", 0) for r in all_region_results]
        ce_colours = [COLOUR_SC if v >= 0 else COLOUR_PC for v in ce_values]
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(
            x=[abbrev(r["Region"]) for r in all_region_results],
            y=ce_values,
            marker_color=ce_colours,
            text=[f"{v:+.1f}%" for v in ce_values],
            textposition="outside",
            textfont=dict(size=11),
            showlegend=False,
        ))
        fig_reg.update_layout(
            **_plotly_layout("Income Equivalent by Region (SC vs PC)", height=450),
        )
        fig_reg.update_yaxes(title_text="Income Equivalent (%)",
                             zeroline=True, zerolinewidth=1.5, zerolinecolor="#888888")
        st.plotly_chart(fig_reg, use_container_width=True)

        st.info(
            "**How to read this chart:** The income equivalent (IE) measures the "
            "permanent percentage change in income that an individual living under PC "
            "would need *every year* to be as well off as under SC. An IE of +10% means SC "
            "delivers the same welfare as PC would if PC\u2019s income were permanently "
            "raised by 10%. Positive values (teal) indicate SC is welfare-superior; negative "
            "values (coral) would indicate PC is welfare-superior."
        )

    # ────────────────────────────────────────────────────────────
    # TAB 4: SENSITIVITY
    # ────────────────────────────────────────────────────────────
    with tab_sensitivity:
        st.subheader("Sensitivity Heatmap")

        if spec_key == "augmented_gdp":
            # Only r and damage parameters are relevant for linear utility
            sens_param_defs = {
                "r": ("Discount rate r (%)", np.arange(0.5, 5.1, 0.5)),
                "phi": ("Output damage (%/\u00B0C)", np.arange(0.0, 31.0, 3.0)),
                "gamma_raw": ("Well-being damage (%/\u00B0C)", np.arange(0.0, 31.0, 3.0)),
            }
        else:
            sens_param_defs = {
                "r": ("Discount rate r (%)", np.arange(0.5, 5.1, 0.5)),
                "sigma": ("Leisure curvature \u03C3", np.arange(-1.0, 3.1, 0.5)),
                "phi": ("Output damage (%/\u00B0C)", np.arange(0.0, 31.0, 3.0)),
                "gamma_raw": ("Well-being damage (%/\u00B0C)", np.arange(0.0, 31.0, 3.0)),
                "epsilon": ("CES elasticity \u03B5", np.arange(0.2, 2.1, 0.2)),
            }
            if spec_key == "ecological":
                sens_param_defs["epsilon_cE"] = ("Substitutability \u03B5(y,E)",
                                                  np.arange(0.05, 2.05, 0.1))
                sens_param_defs["alpha_cE"] = ("Income weight \u03B1(y,E)",
                                                np.arange(0.1, 0.91, 0.1))
                sens_param_defs["kappa"] = ("Env. sensitivity \u03BA",
                                             np.arange(0.03, 0.51, 0.03))

        col_x, col_y = st.columns(2)
        with col_x:
            x_key = st.selectbox("X-axis parameter",
                                  list(sens_param_defs.keys()),
                                  format_func=lambda k: f"{k}: {sens_param_defs[k][0]}")
        with col_y:
            y_key = st.selectbox("Y-axis parameter",
                                  list(sens_param_defs.keys()),
                                  index=1,
                                  format_func=lambda k: f"{k}: {sens_param_defs[k][0]}")

        if x_key == y_key:
            st.warning("Select two different parameters.")
        else:
            sens_region = st.selectbox("Region for sensitivity", regions,
                                        index=regions.index("World") if "World" in regions else 0,
                                        key="sens_region")

            if st.button("Generate Heatmap", type="primary"):
                x_label, x_vals = sens_param_defs[x_key]
                y_label, y_vals = sens_param_defs[y_key]

                df_sc = data["SC"][data["SC"]["Region"] == sens_region].sort_values("Year").reset_index(drop=True)
                df_pc = data["PC"][data["PC"]["Region"] == sens_region].sort_values("Year").reset_index(drop=True)

                ce_grid = np.zeros((len(y_vals), len(x_vals)))
                progress = st.progress(0)
                total = len(y_vals) * len(x_vals)
                count = 0
                for i, y_val in enumerate(y_vals):
                    for j, x_val in enumerate(x_vals):
                        p = params.copy()
                        p[x_key] = float(x_val)
                        p[y_key] = float(y_val)
                        res_sc = compute_welfare(df_sc, p, spec=spec_key)
                        res_pc = compute_welfare(df_pc, p, spec=spec_key)
                        n = len(res_sc["years"])
                        if spec_key == "augmented_gdp":
                            ce_grid[i, j] = income_equivalent_linear(
                                res_sc["welfare"], res_pc["welfare"])
                        else:
                            ce_grid[i, j] = income_equivalent(
                                res_sc["welfare"], res_pc["welfare"], n, p["r"])
                        count += 1
                        progress.progress(count / total)
                progress.empty()

                vmax = max(abs(ce_grid.max()), abs(ce_grid.min()), 1.0)

                fig_heat = go.Figure(data=go.Heatmap(
                    z=ce_grid,
                    x=[f"{v:.2g}" for v in x_vals],
                    y=[f"{v:.2g}" for v in y_vals],
                    colorscale=[
                        [0.0, "#B2182B"], [0.125, "#D6604D"],
                        [0.25, "#F4A582"], [0.375, "#FDDBC7"],
                        [0.5, "#FFFFFF"],
                        [0.625, "#D1E5B0"], [0.75, "#7FBC41"],
                        [0.875, "#4D9221"], [1.0, "#276419"],
                    ],
                    zmin=-vmax, zmax=vmax,
                    text=[[f"{ce_grid[i, j]:+.1f}%" for j in range(len(x_vals))]
                          for i in range(len(y_vals))],
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                    colorbar=dict(title="CE (%)"),
                ))
                fig_heat.update_layout(
                    **_plotly_layout(
                        f"Sensitivity: {sens_region} | "
                        f"{'Augmented GDP' if spec_key == 'augmented_gdp' else 'Additive' if spec_key == 'additive' else 'CES' if spec_key == 'ces' else 'Ecological'}",
                        height=550),
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("\U0001F7E2 Green = SC preferred  |  \U0001F534 Red = PC preferred")

    # ────────────────────────────────────────────────────────────
    # TAB 5: PROGRESS
    # ────────────────────────────────────────────────────────────
    with tab_progress:
        st.subheader("2025 \u2192 2100  |  Change by Region")

        def pct_change(v_old, v_new):
            if abs(v_old) < 1e-10:
                return float("nan")
            return ((v_new - v_old) / abs(v_old)) * 100.0

        def fmt_pct(v):
            if math.isnan(v):
                return "N/A"
            return f"{v:+.1f}%"

        prog_rows = []
        for row in all_region_results:
            prog_rows.append({
                "Region": abbrev(row["Region"]),
                "SC Util %": fmt_pct(pct_change(row.get("SC_utility_2025", 0),
                                                 row.get("SC_utility_2100", 0))),
                "PC Util %": fmt_pct(pct_change(row.get("PC_utility_2025", 0),
                                                 row.get("PC_utility_2100", 0))),
                "SC Inc %": fmt_pct(pct_change(row.get("SC_effinc_2025", 0),
                                                row.get("SC_effinc_2100", 0))),
                "PC Inc %": fmt_pct(pct_change(row.get("PC_effinc_2025", 0),
                                                row.get("PC_effinc_2100", 0))),
                "SC Leis %": fmt_pct(pct_change(row.get("SC_leisure_2025", 0),
                                                 row.get("SC_leisure_2100", 0))),
                "PC Leis %": fmt_pct(pct_change(row.get("PC_leisure_2025", 0),
                                                 row.get("PC_leisure_2100", 0))),
                "SC \u0394T": f"{row.get('SC_temp_2100', 0) - row.get('SC_temp_2025', 0):+.2f}\u00B0C",
                "PC \u0394T": f"{row.get('PC_temp_2100', 0) - row.get('PC_temp_2025', 0):+.2f}\u00B0C",
            })
        st.dataframe(pd.DataFrame(prog_rows), use_container_width=True, hide_index=True)

        # 2x2 bar chart
        metrics = [
            ("Period Utility", "utility", True),
            ("Effective Income", "effinc", True),
            ("Leisure Hours", "leisure", True),
            ("Cumulative Warming", "temp", False),
        ]

        fig_prog = make_subplots(rows=2, cols=2, subplot_titles=[m[0] for m in metrics],
                                  vertical_spacing=0.14, horizontal_spacing=0.08)

        labels = [abbrev(r["Region"]) for r in all_region_results]

        for idx, (title, key, use_pct) in enumerate(metrics):
            r_idx, c_idx = divmod(idx, 2)
            if use_pct:
                sc_vals = [pct_change(r.get(f"SC_{key}_2025", 0),
                                      r.get(f"SC_{key}_2100", 0))
                           for r in all_region_results]
                pc_vals = [pct_change(r.get(f"PC_{key}_2025", 0),
                                      r.get(f"PC_{key}_2100", 0))
                           for r in all_region_results]
            else:
                sc_vals = [r.get(f"SC_{key}_2100", 0) - r.get(f"SC_{key}_2025", 0)
                           for r in all_region_results]
                pc_vals = [r.get(f"PC_{key}_2100", 0) - r.get(f"PC_{key}_2025", 0)
                           for r in all_region_results]

            # NaN to 0
            sc_vals = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in sc_vals]
            pc_vals = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in pc_vals]

            show_leg = (idx == 0)
            fig_prog.add_trace(go.Bar(
                x=labels, y=sc_vals, name="SC",
                marker_color=COLOUR_SC, showlegend=show_leg,
                text=[f"{v:.0f}" if use_pct else f"{v:.1f}" for v in sc_vals],
                textposition="outside", textfont=dict(size=8),
            ), row=r_idx + 1, col=c_idx + 1)
            fig_prog.add_trace(go.Bar(
                x=labels, y=pc_vals, name="PC",
                marker_color=COLOUR_PC, showlegend=show_leg,
                text=[f"{v:.0f}" if use_pct else f"{v:.1f}" for v in pc_vals],
                textposition="outside", textfont=dict(size=8),
            ), row=r_idx + 1, col=c_idx + 1)

            y_title = "% Change" if use_pct else "\u00B0C"
            fig_prog.update_yaxes(title_text=y_title, row=r_idx + 1, col=c_idx + 1)

        fig_prog.update_layout(
            **_plotly_layout("2025 to 2100: Change by Region", height=650),
            barmode="group",
        )
        st.plotly_chart(fig_prog, use_container_width=True)

    # ────────────────────────────────────────────────────────────
    # TAB 6: MODEL
    # ────────────────────────────────────────────────────────────
    with tab_model:
        spec_labels_model = {"augmented_gdp": "Augmented GDP",
                             "additive": "Additive (Log + CRRA)",
                             "ces": "CES", "ecological": "Ecological"}
        st.subheader(f"Welfare Model: {spec_labels_model[spec_key]}")
        st.caption("This documentation updates when you switch specifications in the sidebar.")

        st.markdown("---")

        # 1. Period Utility
        st.markdown("### 1. Period Utility Function")
        if spec_key == "augmented_gdp":
            st.latex(r"u_t = \hat{y}_t + \frac{\hat{y}_t}{h_t} \cdot \ell_t = \hat{y}_t \cdot \frac{T}{h_t}")
            st.markdown("""
This is the **implicit utility function of the augmented-GDP approach** used in GJP.

- $\\hat{y}_t$ = effective (damage-adjusted) per-capita income
- $h_t$ = labour hours per year
- $\\ell_t = T - h_t$ = leisure hours
- $T$ = total time endowment (fixed at 4,000 hrs/yr)
- Leisure is valued at the **implied hourly wage** $y/h$ — so a richer person's leisure is worth more
- **No diminishing returns**: doubling income doubles utility
- **No curvature parameters**: no $\\sigma$, $\\chi$, or $\\varepsilon$ to set

**Limitations:** Linear utility implies that (1) the marginal utility of income is constant
regardless of income level, (2) a dollar of income is worth the same to a billionaire
as to someone in poverty, and (3) leisure time scales proportionally with income.
""")
        elif spec_key == "additive":
            st.latex(r"u_t = \ln(\hat{y}_t) + \chi \cdot \frac{\ell_t^{1-\sigma}}{1-\sigma}")
            st.markdown(r"When $\sigma = 1$:  $u_t = \ln(\hat{y}_t) + \chi \cdot \ln(\ell_t)$")
            st.markdown("""
- $\\hat{y}_t$ = effective (damage-adjusted) per-capita income
- $\\ell_t = T - h_t$ = leisure hours
- $\\chi$ = calibrated leisure weight (per region)
- $\\sigma$ = leisure curvature (CRRA on leisure); $\\sigma > 0$ ⇒ diminishing MU
""")
        elif spec_key == "ces":
            st.latex(r"U_t = \left[\alpha \cdot \hat{y}_t^{\,\rho} + (1-\alpha) \cdot \ell_t^{\,\rho}\right]^{1/\rho}")
            st.latex(r"u_t = \ln(U_t) = \frac{1}{\rho} \ln\!\left[\alpha \cdot \hat{y}_t^{\,\rho} + (1-\alpha) \cdot \ell_t^{\,\rho}\right], \quad \rho = \frac{\varepsilon - 1}{\varepsilon}")
            st.markdown("""
- $\\varepsilon$ = elasticity of substitution between income and leisure
- $\\alpha$ = calibrated share parameter
- $\\varepsilon < 1$: complements  |  $\\varepsilon > 1$: substitutes  |  $\\varepsilon = 1$: Cobb-Douglas
""")
        else:
            st.latex(r"u_t = \frac{1}{\rho_{yE}} \ln\!\left[\alpha_{yE} \cdot \hat{y}_t^{\,\rho_{yE}} + (1-\alpha_{yE}) \cdot E_t^{\,\rho_{yE}}\right] + \chi \cdot \frac{\ell_t^{1-\sigma}}{1-\sigma}")
            st.latex(r"\rho_{yE} = \frac{\varepsilon_{yE} - 1}{\varepsilon_{yE}}")
            st.markdown("""
**Three arguments:** income, environment, and leisure.

- $\\hat{y}_t$ = effective income (output damage **only** — $\\varphi$, not $\\gamma$)
- $E_t$ = environmental quality index (see below)
- $\\ell_t = T - h_t$ = leisure hours
- $\\alpha_{yE}$ = weight on income in the income-environment composite
- $\\varepsilon_{yE}$ = elasticity of substitution between income and environment
- $\\chi$, $\\sigma$ = leisure weight and curvature (same as additive)

**Key limiting cases:**
- $\\varepsilon_{yE} \\to 0$ (Leontief): welfare ≈ min(income, environment) — no compensation possible
- $\\varepsilon_{yE} = 1$ (Cobb-Douglas): moderate substitutability
- $\\varepsilon_{yE} \\to \\infty$: income can fully offset environmental loss
""")

        st.markdown("---")

        # 2. Climate Damage Channels
        st.markdown("### 2. Climate Damage Channels")
        if spec_key == "ecological":
            st.markdown("**In the ecological specification, the two empirical damage channels "
                        "are separated:**")
            st.markdown("**Channel A — Output damage (hits income):**")
            st.latex(r"\hat{y}_t = y_t \cdot \exp\!\left(-\varphi \cdot \Delta T_t\right) \quad \text{(exponential)}")
            st.latex(r"\hat{y}_t = \frac{y_t}{1 + \varphi \cdot \Delta T_t^{\,2}} \quad \text{(quadratic)}")
            st.markdown("**Channel B — Non-market well-being damage (enters via E):**")
            st.latex(r"E_t = \frac{1}{1 + \kappa \cdot \Delta T_t}, \qquad \kappa = \gamma / 100")
            st.markdown("""
- $\\varphi$ = output damage (default 12%/°C, Bilal & Känzig 2024) — only this hits income
- $\\gamma$ sets $\\kappa$, which controls how fast environmental quality degrades (default 13.3%/°C, Dietrich & Nichols 2025)
- $E = 1$ at zero warming (pristine); $E \\to 0$ as warming rises
- Under SC (ΔT ≈ 0.4°C): E ≈ 0.95 — modest degradation
- Under PC (ΔT ≈ 2.8°C): E ≈ 0.73 — significant degradation

**Why separate?** The Dietrich & Nichols evidence shows that non-market channels (health, mental well-being, amenity) dominate the welfare impact of extreme heat. Routing these through income conflates two different mechanisms and assumes unlimited substitutability between income and the environment.
""")
        else:
            st.markdown("**Exponential (default):**")
            st.latex(r"\hat{y}_t = y_t \cdot \exp\!\left(-(\varphi + \gamma) \cdot \Delta T_t\right)")
            st.markdown("**Quadratic / DICE-style (alternative):**")
            st.latex(r"\hat{y}_t = \frac{y_t}{1 + (\varphi + \gamma) \cdot \Delta T_t^{\,2}}")
            st.markdown("""
- $y_t$ = raw per-capita income from scenario data
- $\\Delta T_t$ = cumulative temperature change (summed from annual increments)
- $\\varphi$ = output damage (default 12%/°C, Bilal & Känzig 2024)
- $\\gamma$ = well-being damage (default 13.3%/°C, Dietrich & Nichols 2025)
- The exponential form is **concave** in $\\Delta T$ (each additional degree does proportionally less damage)
- The quadratic (DICE-style) form is **convex** in $\\Delta T$ (each additional degree does proportionally more damage), better capturing tipping points and tail risks. Damage never reaches 100%.
""")
            if params.get("damage_type") == "quadratic":
                st.info("**Example (quadratic):** With total damage coeff = 25.3% and ΔT = 2°C: "
                        "effective income = 1 / (1 + 0.253 × 4) ≈ 50% of raw. "
                        "At 3°C: ≈ 30% of raw. Damage accelerates with warming.")
            else:
                st.info("**Example (exponential):** With total damage = 25.3%/°C and ΔT = 2°C: "
                        "effective income = exp(−0.253 × 2) ≈ 60% of raw.")

        st.markdown("---")

        # 3. Calibration
        st.markdown("### 3. Calibration")
        if spec_key == "augmented_gdp":
            st.markdown("**No calibration needed.** The augmented GDP specification has no free "
                        "preference parameters — leisure is valued at the implied wage rate "
                        "$y/h$ directly from the data. The only user-chosen parameters are "
                        "the discount rate and damage coefficients.")
        elif spec_key == "additive":
            st.latex(r"\text{MRS} = \chi \cdot y_0 \cdot \ell_0^{-\sigma} = \frac{y_0}{h_0} \quad \Rightarrow \quad \chi = \frac{\ell_0^{\,\sigma}}{h_0}")
            st.markdown(
                "By default, $\\chi$ is calibrated separately per region from 2025 baseline data. "
                "A key advantage is that $\\chi$ depends only on the time allocation, "
                "not on income levels.\n\n"
                "Alternatively, $\\chi$ can be set manually to a common value across all regions, "
                "which imposes identical preferences everywhere."
            )
        elif spec_key == "ces":
            st.latex(r"\text{MRS} = \frac{1-\alpha}{\alpha}\left(\frac{y_0}{\ell_0}\right)^{1/\varepsilon} = \frac{y_0}{h_0} \quad \Rightarrow \quad \alpha = \frac{1}{1 + (y_0/h_0) \cdot (\ell_0/y_0)^{1/\varepsilon}}")
            st.markdown("Calibrated separately per region from 2025 baseline data.")
        else:
            st.markdown("**Leisure weight χ:**")
            st.latex(r"\chi = \frac{\ell_0^{\,\sigma}}{h_0}")
            st.markdown("Same calibration as the additive specification — depends only on time "
                        "allocation, not income levels.")
            st.markdown("**Income-environment weight α(y,E):**")
            st.markdown("Set directly by the user (default 0.5). This parameter controls the "
                        "relative importance of income vs. environmental quality in welfare. "
                        "It can be explored over a range to reveal which assumptions drive the "
                        "scenario ranking.")
            st.markdown("**Environmental sensitivity κ:**")
            st.latex(r"\kappa = \gamma / 100 \quad \text{(default calibration)}")
            st.markdown(
                "By default, κ is calibrated from the well-being damage coefficient γ so that the marginal "
                "welfare loss from environmental degradation at 1°C matches the Dietrich & Nichols (2025) "
                "evidence.\n\n"
                "Alternatively, κ can be **set manually** to reflect different calibration strategies:\n"
                "- **κ ≈ 0.03–0.04:** Health/mortality channel only (Carleton et al. 2022)\n"
                "- **κ ≈ 0.13:** Dietrich & Nichols default (all non-income via subjective well-being)\n"
                "- **κ ≈ 0.14–0.25:** Bottom-up sum across mortality, mental health, conflict, "
                "ecosystems, amenity, and other channels\n"
                "- **κ ≈ 0.38+:** Severe ecosystem-collapse view (anchored at E ≈ 0.4 at 4°C)\n\n"
                "See *NonIncomeDamages.docx* for the full evidence base."
            )

        st.markdown("---")

        # 4. Lifetime Welfare
        st.markdown("### 4. Lifetime Welfare")
        st.latex(r"W = \sum_{t=0}^{75} \beta^t \cdot u_t, \qquad \beta = \frac{1}{1+r}")
        st.markdown(f"Default discount rate $r$ = {DEFAULT_PARAMS['r']}% per year (76-year horizon: 2025–2100).")

        st.markdown("---")

        # 5. Income Equivalent
        st.markdown("### 5. Income Equivalent")
        if spec_key == "augmented_gdp":
            st.latex(r"\lambda = \frac{W_{SC}}{W_{PC}} - 1")
            st.markdown("Since utility is **linear** in income, a uniform percentage "
                        "increase in income scales welfare proportionally. "
                        "The IE is simply the ratio of lifetime welfare levels.")
        else:
            st.latex(r"\lambda = \exp\!\left(\frac{W_{SC} - W_{PC}}{\sum_t \beta^t}\right) - 1")
            st.markdown("For **log-based** utility, the IE uses the exponential form to "
                        "convert the welfare difference (in log-income units) back to "
                        "a percentage.")
        st.success("**λ > 0:** SC delivers higher welfare. The PC agent would need λ% more "
                   "income every year to match SC.\n\n"
                   "**λ < 0:** PC delivers higher welfare.")

        st.markdown("---")

        # 6. World Aggregation
        st.markdown("### 6. World Aggregation")
        st.markdown("""
| Method | Formula |
|--------|---------|
| **Raw** | Uses World-level data directly (representative agent) |
| **Utilitarian** | $W = \\sum_i (pop_i / pop_{total}) \\cdot W_i$ |
| **Atkinson** | $W = \\left[\\sum_i w_i \\cdot W_i^{1-e}\\right]^{1/(1-e)}$ |
""")

        st.markdown("---")

        # 7. Parameter Reference
        st.markdown("### 7. Parameter Reference")
        symbols = ["r", "T", "\u03C6", "\u03B3"]
        param_names = [
            "Social discount rate",
            "Annual time endowment",
            "Output damage",
            "Well-being damage",
        ]
        defaults = [
            "2.0%/yr",
            "4,000 hrs (fixed)",
            "12%/\u00B0C (Bilal & K\u00E4nzig 2024)",
            "13.3%/\u00B0C (Dietrich & Nichols 2025)",
        ]
        if spec_key in ("additive", "ecological"):
            symbols.insert(1, "\u03C3")
            param_names.insert(1, "Leisure curvature (CRRA)")
            defaults.insert(1, "1.0")
            symbols.append("\u03C7")
            param_names.append("Leisure weight")
            defaults.append("Calibrated from 2025 MRS (or manual)")
        if spec_key == "ces":
            symbols.insert(1, "\u03B5")
            param_names.insert(1, "CES elasticity")
            defaults.insert(1, "0.8")
        if spec_key == "ecological":
            symbols.append("\u03B5(y,E)")
            param_names.append("Income-environment substitutability")
            defaults.append("0.50 (limited substitution)")
            symbols.append("\u03B1(y,E)")
            param_names.append("Income weight in composite")
            defaults.append("0.50 (equal weight)")
            symbols.append("\u03BA")
            param_names.append("Environmental sensitivity")
            defaults.append("\u03B3/100 (calibrated) or manual (0.03\u20130.50)")
        symbols.append("\u2014")
        param_names.append("Damage function")
        defaults.append("Exponential (or Quadratic)")
        param_ref = {"Symbol": symbols, "Parameter": param_names, "Default": defaults}
        st.dataframe(pd.DataFrame(param_ref), use_container_width=True, hide_index=True)

    # ── Excel Export ──
    st.sidebar.divider()
    if st.sidebar.button("Export to Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(all_region_results).to_excel(writer, sheet_name="Summary", index=False)
            export_dict = {
                "Specification": spec_key,
                "DiscountRate_pct": params["r"],
                "TimeEndowment_hrs": params["T"],
                "OutputDamage_pctPerDegC": params["phi"],
                "WellbeingDamage_pctPerDegC": params["gamma_raw"],
                "DamageFunction": params.get("damage_type", "exponential"),
            }
            if spec_key != "augmented_gdp":
                export_dict["Sigma_LeisureCRRA"] = params["sigma"]
            if spec_key in ("additive", "ecological"):
                export_dict["Chi_Override"] = params.get("chi_override", "calibrated")
            if spec_key == "ces":
                export_dict["CES_Epsilon"] = params["epsilon"]
            if spec_key == "ecological":
                export_dict["Epsilon_yE"] = params.get("epsilon_cE", 0.5)
                export_dict["Alpha_yE"] = params.get("alpha_cE", 0.5)
                export_dict["Kappa"] = params.get("kappa", 0.1333)
                export_dict["Kappa_Source"] = "manual" if kappa_val is not None else "gamma/100"
            param_export = pd.DataFrame([export_dict])
            param_export.to_excel(writer, sheet_name="Parameters", index=False)
        st.sidebar.download_button(
            "\u2B07 Download Excel",
            output.getvalue(),
            file_name="WelfareResults.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
