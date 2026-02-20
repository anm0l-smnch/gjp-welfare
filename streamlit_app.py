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
            if "consumption" in str(cl):
                col_map[c] = "Consumption"
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


def calibrate_chi(c0, h0, T, sigma):
    ell0 = T - h0
    if ell0 <= 0 or h0 <= 0:
        return 0.0
    return (ell0 ** sigma) / h0


def effective_consumption(c, delta_T, phi, gamma, damage_type="exponential"):
    coeff = phi + gamma
    if damage_type == "quadratic":
        # DICE/Nordhaus-style: convex in ΔT, never hits zero
        return c / (1.0 + coeff * delta_T ** 2)
    return c * np.exp(-coeff * delta_T)


def period_utility_additive(c_hat, h, T, sigma, chi):
    ell = T - h
    ell = np.maximum(ell, 1e-6)
    c_hat = np.maximum(c_hat, 1e-6)
    u_cons = np.log(c_hat)
    if abs(sigma - 1.0) < 1e-10:
        u_leisure = chi * np.log(ell)
    else:
        u_leisure = chi * (ell ** (1.0 - sigma)) / (1.0 - sigma)
    return u_cons + u_leisure


def period_utility_ces(c_hat, h, T, epsilon, alpha):
    ell = T - h
    ell = np.maximum(ell, 1e-6)
    c_hat = np.maximum(c_hat, 1e-6)
    rho = (epsilon - 1.0) / epsilon
    if abs(rho) < 1e-10:
        return alpha * np.log(c_hat) + (1 - alpha) * np.log(ell)
    agg = alpha * (c_hat ** rho) + (1 - alpha) * (ell ** rho)
    agg = np.maximum(agg, 1e-30)
    return np.log(agg) / rho


def calibrate_alpha_ces(c0, h0, T, epsilon):
    ell0 = T - h0
    if ell0 <= 0 or h0 <= 0:
        return 0.5
    vot = c0 / h0
    mrs_ratio = vot * (ell0 / c0) ** (1.0 / epsilon)
    return 1.0 / (1.0 + mrs_ratio)


def env_quality(delta_T, kappa):
    """Environmental quality index: E = 1 / (1 + kappa * delta_T).
    E = 1 at zero warming (pristine), declines toward 0 as warming rises."""
    return 1.0 / (1.0 + kappa * np.maximum(delta_T, 0.0))


def effective_consumption_output_only(c, delta_T, phi, damage_type="exponential"):
    """Effective consumption with OUTPUT damage only (no well-being/amenity damage).
    Used in the ecological specification where non-market damages enter via E."""
    if damage_type == "quadratic":
        return c / (1.0 + phi * delta_T ** 2)
    return c * np.exp(-phi * delta_T)


def period_utility_ecological(c_eff, h, E, T, sigma, chi, alpha_cE, epsilon_cE, c_ref=1.0):
    """Ecological utility: CES composite of (normalized consumption, environment) + CRRA leisure.

    u = (1/rho) * ln[alpha * c_norm^rho + (1-alpha) * E^rho] + chi * ell^(1-sigma)/(1-sigma)

    where c_norm = c_eff / c_ref normalises consumption to be on a comparable scale to E (both ~1),
    and rho = (epsilon_cE - 1) / epsilon_cE controls substitutability.

    Without normalisation, c ~ 50,000 vs E ~ 0.7-1.0, and the CES composite
    breaks down for rho < 0 (the c^rho term becomes negligible).
    """
    ell = T - h
    ell = np.maximum(ell, 1e-6)
    c_eff = np.maximum(c_eff, 1e-6)
    E = np.maximum(E, 1e-6)

    # Normalise consumption to be on comparable scale to E
    c_norm = c_eff / c_ref

    # CES composite of normalised consumption and environment
    rho = (epsilon_cE - 1.0) / epsilon_cE
    if abs(rho) < 1e-10:
        # Cobb-Douglas limit: rho -> 0
        u_composite = alpha_cE * np.log(c_norm) + (1.0 - alpha_cE) * np.log(E)
    elif epsilon_cE < 0.02:
        # Near-Leontief: use min
        u_composite = np.log(np.minimum(c_norm ** alpha_cE, E ** (1.0 - alpha_cE)))
    else:
        agg = alpha_cE * (c_norm ** rho) + (1.0 - alpha_cE) * (E ** rho)
        agg = np.maximum(agg, 1e-30)
        u_composite = np.log(agg) / rho

    # Leisure component (same as additive)
    if abs(sigma - 1.0) < 1e-10:
        u_leisure = chi * np.log(ell)
    else:
        u_leisure = chi * (ell ** (1.0 - sigma)) / (1.0 - sigma)

    return u_composite + u_leisure


def calibrate_kappa(gamma_pct):
    """Calibrate kappa so that the marginal welfare loss from E at 1°C warming
    roughly matches the non-market damage evidence (gamma %/°C).

    For small ΔT, E ≈ 1 - kappa*ΔT, so the proportional loss is kappa*ΔT.
    At 1°C we want this to be gamma/100.  Hence kappa ≈ gamma/100.
    """
    return gamma_pct / 100.0


def calibrate_alpha_ecological(c0, E0, gamma_pct):
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
    c = df["Consumption"].values
    h = df["LabourHours"].values
    dT_increments = df["TempChange"].values
    dT = np.cumsum(dT_increments)
    n = len(years)
    t_idx = np.arange(n)

    c_no_damage = c.copy()
    c0, h0 = c[0], h[0]

    if spec == "ecological":
        # Ecological: output damage on consumption only; environment enters utility directly
        kappa = params.get("kappa", calibrate_kappa(params["gamma_raw"]))
        alpha_cE = params.get("alpha_cE", 0.5)
        epsilon_cE = params.get("epsilon_cE", 0.5)
        chi_override = params.get("chi_override", None)
        chi = chi_override if chi_override is not None else calibrate_chi(c0, h0, T, sigma)

        # Only output damage (phi) hits consumption; gamma goes into E
        c_hat = effective_consumption_output_only(c, dT, phi, damage_type)
        E = env_quality(dT, kappa)
        E_pristine = np.ones(n)  # no damage counterfactual

        # Use baseline consumption as reference for normalisation (c and E on same scale)
        c_ref = c0

        u = np.array([period_utility_ecological(c_hat[i], h[i], E[i], T, sigma, chi,
                                                 alpha_cE, epsilon_cE, c_ref)
                       for i in range(n)])
        u_no_damage = np.array([period_utility_ecological(c_no_damage[i], h[i], E_pristine[i],
                                                           T, sigma, chi, alpha_cE, epsilon_cE, c_ref)
                                 for i in range(n)])
    else:
        # Additive and CES: all damage (phi+gamma) hits consumption
        c_hat = effective_consumption(c, dT, phi, gamma, damage_type)
        E = None

        if spec == "additive":
            chi_override = params.get("chi_override", None)
            chi = chi_override if chi_override is not None else calibrate_chi(c0, h0, T, sigma)
            u = np.array([period_utility_additive(c_hat[i], h[i], T, sigma, chi)
                           for i in range(n)])
            u_no_damage = np.array([period_utility_additive(c_no_damage[i], h[i], T, sigma, chi)
                                     for i in range(n)])
        else:
            alpha = calibrate_alpha_ces(c0, h0, T, epsilon)
            u = np.array([period_utility_ces(c_hat[i], h[i], T, epsilon, alpha)
                           for i in range(n)])
            u_no_damage = np.array([period_utility_ces(c_no_damage[i], h[i], T, epsilon, alpha)
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
        "effective_consumption": c_hat,
        "raw_consumption": c,
        "leisure": T - h,
        "temp_change": dT,
        "discount_factors": discount,
        "discounted_utility": discount * u,
    }
    if E is not None:
        result["env_quality"] = E
    return result


def consumption_equivalent(W_sc, W_pc, n_periods, r):
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
    ce = consumption_equivalent(scenario_results["SC"]["welfare"],
                                 scenario_results["PC"]["welfare"], n, params["r"])
    return {"SC": scenario_results["SC"], "PC": scenario_results["PC"],
            "consumption_equivalent": ce, "regions": regions}


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
                        ["Additive (Log + CRRA)", "CES", "Ecological"],
                        horizontal=True)
        if "Additive" in spec:
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
        }
        for k, v in _slider_keys.items():
            if k not in st.session_state:
                st.session_state[k] = v
        if "chi_mode" not in st.session_state:
            st.session_state["chi_mode"] = "Calibrate from 2025 data"
        if "damage_type" not in st.session_state:
            st.session_state["damage_type"] = "Exponential"

        def _reset_params():
            for k, v in _slider_keys.items():
                st.session_state[k] = v
            st.session_state["chi_mode"] = "Calibrate from 2025 data"
            st.session_state["damage_type"] = "Exponential"

        r_val = st.slider("Discount rate r (%)", 0.5, 6.0, step=0.1, key="sl_r")

        if spec_key == "additive":
            sigma_val = st.slider("Leisure curvature \u03C3", -1.0, 3.0,
                                  step=0.1, key="sl_sigma")
            epsilon_val = DEFAULT_PARAMS["epsilon"]

            chi_mode = st.radio("Leisure weight \u03C7",
                                ["Calibrate from 2025 data", "Set manually"],
                                horizontal=True, key="chi_mode")
            chi_override = None
            if chi_mode == "Set manually":
                chi_override = st.slider("Leisure weight \u03C7", 0.5, 10.0,
                                         step=0.1, key="sl_chi")
            else:
                st.caption("\u03C7 = \u2113\u2080\u1D9E / h\u2080 (calibrated per region)")
        elif spec_key == "ces":
            sigma_val = DEFAULT_PARAMS["sigma"]
            epsilon_val = st.slider("CES elasticity \u03B5", 0.2, 2.0,
                                    step=0.1, key="sl_epsilon")
            chi_override = None
        else:
            # Ecological specification
            sigma_val = st.slider("Leisure curvature \u03C3", -1.0, 3.0,
                                  step=0.1, key="sl_sigma")
            epsilon_val = DEFAULT_PARAMS["epsilon"]

            chi_mode = st.radio("Leisure weight \u03C7",
                                ["Calibrate from 2025 data", "Set manually"],
                                horizontal=True, key="chi_mode")
            chi_override = None
            if chi_mode == "Set manually":
                chi_override = st.slider("Leisure weight \u03C7", 0.5, 10.0,
                                         step=0.1, key="sl_chi")
            else:
                st.caption("\u03C7 = \u2113\u2080\u1D9E / h\u2080 (calibrated per region)")

            st.divider()
            st.subheader("Ecological Parameters")
            epsilon_cE_val = st.slider(
                "Substitutability \u03B5(c,E)",
                0.01, 2.0, step=0.01, key="sl_epsilon_cE",
                help="Elasticity of substitution between consumption and environment. "
                     "0 = Leontief (no substitution), 1 = Cobb-Douglas, >1 = substitutes.")
            alpha_cE_val = st.slider(
                "Consumption weight \u03B1(c,E)",
                0.1, 0.9, step=0.05, key="sl_alpha_cE",
                help="Weight on consumption in the CES composite. "
                     "Higher = consumption matters more; lower = environment matters more.")

            # Show interpretation of epsilon_cE
            if epsilon_cE_val < 0.1:
                st.caption("\u2248 Leontief: welfare \u2248 min(consumption, environment)")
            elif abs(epsilon_cE_val - 1.0) < 0.05:
                st.caption("\u2248 Cobb-Douglas: moderate substitutability")
            elif epsilon_cE_val > 1.5:
                st.caption("High substitutability: consumption can offset env. loss")
            else:
                st.caption(f"\u03B5(c,E) = {epsilon_cE_val:.2f}: limited substitutability")

        st.divider()
        st.subheader("Damage Function")
        damage_type_label = st.radio("Damage specification",
                                     ["Exponential", "Quadratic"],
                                     horizontal=True, key="damage_type")
        damage_type = damage_type_label.lower()
        if spec_key == "ecological":
            st.caption("**Ecological mode:** output damage (\u03C6) hits consumption; "
                       "well-being damage (\u03B3) sets environmental quality \u03BA.")
            if damage_type == "exponential":
                st.caption("\u0109 = c \u00B7 exp[\u2013\u03C6\u00B7\u0394T]  |  "
                           "E = 1 / (1 + \u03BA\u00B7\u0394T),  \u03BA = \u03B3/100")
            else:
                st.caption("\u0109 = c / [1 + \u03C6\u00B7\u0394T\u00B2]  |  "
                           "E = 1 / (1 + \u03BA\u00B7\u0394T),  \u03BA = \u03B3/100")
        else:
            if damage_type == "exponential":
                st.caption("\u0109 = c \u00B7 exp[\u2013(\u03C6+\u03B3)\u00B7\u0394T]")
            else:
                st.caption("\u0109 = c / [1 + (\u03C6+\u03B3)\u00B7\u0394T\u00B2]  (DICE-style)")

        phi_val = st.slider("Output damage (%/\u00B0C)", 0.0, 30.0,
                            step=1.0, key="sl_phi")
        gamma_val = st.slider("Well-being damage (%/\u00B0C)", 0.0, 30.0,
                              step=1.0, key="sl_gamma")

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
            params["kappa"] = calibrate_kappa(gamma_val)

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
                                  }[x])
            if agg_method == "atkinson":
                aversion_val = st.slider("Inequality aversion (e)", 0.0, 3.0, 1.0, 0.1)

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
        ce = pw["consumption_equivalent"]
    else:
        for scenario in ["SC", "PC"]:
            df_r = data[scenario][data[scenario]["Region"] == region].sort_values("Year").reset_index(drop=True)
            results[scenario] = compute_welfare(df_r, params, spec=spec_key)
        n = len(results["SC"]["years"])
        ce = consumption_equivalent(results["SC"]["welfare"], results["PC"]["welfare"],
                                     n, params["r"])
    results["consumption_equivalent"] = ce

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
            row[f"{scenario}_avg_cons"] = np.mean(res["raw_consumption"])
            row[f"{scenario}_avg_leisure"] = np.mean(res["leisure"])
            row[f"{scenario}_terminal_temp"] = res["temp_change"][-1]
            row[f"{scenario}_utility_2025"] = res["period_utility"][0]
            row[f"{scenario}_utility_2100"] = res["period_utility"][-1]
            row[f"{scenario}_effcons_2025"] = res["effective_consumption"][0]
            row[f"{scenario}_effcons_2100"] = res["effective_consumption"][-1]
            row[f"{scenario}_leisure_2025"] = res["leisure"][0]
            row[f"{scenario}_leisure_2100"] = res["leisure"][-1]
            row[f"{scenario}_temp_2025"] = res["temp_change"][0]
            row[f"{scenario}_temp_2100"] = res["temp_change"][-1]
        n_yr = len(data["SC"][data["SC"]["Region"] == reg]["Year"].unique())
        row["CE_pct"] = consumption_equivalent(
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
        spec_text = f"**{'Additive (Log + CRRA)' if spec_key == 'additive' else 'CES'}**"
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
        col3.metric("Consumption Equivalent", f"{ce:+.2f}%",
                     delta=f"{'SC' if ce > 0 else 'PC'} preferred")

        summary_metrics = [
            "Lifetime Welfare (with damages)",
            "Lifetime Welfare (no damages)",
            "Climate Damage Cost",
            "Avg Raw Consumption",
            "Avg Effective Consumption",
            "Avg Leisure (hrs/yr)",
            "Terminal Temp Change (\u00B0C)",
        ]
        summary_sc = [
            f"{W_sc:.2f}", f"{W_sc_nd:.2f}",
            f"{results['SC']['damage_cost']:.2f}",
            f"{np.mean(results['SC']['raw_consumption']):,.0f}",
            f"{np.mean(results['SC']['effective_consumption']):,.0f}",
            f"{np.mean(results['SC']['leisure']):,.0f}",
            f"{results['SC']['temp_change'][-1]:.2f}",
        ]
        summary_pc = [
            f"{W_pc:.2f}", f"{W_pc_nd:.2f}",
            f"{results['PC']['damage_cost']:.2f}",
            f"{np.mean(results['PC']['raw_consumption']):,.0f}",
            f"{np.mean(results['PC']['effective_consumption']):,.0f}",
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

        param_str = (f"r={params['r']:.1f}%,  \u03C3={params['sigma']:.1f},  "
                     f"T={params['T']:.0f}h,  output damage={params['phi']:.1f}%/\u00B0C,  "
                     f"well-being damage={params['gamma_raw']:.1f}%/\u00B0C,  "
                     f"damage fn={params.get('damage_type', 'exponential')}")
        if spec_key in ("additive", "ecological"):
            if params.get("chi_override") is not None:
                param_str += f",  \u03C7={params['chi_override']:.1f} (manual)"
            else:
                param_str += ",  \u03C7=calibrated"
        if spec_key == "ces":
            param_str += f",  \u03B5={params['epsilon']:.1f}"
        if spec_key == "ecological":
            param_str += (f",  \u03B5(c,E)={params['epsilon_cE']:.2f}"
                          f",  \u03B1(c,E)={params['alpha_cE']:.2f}"
                          f",  \u03BA={params['kappa']:.4f}")
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
                                subplot_titles=("Per-Capita Consumption", "Leisure (hrs/yr)",
                                                "Environmental Quality (E)", "Temperature Change",
                                                "Period Utility", "Discounted Utility"),
                                vertical_spacing=0.08, horizontal_spacing=0.08)
            chart_height = 900
        else:
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=("Per-Capita Consumption", "Leisure (hrs/yr)",
                                                "Period Utility", "Temperature Change"),
                                vertical_spacing=0.12, horizontal_spacing=0.08)
            chart_height = 650

        # Consumption
        fig.add_trace(go.Scatter(x=years, y=results["SC"]["raw_consumption"],
                                  name="SC (raw)", line=dict(color=COLOUR_SC, width=2.2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=results["PC"]["raw_consumption"],
                                  name="PC (raw)", line=dict(color=COLOUR_PC, width=2.2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=results["SC"]["effective_consumption"],
                                  name="SC (eff.)", line=dict(color=COLOUR_SC_LIGHT, width=1.6, dash="dash"),
                                  showlegend=True),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=results["PC"]["effective_consumption"],
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
        fig.update_yaxes(title_text="Consumption (2025 Euro PPP)", row=1, col=1)
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

        # Bar chart — Consumption Equivalent (%)
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
            **_plotly_layout("Consumption Equivalent by Region (SC vs PC)", height=450),
        )
        fig_reg.update_yaxes(title_text="Consumption Equivalent (%)",
                             zeroline=True, zerolinewidth=1.5, zerolinecolor="#888888")
        st.plotly_chart(fig_reg, use_container_width=True)

        st.info(
            "**How to read this chart:** The consumption equivalent (CE) measures the "
            "permanent percentage change in consumption that an individual living under PC "
            "would need *every year* to be as well off as under SC. A CE of +10% means SC "
            "delivers the same welfare as PC would if PC\u2019s consumption were permanently "
            "raised by 10%. Positive values (teal) indicate SC is welfare-superior; negative "
            "values (coral) would indicate PC is welfare-superior."
        )

    # ────────────────────────────────────────────────────────────
    # TAB 4: SENSITIVITY
    # ────────────────────────────────────────────────────────────
    with tab_sensitivity:
        st.subheader("Sensitivity Heatmap")

        sens_param_defs = {
            "r": ("Discount rate r (%)", np.arange(0.5, 5.1, 0.5)),
            "sigma": ("Leisure curvature \u03C3", np.arange(-1.0, 3.1, 0.5)),
            "phi": ("Output damage (%/\u00B0C)", np.arange(0.0, 31.0, 3.0)),
            "gamma_raw": ("Well-being damage (%/\u00B0C)", np.arange(0.0, 31.0, 3.0)),
            "epsilon": ("CES elasticity \u03B5", np.arange(0.2, 2.1, 0.2)),
        }
        if spec_key == "ecological":
            sens_param_defs["epsilon_cE"] = ("Substitutability \u03B5(c,E)",
                                              np.arange(0.05, 2.05, 0.1))
            sens_param_defs["alpha_cE"] = ("Consumption weight \u03B1(c,E)",
                                            np.arange(0.1, 0.91, 0.1))

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
                        ce_grid[i, j] = consumption_equivalent(
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
                        f"{'Additive' if spec_key == 'additive' else 'CES' if spec_key == 'ces' else 'Ecological'}",
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
                "SC Cons %": fmt_pct(pct_change(row.get("SC_effcons_2025", 0),
                                                 row.get("SC_effcons_2100", 0))),
                "PC Cons %": fmt_pct(pct_change(row.get("PC_effcons_2025", 0),
                                                 row.get("PC_effcons_2100", 0))),
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
            ("Effective Consumption", "effcons", True),
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
        spec_labels = {"additive": "Additive (Log + CRRA)",
                       "ces": "CES", "ecological": "Ecological"}
        st.subheader(f"Welfare Model: {spec_labels[spec_key]}")
        st.caption("This documentation updates when you switch specifications in the sidebar.")

        st.markdown("---")

        # 1. Period Utility
        st.markdown("### 1. Period Utility Function")
        if spec_key == "additive":
            st.latex(r"u_t = \ln(\hat{c}_t) + \chi \cdot \frac{\ell_t^{1-\sigma}}{1-\sigma}")
            st.markdown(r"When $\sigma = 1$:  $u_t = \ln(\hat{c}_t) + \chi \cdot \ln(\ell_t)$")
            st.markdown("""
- $\\hat{c}_t$ = effective (damage-adjusted) consumption
- $\\ell_t = T - h_t$ = leisure hours
- $\\chi$ = calibrated leisure weight (per region)
- $\\sigma$ = leisure curvature (CRRA on leisure); $\\sigma > 0$ ⇒ diminishing MU
""")
        elif spec_key == "ces":
            st.latex(r"U_t = \left[\alpha \cdot \hat{c}_t^{\,\rho} + (1-\alpha) \cdot \ell_t^{\,\rho}\right]^{1/\rho}")
            st.latex(r"u_t = \ln(U_t) = \frac{1}{\rho} \ln\!\left[\alpha \cdot \hat{c}_t^{\,\rho} + (1-\alpha) \cdot \ell_t^{\,\rho}\right], \quad \rho = \frac{\varepsilon - 1}{\varepsilon}")
            st.markdown("""
- $\\varepsilon$ = elasticity of substitution between consumption and leisure
- $\\alpha$ = calibrated share parameter
- $\\varepsilon < 1$: complements  |  $\\varepsilon > 1$: substitutes  |  $\\varepsilon = 1$: Cobb-Douglas
""")
        else:
            st.latex(r"u_t = \frac{1}{\rho_{cE}} \ln\!\left[\alpha_{cE} \cdot \hat{c}_t^{\,\rho_{cE}} + (1-\alpha_{cE}) \cdot E_t^{\,\rho_{cE}}\right] + \chi \cdot \frac{\ell_t^{1-\sigma}}{1-\sigma}")
            st.latex(r"\rho_{cE} = \frac{\varepsilon_{cE} - 1}{\varepsilon_{cE}}")
            st.markdown("""
**Three arguments:** consumption, environment, and leisure.

- $\\hat{c}_t$ = effective consumption (output damage **only** — $\\varphi$, not $\\gamma$)
- $E_t$ = environmental quality index (see below)
- $\\ell_t = T - h_t$ = leisure hours
- $\\alpha_{cE}$ = weight on consumption in the consumption-environment composite
- $\\varepsilon_{cE}$ = elasticity of substitution between consumption and environment
- $\\chi$, $\\sigma$ = leisure weight and curvature (same as additive)

**Key limiting cases:**
- $\\varepsilon_{cE} \\to 0$ (Leontief): welfare ≈ min(consumption, environment) — no compensation possible
- $\\varepsilon_{cE} = 1$ (Cobb-Douglas): moderate substitutability
- $\\varepsilon_{cE} \\to \\infty$: consumption can fully offset environmental loss
""")

        st.markdown("---")

        # 2. Climate Damage Channels
        st.markdown("### 2. Climate Damage Channels")
        if spec_key == "ecological":
            st.markdown("**In the ecological specification, the two empirical damage channels "
                        "are separated:**")
            st.markdown("**Channel A — Output damage (hits consumption):**")
            st.latex(r"\hat{c}_t = c_t \cdot \exp\!\left(-\varphi \cdot \Delta T_t\right) \quad \text{(exponential)}")
            st.latex(r"\hat{c}_t = \frac{c_t}{1 + \varphi \cdot \Delta T_t^{\,2}} \quad \text{(quadratic)}")
            st.markdown("**Channel B — Non-market well-being damage (enters via E):**")
            st.latex(r"E_t = \frac{1}{1 + \kappa \cdot \Delta T_t}, \qquad \kappa = \gamma / 100")
            st.markdown("""
- $\\varphi$ = output damage (default 12%/°C, Bilal & Känzig 2024) — only this hits consumption
- $\\gamma$ sets $\\kappa$, which controls how fast environmental quality degrades (default 13.3%/°C, Dietrich & Nichols 2025)
- $E = 1$ at zero warming (pristine); $E \\to 0$ as warming rises
- Under SC (ΔT ≈ 0.4°C): E ≈ 0.95 — modest degradation
- Under PC (ΔT ≈ 2.8°C): E ≈ 0.73 — significant degradation

**Why separate?** The Dietrich & Nichols evidence shows that non-income channels (health, mental well-being, amenity) dominate the welfare impact of extreme heat. Routing these through consumption conflates two different mechanisms and assumes unlimited substitutability between consumption and the environment.
""")
        else:
            st.markdown("**Exponential (default):**")
            st.latex(r"\hat{c}_t = c_t \cdot \exp\!\left(-(\varphi + \gamma) \cdot \Delta T_t\right)")
            st.markdown("**Quadratic / DICE-style (alternative):**")
            st.latex(r"\hat{c}_t = \frac{c_t}{1 + (\varphi + \gamma) \cdot \Delta T_t^{\,2}}")
            st.markdown("""
- $c_t$ = raw consumption from scenario data
- $\\Delta T_t$ = cumulative temperature change (summed from annual increments)
- $\\varphi$ = output damage (default 12%/°C, Bilal & Känzig 2024)
- $\\gamma$ = well-being damage (default 13.3%/°C, Dietrich & Nichols 2025)
- The exponential form is **concave** in $\\Delta T$ (each additional degree does proportionally less damage)
- The quadratic (DICE-style) form is **convex** in $\\Delta T$ (each additional degree does proportionally more damage), better capturing tipping points and tail risks. Damage never reaches 100%.
""")
            if params.get("damage_type") == "quadratic":
                st.info("**Example (quadratic):** With total damage coeff = 25.3% and ΔT = 2°C: "
                        "effective consumption = 1 / (1 + 0.253 × 4) ≈ 50% of raw. "
                        "At 3°C: ≈ 30% of raw. Damage accelerates with warming.")
            else:
                st.info("**Example (exponential):** With total damage = 25.3%/°C and ΔT = 2°C: "
                        "effective consumption = exp(−0.253 × 2) ≈ 60% of raw.")

        st.markdown("---")

        # 3. Calibration
        st.markdown("### 3. Calibration")
        if spec_key == "additive":
            st.latex(r"\text{MRS} = \chi \cdot c_0 \cdot \ell_0^{-\sigma} = \frac{c_0}{h_0} \quad \Rightarrow \quad \chi = \frac{\ell_0^{\,\sigma}}{h_0}")
            st.markdown(
                "By default, $\\chi$ is calibrated separately per region from 2025 baseline data. "
                "A key advantage is that $\\chi$ depends only on the time allocation, "
                "not on consumption levels.\n\n"
                "Alternatively, $\\chi$ can be set manually to a common value across all regions, "
                "which imposes identical preferences everywhere."
            )
        elif spec_key == "ces":
            st.latex(r"\text{MRS} = \frac{1-\alpha}{\alpha}\left(\frac{c_0}{\ell_0}\right)^{1/\varepsilon} = \frac{c_0}{h_0} \quad \Rightarrow \quad \alpha = \frac{1}{1 + (c_0/h_0) \cdot (\ell_0/c_0)^{1/\varepsilon}}")
            st.markdown("Calibrated separately per region from 2025 baseline data.")
        else:
            st.markdown("**Leisure weight χ:**")
            st.latex(r"\chi = \frac{\ell_0^{\,\sigma}}{h_0}")
            st.markdown("Same calibration as the additive specification — depends only on time "
                        "allocation, not consumption levels.")
            st.markdown("**Consumption-environment weight α(c,E):**")
            st.markdown("Set directly by the user (default 0.5). This parameter controls the "
                        "relative importance of consumption vs. environmental quality in welfare. "
                        "It can be explored over a range to reveal which assumptions drive the "
                        "scenario ranking.")
            st.markdown("**Environmental sensitivity κ:**")
            st.latex(r"\kappa = \gamma / 100")
            st.markdown("Calibrated from the well-being damage coefficient γ so that the marginal "
                        "welfare loss from environmental degradation at 1°C matches the Dietrich & "
                        "Nichols evidence.")

        st.markdown("---")

        # 4. Lifetime Welfare
        st.markdown("### 4. Lifetime Welfare")
        st.latex(r"W = \sum_{t=0}^{75} \beta^t \cdot u_t, \qquad \beta = \frac{1}{1+r}")
        st.markdown(f"Default discount rate $r$ = {DEFAULT_PARAMS['r']}% per year (76-year horizon: 2025–2100).")

        st.markdown("---")

        # 5. Consumption Equivalent
        st.markdown("### 5. Consumption Equivalent")
        st.latex(r"\lambda = \exp\!\left(\frac{W_{SC} - W_{PC}}{\sum_t \beta^t}\right) - 1")
        st.success("**λ > 0:** SC delivers higher welfare. The PC agent would need λ% more "
                   "consumption every year to match SC.\n\n"
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
            symbols.append("\u03B5(c,E)")
            param_names.append("Consumption-environment substitutability")
            defaults.append("0.50 (limited substitution)")
            symbols.append("\u03B1(c,E)")
            param_names.append("Consumption weight in composite")
            defaults.append("0.50 (equal weight)")
            symbols.append("\u03BA")
            param_names.append("Environmental sensitivity")
            defaults.append("\u03B3/100 (calibrated from well-being damage)")
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
                "Sigma_LeisureCRRA": params["sigma"],
                "TimeEndowment_hrs": params["T"],
                "OutputDamage_pctPerDegC": params["phi"],
                "WellbeingDamage_pctPerDegC": params["gamma_raw"],
                "CES_Epsilon": params["epsilon"],
                "DamageFunction": params.get("damage_type", "exponential"),
                "Chi_Override": params.get("chi_override", "calibrated"),
            }
            if spec_key == "ecological":
                export_dict["Epsilon_cE"] = params.get("epsilon_cE", 0.5)
                export_dict["Alpha_cE"] = params.get("alpha_cE", 0.5)
                export_dict["Kappa"] = params.get("kappa", 0.1333)
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
