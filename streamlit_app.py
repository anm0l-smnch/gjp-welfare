#!/usr/bin/env python3
"""
Welfare Evaluation App — Streamlit Web Version
================================================
Evaluates lifetime welfare across climate-work scenarios using
additive log + CRRA-in-leisure and CES specifications.

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

    c_hat = effective_consumption(c, dT, phi, gamma, damage_type)
    c_no_damage = c.copy()
    c0, h0 = c[0], h[0]

    if spec == "additive":
        chi_override = params.get("chi_override", None)
        chi = chi_override if chi_override is not None else calibrate_chi(c0, h0, T, sigma)
        u = np.array([period_utility_additive(c_hat[i], h[i], T, sigma, chi)
                       for i in range(n)])
        u_no_damage = np.array([period_utility_additive(c_no_damage[i], h[i], T, sigma, chi)
                                 for i in range(n)])
        u_cons_only = np.log(np.maximum(c_hat, 1e-6))
    else:
        alpha = calibrate_alpha_ces(c0, h0, T, epsilon)
        u = np.array([period_utility_ces(c_hat[i], h[i], T, epsilon, alpha)
                       for i in range(n)])
        u_no_damage = np.array([period_utility_ces(c_no_damage[i], h[i], T, epsilon, alpha)
                                 for i in range(n)])
        u_cons_only = np.log(np.maximum(c_hat, 1e-6))

    discount = beta ** t_idx
    W = np.sum(discount * u)
    W_no_damage = np.sum(discount * u_no_damage)

    return {
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
        spec = st.radio("Specification", ["Additive (Log + CRRA)", "CES"],
                        horizontal=True)
        spec_key = "additive" if "Additive" in spec else "ces"

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
        else:
            sigma_val = DEFAULT_PARAMS["sigma"]
            epsilon_val = st.slider("CES elasticity \u03B5", 0.2, 2.0,
                                    step=0.1, key="sl_epsilon")
            chi_override = None

        st.divider()
        st.subheader("Damage Function")
        damage_type_label = st.radio("Damage specification",
                                     ["Exponential", "Quadratic"],
                                     horizontal=True, key="damage_type")
        damage_type = damage_type_label.lower()
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

        summary_data = {
            "Metric": [
                "Lifetime Welfare (with damages)",
                "Lifetime Welfare (no damages)",
                "Climate Damage Cost",
                "Avg Raw Consumption",
                "Avg Effective Consumption",
                "Avg Leisure (hrs/yr)",
                "Terminal Temp Change (\u00B0C)",
            ],
            "SC": [
                f"{W_sc:.2f}", f"{W_sc_nd:.2f}",
                f"{results['SC']['damage_cost']:.2f}",
                f"{np.mean(results['SC']['raw_consumption']):,.0f}",
                f"{np.mean(results['SC']['effective_consumption']):,.0f}",
                f"{np.mean(results['SC']['leisure']):,.0f}",
                f"{results['SC']['temp_change'][-1]:.2f}",
            ],
            "PC": [
                f"{W_pc:.2f}", f"{W_pc_nd:.2f}",
                f"{results['PC']['damage_cost']:.2f}",
                f"{np.mean(results['PC']['raw_consumption']):,.0f}",
                f"{np.mean(results['PC']['effective_consumption']):,.0f}",
                f"{np.mean(results['PC']['leisure']):,.0f}",
                f"{results['PC']['temp_change'][-1]:.2f}",
            ],
        }
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
        if spec_key == "additive":
            if params.get("chi_override") is not None:
                param_str += f",  \u03C7={params['chi_override']:.1f} (manual)"
            else:
                param_str += ",  \u03C7=calibrated"
        if spec_key == "ces":
            param_str += f",  \u03B5={params['epsilon']:.1f}"
        st.caption(f"Parameters: {param_str}")

    # ────────────────────────────────────────────────────────────
    # TAB 2: CHARTS
    # ────────────────────────────────────────────────────────────
    with tab_charts:
        st.subheader(f"Time Series — {region}")
        years = results["SC"]["years"]

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Per-Capita Consumption", "Leisure (hrs/yr)",
                                            "Period Utility", "Temperature Change"),
                            vertical_spacing=0.12, horizontal_spacing=0.08)

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

        # Period Utility
        fig.add_trace(go.Scatter(x=years, y=results["SC"]["period_utility"],
                                  name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                  showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=years, y=results["PC"]["period_utility"],
                                  name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                  showlegend=False),
                      row=2, col=1)

        # Temperature
        fig.add_trace(go.Scatter(x=years, y=results["SC"]["temp_change"],
                                  name="SC", line=dict(color=COLOUR_SC, width=2.2),
                                  showlegend=False),
                      row=2, col=2)
        fig.add_trace(go.Scatter(x=years, y=results["PC"]["temp_change"],
                                  name="PC", line=dict(color=COLOUR_PC, width=2.2),
                                  showlegend=False),
                      row=2, col=2)

        layout_opts = _plotly_layout(f"Welfare Analysis — {region}", height=650)
        layout_opts["legend"] = dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                     bgcolor="rgba(255,255,255,0.9)",
                                     bordercolor="#DDDDDD", borderwidth=1, font=dict(size=10))
        fig.update_layout(**layout_opts)
        fig.update_yaxes(title_text="Consumption (2025 Euro PPP)", row=1, col=1)
        fig.update_yaxes(title_text="Hours", row=1, col=2)
        fig.update_yaxes(title_text="u(t)", row=2, col=1)
        fig.update_yaxes(title_text="\u0394T (\u00B0C)", row=2, col=2)

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
                        f"{'Additive' if spec_key == 'additive' else 'CES'}",
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
        st.subheader(f"Welfare Model: {'Additive (Log + CRRA)' if spec_key == 'additive' else 'CES'}")
        st.caption("This documentation updates when you switch between Additive and CES "
                   "in the sidebar.")

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
        else:
            st.latex(r"U_t = \left[\alpha \cdot \hat{c}_t^{\,\rho} + (1-\alpha) \cdot \ell_t^{\,\rho}\right]^{1/\rho}")
            st.latex(r"u_t = \ln(U_t) = \frac{1}{\rho} \ln\!\left[\alpha \cdot \hat{c}_t^{\,\rho} + (1-\alpha) \cdot \ell_t^{\,\rho}\right], \quad \rho = \frac{\varepsilon - 1}{\varepsilon}")
            st.markdown("""
- $\\varepsilon$ = elasticity of substitution between consumption and leisure
- $\\alpha$ = calibrated share parameter
- $\\varepsilon < 1$: complements  |  $\\varepsilon > 1$: substitutes  |  $\\varepsilon = 1$: Cobb-Douglas
""")

        st.markdown("---")

        # 2. Climate Damage Wedge
        st.markdown("### 2. Climate Damage Wedge")
        st.markdown("**Exponential (default):**")
        st.latex(r"\hat{c}_t = c_t \cdot \exp\!\left(-(\varphi + \gamma) \cdot \Delta T_t\right)")
        st.markdown("**Quadratic / DICE-style (alternative):**")
        st.latex(r"\hat{c}_t = \frac{c_t}{1 + (\varphi + \gamma) \cdot \Delta T_t^{\,2}}")
        st.markdown("""
- $c_t$ = raw consumption from scenario data
- $\\Delta T_t$ = cumulative temperature change (summed from annual increments)
- $\\varphi$ = output damage (default 12%/°C, Bilal & Känzig 2026)
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
        else:
            st.latex(r"\text{MRS} = \frac{1-\alpha}{\alpha}\left(\frac{c_0}{\ell_0}\right)^{1/\varepsilon} = \frac{c_0}{h_0} \quad \Rightarrow \quad \alpha = \frac{1}{1 + (c_0/h_0) \cdot (\ell_0/c_0)^{1/\varepsilon}}")
            st.markdown("Calibrated separately per region from 2025 baseline data.")

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
        symbols = ["r", "\u03C3" if spec_key == "additive" else "\u03B5",
                    "T", "\u03C6", "\u03B3"]
        param_names = [
            "Social discount rate",
            "Leisure curvature (CRRA)" if spec_key == "additive" else "CES elasticity",
            "Annual time endowment",
            "Output damage",
            "Well-being damage",
        ]
        defaults = [
            "2.0%/yr",
            "1.0" if spec_key == "additive" else "0.8",
            "4,000 hrs (fixed)",
            "12%/\u00B0C (Bilal & K\u00E4nzig 2026)",
            "13.3%/\u00B0C (Dietrich & Nichols 2025)",
        ]
        if spec_key == "additive":
            symbols.append("\u03C7")
            param_names.append("Leisure weight")
            defaults.append("Calibrated from 2025 MRS (or manual)")
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
            param_export = pd.DataFrame([{
                "Specification": spec_key,
                "DiscountRate_pct": params["r"],
                "Sigma_LeisureCRRA": params["sigma"],
                "TimeEndowment_hrs": params["T"],
                "OutputDamage_pctPerDegC": params["phi"],
                "WellbeingDamage_pctPerDegC": params["gamma_raw"],
                "CES_Epsilon": params["epsilon"],
                "DamageFunction": params.get("damage_type", "exponential"),
                "Chi_Override": params.get("chi_override", "calibrated"),
            }])
            param_export.to_excel(writer, sheet_name="Parameters", index=False)
        st.sidebar.download_button(
            "\u2B07 Download Excel",
            output.getvalue(),
            file_name="WelfareResults.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
