import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gumbel_r
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="LankaFlood-AI | National Flood Risk Engine",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background-color: #1a2234;
        border-radius: 8px;
        padding: 10px 15px;
        border-left: 4px solid #3498db;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 2. Load & Cache Modeled Data
@st.cache_data
def load_data():
  df = pd.read_csv("sri_lanka_flood_risk_modeled.csv")
  df["date"] = pd.to_datetime(df["date"])
  return df


try:
  df = load_data()
except Exception:
  st.error(
      "⚠️ Dataset 'sri_lanka_flood_risk_modeled.csv' not found. Please ensure"
      " the CSV is in the root directory."
  )
  st.stop()


# 3. Precompute Gumbel Parameters per District
@st.cache_data
def get_gumbel_parameters(data):
  params = {}
  annual_max = (
      data.groupby(["district", "year"])[["rain_24h", "rain_72h"]]
      .max()
      .reset_index()
  )
  for district, group in annual_max.groupby("district"):
    loc_24, scale_24 = gumbel_r.fit(group["rain_24h"])
    loc_72, scale_72 = gumbel_r.fit(group["rain_72h"])
    t5_72 = gumbel_r.ppf(0.80, loc_72, scale_72)
    t10_24 = gumbel_r.ppf(0.90, loc_24, scale_24)
    t25_24 = gumbel_r.ppf(0.96, loc_24, scale_24)
    params[district] = {
        "loc_24": loc_24,
        "scale_24": scale_24,
        "loc_72": loc_72,
        "scale_72": scale_72,
        "t5_72": t5_72,
        "t10_24": t10_24,
        "t25_24": t25_24,
    }
  return params


gumbel_params = get_gumbel_parameters(df)

# --- HEADER SECTION ---
st.title("🌧️ LankaFlood-AI: National Extreme Weather Risk Engine")
st.markdown(
    "**Applied Statistics & Extreme Value Theory (GEV) for Sri Lanka's 25"
    " Districts** | *Rasindu Pramith • BSc (Hons) in Applied Statistics, UoC*"
)
st.divider()

# Sidebar Navigation
mode = st.sidebar.radio(
    "Navigation",
    [
        "🗺️ National Risk Map (Historical)",
        "🧪 Interactive Flood Simulator",
        "📊 Statistical Return Periods",
    ],
)

# ==========================================
# TAB 1: NATIONAL RISK MAP (HISTORICAL VIEWER)
# ==========================================
if mode == "🗺️ National Risk Map (Historical)":
  st.subheader("🗺️ Island-Wide Flood & Inundation Risk Map")

  col_ctrl, col_info = st.columns([1, 2])
  with col_ctrl:
    quick_pick = st.selectbox(
        "Select Notable Historical Event:",
        [
            "Custom Date",
            "May 16, 2016 (Cyclone Roanu Catastrophe)",
            "May 25, 2017 (Southwest Monsoon Surge)",
            "Nov 10, 2023 (Inter-Monsoon Bursts)",
        ],
    )
    if quick_pick == "May 16, 2016 (Cyclone Roanu Catastrophe)":
      selected_date = pd.to_datetime("2016-05-16")
    elif quick_pick == "May 25, 2017 (Southwest Monsoon Surge)":
      selected_date = pd.to_datetime("2017-05-25")
    elif quick_pick == "Nov 10, 2023 (Inter-Monsoon Bursts)":
      selected_date = pd.to_datetime("2023-11-10")
    else:
      selected_date = pd.to_datetime(
          st.date_input(
              "Choose Date:",
              value=pd.to_datetime("2016-05-16"),
              min_value=df["date"].min(),
              max_value=df["date"].max(),
          )
      )

  day_df = df[df["date"] == selected_date]

  critical_count = len(
      day_df[day_df["flood_category"].str.contains("Critical")]
  )
  max_rain_district = day_df.loc[day_df["rain_24h"].idxmax()]

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("Date Inspected", selected_date.strftime("%Y-%m-%d"))
  m2.metric(
      "Critical Emergency Districts",
      f"{critical_count} / 25",
      delta=f"{critical_count} Alert" if critical_count > 0 else "Normal",
      delta_color="inverse",
  )
  m3.metric(
      "Peak 24h Rain",
      f"{max_rain_district['rain_24h']:.1f} mm",
      max_rain_district["district"],
  )
  m4.metric(
      "Average Soil Saturation", f"{day_df['soil_saturation_index'].mean():.2%}"
  )

  # Plotly Map
  fig = px.scatter_geo(
      day_df,
      lat="latitude",
      lon="longitude",
      color="flood_risk_score",
      size="rain_72h",
      hover_name="district",
      hover_data={
          "latitude": False,
          "longitude": False,
          "climatic_zone": True,
          "rain_24h": ":.1f mm",
          "rain_72h": ":.1f mm",
          "soil_saturation_index": ":.3f",
          "flood_risk_score": ":.1f",
          "flood_category": True,
      },
      color_continuous_scale=[
          (0.0, "#2ecc71"),
          (0.3, "#f1c40f"),
          (0.6, "#e67e22"),
          (1.0, "#e74c3c"),
      ],
      range_color=[0, 100],
      size_max=35,
      height=650,
  )
  fig.update_geos(
      center=dict(lat=7.8731, lon=80.7718),
      projection_scale=38,
      visible=True,
      showcountries=False,
      showcoastlines=True,
      coastlinecolor="#7f8c8d",
      showland=True,
      landcolor="#1e272e",
      showocean=True,
      oceancolor="#0f141d",
      bgcolor="#0b0f19",
  )
  fig.update_layout(
      paper_bgcolor="#0b0f19",
      font_color="#ecf0f1",
      margin=dict(r=0, t=10, b=0, l=0),
  )
  st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: INTERACTIVE WHAT-IF SIMULATOR
# ==========================================
elif mode == "🧪 Interactive Flood Simulator":
  st.subheader("🧪 Real-Time Flood Scenario Simulator")
  st.markdown(
      "Simulate intense precipitation & ground saturation scenarios to"
      " calculate localized inundation probability."
  )

  col_input, col_result = st.columns([1, 1])

  with col_input:
    district_choice = st.selectbox(
        "Select Target District:", sorted(df["district"].unique()), index=4
    )
    meta = df[df["district"] == district_choice].iloc[0]
    st.caption(
        f"📍 **Province:** {meta['province']} | **Zone:**"
        f" {meta['climatic_zone']}"
    )

    sim_rain_24 = st.slider(
        "Simulated 24-Hour Rainfall (mm):",
        min_value=0.0,
        max_value=350.0,
        value=135.0,
        step=5.0,
    )
    sim_rain_72 = st.slider(
        "Past 72-Hour Cumulative Rainfall (mm):",
        min_value=float(sim_rain_24),
        max_value=600.0,
        value=max(sim_rain_24, 210.0),
        step=10.0,
    )
    sim_soil = st.slider(
        "Current Soil Moisture Saturation:",
        min_value=0.10,
        max_value=0.55,
        value=0.46,
        step=0.01,
        help=(
            "Values > 0.40 indicate near field capacity (water cannot seep in)."
        ),
    )

  # Calculation
  p = gumbel_params[district_choice]
  p_extreme_24 = gumbel_r.cdf(sim_rain_24, p["loc_24"], p["scale_24"])
  soil_factor = np.clip((sim_soil - 0.20) / (0.50 - 0.20), 0.0, 1.0)
  sat_ratio = np.clip(sim_rain_72 / (p["t5_72"] + 1e-5), 0.0, 1.5)

  raw_score = (
      (0.40 * p_extreme_24)
      + (0.35 * soil_factor * (sim_rain_24 > 15))
      + (0.25 * (sat_ratio / 1.5))
  )
  risk_score = round(float(np.clip(raw_score * 100, 0, 100)), 1)

  non_exc = np.clip(p_extreme_24, 0.01, 0.99)
  est_return_period = round(1.0 / (1.0 - non_exc), 1)

  with col_result:
    st.markdown("### 🚨 Engine Prediction Output")

    if risk_score < 30:
      color = "#2ecc71"
      badge = "🟢 NORMAL / LOW RISK"
      rec = "No danger detected. Standard drainage capacity adequate."
    elif risk_score < 60:
      color = "#f1c40f"
      badge = "🟡 ADVISORY: WATERLOGGING"
      rec = (
          "Localized surface runoff. Flash pooling on low-lying secondary"
          " roads."
      )
    elif risk_score < 80:
      color = "#e67e22"
      badge = "🟠 HIGH WARNING: RIVER INUNDATION"
      rec = (
          "River basins approaching bankfull. Activate flood pumps and prepare"
          " staging areas."
      )
    else:
      color = "#e74c3c"
      badge = "🔴 CRITICAL EMERGENCY: CATASTROPHIC OVERFLOW"
      rec = (
          "Severe inundation imminent. Issue immediate evacuation alerts for"
          " low-lying zones."
      )

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={
                "text": f"Risk Score: {badge}",
                "font": {"size": 17, "color": color},
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(46, 204, 113, 0.2)"},
                    {"range": [30, 60], "color": "rgba(241, 196, 15, 0.2)"},
                    {"range": [60, 80], "color": "rgba(230, 126, 34, 0.2)"},
                    {"range": [80, 100], "color": "rgba(231, 76, 60, 0.2)"},
                ],
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="#0b0f19",
        font_color="#ffffff",
        height=300,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.info(
        f"📊 **Statistical Return Period:** This storm volume represents an"
        f" estimated **1-in-{est_return_period}-Year Event** for"
        f" {district_choice}."
    )
    st.warning(f"🛡️ **Action Protocol:** {rec}")

# ==========================================
# TAB 3: STATISTICAL RETURN PERIODS
# ==========================================
elif mode == "📊 Statistical Return Periods":
  st.subheader("📊 Gumbel Extreme Value Analysis by Climatic Zone")
  st.markdown(
      "Return period thresholds calculated using the **Block Maxima method**"
      " across 10 years of ECMWF ERA5 reanalysis data."
  )

  summary_rows = []
  for d, p in gumbel_params.items():
    z = df[df["district"] == d]["climatic_zone"].iloc[0]
    summary_rows.append({
        "District": d,
        "Climatic Zone": z,
        "10-Year Extreme (mm)": round(p["t10_24"], 1),
        "25-Year Catastrophic (mm)": round(p["t25_24"], 1),
    })
  s_df = pd.DataFrame(summary_rows).sort_values(
      by=["Climatic Zone", "10-Year Extreme (mm)"], ascending=[True, True]
  )

  fig_b = px.bar(
      s_df,
      x="10-Year Extreme (mm)",
      y="District",
      color="Climatic Zone",
      orientation="h",
      text="10-Year Extreme (mm)",
      height=720,
      color_discrete_map={
          "Wet": "#3498db",
          "Intermediate": "#f39c12",
          "Dry": "#e74c3c",
      },
  )
  fig_b.update_layout(
      paper_bgcolor="#0b0f19",
      plot_bgcolor="#1e272e",
      font_color="#ecf0f1",
      xaxis=dict(gridcolor="#2c3e50"),
  )
  st.plotly_chart(fig_b, use_container_width=True)
