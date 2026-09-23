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

# 2. Comprehensive High-Contrast Dark Theme CSS
st.markdown(
    """
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
    }

    /* FORCED CRISP TEXT ON ALL WIDGET LABELS (Dropdowns, Sliders, Inputs) */
    label[data-testid="stWidgetLabel"] p, 
    label[data-testid="stWidgetLabel"] span,
    .stSelectbox label p,
    .stSlider label p,
    .stDateInput label p {
        color: #f0f6fc !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
    }

    /* SIDEBAR TEXT & BULLET POINTS FIX */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] li {
        margin-bottom: 6px !important;
        font-size: 14px !important;
    }

    /* NAVIGATION RADIO BUTTONS: HIGH CONTRAST & CARD LOOK */
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        margin-bottom: 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #58a6ff !important;
        background-color: #30363d !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        color: #ffffff !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* KPI CARDS ENHANCED CONTRAST */
    .kpi-card {
        background: linear-gradient(135deg, #1c2128 0%, #252c37 100%);
        border: 1px solid #384252;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .kpi-label {
        color: #94a3b8 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 12px !important;
        font-weight: 600;
        margin-top: 8px;
    }
    .kpi-alert { color: #ff6b6b !important; }
    .kpi-good  { color: #4cd964 !important; }
    .kpi-info  { color: #58a6ff !important; }

    /* AUTHOR BADGE IN SIDEBAR */
    .sidebar-author {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 20px;
    }
    .sys-badge {
        display: inline-block;
        background-color: #21262d;
        color: #79c0ff;
        border: 1px solid #388bfd44;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 3. Data Loading & GEV Calibration
@st.cache_data
def load_data():
  df = pd.read_csv("sri_lanka_flood_risk_modeled.csv")
  df["date"] = pd.to_datetime(df["date"])
  return df


try:
  df = load_data()
except Exception:
  st.error("⚠️ Dataset not found. Please ensure CSV is present.")
  st.stop()


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

# ==========================================
# SIDEBAR: COMMAND CENTER & NAVIGATION
# ==========================================
with st.sidebar:
  st.markdown(
      """
        <div class="sidebar-author">
            <h4 style="margin:0 0 4px 0; color:#ffffff; font-size:17px;">Rasindu Pramith</h4>
            <p style="margin:0 0 10px 0; font-size:12px; color:#8b949e; line-height:1.4;">
                BSc (Hons) in Applied Statistics<br>
                Faculty of Science • University of Colombo
            </p>
            <div>
                <span class="sys-badge">🎓 UoC Stats</span>
                <span class="sys-badge">⚡ GEV Theory</span>
                <span class="sys-badge">🛰️ ERA5 Reanalysis</span>
            </div>
        </div>
    """,
      unsafe_allow_html=True,
  )

  st.markdown(
      "<h4 style='color:#ffffff; margin-bottom:8px;'>🧭 Control Center</h4>",
      unsafe_allow_html=True,
  )
  mode = st.radio(
      label="Navigation",
      options=[
          "🗺️ National Risk Map (Historical)",
          "🧪 Interactive Flood Simulator",
          "📊 Statistical Return Periods",
      ],
      label_visibility="collapsed",
  )

  st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
  st.markdown(
      "<h4 style='color:#ffffff; margin-bottom:8px;'>📡 System Telemetry</h4>",
      unsafe_allow_html=True,
  )
  st.markdown("""
        * **Monitored Nodes:** `25 Districts`
        * **Temporal Depth:** `10 Years (2015–2024)`
        * **Total Records:** `91,325 Days`
        * **Physical Benchmark:** `Cyclone Roanu Disaster`
    """)

  st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
  col_btn1, col_btn2 = st.columns(2)
  with col_btn1:
    st.link_button(
        "💼 LinkedIn",
        "https://www.linkedin.com/in/rasindu-pramith/",
        use_container_width=True,
    )
  with col_btn2:
    st.link_button(
        "💻 GitHub",
        "https://github.com/rasindupramith-oss",
        use_container_width=True,
    )

# ==========================================
# TAB 1: NATIONAL RISK MAP (HISTORICAL VIEWER)
# ==========================================
if mode == "🗺️ National Risk Map (Historical)":
  st.markdown(
      "<h1 style='color:#ffffff; margin-bottom:2px;'>🌧️ LankaFlood-AI:"
      " National Risk Map</h1>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<p style='color:#94a3b8; font-size:15px; margin-bottom:20px;'>Spatial-temporal"
      " inundation probability modeled via Generalized Extreme Value theory"
      " across Sri Lanka.</p>",
      unsafe_allow_html=True,
  )

  c_sel1, c_sel2 = st.columns([1.6, 1])
  with c_sel1:
    quick_pick = st.selectbox(
        "⚡ Select Disaster Benchmark Event:",
        [
            "May 16, 2016 (Cyclone Roanu Catastrophe)",
            "May 25, 2017 (Southwest Monsoon Surge)",
            "Nov 10, 2023 (Inter-Monsoon Bursts)",
            "Custom Date",
        ],
    )
    if "2016" in quick_pick:
      selected_date = pd.to_datetime("2016-05-16")
    elif "2017" in quick_pick:
      selected_date = pd.to_datetime("2017-05-25")
    elif "2023" in quick_pick:
      selected_date = pd.to_datetime("2023-11-10")
    else:
      selected_date = pd.to_datetime(
          st.date_input(
              "Custom Date:",
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
  mean_soil = day_df["soil_saturation_index"].mean()

  st.write("")
  # High-Contrast KPI Cards
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.markdown(
        f"""
            <div class="kpi-card">
                <div class="kpi-label">Date Inspected</div>
                <div class="kpi-value">{selected_date.strftime('%Y-%m-%d')}</div>
                <div class="kpi-sub kpi-info">📅 ECMWF ERA5 Reanalysis</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
  with col2:
    alert_color = "kpi-alert" if critical_count > 0 else "kpi-good"
    alert_text = (
        f"🚨 {critical_count} Districts in Critical State"
        if critical_count > 0
        else "🟢 All 25 Districts Normal"
    )
    st.markdown(
        f"""
            <div class="kpi-card">
                <div class="kpi-label">Critical Overflow Alerts</div>
                <div class="kpi-value">{critical_count} <span style="font-size:16px; color:#8b949e;">/ 25</span></div>
                <div class="kpi-sub {alert_color}">{alert_text}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
  with col3:
    st.markdown(
        f"""
            <div class="kpi-card">
                <div class="kpi-label">Peak 24h Rainfall</div>
                <div class="kpi-value">{max_rain_district['rain_24h']:.1f} <span style="font-size:16px; color:#8b949e;">mm</span></div>
                <div class="kpi-sub kpi-info">📍 {max_rain_district['district']} ({max_rain_district['province']})</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
  with col4:
    sat_status = "kpi-alert" if mean_soil > 0.40 else "kpi-good"
    sat_note = (
        "⚠️ High Runoff Potential"
        if mean_soil > 0.40
        else "🟢 Ground Absorptive"
    )
    st.markdown(
        f"""
            <div class="kpi-card">
                <div class="kpi-label">Island Mean Soil Saturation</div>
                <div class="kpi-value">{mean_soil:.1%}</div>
                <div class="kpi-sub {sat_status}">{sat_note}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

  st.write("")

  # Mapbox / Geo Scatter
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
          (0.0, "#3fb950"),
          (0.3, "#d29922"),
          (0.6, "#db6d28"),
          (1.0, "#f85149"),
      ],
      range_color=[0, 100],
      size_max=36,
      height=640,
  )
  fig.update_geos(
      center=dict(lat=7.8731, lon=80.7718),
      projection_scale=38,
      visible=True,
      showcountries=False,
      showcoastlines=True,
      coastlinecolor="#484f58",
      showland=True,
      landcolor="#161b22",
      showocean=True,
      oceancolor="#090d16",
      bgcolor="#0d1117",
  )
  fig.update_layout(
      paper_bgcolor="#0d1117",
      font_color="#f0f6fc",
      margin=dict(r=0, t=10, b=0, l=0),
  )
  st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: INTERACTIVE WHAT-IF SIMULATOR
# ==========================================
elif mode == "🧪 Interactive Flood Simulator":
  st.markdown(
      "<h1 style='color:#ffffff; margin-bottom:2px;'>🧪 Real-Time Flood"
      " Scenario Simulator</h1>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<p style='color:#94a3b8; font-size:15px; margin-bottom:20px;'>Simulate"
      " precipitation intensity & ground saturation to compute localized"
      " inundation probability in real time.</p>",
      unsafe_allow_html=True,
  )

  col_input, col_result = st.columns([1, 1.1])

  with col_input:
    st.markdown(
        "<h4 style='color:#ffffff; border-bottom:1px solid #30363d;"
        " padding-bottom:6px;'>⚙️ Input Scenario</h4>",
        unsafe_allow_html=True,
    )
    district_choice = st.selectbox(
        "Target District:", sorted(df["district"].unique()), index=4
    )
    meta = df[df["district"] == district_choice].iloc[0]
    st.markdown(
        f"<span style='color:#79c0ff; font-weight:600;'>Province:"
        f" {meta['province']}</span> | <span style='color:#e3b341;"
        f" font-weight:600;'>Climatic Zone: {meta['climatic_zone']}</span>",
        unsafe_allow_html=True,
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
        "Current Soil Saturation Index:",
        min_value=0.10,
        max_value=0.55,
        value=0.46,
        step=0.01,
        help="Soil field capacity threshold is ~0.40. Above this, rain turns directly to runoff.",
    )

  # Calculations
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
    st.markdown(
        "<h4 style='color:#ffffff; border-bottom:1px solid #30363d;"
        " padding-bottom:6px;'>🚨 Predictive Output</h4>",
        unsafe_allow_html=True,
    )

    if risk_score < 30:
      color = "#3fb950"
      badge = "🟢 NORMAL / LOW RISK"
      rec = "Standard municipal drainage adequate. No public warnings required."
    elif risk_score < 60:
      color = "#d29922"
      badge = "🟡 ADVISORY: WATERLOGGING"
      rec = (
          "Localized pooling on low-lying road networks. Advise transit"
          " caution."
      )
    elif risk_score < 80:
      color = "#db6d28"
      badge = "🟠 HIGH WARNING: RIVER INUNDATION"
      rec = (
          "Basins approaching bankfull. Stage emergency pumps and prepare"
          " low-lying sectors."
      )
    else:
      color = "#f85149"
      badge = "🔴 CRITICAL EMERGENCY: CATASTROPHIC OVERFLOW"
      rec = (
          "Severe flooding imminent. Issue immediate evacuation alerts for"
          " floodplains."
      )

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": f"{badge}", "font": {"size": 15, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(63, 185, 80, 0.15)"},
                    {"range": [30, 60], "color": "rgba(210, 153, 34, 0.15)"},
                    {"range": [60, 80], "color": "rgba(219, 109, 40, 0.15)"},
                    {"range": [80, 100], "color": "rgba(248, 81, 73, 0.15)"},
                ],
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="#0d1117",
        font_color="#ffffff",
        height=270,
        margin=dict(t=25, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(f"""
        <div style="background:#161b22; border-left:4px solid #58a6ff; border-radius:8px; padding:12px 16px; margin-bottom:12px; border:1px solid #30363d; border-left:4px solid #58a6ff;">
            <b style="color:#58a6ff; font-size:14px;">📊 Statistical Return Period:</b><br>
            <span style="color:#f0f6fc; font-size:13px;">A <b>{sim_rain_24:.1f} mm</b> daily storm represents an estimated <b>1-in-{est_return_period}-Year Extreme Event</b> for {district_choice}.</span>
        </div>
        <div style="background:#161b22; border-left:4px solid {color}; border-radius:8px; padding:12px 16px; border:1px solid #30363d; border-left:4px solid {color};">
            <b style="color:{color}; font-size:14px;">🛡️ Action Protocol:</b><br>
            <span style="color:#f0f6fc; font-size:13px;">{rec}</span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: STATISTICAL RETURN PERIODS
# ==========================================
elif mode == "📊 Statistical Return Periods":
  st.markdown(
      "<h1 style='color:#ffffff; margin-bottom:2px;'>📊 Gumbel Extreme Value"
      " Analysis</h1>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<p style='color:#94a3b8; font-size:15px; margin-bottom:20px;'>Block"
      " Maxima return period thresholds across Sri Lanka's Wet, Dry, and"
      " Intermediate Climatic Zones.</p>",
      unsafe_allow_html=True,
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
      height=740,
      color_discrete_map={
          "Wet": "#58a6ff",
          "Intermediate": "#d29922",
          "Dry": "#f85149",
      },
  )
  fig_b.update_layout(
      paper_bgcolor="#0d1117",
      plot_bgcolor="#161b22",
      font_color="#f0f6fc",
      xaxis=dict(gridcolor="#30363d"),
  )
  st.plotly_chart(fig_b, use_container_width=True)
