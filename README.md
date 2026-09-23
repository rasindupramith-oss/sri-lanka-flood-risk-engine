# 🌧️ LankaFlood-AI: National Extreme Weather & Flood Risk Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sri-lanka-flood-risk-engine.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Data Source](https://img.shields.io/badge/Data-ECMWF%20ERA5%20Reanalysis-orange?style=flat)](https://open-meteo.com/en/docs/historical-weather-api)
[![Methodology](https://img.shields.io/badge/Theory-Extreme%20Value%20Theory%20(EVT)-blueviolet?style=flat)](https://en.wikipedia.org/wiki/Extreme_value_theory)
[![Institution](https://img.shields.io/badge/Affiliation-University%20of%20Colombo-gold?style=flat)](https://science.cmb.ac.lk/)

> **A calibrated spatial-temporal decision-support engine applying Extreme Value Theory (GEV/Gumbel) and antecedent soil saturation physics across all 25 districts of Sri Lanka (2015–2024).**

🔗 **Live Interactive Application:** [sri-lanka-flood-risk-engine.streamlit.app](https://sri-lanka-flood-risk-engine.streamlit.app/)

---

## 📌 Executive Summary

Sri Lanka frequently experiences severe monsoon flash floods, river overflows (Kelani, Kalu, Gin, Mahaweli), and localized inundations that cause massive socio-economic disruptions. Traditional early warnings often evaluate rainfall in isolation—ignoring the critical physical fact that **100 mm of rain on dry ground infiltrates harmlessly, whereas the same rain on saturated soil triggers catastrophic surface runoff.**

**LankaFlood-AI** addresses this by coupling:
1. **Extreme Value Theory (EVT):** Calibrating localized precipitation return periods ($T$-year events) using 10 years of unbroken ECMWF ERA5 reanalysis data.
2. **Antecedent Hydrology:** Modeling multi-day cumulative precipitation ($R_{72}$) and volumetric soil moisture saturation.
3. **Climatic Zone Stratification:** Tailoring baseline risk thresholds independently across Sri Lanka's **Wet, Intermediate, and Dry Zones**.

---

## 🚀 Key Features

* **🗺️ Island-Wide Geospatial Risk Mapping:** Interactive monitoring across all 25 districts with dynamic bubble scaling (72h accumulation) and risk color classification (Green to Red).
* **🧪 Real-Time "What-If" Scenario Simulator:** Adjust sliders for 24h rainfall, 72h accumulation, and soil moisture saturation to compute real-time risk scores (0–100%) and localized return period estimates.
* **🧠 Model Explainability (Component Breakdown):** Transparent attribution revealing the exact percentage weights driving the flood score (Tail Rain vs. Soil Moisture vs. Antecedent Runoff).
* **📊 Gumbel Return Period Tail Analysis:** Comparative benchmarks for 10-Year Extreme and 25-Year Catastrophic storms across all districts.
* **🚨 Real-World Historical Disaster Validation:** Autonomous ground-truth validation against landmark historical disasters (including May 2016 Cyclone Roanu).

---

## 🔬 Mathematical & Statistical Formulation

### 1. Extreme Value Modeling (Block Maxima Method)
Floods are tail-event phenomena. The annual maximum daily precipitation for each district $i$ is modeled via the **Gumbel Extreme Value Distribution** (GEV Type I):

$$F(x; \mu, \beta) = \exp\left(-\exp\left(-\frac{x - \mu}{\beta}\right)\right)$$

Where:
* $\mu$ = Location parameter (characteristic seasonal peak rainfall).
* $\beta$ = Scale parameter (dispersion and tail weight of extreme bursts).

The Return Period $T$ for a precipitation threshold $x_T$ is computed analytically via the quantile function (PPF):

$$x_T = \mu - \beta \ln\left(-\ln\left(1 - \frac{1}{T}\right)\right)$$

### 2. Composite Hydrological Risk Index
The unified Flood Risk Score ($S \in [0, 100]$) combines probability of rainfall exceedance with physical soil saturation:

$$S = \text{clip}\Big( 0.40 \cdot F(R_{24}) + 0.35 \cdot \Phi(\theta_{\text{soil}}) \cdot \mathbb{I}(R_{24} > 15) + 0.25 \cdot \min\left( \frac{R_{72}}{x_{T=5}}, 1.5 \right), 0, 1 \Big) \times 100$$

Where:
* $F(R_{24})$: Gumbel CDF probability for 24-hour rainfall.
* $\Phi(\theta_{\text{soil}})$: Soil moisture saturation factor normalized relative to field capacity ($\theta \approx 0.40\text{--}0.50 \text{ m}^3/\text{m}^3$).
* $R_{72} / x_{T=5}$: Ratio of 3-day antecedent rainfall against the district's 5-year extreme storm threshold.

---

## 🎯 Ground Truth Empirical Validation

When evaluated across the entire 10-year dataset (91,325 district-days), the engine **autonomously flagged May 15–16, 2016 (Cyclone Roanu)** as the most catastrophic flood event of the decade without any external supervision:

| Date | District | Climatic Zone | 24h Rain | 72h Rain | Soil Moisture | Risk Score | Official Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **2016-05-15** | **Gampaha** | Wet | 231.4 mm | 311.4 mm | 0.496 m³/m³ | **99.3%** | 🔴 Critical Emergency |
| **2016-05-15** | **Kalutara** | Wet | 195.9 mm | 300.3 mm | 0.495 m³/m³ | **99.3%** | 🔴 Critical Emergency |
| **2016-05-16** | **Kurunegala** | Intermediate | 227.5 mm | 514.4 mm | 0.492 m³/m³ | **98.7%** | 🔴 Critical Emergency |
| **2016-05-15** | **Kegalle** | Wet | 259.5 mm | 330.6 mm | 0.494 m³/m³ | **97.6%** | 🔴 Critical Emergency |

*Physical Validation:* Soil saturation across all affected districts reached $\approx 0.495$ (complete physical saturation), confirming that near 100% of incoming precipitation converted directly into lethal surface runoff.

---

## 📊 Dataset Specifications

* **Coverage:** All 25 Administrative Districts of Sri Lanka.
* **Temporal Span:** January 1, 2015 – December 31, 2024 (10 Years / 3,653 Days).
* **Granularity:** Daily observations ($3,653 \times 25 = 91,325$ records, zero missing data).
* **Variables:** Total Precipitation, Liquid Rain, Topsoil Moisture (0–7 cm), Subsurface Soil Moisture (7–28 cm), Max Temperature, Wind Speed.
* **Source:** ECMWF ERA5 Atmospheric Reanalysis via Open-Meteo Archive API.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.10+
* **Statistical Modeling:** `SciPy` (`scipy.stats.gumbel_r`), `NumPy`
* **Data Engineering:** `Pandas` (rolling multi-day window aggregations)
* **Geospatial & Visualizations:** `Plotly Express`, `Plotly Graph Objects`
* **Web Deployment:** `Streamlit Community Cloud`

---

## 💻 Local Installation & Setup

To run LankaFlood-AI locally on your machine:

```bash
# 1. Clone the repository
git clone https://github.com/rasindupramith-oss/lanka-flood-risk-engine.git
cd lanka-flood-risk-engine

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit web application
streamlit run app.py
```

---

👨‍💻 Author: Rasindu Pramith | Undergraduate, BSc (Hons) in Applied Statistics | Faculty of Science, University of Colombo, Sri Lanka

🌐 Portfolio: http://rasindupramith-oss.github.io

💼 LinkedIn: http://linkedin.com/in/rasindu-pramith

💻 GitHub: http://github.com/rasindupramith-oss

📄 License : This project is open-source and available under the MIT License.
