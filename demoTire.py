import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# ---------------------- Load Model + Artifacts -------------------------
MODEL_FILE = "tire_maintenance_model.joblib"
SCALER_FILE = "tire_maintenance_scaler.joblib"
FEATURE_FILE = "tire_features.joblib"

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
FEATURE_COLS = joblib.load(FEATURE_FILE)

st.set_page_config(page_title="Tire RUL Digital Twin", layout="wide")

# ---------------------- Session State Initialization -------------------
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

if "live_time" not in st.session_state:
    st.session_state.live_time = []

if "live_rul" not in st.session_state:
    st.session_state.live_rul = []

# ---------------------- UI Layout ----------------------
st.title("🚗 Tire Maintenance - Digital Twin Dashboard")
st.write("Predict and simulate Tire Remaining Useful Life with live telemetry tracking")

col1, col2 = st.columns(2)

with col1:
    Distance_Traveled = st.number_input("Distance Traveled (km)", 0, 50000, 10000)
    Tire_Pressure = st.number_input("Tire Pressure (PSI)", 20.0, 50.0, 32.0)
    Tire_Temperature = st.number_input("Tire Temperature (°C)", 10.0, 120.0, 45.0)
    Driving_Speed = st.number_input("Driving Speed (km/h)", 0, 200, 60)

with col2:
    Route_Roughness = st.number_input("Road Roughness (0-1 scale)", 0.0, 1.0, 0.3)
    Load_Weight = st.number_input("Vehicle Load Weight (kg)", 0, 3000, 250)
    Ambient_Temperature = st.number_input("Ambient Temperature (°C)", -10.0, 60.0, 25.0)

# ----------------------- Manual Prediction -----------------------
if st.button("🔍 Predict RUL"):
    input_data = np.array([[Distance_Traveled, Tire_Pressure, Tire_Temperature, Driving_Speed,
                            Route_Roughness, Load_Weight, Ambient_Temperature]])

    scaled = scaler.transform(input_data)
    predicted_rul = model.predict(scaled)[0]

    st.session_state.prediction_log.append({
        "Distance": Distance_Traveled,
        "Speed": Driving_Speed,
        "Pressure": Tire_Pressure,
        "RUL_KM": round(predicted_rul, 2)
    })

    st.success(f"### 🛞 Predicted Tire RUL: **{predicted_rul:.2f} km**")

# ----------------------- Simulation -----------------------
st.subheader("📡 Live Telemetry Simulation")
run_sim = st.toggle("Enable Live Simulation")

placeholder_chart = st.empty()
placeholder_metric = st.empty()

if run_sim:
    for t in range(30):  # 30 real-time cycles
        random_values = np.array([[np.random.randint(1000, 45000),
                                   np.random.uniform(28, 38),
                                   np.random.uniform(30, 75),
                                   np.random.randint(20, 120),
                                   np.random.uniform(0.1, 0.9),
                                   np.random.randint(100, 1200),
                                   np.random.uniform(15, 45)]])

        scaled = scaler.transform(random_values)
        rul_pred = model.predict(scaled)[0]

        # Update session state time series
        st.session_state.live_time.append(len(st.session_state.live_time))
        st.session_state.live_rul.append(rul_pred)

        placeholder_metric.metric("Live RUL Prediction (km)", f"{rul_pred:.2f}")

        # ---------------- Live Plotly Line Chart -----------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.live_time,
            y=st.session_state.live_rul,
            mode="lines+markers",
            name="RUL"
        ))

        fig.update_layout(
            title="📈 Live RUL Trend During Simulation",
            xaxis_title="Time Step",
            yaxis_title="RUL (km)",
            template="plotly_white",
            height=350
        )

        placeholder_chart.plotly_chart(fig, use_container_width=True)

        time.sleep(0.5)

# ---------------- Logs Table --------------------
st.subheader("📄 Prediction Logs")
df_log = pd.DataFrame(st.session_state.prediction_log)
st.dataframe(df_log, use_container_width=True)
