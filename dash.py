import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib
from pathlib import Path

st.set_page_config(page_title="EV Digital Twin", layout="wide")

# ---------------------------------------------------
# LOAD MODELS (BEST-EFFORT)
# ---------------------------------------------------
MODEL_DIR = Path(".")

def try_load(path, loader):
    try:
        if path.exists():
            return loader(path)
    except:
        return None
    return None

battery_model = try_load(MODEL_DIR / "rul_lstm_model.keras", lambda p: None)   # Battery model likely needs seq input, so disabled
battery_scaler = try_load(MODEL_DIR / "feature_scaler.joblib", joblib.load)

motor_pack = try_load(MODEL_DIR / "motor_model_improved2.pkl", joblib.load)
motor_model = None
if isinstance(motor_pack, dict):
    motor_model = motor_pack.get("model")

tire_model = try_load(MODEL_DIR / "tire_maintenance_model.joblib", joblib.load)
tire_scaler = try_load(MODEL_DIR / "tire_maintenance_scaler.joblib", joblib.load)
tire_features = try_load(MODEL_DIR / "tire_features.joblib", joblib.load)

# ---------------------------------------------------
# SYNTHETIC DATA GENERATORS
# ---------------------------------------------------
def next_battery(state):
    state["voltage"] += np.random.normal(0, 0.01)
    state["current"] = max(0.1, state["current"] + np.random.normal(0, 0.05))
    state["temperature"] += np.random.normal(0, 0.02)
    state["soh"] = max(0, state["soh"] - 0.0004)
    state["rul"] = state["soh"] * 1000
    return state

def next_motor(state):
    state["vibration"] += np.random.normal(0, 0.02)
    state["temp"] += np.random.normal(0, 0.05)
    state["load"] = np.clip(state["load"] + np.random.normal(0, .03), 0, 1)
    state["health"] = max(0, 1 - state["vibration"] * 0.9)
    return state

def next_tire(state):
    inc = np.random.uniform(0.2, 1.1)
    state["distance"] += inc
    state["pressure"] += np.random.normal(0, 0.01)
    state["temp"] += np.random.normal(0, 0.02)
    state["rul_km"] = max(0, 50000 - state["distance"])
    return state

# ---------------------------------------------------
# UI / DASHBOARD STRUCTURE
# ---------------------------------------------------
st.title("⚡ EV Digital Twin — Predictive Maintenance Dashboard")
st.caption("Demonstration of multi-component health monitoring for Battery, Motor, & Tires.")

tabs = st.tabs(["Home", "Battery", "Motor", "Tire", "About"])

# ---------------------------------------------------
# HOME TAB
# ---------------------------------------------------
with tabs[0]:
    st.header("Overall EV Health")
    start = st.button("Start Live Simulation")
    stop = st.button("Stop")

    if "run" not in st.session_state:
        st.session_state.run = False

    if start: st.session_state.run = True
    if stop: st.session_state.run = False

    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=["overall","battery","motor","tire"])

    if st.session_state.run:
        battery = {"soh": 0.98}
        motor = {"health": 0.99}
        tire = {"rul_km": 45000}

        placeholder = st.empty()

        for _ in range(200):
            battery["soh"] -= 0.0004
            motor["health"] -= 0.0003
            tire["rul_km"] -= np.random.uniform(10, 40)

            overall = 0.5*battery["soh"] + 0.3*motor["health"] + 0.2*(tire["rul_km"]/50000)

            st.session_state.history.loc[len(st.session_state.history)] = [
                overall, battery["soh"], motor["health"], tire["rul_km"]/50000
            ]

            with placeholder.container():
                st.subheader("Overall EV Health Trend")
                st.line_chart(st.session_state.history)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Overall", f"{overall:.2f}")
                col2.metric("Battery SoH", f"{battery['soh']:.2f}")
                col3.metric("Motor Health", f"{motor['health']:.2f}")
                col4.metric("Tire life", f"{tire['rul_km']:.0f} km")

            if overall < 0.65:
                st.error("⚠ ALERT: Vehicle requires maintenance soon")
            time.sleep(0.15)
            if not st.session_state.run:
                break

# ---------------------------------------------------
# BATTERY TAB
# ---------------------------------------------------
with tabs[1]:
    st.header("🔋 Battery Digital Twin")
    st.write("Live synthetic battery simulation with SoH & RUL graph.")

    if "b_hist" not in st.session_state:
        st.session_state.b_hist = pd.DataFrame(columns=["soh","rul"])

    start_b = st.button("Start Battery Simulation")
    stop_b = st.button("Stop Battery")
    if start_b: st.session_state.batt = True
    if stop_b: st.session_state.batt = False

    battery = {"voltage":3.6, "current":1.0, "temperature":28, "soh":0.98, "rul":900}
    ph = st.empty()

    if st.session_state.get("batt", False):
        for _ in range(200):
            battery = next_battery(battery)
            st.session_state.b_hist.loc[len(st.session_state.b_hist)] = [battery["soh"], battery["rul"]]

            with ph.container():
                st.line_chart(st.session_state.b_hist)
                st.metric("SOH", f"{battery['soh']:.2f}")
                st.metric("Predicted RUL", f"{battery['rul']:.0f}")
            time.sleep(0.15)
            if not st.session_state.batt: break

# ---------------------------------------------------
# MOTOR TAB
# ---------------------------------------------------
with tabs[2]:
    st.header("⚙ Motor Digital Twin")

    if "m_hist" not in st.session_state:
        st.session_state.m_hist = pd.DataFrame(columns=["vibration","health"])

    start_m = st.button("Start Motor Simulation")
    stop_m = st.button("Stop Motor")
    if start_m: st.session_state.motor = True
    if stop_m: st.session_state.motor = False

    motor = {"vibration":0.03, "temp":38, "load":0.4, "health":0.98}
    phm = st.empty()

    if st.session_state.get("motor", False):
        for _ in range(200):
            motor = next_motor(motor)
            st.session_state.m_hist.loc[len(st.session_state.m_hist)] = [motor["vibration"], motor["health"]]

            with phm.container():
                st.line_chart(st.session_state.m_hist)
                st.metric("Motor Health", f"{motor['health']:.2f}")
            time.sleep(0.15)
            if not st.session_state.motor: break

# ---------------------------------------------------
# TIRE TAB
# ---------------------------------------------------
with tabs[3]:
    st.header("🚗 Tire Digital Twin")

    if "t_hist" not in st.session_state:
        st.session_state.t_hist = pd.DataFrame(columns=["pressure","temp","rul"])

    start_t = st.button("Start Tire Sim")
    stop_t = st.button("Stop tire")
    if start_t: st.session_state.tire = True
    if stop_t: st.session_state.tire = False

    tire = {"distance":1000, "pressure":2.3, "temp":29, "rul_km":45000}
    pht = st.empty()

    if st.session_state.get("tire", False):
        for _ in range(200):
            tire = next_tire(tire)
            st.session_state.t_hist.loc[len(st.session_state.t_hist)] = [tire["pressure"], tire["temp"], tire["rul_km"]]

            with pht.container():
                st.line_chart(st.session_state.t_hist)
                st.metric("RUL (km)", f"{tire['rul_km']:.0f}")
            time.sleep(.15)
            if not st.session_state.tire: break

# ---------------------------------------------------
# ABOUT TAB
# ---------------------------------------------------
with tabs[4]:
    st.header("📦 About Dataset & Models")
    st.markdown("""
   This section provides an overview of the data used to train the predictive models for the EV Car Predictive Maintenance.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔋 Battery Dataset")
        st.write("""
        * **Dataset name:** NASA Li ion Battery Degradation Dataset.   
        * **Algorithm Used:** LSTM Neural Network      
        * **Purpose:** Battery Remaining Useful Life (RUL) Prediction.
        * **Key Features:** Discharge Cycles, Voltage, Current, Temperature, Capacity
        * **Target:** Remaining Cycles until End-of-Life (EOL).
        * **Model:** `rul_lstm_model.keras`, `feature_scaler.joblib`
        * **Evaluation Metrics:** R2 Score: 0.9904, MSE: 18.8628, RMSE: 4.3431, MAE: 2.9178
        """)
    
    with col2:
        st.subheader("⚡ Motor Dataset")
        st.write("""
        * **Dataset name:** University of Ottawa Electric Motor Dataset- Vibration and Acoustic Faults under Constant and Variable Speed Conditions (UOEMD-VAFCVS)
        * **Algorithm Used:** Random Forest Classifier, GridSearchCV and Stratified cross-validation
        * **Purpose:** Motor Fault Classification.
        * **Key Features:** Motor Temperature, Stator Temperature, Torque, Speed, Motor Current/Voltage, Vibration readings.
        * **Target:** Fault Class ('Healthy Motor', 'Stator Winding Fault', 'Bowed Rotor', 'Faulty Bearing', 'Broken Rotor Bars', 'Rotor Misalignment', 'Rotor Unbalance', 'Voltage Un balance' ).
        * **Model:** `motor_model_improved2.pkl`
        * **Evaluation Metrics:** Accuracy: 0.9453125, F1-Score, Precision, recall
        """)
        
    with col3:
        st.subheader("⚫ Tire Dataset")
        st.write("""
        * **Dataset name:** EV_Predictive_Maintenance_Dataset_15min 
        * **Algorithm Used:** Random Forest Regressor
        * **Purpose:** Tire Maintenance Prediction.
        * **Key Features:** Tire Pressure, Tire Temperature, Distance travelled, Load, Vehicle Speed, Mileage.
        * **Target:** Remaining distance (in km) until replacement is required.
        * **Model:** `tire_maintenance_model.joblib`, `tire_features.joblib`, `tire_maintenance_scaler.joblib`
        * **Evaluation Metrics:** MSE: 0.00km, R2 Score: 1.00
        """)

    st.info("This dashboard demonstrates real-time Digital Twin simulation for EV predictive maintenance.")
