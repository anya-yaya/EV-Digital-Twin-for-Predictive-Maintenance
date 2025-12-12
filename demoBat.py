import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Battery Digital Twin", layout="wide")

st.title("🔋 Battery RUL Digital Twin Simulation")

# --- Load pre-trained model and scaler ---
@st.cache_data
def load_battery_model():
    lstm_model = load_model("rul_lstm_model.keras")  # your trained LSTM model
    scaler = joblib.load("feature_scaler.joblib")       # scaler used during training
    feature_cols = scaler.feature_names_in_            # feature columns
    return lstm_model, scaler, feature_cols

model, scaler, feature_cols = load_battery_model()

SEQ_LEN = 20  # Same as LSTM training

# --- Function to simulate battery features ---
def generate_simulated_features(step):
    """Simulate one cycle's feature dictionary."""
    features = {}
    # Voltage
    v_mean = max(3.0, 4.2 - 0.01*step + np.random.normal(0,0.01))
    features["Voltage_mean"] = v_mean
    features["Voltage_std"] = np.random.normal(0.01, 0.002)
    features["Voltage_min"] = v_mean - np.random.uniform(0.05,0.1)
    features["Voltage_max"] = v_mean + np.random.uniform(0.0,0.05)
    features["Voltage_drop"] = features["Voltage_max"] - features["Voltage_min"]
    features["dV_dt_mean"] = np.random.normal(-0.01, 0.002)
    
    # Current
    c_mean = np.random.normal(1.0, 0.05)
    features["Current_mean"] = c_mean
    features["Current_std"] = np.random.normal(0.05, 0.01)
    features["Current_min"] = c_mean - np.random.uniform(0.05,0.1)
    features["Current_max"] = c_mean + np.random.uniform(0.0,0.05)
    
    # Temperature
    t_mean = np.random.normal(25, 0.5)
    features["Temperature_mean"] = t_mean
    features["Temperature_std"] = np.random.normal(0.5, 0.1)
    features["Temperature_max"] = t_mean + np.random.uniform(0.0,1.0)
    features["Temperature_rise_rate"] = np.random.normal(0.01, 0.002)
    
    # Load
    features["Current_load_mean"] = np.random.normal(0.95, 0.02)
    features["Voltage_load_mean"] = np.random.normal(3.8, 0.02)
    
    # Cycle duration
    features["cycle_duration"] = 1.0
    
    # Capacity
    features["Capacity"] = max(0.5, 1.0 - 0.005*step)
    features["Delta_Capacity"] = -0.005
    features["Normalized_Capacity"] = features["Capacity"] / 1.0
    
    # Initial state
    features["Init_Capacity"] = 1.0
    features["Init_V_Mean"] = 4.2
    features["Init_C_Mean"] = 1.0
    features["Init_Amb_Temp"] = 25
    
    return features

# --- Initialize simulation ---
st.subheader("Live Battery RUL Simulation")

placeholder = st.empty()
graph_placeholder = st.empty()

# Buffer to store last SEQ_LEN cycles for LSTM input
feature_buffer = []

num_cycles = 50  # total simulation cycles

for step in range(num_cycles):
    feat_dict = generate_simulated_features(step)
    feature_buffer.append(feat_dict)
    
    # Keep only last SEQ_LEN cycles
    if len(feature_buffer) > SEQ_LEN:
        feature_buffer = feature_buffer[-SEQ_LEN:]
    
    # Prepare DataFrame for scaler/model
    df_feat = pd.DataFrame([feat_dict])[feature_cols]
    if len(feature_buffer) >= SEQ_LEN:
        # Create sequence for LSTM
        X_seq = pd.DataFrame(feature_buffer)[feature_cols].values
        X_seq_scaled = scaler.transform(X_seq).reshape(1, SEQ_LEN, len(feature_cols))
        rul_pred = model.predict(X_seq_scaled, verbose=0)[0,0]
    else:
        rul_pred = None
    
    # Display metrics
    with placeholder.container():
        st.metric(label="Current Cycle", value=f"{step+1}/{num_cycles}")
        st.metric(label="Predicted RUL", value=f"{rul_pred:.2f}" if rul_pred else "Waiting...")
        st.dataframe(pd.DataFrame(feature_buffer).tail(1))
    
    # Plot RUL evolution
    if step == 0:
        rul_history = []
    rul_history.append(rul_pred if rul_pred else np.nan)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(rul_history, marker='o')
    ax.set_title("Predicted RUL over Time")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.grid(True)
    graph_placeholder.pyplot(fig)
    
    time.sleep(0.2)  # simulate real-time update
