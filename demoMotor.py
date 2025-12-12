# app.py
import streamlit as st
import numpy as np
import pandas as pd
import time
from scipy import stats
from scipy.signal import welch
import joblib

st.set_page_config(page_title="Motor Fault Prediction Dashboard", layout="wide")

# ---------------------- Load pre-trained motor model ----------------------
model_bundle = joblib.load("motor_model_improved2.pkl")
model = model_bundle['model']
scaler = model_bundle['scaler']
encoder = model_bundle['label_encoder']
label_map = model_bundle.get('label_map', {})  # fallback
feat_cols = model_bundle['feature_columns']

FS = 42000  # sampling frequency

# ---------------------- Feature extraction (exact as training) ----------------------
def extract_features_exact(x):
    x = np.asarray(x).astype(float)
    f = {}
    f['mean'] = float(np.mean(x))
    f['std'] = float(np.std(x))
    f['rms'] = float(np.sqrt(np.mean(x**2)))
    f['ptp'] = float(np.ptp(x))
    f['skew'] = float(stats.skew(x))
    f['kurtosis'] = float(stats.kurtosis(x))
    f['median'] = float(np.median(x))
    f['max'] = float(np.max(x))
    f['min'] = float(np.min(x))
    f['crest'] = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-9))
    f['zcr'] = ((x[:-1] * x[1:]) < 0).sum() / (len(x) + 1e-9)
    # spectral features
    freqs, psd = welch(x, fs=FS, nperseg=min(len(x), 4096))
    idx = np.argmax(psd)
    f['dominant_freq'] = float(freqs[idx])
    f['band0'] = float(np.sum(psd[(freqs >= 0) & (freqs < 200)]))
    f['band1'] = float(np.sum(psd[(freqs >= 200) & (freqs < 1000)]))
    f['band2'] = float(np.sum(psd[(freqs >= 1000) & (freqs < 5000)]))
    f['centroid'] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
    f['spec_entropy'] = -np.sum((psd / (np.sum(psd) + 1e-12)) *
                                np.log2(psd / (np.sum(psd) + 1e-12) + 1e-12))
    f['peak_power'] = float(psd[idx])
    return f

# ---------------------- Simulate motor sensor data ----------------------
def simulate_motor_window():
    # 1-second window signals at FS
    t = np.linspace(0, 1, FS)
    # generate synthetic signals with some noise (digital twin)
    acc1 = np.sin(2*np.pi*50*t) + 0.05*np.random.randn(FS)
    acc2 = np.sin(2*np.pi*55*t) + 0.05*np.random.randn(FS)
    acc3 = np.sin(2*np.pi*60*t) + 0.05*np.random.randn(FS)
    mic  = np.sin(2*np.pi*70*t) + 0.05*np.random.randn(FS)
    temp = 40 + 2*np.random.randn(FS)
    return acc1, acc2, acc3, mic, temp

# ---------------------- Predict motor condition ----------------------
def predict_motor_condition(acc1, acc2, acc3, mic, temp):
    feat_dict = {}
    for ch_name, sig in zip(["acc1", "acc2", "acc3", "mic"], [acc1, acc2, acc3, mic]):
        fch = extract_features_exact(sig)
        for k, v in fch.items():
            feat_dict[f"{ch_name}_{k}"] = v
    feat_dict['temp_mean'] = float(np.mean(temp))
    feat_dict['temp_std'] = float(np.std(temp))

    # ensure all columns match training
    X = np.array([feat_dict[c] for c in feat_cols]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    condition = encoder.inverse_transform(pred)[0]
    return condition

# ---------------------- Streamlit UI ----------------------
st.title("⚡ Motor Fault Prediction Dashboard")

tabs = st.tabs(["Digital Twin Simulation"])

# ---------------------- Tab 1: Digital Twin Simulation ----------------------
with tabs[0]:
    st.subheader("Motor Digital Twin Simulation")
    st.markdown("Simulating motor sensor data in real-time...")

    condition_placeholder = st.empty()
    acc_placeholder = st.empty()
    temp_placeholder = st.empty()

    # live update loop
    for _ in range(100):  # 100 iterations (~simulate 100 seconds)
        acc1, acc2, acc3, mic, temp = simulate_motor_window()
        cond = predict_motor_condition(acc1, acc2, acc3, mic, temp)

        condition_placeholder.markdown(f"**Motor Condition:** 🟢 {cond}")
        acc_placeholder.line_chart({
            'acc1': acc1[:500],  # only show first 500 samples for clarity
            'acc2': acc2[:500],
            'acc3': acc3[:500],
            'mic': mic[:500]
        })
        temp_placeholder.line_chart({'temp': temp[:500]})
        time.sleep(0.5)

