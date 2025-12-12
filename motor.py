# improve motor model.py
"""
Improved motor model training:
- windowing (1s windows at 42kHz => 42000 samples/window)
- richer features (time + freq + crest + zcr + spectral entropy)
- stratified k-fold CV + GridSearch for RandomForest
- optional simple augmentation
- saves model bundle to motor_model_improved2.pkl
Usage:
    python motor.py --data_root /path/to/CSV_Data_Files --out motor_model_improved.pkl --do_augment
"""
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

FS = 42000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
NFFT = 4096

# mapping from short code to human label
LABEL_MAP = {
    "H_H": "Healthy",
    "B_R": "Bowed_Rotor",
    "F_B": "Faulty_Bearings",
    "K_A": "Broken_Rotor_Bars",
    "R_M": "Rotor_Misalignment",
    "R_U": "Rotor_Unbalance",
    "S_W": "Stator_Winding_Fault",
    "V_U": "Voltage_Unbalance"
}

# ---------------- feature helpers ----------------
def rms(x): return np.sqrt(np.mean(x**2))
def crest_factor(x): return np.max(np.abs(x)) / (rms(x) + 1e-9)
def zero_crossing_rate(x):
    return ((x[:-1] * x[1:]) < 0).sum() / (len(x) + 1e-9)
def spectral_entropy(psd):
    psd = np.asarray(psd)
    psd = psd / (psd.sum() + 1e-12)
    ent = -np.sum(psd * np.log2(psd + 1e-12))
    return float(ent)

def spec_features(x, fs=FS):
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), NFFT))
    idx = np.argmax(psd)
    domf = float(freqs[idx])
    e0 = float(np.sum(psd[(freqs >= 0) & (freqs < 200)]))
    e1 = float(np.sum(psd[(freqs >= 200) & (freqs < 1000)]))
    e2 = float(np.sum(psd[(freqs >= 1000) & (freqs < 5000)]))
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
    ent = spectral_entropy(psd)
    peak_power = float(psd[idx])
    return dict(dominant_freq=domf, band0=e0, band1=e1, band2=e2, centroid=centroid, spec_entropy=ent, peak_power=peak_power)

def features_from_signal(x):
    x = np.asarray(x).astype(float)
    f = {}
    f['mean'] = float(np.mean(x))
    f['std'] = float(np.std(x))
    f['rms'] = float(rms(x))
    f['ptp'] = float(np.ptp(x))
    f['skew'] = float(stats.skew(x))
    f['kurtosis'] = float(stats.kurtosis(x))
    f['median'] = float(np.median(x))
    f['max'] = float(np.max(x))
    f['min'] = float(np.min(x))
    f['crest'] = float(crest_factor(x))
    f['zcr'] = float(zero_crossing_rate(x))
    specf = spec_features(x)
    f.update(specf)
    return f

# ---------------- IO helpers ----------------
def read_motor_csv_for_window(filepath):
    # Accept CSV with/without header. Try common separators.
    # Return five columns as numpy arrays (acc1, mic, acc2, acc3, temp)
    df = None
    try:
        # first try without header (raw signal data)
        df = pd.read_csv(filepath, header=None, on_bad_lines='skip')
    except Exception:
        # fallback to pandas auto header
        df = pd.read_csv(filepath, on_bad_lines='skip')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all', axis=1).dropna()
    if df.shape[1] < 5:
        raise ValueError(f"{filepath} has {df.shape[1]} columns; need at least 5 (acc1,mic,acc2,acc3,temp)")
    # use first five columns
    return df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values, df.iloc[:, 3].values, df.iloc[:, 4].values

def parse_label(fname):
    # extract first two tokens separated by '-', '_' or '.' and map using LABEL_MAP
    stem = Path(fname).stem
    parts = re.split(r'[-_.]', stem)
    # get first two non-empty tokens
    tokens = [p for p in parts if p != ""]
    if len(tokens) >= 2:
        code = f"{tokens[0]}_{tokens[1]}"
        code = code.upper()
        if code in LABEL_MAP:
            return LABEL_MAP[code]
        # also try single token (e.g., H or HH)
        if tokens[0].upper() in LABEL_MAP:
            return LABEL_MAP[tokens[0].upper()]
    # fallback: return original stem
    return stem

# ---------------- feature extraction ----------------
def extract_windows_from_file(filepath, window_samples=WINDOW_SAMPLES, fs=FS):
    try:
        acc1, mic, acc2, acc3, temp = read_motor_csv_for_window(filepath)
    except Exception as e:
        print(f"skip {filepath}: {e}")
        return []
    n = len(acc1)
    n_windows = n // window_samples
    rows = []
    label = parse_label(filepath)
    for w in range(n_windows):
        s = w * window_samples
        e = s + window_samples
        a1 = acc1[s:e]
        m = mic[s:e]
        a2 = acc2[s:e]
        a3 = acc3[s:e]
        tseg = temp[s:e]
        feat = {}
        for ch_name, ch_sig in (("acc1", a1), ("acc2", a2), ("acc3", a3), ("mic", m)):
            fch = features_from_signal(ch_sig)
            for k, v in fch.items():
                feat[f"{ch_name}_{k}"] = v
        feat['temp_mean'] = float(np.mean(tseg))
        feat['temp_std'] = float(np.std(tseg))
        feat['label'] = label
        feat['source_file'] = str(filepath)
        feat['window_idx'] = int(w)
        rows.append(feat)
    return rows

def extract_all(data_root):
    data_root = Path(data_root)
    files = list(data_root.rglob("*.csv"))
    print(f"Found {len(files)} CSV files.")
    rows = []
    for f in tqdm(files):
        rows.extend(extract_windows_from_file(f))
    df = pd.DataFrame(rows)
    return df

# ---------------- augmentation (feature-level) ----------------
def simple_augment_features(X, y, n_copies_per_sample=2, noise_scale=0.02, scale_range=(0.98, 1.02), seed=123):
    rng = np.random.RandomState(seed)
    X_aug = []
    y_aug = []
    for i in range(len(X)):
        for _ in range(n_copies_per_sample):
            noise = rng.normal(0, noise_scale, size=X.shape[1])
            scale = rng.uniform(scale_range[0], scale_range[1])
            X_aug.append(X[i] * (1 + (scale - 1)) + noise)
            y_aug.append(y[i])
    if X_aug:
        X2 = np.vstack([X, np.array(X_aug)])
        y2 = np.concatenate([y, np.array(y_aug)])
        return X2, y2
    return X, y

# ---------------- main ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="path to root folder containing CSV files")
    ap.add_argument("--out", default="motor_model_improved2.pkl", help="output model file")
    ap.add_argument("--do_augment", action="store_true", help="augment training features")
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()

    df_feat = extract_all(args.data_root)
    if df_feat.empty:
        raise SystemExit("No features extracted. Check dataset path and file formats.")

    print("Total windows extracted:", len(df_feat))
    df_feat = df_feat.dropna(axis=1, how='all').fillna(0)

    label_col = 'label'
    meta_cols = ('label', 'source_file', 'window_idx')
    feat_cols = [c for c in df_feat.columns if c not in meta_cols]
    X = df_feat[feat_cols].values
    y = df_feat[label_col].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes (label encoder):", list(le.classes_))

    if args.do_augment:
        print("Performing simple augmentation in feature space...")
        X, y_enc = simple_augment_features(X, y_enc, n_copies_per_sample=2)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    print("Train size:", Xtr.shape[0], "Test size:", Xte.shape[0])

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [12, 20],
        'class_weight': ['balanced']
    }
    clf = RandomForestClassifier(random_state=42, n_jobs=args.n_jobs)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=skf, scoring='accuracy', n_jobs=args.n_jobs, verbose=1)
    print("Starting GridSearchCV ...")
    grid.fit(Xtr, ytr)
    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    ypred = best.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print("Test accuracy:", acc)
    print("Classification report:")
    print(classification_report(yte, ypred, target_names=le.classes_))

    # simulated robustness check
    print("Creating simulated test variants (noisy/scaled) ...")
    rng = np.random.RandomState(999)
    X_sim = []
    y_sim = []
    n_sim_per = 3
    for i in range(min(50, Xte.shape[0])):
        base = Xte[i]
        for _ in range(n_sim_per):
            noise = rng.normal(0, 0.02, size=base.shape)
            scale = rng.uniform(0.98, 1.02)
            X_sim.append(base * scale + noise)
            y_sim.append(yte[i])
    if X_sim:
        X_sim = np.vstack(X_sim)
        y_sim = np.array(y_sim)
        ysim_pred = best.predict(X_sim)
        print("Simulated test acc:", accuracy_score(y_sim, ysim_pred))

    save_pack = {
        'model': best,
        'scaler': scaler,
        'label_encoder': le,
        'feature_columns': feat_cols,
        'label_map': LABEL_MAP
    }
    joblib.dump(save_pack, args.out)
    print("Saved improved model to", args.out)
