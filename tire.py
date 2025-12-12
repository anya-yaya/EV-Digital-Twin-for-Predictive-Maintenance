import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_tire_maintenance_model(file_path):
    # --- Configuration ---
    # Define a maximum tire life for engineering the target RUL (in km)
    MAX_TIRE_LIFE_KM = 50000
    FEATURE_COLS = [
        'Distance_Traveled',
        'Tire_Pressure',
        'Tire_Temperature',
        'Driving_Speed',
        'Route_Roughness',
        'Load_Weight',
        'Ambient_Temperature'
    ]
    
    MODEL_FILE = 'tire_maintenance_model.joblib'
    SCALER_FILE = 'tire_maintenance_scaler.joblib'
    FEATURES_FILE = 'tire_features.joblib'

    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 2. Feature Engineering: Create the RUL Target
    # RUL_Tire_km = Max Life - Current Distance Traveled
    df['RUL_Tire_km'] = np.maximum(0, MAX_TIRE_LIFE_KM - df['Distance_Traveled'])

    # 3. Data Preprocessing
    df_model = df[FEATURE_COLS + ['RUL_Tire_km']].copy()
    
    # Handle missing values (Forward Fill for time series, then drop remaining)
    df_model.fillna(method='ffill', inplace=True)
    df_model.dropna(inplace=True)

    X = df_model[FEATURE_COLS]
    y = df_model['RUL_Tire_km']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training (Random Forest Regressor)
    print("Starting Model Training...")
    # Using a simple setup for quick training and demonstration
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train_scaled, y_train)
    print("Model Training Complete.")

    # 6. Evaluate Model
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation (Random Forest Regressor):")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} km")
    print(f"  R-squared (R2): {r2:.4f}")

    # 7. Save Model Artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(FEATURE_COLS, FEATURES_FILE)
    
    print(f"\nModel, Scaler, and Features saved to: {MODEL_FILE}, {SCALER_FILE}, {FEATURES_FILE}")

if __name__ == "__main__":
    # Ensure you run this script with your dataset in the same directory
    train_tire_maintenance_model("EV_Predictive_Maintenance_Dataset_15min.csv")