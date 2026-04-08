import pandas as pd
import numpy as np
import joblib
import time
import os
# The core metrics functions (MAE, RMSE, R2, SMAPE, RMSLE) are assumed to be 
# imported from a separate 'metrics_helper.py' file.

# --- Required Imports for Model Training ---
from sklearn.linear_model import Ridge 
import lightgbm as lgb 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR 
from catboost import CatBoostRegressor # <<< FINAL MODEL ADDITION >>>

# --- 0. Setup and Data Loading (Ensure directories and paths are correct) ---

# Define Directories (ABSOLUTE PATHS)
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"      # Source of scaled CSVs
MODELS_DIR = r"E:\Regg Thesis\Model_Trained"    # Destination for .pkl models and times CSV

# Ensure the output directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
print("Loading training data...")
# X_train is scaled features
X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
# y_train is log-transformed target (y_log = log(y_raw + 1))
# .squeeze() converts a single-column DataFrame to a Series, which is ideal for y_train
y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze() 

# --- 1. Model Definitions (Final 6 ML Models for Baseline) ---

MODEL_DEFS = {
    # 4/5 Models selected for final HPO
    'LightGBM_Default': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    'RandomForest_Default': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost_Default': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
    'CatBoost_Default': CatBoostRegressor(random_state=42, verbose=0, thread_count=-1), 
    'SVR_Default': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    
    # 6th Model (Linear Baseline - Excluded from HPO)
    'Ridge_Default': Ridge(random_state=42) 
}

training_times = {}

# --- 2. Training Loop for ALL 6 ML Models ---
print("\nStarting training for ALL 6 Machine Learning Models (Trained on Log Target)...")
for name, model in MODEL_DEFS.items():
    print(f"-> Training {name}...")
    
    start_time = time.time()
    
    # Fit the model using the log-transformed target (y_train)
    model.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    training_times[name] = round(elapsed_time, 4)
    
    # Save the trained model object
    joblib.dump(model, os.path.join(MODELS_DIR, f'{name.lower()}.pkl'))
    print(f"  Trained and saved in {elapsed_time:.4f} seconds.")

# --- 3. Output Summary ---
print("\n--- Final 6-Model Baseline Training Complete ---")
print(f"Trained models saved to: {MODELS_DIR}")

# Save training times to a CSV
pd.Series(training_times, name='Training_Time_s').to_csv(os.path.join(MODELS_DIR, 'baseline_training_times_v4.csv'), index=True, header=True)
print("\nTraining times saved to 'baseline_training_times_v4.csv'.")