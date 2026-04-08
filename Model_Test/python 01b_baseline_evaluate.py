import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union

# --- Metric Functions (Final Set - All correct) ---

def calculate_mae(y_true_raw, y_pred_raw):
    """Calculates Mean Absolute Error on raw counts."""
    return mean_absolute_error(y_true_raw, y_pred_raw)

def calculate_rmse(y_true_raw, y_pred_raw):
    """Calculates Root Mean Squared Error on raw counts."""
    return np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))

def calculate_r2(y_true_raw, y_pred_raw):
    """Calculates R-squared on raw counts."""
    return r2_score(y_true_raw, y_pred_raw)

def calculate_rmsle(y_true_log, y_pred_log):
    """Calculates Root Mean Squared Logarithmic Error (RMSLE) on log-transformed values."""
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))

# MAPE calculated only on non-zero true values
def calculate_mape_non_zero(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE) only for observations where the true count > 0."""
    non_zero_indices = y_true_raw != 0
    y_true_nz = y_true_raw[non_zero_indices]
    y_pred_nz = y_pred_raw[non_zero_indices]
    
    if len(y_true_nz) == 0:
        return np.nan
    
    absolute_percentage_error = np.abs((y_true_nz - y_pred_nz) / y_true_nz)
    mape = np.mean(absolute_percentage_error) * 100
    return mape

def calculate_metrics(y_test_log, y_test_raw, y_pred_log):
    metrics = {}
    y_pred_raw = np.maximum(0, np.expm1(y_pred_log))
    
    metrics['MAE'] = calculate_mae(y_test_raw, y_pred_raw)
    metrics['RMSE'] = calculate_rmse(y_test_raw, y_pred_raw)
    metrics['R2'] = calculate_r2(y_test_raw, y_pred_raw)
    metrics['MAPE (Non-Zero %)'] = calculate_mape_non_zero(y_test_raw, y_pred_raw)
    metrics['RMSLE'] = calculate_rmsle(y_test_log, y_pred_log) 
    
    return metrics

# --- 0. Setup and Data Loading (Path Correction Applied) ---

TEST_DIR = r"E:\Regg Thesis\Results" # Destination for metrics CSV
MODELS_DIR = r"E:\Regg Thesis\Model_Trained" # Source for trained models
TEST_DATA_DIR = r"E:\Regg Thesis\Model_Test" # <<< CORRECTED SOURCE FOR TEST DATA >>>

os.makedirs(TEST_DIR, exist_ok=True)

# Corrected 6-model list: EBM replaced by CatBoost
MODEL_NAMES = [
    'LightGBM_Default', 'RandomForest_Default', 'XGBoost_Default', 
    'CatBoost_Default', 'SVR_Default', 'Ridge_Default' 
]

# Paths for input/output
TIMES_FILE = os.path.join(MODELS_DIR, 'baseline_training_times_v4.csv')
METRICS_FILE = os.path.join(TEST_DIR, 'baseline_results_table1_final_v4.csv') 

# Load data
print("Loading testing data...")
# Load X_test, y_test_log from the CORRECT directory (TEST_DATA_DIR)
X_test = pd.read_csv(os.path.join(TEST_DATA_DIR, "X_test_scaled.csv")) 
y_test_log = pd.read_csv(os.path.join(TEST_DATA_DIR, "y_test_log.csv")).squeeze()

# Create raw target for raw-space metric calculation
# NOTE: The y_test_raw is derived from y_test_log using expm1, assuming it wasn't saved separately.
y_test_raw = np.expm1(y_test_log)

# Load training times
try:
    training_times = pd.read_csv(TIMES_FILE, index_col=0).squeeze()
except FileNotFoundError:
    print(f"Error: Training times file not found at {TIMES_FILE}. Please ensure it was created by the training script.")
    exit()

results_list = []

# --- 1. Evaluation Loop ---
print("\nStarting final evaluation for 6 ML models...")
for name in MODEL_NAMES:
    print(f"-> Evaluating {name}...")
    
    model_path = os.path.join(MODELS_DIR, f'{name.lower()}.pkl')
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"  ERROR: Model file not found at {model_path}. Skipping.")
        continue

    result = {'Model': name.replace('_Default', '')}
    
    # --- Unified ML Model Handling ---
    y_pred_log = model.predict(X_test)
    metrics = calculate_metrics(y_test_log, y_test_raw, y_pred_log)
    
    # Combine results and training time
    result.update(metrics)
    result['Training Time (s)'] = training_times.get(name, np.nan)
    results_list.append(result)

# --- 2. Final Output ---
results_df = pd.DataFrame(results_list)
results_df = results_df.set_index('Model')
results_df.index.name = None

# Save the complete results, including RMSLE for the appendix
results_df.to_csv(METRICS_FILE)
print("\n--- Final Evaluation Complete ---")
print(f"Full results saved to: {METRICS_FILE}\n")

# Print the final 5-metric main table structure
final_main_table = results_df[['MAE', 'RMSE', 'R2', 'MAPE (Non-Zero %)', 'Training Time (s)']]
print("Final 5-Metric Main Table (Partial Printout):")
print(final_main_table.to_markdown(floatfmt=".4f"))