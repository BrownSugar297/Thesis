import pandas as pd
import numpy as np
import joblib
import os
# --- CHANGE 1: Added mean_absolute_error import ---
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator

# =========================================================================
# === CRITICAL FIX: Custom Class Definition for LightGBM HPO ===
# =========================================================================
class LGBMRegressorWithES(LGBMRegressor):
    """
    Custom class that inherits from LGBMRegressor. 
    This allows joblib to correctly unpickle the LightGBM model 
    that was saved with this custom wrapper class name.
    """
    def __init__(self, **params):
        super().__init__(random_state=42, **params)
        
# =========================================================================

# --- 0. Setup Directories and Files ---
TEST_DATA_DIR = r"E:\Regg Thesis\Model_Test"
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results" 
FINAL_TABLE_DIR = r"E:\Regg Thesis\Results"

os.makedirs(FINAL_TABLE_DIR, exist_ok=True)

print("Starting Final Test Set Evaluation for Tuned Models...")

# --- 1. Load Test Data ---
try:
    X_test = pd.read_csv(os.path.join(TEST_DATA_DIR, "X_test_scaled.csv"))
    y_test_log = pd.read_csv(os.path.join(TEST_DATA_DIR, "y_test_log.csv")).squeeze()
    y_test_raw = np.expm1(y_test_log)
    
except FileNotFoundError:
    print(f"FATAL ERROR: Test Set data files not found in the specified directory: {TEST_DATA_DIR}")
    exit() 

# --- 2. Thesis-Specific Metric Calculation Functions (Metrics will be correct) ---
def rmsle(y_true_log, y_pred_log):
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))

def mape_nonzero(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    y_pred_raw = np.clip(y_pred_raw, a_min=0, a_max=None)
    non_zero_mask = y_true_raw > 0
    y_true_nz = y_true_raw[non_zero_mask]
    y_pred_nz = y_pred_raw[non_zero_mask]
    
    if len(y_true_nz) == 0: return np.nan 
        
    percentage_error = np.abs(y_true_nz - y_pred_nz) / y_true_nz
    return 100 * np.mean(percentage_error)

def evaluate_model(y_test_log, y_test_raw, y_pred_log):
    y_pred_raw = np.expm1(y_pred_log)
    return {
        # --- CHANGE 2: Added MAE calculation ---
        'MAE': mean_absolute_error(y_test_raw, y_pred_raw), 
        'RMSE': np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)), 
        'R^2': r2_score(y_test_raw, y_pred_raw),                      
        'MAPE (Non-Zero %)': mape_nonzero(y_test_raw, y_pred_raw),   
        'RMSLE': rmsle(y_test_log, y_pred_log)                       
    }

# --- 3. Define Models and Map to EXACT File Prefixes ---
MODELS_TO_EVALUATE = [
    'Random Forest', 'LightGBM', 'CatBoost', 'XGBoost', 'Ridge', 'SVR'
]
# CRITICAL: MAPPING to the EXACT prefix of your .pkl file names.
MODEL_FILE_MAP = {
    'Random Forest': 'randomforest_tuned', # Matches 'randomforest_tuned.pkl'
    'LightGBM': 'lightgbm_tuned',          # Matches 'lightgbm_tuned.pkl'
    'CatBoost': 'catboost_tuned',          # Matches 'catboost_tuned.pkl'
    'XGBoost': 'xgboost_tuned_full',       # Matches 'xgboost_tuned_full.pkl'
    'Ridge': 'ridge_tuned',                # Matches 'ridge_tuned.pkl'
    'SVR': 'svr_tuned'                     # Matches 'svr_tuned.pkl'
}
FINAL_RESULTS = []

# --- 4. Loop through Tuned Models, Predict, and Compile Results ---
print("\nEvaluating Tuned Models on Test Set...")
for model_name in MODELS_TO_EVALUATE:
    # Use the full prefix from the map to construct the model path
    pkl_file_prefix = MODEL_FILE_MAP[model_name]
    model_path = os.path.join(MODELS_DIR, f'{pkl_file_prefix}.pkl') 
    
    # Use the simplified name for the HPO results CSV (e.g., 'xgboost_hpo_results.csv')
    hpo_csv_prefix = pkl_file_prefix.replace('_tuned', '').replace('_full', '')
    results_path = os.path.join(RESULTS_DIR, f'{hpo_csv_prefix}_hpo_results.csv') 
    
    print(f"-> Processing {model_name}...")
    
    try:
        # Load Model
        tuned_model = joblib.load(model_path)
        
        # Load Training Time
        hpo_df = pd.read_csv(results_path)
        
        # === ROBUST FIX for KeyError and Missing Time ===
        TIME_COLUMN = 'final_model_training_time_s'
        PROXY_COLUMN = 'mean_fit_time'

        if TIME_COLUMN in hpo_df.columns:
            # Use the dedicated final training time if available
            training_time = hpo_df[TIME_COLUMN].iloc[0]
        elif PROXY_COLUMN in hpo_df.columns and 'rank_test_score' in hpo_df.columns:
            # Use the mean fit time of the best model (rank 1) as a proxy
            best_model_row = hpo_df[hpo_df['rank_test_score'] == 1]
            if not best_model_row.empty:
                training_time = best_model_row[PROXY_COLUMN].iloc[0]
                print(f"   WARNING: Using '{PROXY_COLUMN}' of the best model ({training_time:.4f}s) as proxy for {model_name}.")
            else:
                training_time = np.nan
                print(f"   WARNING: Could not find rank 1 for {model_name}. Setting time to NaN.")
        else:
            # If no time column is available at all
            print(f"   WARNING: No training time data found for {model_name}. Setting time to NaN.")
            training_time = np.nan
        # ====================================

        # Predict (in log space) and Evaluate
        y_pred_log = tuned_model.predict(X_test)
        metrics = evaluate_model(y_test_log, y_test_raw, y_pred_log)
        
        # Store Results
        result = {
            'Model': model_name,
            'Training Time (s)': training_time,
            **metrics
        }
        FINAL_RESULTS.append(result)
        
    except FileNotFoundError as e:
        print(f"SKIPPING: Required file not found for {model_name}. Please check paths. Error: {e}")
        
# --- 5. Compile Final Table ---
table_df = pd.DataFrame(FINAL_RESULTS)
table_df = table_df.set_index('Model')

# --- CHANGE 3: Added MAE to the column order ---
COLUMN_ORDER = ['MAE', 'RMSE', 'R^2', 'MAPE (Non-Zero %)', 'RMSLE', 'Training Time (s)'] 
table_df = table_df[COLUMN_ORDER]
table_df = table_df.sort_values(by='RMSE', ascending=True)

output_path = os.path.join(FINAL_TABLE_DIR, "Table_Tuned_Models_Performance.csv")
table_df.to_csv(output_path)

print("\n--- FINAL TABLE COMPILATION COMPLETE ---")
print(f"Full results saved to: {output_path}")

print("\n--- FINAL RESULTS (TUNED MODELS) ---")
print(table_df.to_markdown(floatfmt=".4f"))