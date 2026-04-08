import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union

# --- 1. CORE PHYSICAL METRICS (Require Raw Counts: y_pred_raw vs. y_true_raw) ---

def calculate_mae(y_true_raw, y_pred_raw):
    """Calculates Mean Absolute Error (MAE)."""
    return mean_absolute_error(y_true_raw, y_pred_raw)

def calculate_rmse(y_true_raw, y_pred_raw):
    """Calculates Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))

def calculate_r2(y_true_raw, y_pred_raw):
    """Calculates R-squared (Coefficient of Determination)."""
    return r2_score(y_true_raw, y_pred_raw)

def calculate_smape(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    """
    Calculates Symmetric Mean Absolute Percentage Error (SMAPE) robustly.
    Handles the (0, 0) case where the error should be 0.
    """
    # Ensure all values are non-negative, as patient counts cannot be negative
    y_true_raw = np.maximum(0, y_true_raw)
    y_pred_raw = np.maximum(0, y_pred_raw)
    
    numerator = np.abs(y_pred_raw - y_true_raw)
    denominator = (np.abs(y_true_raw) + np.abs(y_pred_raw)) / 2
    
    # Identify indices where the denominator is zero (i.e., y_true=0 and y_pred=0)
    zero_denominator_idx = (denominator == 0)
    
    # Calculate SMAPE term, setting it to 0 where the denominator is zero
    smape_terms = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=~zero_denominator_idx)
    
    # SMAPE formula: Mean of the terms * 100
    smape_val = np.mean(smape_terms) * 100
    
    return smape_val

# --- 2. LOG-SPACE/STATISTICAL METRICS (Require Log Counts) ---

def calculate_rmsle(y_true_log, y_pred_log):
    """
    Calculates Root Mean Squared Logarithmic Error (RMSLE).
    It operates directly on the log-transformed data.
    """
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))


# --- 3. UNIFIED WRAPPER FUNCTION (Mandatory for Clean Pipeline) ---

# NOTE: The model_name argument is no longer needed but kept for compatibility.
def calculate_metrics(y_test_log, y_test_raw, y_pred_log, model_name=None):
    """
    Calculates all required metrics for a given prediction.
    """
    metrics = {}
    
    # 1. Physical Space Metrics (Calculated on Raw Counts after Inverse Transform)
    y_pred_raw = np.maximum(0, np.expm1(y_pred_log))
    
    metrics['MAE'] = calculate_mae(y_test_raw, y_pred_raw)
    metrics['RMSE'] = calculate_rmse(y_test_raw, y_pred_raw)
    metrics['R2'] = calculate_r2(y_test_raw, y_pred_raw)
    metrics['SMAPE (%)'] = calculate_smape(y_test_raw, y_pred_raw)
    
    # 2. Log Space Metrics
    metrics['RMSLE'] = calculate_rmsle(y_test_log, y_pred_log)
    
    # 3. Statistical Loss Metric is RMSLE for all current ML models
    metrics['Statistical Loss Metric'] = metrics['RMSLE']
        
    return metrics