import pandas as pd
import numpy as np
import joblib
import os
import time

from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb 

# --- 0. Setup and Data Loading ---
# Define directory paths
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"     # Destination for tuned models (.pkl)
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"      # Source of scaled CSVs
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results"    # Dedicated directory for HPO logs (.csv)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) 

print("Loading training data...")
X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze() 

# --- CRITICAL: Splitting Data for Early Stopping Monitoring ---
# 90% for training (used in CV), 10% for validation (used for early stopping monitoring)
X_train_monitor, X_val, y_train_monitor, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, shuffle=True
)

# --- 1. Model Definition and Search Space ---

MODEL_NAME = 'LightGBM'

# --- 1.1 FINAL FIX: Custom Class with Overridden .fit() ---
# This class ensures early stopping parameters are passed every time RandomizedSearchCV calls .fit()
class LGBMRegressorWithES(LGBMRegressor):
    def fit(self, X, y, **kwargs):
        return super().fit(
            X, y,
            eval_set=[(X_val, y_val)], # Uses the dedicated validation set for monitoring
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.callback.log_evaluation(period=0)
            ]
        )

# Instantiate the custom model estimator
model_estimator = LGBMRegressorWithES(
    random_state=42, 
    n_jobs=-1, 
    objective='regression', 
    boosting_type='gbdt'
)

# 1.2 Define the search space (Robust and Research-Grade)
param_distributions = {
    'n_estimators': sp_randint(500, 1500),              
    'num_leaves': sp_randint(20, 100),              
    'max_depth': sp_randint(6, 12),                 
    'min_child_samples': sp_randint(10, 50),        
    'learning_rate': sp_uniform(loc=0.005, scale=0.15), 
    'subsample': sp_uniform(loc=0.6, scale=0.4),        
    'colsample_bytree': sp_uniform(loc=0.6, scale=0.4), 
    'reg_lambda': sp_uniform(loc=0.0, scale=1.0),       
    'reg_alpha': sp_uniform(loc=0.0, scale=1.0),        
    'min_split_gain': sp_uniform(loc=0.0, scale=0.4)    
}

# --- 2. HPO Setup and Execution ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
SCORING_METRIC = 'neg_mean_squared_error' 

hpo = RandomizedSearchCV(
    estimator=model_estimator,
    param_distributions=param_distributions,
    n_iter=100,                       
    scoring=SCORING_METRIC,
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print(f"\nStarting Randomized HPO for {MODEL_NAME} (n_iter=100, 5-Fold CV) with Early Stopping via Custom Class...")
start_time = time.time()

# --- EXECUTION: Simply call fit with the main training data ---
# The Early Stopping parameters are now safely contained within the LGBMRegressorWithES class.
hpo.fit(X_train_monitor, y_train_monitor)

elapsed_time = time.time() - start_time
print(f"\nHPO Search Time: {elapsed_time:.2f} seconds.")

# --- 3. Saving Results ---
best_model = hpo.best_estimator_
model_save_path = os.path.join(MODELS_DIR, f'{MODEL_NAME.lower()}_tuned.pkl')
joblib.dump(best_model, model_save_path)

results_df = pd.DataFrame(hpo.cv_results_)
results_df['total_hpo_time_s'] = elapsed_time
results_df.to_csv(os.path.join(RESULTS_DIR, f'{MODEL_NAME.lower()}_hpo_results.csv'), index=False)

print(f"\n[{MODEL_NAME} HPO Complete]")
print(f"Best parameters found: {hpo.best_params_}")
print(f"Tuned model saved to: {model_save_path}")
print(f"HPO results log saved to: {os.path.join(RESULTS_DIR, f'{MODEL_NAME.lower()}_hpo_results.csv')}")