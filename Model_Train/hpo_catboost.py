import pandas as pd
import numpy as np
import joblib
import os
import time

from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# --- 0. Setup and Data Loading ---
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"     
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"      
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results"    

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) 

print("Loading training data...")
X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze() 

# --- CRITICAL: Splitting Data for Early Stopping Monitoring ---
X_train_monitor, X_val, y_train_monitor, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, shuffle=True
)

# --- 1. Model Definition and Search Space ---

MODEL_NAME = 'CatBoost'

# 1.1 Define the model object (Fixed parameters - UPDATED with explicit eval_metric)
model_estimator = CatBoostRegressor(
    loss_function='RMSE',           
    eval_metric='RMSE',              # FIX 2: Explicitly defining the evaluation metric
    random_state=42, 
    thread_count=-1,                
    verbose=0,                      
    allow_writing_files=False       
)

# 1.2 Define the search space (UPDATED with grow_policy, min_data_in_leaf, subsampling)
param_distributions = {
    'iterations': sp_randint(500, 1500),              
    'depth': sp_randint(6, 12),                       
    'learning_rate': sp_uniform(loc=0.005, scale=0.15), 
    'l2_leaf_reg': sp_uniform(loc=1.0, scale=5.0),      
    'border_count': sp_randint(50, 255),               
    'random_strength': sp_uniform(loc=0.1, scale=1.0), 
    # FIX 1: Essential Tree Structure Parameters for numeric-only data
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
    'min_data_in_leaf': sp_randint(1, 50),
    # OPTIONAL 1: Bagging / Subsampling
    'subsample': sp_uniform(loc=0.7, scale=0.3),        # Row subsampling (Range 0.7 to 1.0)
    'colsample_bylevel': sp_uniform(loc=0.7, scale=0.3) # Feature subsampling (Range 0.7 to 1.0)
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

print(f"\nStarting Randomized HPO for {MODEL_NAME} (n_iter=100, 5-Fold CV) with full parameter exploration...")
start_time = time.time()

# --- EXECUTION: Increased Early Stopping Rounds ---
hpo.fit(
    X_train_monitor, y_train_monitor,
    # Increased patience for stability (OPTIONAL 2)
    eval_set=[(X_val, y_val)],       
    early_stopping_rounds=100,        
)

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