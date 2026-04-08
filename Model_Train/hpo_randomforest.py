import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# --- 0. Setup and Data Loading ---
# Define directory paths (Must match your setup)
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"     
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"      
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results"    

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) 

print("Loading training data...")
# Load the full 2022-2024 training set (Features and Log-transformed Target)
X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze() 

# --- 1. Model Definition and Search Space ---

MODEL_NAME = 'RandomForest'

# 1.1 Define the model object
model_estimator = RandomForestRegressor(
    random_state=42, 
    n_jobs=-1,
    bootstrap=True 
)

# 1.2 Define the search space (RandomForest specific parameters)
param_distributions = {
    'n_estimators': sp_randint(500, 1500),              
    'max_depth': sp_randint(6, 12),                     
    'min_samples_split': sp_randint(2, 20),             
    'min_samples_leaf': sp_randint(1, 10),              
    'max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0], 
    'max_leaf_nodes': sp_randint(10, 100),              
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

print(f"\nStarting Randomized HPO for {MODEL_NAME} (n_iter=100, 5-Fold CV)...")
start_time = time.time()

# --- EXECUTION ---
hpo.fit(X_train, y_train)

elapsed_time = time.time() - start_time
print(f"\nHPO Search Time: {elapsed_time:.2f} seconds.")

# --- 3. Final Model Training and Saving Results (CORRECTED) ---

# 1. Extract the best model from HPO
best_model = hpo.best_estimator_

# 2. Time the training of the FINAL model on the entire training set (X_train, y_train) 
#    This captures the exact 'Training Time (s)' for Table II.
start_final_fit = time.time()
# Re-fit the final best model on the full data to capture the most accurate time
final_model = best_model.fit(X_train, y_train) 
final_fit_time = time.time() - start_final_fit

# Save the final optimized model (.pkl)
model_save_path = os.path.join(MODELS_DIR, f'{MODEL_NAME.lower()}_tuned.pkl')
joblib.dump(final_model, model_save_path)


# Save the HPO results log (.csv)
results_df = pd.DataFrame(hpo.cv_results_)
results_df['total_hpo_search_time_s'] = elapsed_time # Total time for all 500 CV fits
results_df['final_model_training_time_s'] = final_fit_time # New metric for Table II
results_df.to_csv(os.path.join(RESULTS_DIR, f'{MODEL_NAME.lower()}_hpo_results.csv'), index=False)

print(f"\n[{MODEL_NAME} HPO Complete]")
print(f"Best parameters found: {hpo.best_params_}")
print(f"Tuned model saved to: {model_save_path}")
print(f"Final model training time (for Table II): {final_fit_time:.4f} seconds.")