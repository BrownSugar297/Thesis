import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import loguniform

# --- 0. Setup and Data Loading ---
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results"
MODEL_NAME = 'SVR'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading training data...")
X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze()

# Use the full training set for HPO
X_hpo, y_hpo = X_train.copy(), y_train.copy()


# --- 1. Model Definition and Search Space ---

# 1.1 Model Estimator
model_estimator = SVR(kernel='rbf')

# 1.2 Define the search space (including the new epsilon parameter)
param_distributions = {
    # C: Regularization parameter. Log search from 0.1 to 1000.
    'C': loguniform(1e-1, 1e3),      
    # gamma: Kernel coefficient. Log search from 0.0001 to 0.1.
    'gamma': loguniform(1e-4, 1e-1), 
    # epsilon: Margin of tolerance. Log search from 0.001 to 1.0.
    'epsilon': loguniform(1e-3, 1),
    # Kernel is fixed to RBF.
    'kernel': ['rbf']                
}

# --- 2. HPO Setup and Execution ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- REFINEMENT 1: Multiple Scoring Metrics ---
SCORING_METRIC = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2'
}

hpo = RandomizedSearchCV(
    estimator=model_estimator,
    param_distributions=param_distributions,
    n_iter=100,
    scoring=SCORING_METRIC,
    cv=kf,
    random_state=42,
    n_jobs=-1,
    refit='rmse', # --- REFINEMENT 2: Refit based on RMSE (primary metric)
    verbose=1
)

print(f"\nStarting Randomized HPO for {MODEL_NAME} (n_iter=100, 5-Fold CV)...")
print("⚠️ WARNING: SVR HPO is computationally expensive and may take several hours to complete.")
start_time = time.time()

# --- EXECUTION: HPO runs the CV folds ---
hpo.fit(X_hpo, y_hpo)

elapsed_time = time.time() - start_time
print(f"\nHPO Search Time: {elapsed_time:.2f} seconds.")

# --- 3. Final Model Training and Saving Results ---

# 1. Extract the best hyperparameters found during the HPO search
best_params = hpo.best_params_

# 2. The hpo object automatically refits the best model using the 'rmse' metric.
# We extract the refitted model.
final_model = hpo.best_estimator_

# 3. Time the training of the FINAL model 
# Since refit=True was used, the timing below is not strictly needed for the final fit, 
# but we time a full fit explicitly for consistency with Table II.
start_final_fit = time.time()
final_model.fit(X_hpo, y_hpo)
final_fit_time = time.time() - start_final_fit

# Save the final optimized model (.pkl)
model_save_path = os.path.join(MODELS_DIR, f'{MODEL_NAME.lower()}_tuned.pkl')
joblib.dump(final_model, model_save_path)


# Save the HPO results log (.csv)
results_df = pd.DataFrame(hpo.cv_results_)
results_df['total_hpo_search_time_s'] = elapsed_time
results_df['final_model_training_time_s'] = final_fit_time
results_df['best_score_rmse'] = hpo.best_score_ # --- REFINEMENT 3: Log best score
results_df.to_csv(os.path.join(RESULTS_DIR, f'{MODEL_NAME.lower()}_hpo_results.csv'), index=False)

print(f"\n[{MODEL_NAME} HPO Complete]")
print(f"Best parameters found: {best_params}")
print(f"Best cross-validation RMSE score: {np.sqrt(-hpo.best_score_):.4f}")
print(f"Tuned model saved to: {model_save_path}")
print(f"Final model training time (for Table II): {final_fit_time:.4f} seconds.")
