import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from scipy.stats import loguniform

# --- 0. Setup and Data Loading ---
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results"
MODEL_NAME = 'Ridge'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading training data...")
X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze()

# For Ridge, we use the full training set for HPO, as there's no early stopping.
X_hpo, y_hpo = X_train.copy(), y_train.copy()


# --- 1. Model Definition and Search Space ---

# 1.1 Model Estimator
model_estimator = Ridge(random_state=42)

# 1.2 Define the search space
param_distributions = {
    # 'alpha' (regularization strength) is the key hyperparameter.
    # We search over a wide log-uniform range (10^-4 to 10^2).
    'alpha': loguniform(1e-4, 1e2)
}

# --- 2. HPO Setup and Execution ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
SCORING_METRIC = 'neg_mean_squared_error'

hpo = RandomizedSearchCV(
    estimator=model_estimator,
    param_distributions=param_distributions,
    n_iter=100,                       # 100 iterations for consistency
    scoring=SCORING_METRIC,
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    refit=False
)

print(f"\nStarting Randomized HPO for {MODEL_NAME} (n_iter=100, 5-Fold CV)...")
start_time = time.time()

# --- EXECUTION: HPO runs the CV folds ---
hpo.fit(X_hpo, y_hpo)

elapsed_time = time.time() - start_time
print(f"\nHPO Search Time: {elapsed_time:.2f} seconds.")

# --- 3. Final Model Training and Saving Results ---

# 1. Extract the best hyperparameters found during the HPO search
best_params = hpo.best_params_

# 2. Instantiate the FINAL model with the best parameters
final_model = Ridge(random_state=42, **best_params)

# 3. Time the training of the FINAL model (single fit on the full data)
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
results_df.to_csv(os.path.join(RESULTS_DIR, f'{MODEL_NAME.lower()}_hpo_results.csv'), index=False)

print(f"\n[{MODEL_NAME} HPO Complete]")
print(f"Best parameter found: {best_params}")
print(f"Tuned model saved to: {model_save_path}")
print(f"Final model training time (for Table II): {final_fit_time:.4f} seconds.")