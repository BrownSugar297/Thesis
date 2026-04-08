import pandas as pd
import numpy as np
import joblib
import os
import time

from xgboost import XGBRegressor, DMatrix, train as xgb_train
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================
# 0. Paths & Data
# ==========================
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading training data...")
X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze()

# Optional split for early stopping monitoring
X_train_monitor, X_val, y_train_monitor, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, shuffle=True
)

# ==========================
# 1. XGBoost Wrapper for HPO
# ==========================
class XGBoostHPOWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for XGBRegressor without early stopping (safe for RandomizedSearchCV)."""
    def __init__(self, **params):
        self.params = params
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        final_params = {**base_params, **params}
        self.model = XGBRegressor(**final_params)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.__init__(**params)
        return self

model_estimator = XGBoostHPOWrapper()

# ==========================
# 2. HPO Search Space
# ==========================
param_distributions = {
    'n_estimators': sp_randint(500, 1500),
    'max_depth': sp_randint(6, 12),
    'min_child_weight': sp_randint(1, 10),
    'learning_rate': sp_uniform(loc=0.005, scale=0.15),
    'subsample': sp_uniform(loc=0.6, scale=0.4),
    'colsample_bytree': sp_uniform(loc=0.6, scale=0.4),
    'gamma': sp_uniform(loc=0.0, scale=0.5),
    'reg_lambda': sp_uniform(loc=0.0, scale=1.0),
    'reg_alpha': sp_uniform(loc=0.0, scale=1.0)
}

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
    n_jobs=-1,
    refit=False  # We'll train the final model separately
)

print(f"\nStarting Randomized HPO for XGBoost (n_iter=100, 5-Fold CV)...")
start_time = time.time()
hpo.fit(X_train_monitor, y_train_monitor)
elapsed_time = time.time() - start_time
print(f"\nHPO Search Time: {elapsed_time:.2f} seconds.")

best_params = hpo.best_params_
print(f"Best hyperparameters found: {best_params}")

# ==========================
# 3. Final Full-Data Training
# ==========================
print("\nTraining final XGBoost model on full 2022–2024 data with early stopping...")

# Convert to DMatrix
dtrain_full = DMatrix(X_train, label=y_train)
dval = DMatrix(X_val, label=y_val)

start_final_fit = time.time()
bst = xgb_train(
    params=best_params,
    dtrain=dtrain_full,
    num_boost_round=best_params['n_estimators'],
    evals=[(dtrain_full, 'train'), (dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=False
)
final_fit_time = time.time() - start_final_fit

# Wrap into XGBRegressor for saving (compatible with sklearn)
final_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    **best_params
)
final_model._Booster = bst  # attach trained booster
# Note: Do NOT set best_ntree_limit (not available in 3.1.2)

# Save the model
model_save_path = os.path.join(MODELS_DIR, 'xgboost_tuned_full.pkl')
joblib.dump(final_model, model_save_path)

# Save HPO results
results_df = pd.DataFrame(hpo.cv_results_)
results_df['total_hpo_time_s'] = elapsed_time
results_df['final_model_training_time_s'] = final_fit_time
results_df.to_csv(os.path.join(RESULTS_DIR, 'xgboost_hpo_results.csv'), index=False)

print(f"\n[XGBoost HPO + Full Training Complete]")
print(f"Tuned model saved to: {model_save_path}")
print(f"Final training time (s): {final_fit_time:.4f}")

# ==========================
# 4. Prediction Example
# ==========================
# To predict later using early-stopping rounds:
# y_pred = final_model._Booster.predict(DMatrix(X_test), ntree_limit=final_model._Booster.best_iteration)
