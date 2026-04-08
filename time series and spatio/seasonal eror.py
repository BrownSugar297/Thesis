
#Figure 5: Seasonal Error Analysis (MAE by Division and Season)#


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib 

# ------------------ USER CONFIG (CRITICAL) ------------------
# !!! CRITICAL: ENSURE THESE ABSOLUTE PATHS ARE CORRECT IN YOUR ENVIRONMENT !!!
# Replace these with the actual absolute paths on your system if they differ.
X_TEST_SCALED = r'E:\Regg Thesis\Figures\X_test_scaled.csv' 
X_TEST_WITH_DATES = r'E:\Regg Thesis\Figures\X_test_scaled_for_plotting.csv' # File containing 'date', 'max_temp', 'total_rainfall'
Y_TEST_LOG = r'E:\Regg Thesis\Figures\y_test_log.csv'
Y_TEST_ORIGINAL = r'E:\Regg Thesis\Figures\y_test_original_patients.csv'
MODEL_PATH = r'E:\Regg Thesis\Figures\lightgbm_tuned.pkl'
SAVE_FOLDER = r'E:\Regg Thesis\time series and spatio\Fig' # Where to save the output PDF/PNG
FIG_WIDTH = 7.0 
# ------------------------------------------------------------


# ------------------ UTILITY & STYLING ------------------

# Simplified save function using the configured SAVE_FOLDER
def save_figure(fname_base):
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    pdf_path = os.path.join(SAVE_FOLDER, f"{fname_base}.pdf")
    png_path = os.path.join(SAVE_FOLDER, f"{fname_base}.png")
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    plt.savefig(png_path, bbox_inches='tight', format='png', dpi=300)
    print(f"Saved: {pdf_path} and {png_path}")

# Inverse transformation used for your log-transformed targets
def inverse_transform(y_log_array):
    """Inverse of natural log transformation used in training (assumes y = log(original))."""
    y = np.exp(y_log_array)
    y = np.maximum(0, y)
    return np.round(y).astype(int)

# Placeholder class for joblib.load if your model uses a custom wrapper
class LGBMRegressorWithES:
    def predict(self, X):
        if hasattr(self, '_Booster') and hasattr(self._Booster, 'predict'):
            # LightGBM booster object is usually stored in _Booster attribute
            return self._Booster.predict(X)
        else:
            raise NotImplementedError("Model object is not initialized correctly.")

# ------------------ LOAD DATA & MODEL (SIMULATION) ------------------

# Global DataFrame to hold data, features, and results
X_test = pd.DataFrame() 

try:
    # 1. Load scaled features
    X_test_scaled = pd.read_csv(X_TEST_SCALED)
    
    # 2. Load metadata (Date, etc.)
    df_dates_and_info = pd.read_csv(X_TEST_WITH_DATES)
    
    # Rename for consistency
    if 'date' in df_dates_and_info.columns: df_dates_and_info.rename(columns={'date': 'Date'}, inplace=True)
    
    X_test = X_test_scaled.copy()
    
    # Add metadata back
    INFO_COLS_TO_ADD = ['Date', 'max_temp', 'total_rainfall'] 
    for col in INFO_COLS_TO_ADD:
        if col in df_dates_and_info.columns and len(df_dates_and_info) >= len(X_test):
            X_test[col] = df_dates_and_info[col].iloc[:len(X_test)].values 

    # 3. Load Targets and Model
    y_test_log = pd.read_csv(Y_TEST_LOG).iloc[:, 0]
    y_test_original = pd.read_csv(Y_TEST_ORIGINAL).iloc[:, 0]
    best_model = joblib.load(MODEL_PATH) 
    
    # 4. Feature Engineering (CRITICAL FOR FIGURE 5)
    
    # Reconstruct Division column from one-hot encoding
    division_cols = [c for c in X_test.columns if c.startswith('division_')]
    if division_cols:
        X_test['Division'] = X_test[division_cols].idxmax(axis=1).str.replace('division_', '')
    
    # Reconstruct Season column
    if 'Date' in X_test.columns:
        X_test['Date'] = pd.to_datetime(X_test['Date'])
        X_test['Month'] = X_test['Date'].dt.month
        X_test['Season'] = X_test['Month'].apply(lambda m:
            'Winter' if m in [12,1,2] else
            'Spring' if m in [3,4] else
            'Pre-Monsoon' if m in [5,6] else
            'Monsoon' if m in [7,8] else
            'Post-Monsoon'
        )

    # 5. Predict and Calculate Error
    NON_FEATURE_COLS = ['Date', 'Division', 'Year', 'Month', 'Season']
    PLOTTING_ONLY_COLS = ['max_temp', 'total_rainfall']
    
    X_test_features = X_test.drop(columns=NON_FEATURE_COLS + PLOTTING_ONLY_COLS, errors='ignore')

    # Ensure feature order matches the model's expected features
    if hasattr(best_model, '_Booster') and best_model._Booster:
        expected_features = best_model._Booster.feature_name()
        X_test_features = X_test_features.reindex(columns=expected_features, fill_value=0.0)

    y_pred_log = best_model.predict(X_test_features)
    y_pred_original = inverse_transform(y_pred_log)

    X_test['y_actual'] = y_test_original.values
    X_test['y_pred'] = y_pred_original
    X_test['abs_error'] = np.abs(X_test['y_actual'] - X_test['y_pred'])

    print("Data loading, feature engineering, and prediction successful.")

except FileNotFoundError as e:
    print(f"File not found: {e}. Cannot run Figure 5.")
    X_test = pd.DataFrame() # Set empty to prevent errors
except Exception as e:
    print(f"General error during setup: {e}. Cannot run Figure 5.")
    X_test = pd.DataFrame()


# ------------------ FIGURE 5: Seasonal Error Heatmap (IEEE Color Corrected) ------------------

def figure5_seasonal_error():

    if 'Division' not in X_test.columns or 'Season' not in X_test.columns or X_test.empty:
        print("Skipping Figure 5 (Required columns missing or data is empty).")
        return

    # Calculate Mean Absolute Error (MAE) by Division and Season
    df_err = X_test.groupby(['Division','Season'])['abs_error'].mean().unstack(fill_value=0)

    # Ensure correct seasonal order
    season_order = ['Winter','Spring','Pre-Monsoon','Monsoon','Post-Monsoon']
    df_err = df_err.reindex(columns=[c for c in season_order if c in df_err.columns])
    
    # Sort divisions by max error for better visual clarity
    df_err['Max_Error'] = df_err.max(axis=1)
    df_err = df_err.sort_values(by='Max_Error', ascending=False).drop(columns=['Max_Error'])


    plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 1.3)) 
    
    # 🚨 UPDATED COLORMAP: Using 'YlOrRd' for high-contrast risk visualization (IEEE standard)
    sns.heatmap(df_err, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=0.4, cbar_kws={'label': 'MAE'})
    
    
    ax = plt.gca()

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.tick_params(axis='y', labelsize=8.5) 


   
    plt.ylabel('Division')

    plt.xlabel('Season')


    # Annotate top-10% errors: bold text
    flat_vals = df_err.values.flatten()

    nonzero_vals = flat_vals[flat_vals > 0] if flat_vals.size > 0 else flat_vals

    if nonzero_vals.size > 0:

        thr = np.percentile(nonzero_vals, 90)

        for txt in plt.gca().texts:

            try:

                val = float(txt.get_text())

                if val >= thr:

                    txt.set_weight('bold')

                    txt.set_color('black') 

            except Exception:

                pass


    save_figure('Figure5_Seasonal_Error_YlOrRd')

    plt.close()

# ------------------ EXECUTION ------------------

if not X_test.empty:
    figure5_seasonal_error()
    print("\nAttempted to generate Figure5_Seasonal_Error_YlOrRd.pdf and .png.")