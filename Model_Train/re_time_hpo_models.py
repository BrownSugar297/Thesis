import pandas as pd
import joblib
import os
import time

# --- Configuration (Must Match Your Setup) ---
# NOTE: Ensure these paths are exactly correct for your system!
TRAIN_DIR = r"E:\Regg Thesis\Model_Train"      
MODELS_DIR = r"E:\Regg Thesis\Model_Tuned"     
RESULTS_DIR = r"E:\Regg Thesis\HPO_Results"    

# List of models to re-time (the ones already completed)
MODELS_TO_RETIME = ['lightgbm', 'catboost']

# --- Data Loading ---
print("Loading data for re-timing...")
try:
    X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train_log.csv")).squeeze() 
except FileNotFoundError:
    print("Error: Could not find training data files. Check TRAIN_DIR path.")
    exit()

# --- Re-Timing and Updating Logs ---
training_times = {}

for model_name_short in MODELS_TO_RETIME:
    print(f"\nProcessing {model_name_short}...")
    
    model_pkl_path = os.path.join(MODELS_DIR, f'{model_name_short}_tuned.pkl')
    results_csv_path = os.path.join(RESULTS_DIR, f'{model_name_short}_hpo_results.csv')
    
    try:
        # 1. Load the best estimator
        best_model = joblib.load(model_pkl_path)
        print(f"Model {model_name_short} loaded. Re-timing final fit...")

        # 2. Time the final training on the full dataset
        start_time = time.time()
        
        # We use the .fit() method here to accurately measure the training time.
        best_model.fit(X_train, y_train)
        
        final_fit_time = time.time() - start_time
        training_times[model_name_short] = final_fit_time
        print(f"Final fit time: {final_fit_time:.4f} seconds.")
        
        # 3. Load the results CSV and update the final training time
        results_df = pd.read_csv(results_csv_path)
        
        # Add the new metric to the first row of the HPO results log
        results_df['final_model_training_time_s'] = final_fit_time
        
        # Save the updated CSV
        results_df.to_csv(results_csv_path, index=False)
        print(f"Log updated successfully: {results_csv_path}")

    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}. Check your directory paths.")
    except Exception as e:
        print(f"An unexpected error occurred for {model_name_short}: {e}")

print("\n--- Re-timing Complete ---")
print("Results:")
for model, t in training_times.items():
    print(f"{model.capitalize()} Training Time: {t:.4f} seconds")

print("\nYour HPO logs are now fully ready for the Final Results Table (Table II).")