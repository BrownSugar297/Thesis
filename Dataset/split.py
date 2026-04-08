import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os 
import joblib # CRITICAL ADDITION: Import joblib for saving the scaler

# --- 1. Load Data ---
# Note: Ensure this path is correct on your local machine.
file_name = "E:/Regg Thesis/Dataset/Final_Dataset.csv"
df = pd.read_csv(file_name)

# --- 2. Target Transformation & Split Setup ---
# Target Transformation: log(1 + Patients)
df['log_Patients'] = np.log1p(df['Patients']) 
SPLIT_DATE = '2025-01-01'

# Targets
y = df['log_Patients']
y_original = df['Patients']

# --- 3. Feature Identification ---
# Ensure these lists are consistent throughout the project.
continuous_features = [
    'max_temp', 'min_temp', 'rainfall', 'humidity', 'avg_temp', 'temp_range', 'humidity_norm',
    'Patients_lag3', 'Patients_lag7', 'Patients_lag14', 'avg_temp_lag3', 'avg_temp_lag7', 'avg_temp_lag14',
    'humidity_norm_lag3', 'humidity_norm_lag7', 'humidity_norm_lag14',
    'rolling_Patients_7d_mean', 'rolling_Patients_14d_mean', 'rolling_avg_temp_7d_mean', 'rolling_avg_temp_14d_mean',
    'rolling_humidity_norm_7d_mean', 'rolling_humidity_norm_14d_mean', 'temp_humidity_index', 'rain_temp_index', 'rain_humidity',
    'day_of_year', 'fourier_sin', 'fourier_cos'
]

categorical_features = [
    'rain_binary', 'rain_binary_lag3', 'rain_binary_lag7', 'rain_binary_lag14',
    'rolling_rain_binary_7d_mean', 'rolling_rain_binary_14d_mean',
    'season_Monsoon', 'season_Post-Monsoon', 'season_Pre-Monsoon', 'season_Winter',
    'division_Barishal', 'division_Chattogram', 'division_Dhaka', 'division_Khulna',
    'division_Mymensingh', 'division_Rajshahi', 'division_Rangpur', 'division_Sylhet'
]

all_features = continuous_features + categorical_features
X = df[all_features].copy()

# Ensure categorical/binary features are integers
for col in X.select_dtypes(include=['bool', 'object']).columns:
    if col in categorical_features:
        X[col] = X[col].astype(int)

# --- 4. Train/Test Split (Time-Series) ---
df['date'] = pd.to_datetime(df['date'])

X_train = X[df['date'] < SPLIT_DATE].copy()
X_test = X[df['date'] >= SPLIT_DATE].copy()
y_train = y[df['date'] < SPLIT_DATE].copy()
y_test = y[df['date'] >= SPLIT_DATE].copy()
y_test_original = y_original[df['date'] >= SPLIT_DATE].copy() 

# Separate continuous and categorical subsets for scaling
X_train_cont = X_train[continuous_features]
X_test_cont = X_test[continuous_features]
X_train_cat = X_train[categorical_features]
X_test_cat = X_test[categorical_features]

# --- 5. Scaling (Fit on Train, Transform Both) ---
scaler = StandardScaler()
scaler.fit(X_train_cont) # Fit on TRAIN ONLY

X_train_cont_scaled = pd.DataFrame(scaler.transform(X_train_cont), columns=continuous_features, index=X_train_cont.index)
X_test_cont_scaled = pd.DataFrame(scaler.transform(X_test_cont), columns=continuous_features, index=X_test_cont.index)

# --- 6. Final Assembly ---
X_train_final = pd.concat([X_train_cont_scaled, X_train_cat], axis=1)
X_test_final = pd.concat([X_test_cont_scaled, X_test_cat], axis=1)

X_train_final = X_train_final[all_features]
X_test_final = X_test_final[all_features]

# Reset indices for clean CSV export
X_train_final.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test_final.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
y_test_original.reset_index(drop=True, inplace=True)


# --- 7. Save Final Datasets to CSV and Save the Scaler ---
TRAIN_DIR = r"E:\Regg Thesis\Model Train"
TEST_DIR = r"E:\Regg Thesis\Model Test"

# It's recommended to uncomment these lines if the folders might not exist
# os.makedirs(TRAIN_DIR, exist_ok=True)
# os.makedirs(TEST_DIR, exist_ok=True)

# Training Files
X_train_final.to_csv(os.path.join(TRAIN_DIR, "X_train_scaled.csv"), index=False)
y_train.to_csv(os.path.join(TRAIN_DIR, "y_train_log.csv"), index=False, header=['log_Patients']) 

# Testing Files
X_test_final.to_csv(os.path.join(TEST_DIR, "X_test_scaled.csv"), index=False)
y_test_original.to_csv(os.path.join(TEST_DIR, "y_test_original_patients.csv"), index=False, header=['Patients'])
y_test.to_csv(os.path.join(TEST_DIR, "y_test_log.csv"), index=False, header=['log_Patients']) 

# CRITICAL ADDITION: Save the fitted StandardScaler object for interpretation/deployment
joblib.dump(scaler, os.path.join(TRAIN_DIR, 'scaler_for_X_cont.pkl'))

print("Data preparation and file saving done.")
print(f"Train files saved to: {TRAIN_DIR}")
print(f"Test files saved to: {TEST_DIR}")
print(f"Scaler saved as: {os.path.join(TRAIN_DIR, 'scaler_for_X_cont.pkl')}")