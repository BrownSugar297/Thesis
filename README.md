# 🦟 Explainable Ensemble Models for Spatio-Temporal Dengue Outbreak Forecasting in Bangladesh

> **Bachelor's Thesis** — Faculty of Computer Science and Engineering, Patuakhali Science and Technology University (PSTU)  
> **Authors:** Ashikur Rahman Ashik (2002026) · Sadman Sakib (2002070)  
> **Supervisor:** Prof. Dr. Md. Abdul Masud, Dept. of CSIT  
> **Submitted:** February 2026

---

## 📌 Overview

This repository contains the full implementation of a spatio-temporal machine learning framework for forecasting dengue fever outbreaks across all **8 administrative divisions of Bangladesh**. The system integrates daily epidemiological surveillance data (DGHS) with high-resolution meteorological reanalysis (Open-Meteo & ERA5), and applies explainable AI (SHAP) to provide interpretable, actionable insights for public health decision-making.

**Best model:** LightGBM — R² = **0.900**, MAE = **8.927**, RMSE = **20.094**, RMSLE = **0.765**

---

## 🗂️ Repository Structure

```
dengue-forecasting-bangladesh/
│
├── data/
│   ├── raw/
│   │   ├── dghs_dengue_cases.csv          # Daily division-wise dengue cases (DGHS)
│   │   └── BD_Daily_Weather_2022-2025.csv  # Meteorological data (Open-Meteo/ERA5)
│   └── processed/
│       └── integrated_dataset.csv          # Cleaned & feature-engineered dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb              # Data cleaning & preprocessing
│   ├── 03_feature_engineering.ipynb        # Lag features, Fourier encoding, climate indices
│   ├── 04_model_training.ipynb             # Baseline & tuned model training
│   ├── 05_evaluation.ipynb                 # Performance metrics & comparison
│   └── 06_explainability.ipynb             # SHAP analysis & visualization
│
├── src/
│   ├── data_collection/
│   │   └── fetch_weather.py                # Open-Meteo API script for weather retrieval
│   ├── preprocessing/
│   │   ├── cleaning.py                     # Missing value handling, imputation
│   │   └── transforms.py                   # Log transformation, scaling
│   ├── features/
│   │   ├── lag_features.py                 # Autoregressive lag generation
│   │   ├── rolling_stats.py                # Rolling window statistics
│   │   ├── fourier_encoding.py             # Seasonal Fourier features
│   │   └── climate_indices.py              # Heat Index, Rain-Temp Index
│   ├── models/
│   │   ├── train_baseline.py               # Ridge, RF, SVR training
│   │   ├── train_boosting.py               # LightGBM, CatBoost, XGBoost
│   │   └── hyperparameter_tuning.py        # RandomizedSearchCV + early stopping
│   └── evaluation/
│       ├── metrics.py                      # MAE, RMSE, RMSLE, R² computation
│       └── explainability.py               # SHAP beeswarm & summary plots
│
├── outputs/
│   ├── figures/                            # All plots and visualizations
│   └── models/                             # Saved model files (.pkl / .json)
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 📊 Dataset

| Source | Type | Resolution | Period |
|---|---|---|---|
| DGHS Bangladesh | Daily dengue case counts | Division-level | Jan 2022 – Oct 2025 |
| Open-Meteo API | Temperature, rainfall, humidity | Daily (gridded) | Jan 2022 – Oct 2025 |
| ERA5 (ECMWF) | Meteorological reanalysis | Daily (gridded) | Jan 2022 – Oct 2025 |

**Total observations:** 11,088 (8 divisions × ~1,386 days)  
**Zero inflation:** 42.29% of daily records had zero cases  
**Target variable:** Daily dengue case count per division (log-transformed for modeling)

### Division-wise Summary (2022–2025)

| Division | Mean | Max | Std Dev |
|---|---|---|---|
| Dhaka | 217.07 | 1,703 | 358.25 |
| Chattogram | 57.34 | 547 | 96.32 |
| Barishal | 48.29 | 477 | 84.81 |
| Khulna | 37.24 | 436 | 79.12 |
| Rajshahi | 21.49 | 246 | 44.66 |
| Mymensingh | 10.49 | 105 | 19.21 |
| Rangpur | 5.79 | 88 | 14.05 |
| Sylhet | 1.53 | 30 | 3.97 |

---

## ⚙️ Methodology

### Pipeline Overview

```
Data Collection → Preprocessing → Feature Engineering → Train/Test Split → Model Training → HPO → Evaluation → SHAP Explainability
```

### 1. Preprocessing
- Division-wise monthly average imputation for July 18–23, 2024 (network blackout)
- Log transformation: `y' = log(1 + y)` to handle zero-inflation and right skew
- StandardScaler fit only on training data to prevent leakage

### 2. Feature Engineering

| Category | Features |
|---|---|
| Autoregressive lags | `Patients_lag3`, `lag7`, `lag14`; temp & rain lags |
| Rolling statistics | 7-day and 14-day rolling means for cases, temp, humidity, rain |
| Seasonal encoding | Fourier `sin`/`cos` terms, one-hot season labels |
| Climate indices | Heat Index (HI), Rain-Temperature Index (RTI) |
| Spatial features | One-hot encoding for all 8 divisions |
| Temporal | `day_of_year` |

### 3. Train-Test Split (Time-aware, no leakage)

- **Training:** January 2022 – December 2024
- **Testing:** January 2025 – October 2025

### 4. Models Evaluated

| Model | Type |
|---|---|
| Ridge Regression | Linear baseline |
| Random Forest | Bagging ensemble |
| Support Vector Regression (SVR) | Kernel-based |
| XGBoost | Gradient boosting |
| CatBoost | Gradient boosting |
| LightGBM ⭐ | Gradient boosting (best) |

### 5. Hyperparameter Optimization
- `RandomizedSearchCV` with 5-fold cross-validation, 100 iterations per model
- Early stopping (patience: 50–100 rounds) on a held-out 10% validation split
- RMSLE used as the optimization objective (log-domain)

---

## 📈 Results

### Baseline Performance

| Model | R² | MAE | RMSE | RMSLE |
|---|---|---|---|---|
| LightGBM | 0.899 | 9.148 | 20.236 | 0.773 |
| Random Forest | 0.896 | 9.173 | 20.476 | 0.778 |
| CatBoost | 0.872 | 9.852 | 22.739 | 0.777 |
| XGBoost | 0.849 | 10.759 | 24.708 | 0.812 |
| SVR | 0.689 | 14.389 | 35.478 | 0.957 |
| Ridge Regression | 0.640 | 17.206 | 38.203 | 1.097 |

### After Hyperparameter Tuning

| Model | R² | MAE | RMSE | RMSLE |
|---|---|---|---|---|
| **LightGBM** ⭐ | **0.900** | **8.927** | **20.094** | **0.765** |
| CatBoost | 0.898 | 9.026 | 20.310 | 0.768 |
| XGBoost | 0.891 | 9.366 | 20.998 | 0.772 |
| Random Forest | 0.888 | 8.984 | 21.234 | 0.769 |
| Ridge Regression | 0.637 | 17.275 | 38.322 | 1.097 |
| SVR | 0.383 | 18.702 | 49.997 | 0.910 |

---

## 🔍 Explainability (SHAP)

Top predictors identified via SHAP beeswarm analysis on the best LightGBM model:

1. `rolling_Patients_7d_mean` — 7-day rolling average of past cases (strongest signal)
2. `Patients_lag7` — dengue case count 7 days prior
3. `rolling_Patients_14d_mean` — 14-day rolling average
4. `Patients_lag14` — dengue case count 14 days prior
5. `fourier_cos` — seasonal encoding (cosine term)
6. `avg_temp_lag14` — 14-day lagged average temperature
7. `rolling_avg_temp_*` — rolling temperature averages

Key findings: **past case counts dominate**, followed by **Fourier seasonality** and **lagged temperature**, consistent with known dengue epidemiology.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/<your-username>/dengue-forecasting-bangladesh.git
cd dengue-forecasting-bangladesh
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate dengue-env
```

### Fetch Weather Data

```bash
python src/data_collection/fetch_weather.py
```

This retrieves daily meteorological data from Open-Meteo for all 8 divisions.

### Run the Full Pipeline

```bash
# Step 1: Preprocess & engineer features
python src/preprocessing/cleaning.py
python src/features/lag_features.py

# Step 2: Train and tune models
python src/models/train_boosting.py
python src/models/hyperparameter_tuning.py

# Step 3: Evaluate and explain
python src/evaluation/metrics.py
python src/evaluation/explainability.py
```

Or simply run the Jupyter notebooks in order under `notebooks/`.

---

## 📦 Requirements

```
lightgbm>=4.0
catboost>=1.2
xgboost>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
geopandas>=0.14
requests>=2.31
jupyter
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📁 Key Outputs

| File | Description |
|---|---|
| `outputs/figures/shap_beeswarm.png` | SHAP feature importance plot |
| `outputs/figures/observed_vs_predicted.png` | Temporal fit comparison |
| `outputs/figures/division_mae_heatmap.png` | Division × Season MAE grid |
| `outputs/figures/choropleth_2022_2025.png` | Spatial burden map |
| `outputs/figures/climate_trends.png` | Stacked climate & case trends |
| `outputs/models/lightgbm_tuned.pkl` | Best tuned LightGBM model |

---

## 🌍 Divisions Covered

Barishal · Chattogram · Dhaka · Khulna · Mymensingh · Rajshahi · Rangpur · Sylhet

---

## ⚠️ Limitations

- Division-level aggregation suppresses urban-rural heterogeneity
- Surveillance data may contain underreporting or delays
- No entomological (mosquito vector) or serotype data included
- The July 2024 data gap (network blackout) was imputed using monthly averages
- SHAP provides correlation-based explanations; causal inference requires further work

---

## 🔮 Future Work

- Sub-district or upazila-level spatial resolution
- Integration of real-time data streams
- Long-term forecasting under climate change scenarios
- Cross-regional adaptation (Southeast Asia, Latin America)
- Causal inference modeling for climatic–epidemiological relationships

---

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@thesis{ashik_sakib_2026_dengue,
  title     = {Explainable Ensemble Models for Spatio-Temporal Dengue Outbreak Forecasting in Bangladesh},
  author    = {Ashikur Rahman Ashik and Md. Sadman Sakib},
  school    = {Patuakhali Science and Technology University},
  year      = {2026},
  type      = {B.Sc. Thesis},
  address   = {Patuakhali, Bangladesh}
}
```

---

## 📬 Contact

- **Ashikur Rahman Ashik** — Student ID: 2002026, PSTU
- **Md. Sadman Sakib** — Student ID: 2002070, PSTU
- **Supervisor:** Dr. Md. Abdul Masud — Professor, Dept. of CSIT, PSTU

---

## 🙏 Acknowledgements

Sincere gratitude to **Prof. Dr. Md. Abdul Masud** for his guidance, to the **DGHS Bangladesh** for providing surveillance data, and to the **Open-Meteo** and **ERA5/ECMWF** platforms for open meteorological data access.

---

<p align="center">Faculty of Computer Science and Engineering · Patuakhali Science and Technology University · 2026</p>

