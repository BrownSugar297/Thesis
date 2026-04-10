<div align="center">

<br/>

```
██████╗ ███████╗███╗   ██╗ ██████╗ ██╗   ██╗███████╗
██╔══██╗██╔════╝████╗  ██║██╔════╝ ██║   ██║██╔════╝
██║  ██║█████╗  ██╔██╗ ██║██║  ███╗██║   ██║█████╗  
██║  ██║██╔══╝  ██║╚██╗██║██║   ██║██║   ██║██╔══╝  
██████╔╝███████╗██║ ╚████║╚██████╔╝╚██████╔╝███████╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚══════╝
        F O R E C A S T I N G · B A N G L A D E S H
```

# Explainable Ensemble Models for Spatio-Temporal Dengue Outbreak Forecasting in Bangladesh

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Best_Model-2ECC71?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-E74C3C?style=for-the-badge)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-Academic-F39C12?style=for-the-badge)](#)
[![Thesis](https://img.shields.io/badge/B.Sc._Thesis-PSTU_2026-8E44AD?style=for-the-badge)](#)

<br/>

> **A spatio-temporal machine learning framework for daily dengue outbreak prediction  
> across all 8 administrative divisions of Bangladesh — powered by LightGBM + SHAP.**

<br/>

| 🎯 R² Score | 📉 MAE | 📊 RMSE | 📐 RMSLE |
|:-----------:|:------:|:-------:|:--------:|
| **0.900** | **8.927** | **20.094** | **0.765** |

<sub>Best model: LightGBM (tuned) · Test period: January – October 2025</sub>

<br/>

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Explainability (SHAP)](#-explainability-shap)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [Key Outputs](#-key-outputs)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Dengue fever poses a severe and escalating public health challenge in Bangladesh, driven by Aedes mosquito proliferation under warm, wet monsoon conditions. Classical compartmental models (SIR, SEIR) and linear regression approaches fail to capture the **non-linear, time-lagged, spatially heterogeneous** dynamics of dengue transmission at national scale.

This research presents a fully integrated, **leakage-aware machine learning pipeline** that:

- Fuses **daily epidemiological surveillance** (DGHS) with **high-resolution meteorological reanalysis** (Open-Meteo & ERA5)
- Applies **advanced feature engineering** — autoregressive lags, rolling statistics, Fourier seasonality, and composite climate indices
- Benchmarks **6 regression models** under strict chronological validation
- Delivers **SHAP-based explainability** for transparent, actionable public health insights

The framework is designed as a **decision-support tool** for dengue early warning and healthcare resource allocation across Bangladesh's 8 administrative divisions.

---

## 🏆 Key Contributions

```
┌─────────────────────────────────────────────────────────────────┐
│  1.  Integrated Dataset      Daily dengue + climate data for    │
│                              all 8 divisions, 2022–2025         │
│                                                                 │
│  2.  Leakage-Aware Pipeline  Strict chronological split,        │
│                              training-only scaling              │
│                                                                 │
│  3.  Model Benchmarking      6 models compared baseline → HPO   │
│                              LightGBM identified as best        │
│                                                                 │
│  4.  SHAP Interpretability   Global + local explanations        │
│                              aligned with epidemiology          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### Sources

| Source | Description | Resolution | Coverage |
|--------|-------------|------------|----------|
| [DGHS Bangladesh](https://dghs.gov.bd) | Daily division-wise dengue case counts | Division-level | Jan 2022 – Oct 2025 |
| [Open-Meteo API](https://open-meteo.com) | Temperature, rainfall, relative humidity | Daily (gridded) | Jan 2022 – Oct 2025 |
| [ERA5 / ECMWF](https://cds.climate.copernicus.eu) | Meteorological reanalysis | Daily (gridded) | Jan 2022 – Oct 2025 |

### Dataset Statistics

```
Total Observations  :  11,088   (8 divisions × ~1,386 days)
Zero Inflation      :  42.29%   daily records with zero cases
Target Variable     :  Daily dengue case count per division
Transformation      :  log(1 + y) for training; exp(ŷ) − 1 for evaluation
```

### Division-Wise Case Summary (2022–2025)

| Division | Mean Cases/Day | Max Cases/Day | Std Dev |
|----------|:--------------:|:-------------:|:-------:|
| 🔴 Dhaka | 217.07 | 1,703 | 358.25 |
| 🟠 Chattogram | 57.34 | 547 | 96.32 |
| 🟠 Barishal | 48.29 | 477 | 84.81 |
| 🟡 Khulna | 37.24 | 436 | 79.12 |
| 🟡 Rajshahi | 21.49 | 246 | 44.66 |
| 🟢 Mymensingh | 10.49 | 105 | 19.21 |
| 🟢 Rangpur | 5.79 | 88 | 14.05 |
| 🟢 Sylhet | 1.53 | 30 | 3.97 |

> **Note:** The dataset contains a short gap in July 2024 due to a nationwide network outage. Missing values were imputed using division-wise monthly averages to preserve regional variation without introducing look-ahead bias.

---

## ⚙️ Methodology

### End-to-End Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  Data        │───▶│  Preprocess  │───▶│  Feature Engineering │
│  Collection  │    │  & Clean     │    │  (Lags, Fourier, HI) │
└──────────────┘    └──────────────┘    └──────────┬───────────┘
                                                   │
                         ┌─────────────────────────▼────────────┐
                         │   Chronological Train / Test Split   │
                         │   Train: Jan 2022 – Dec 2024         │
                         │   Test:  Jan 2025 – Oct 2025         │
                         └─────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │           Model Training & HPO               │
                    │  Ridge · RF · SVR · XGBoost · CatBoost ·    │
                    │  LightGBM  │  RandomizedSearchCV (100 iter)  │
                    └──────────────────────┬───────────────────────┘
                                           │
                         ┌─────────────────▼──────────────────┐
                         │   Evaluation: MAE · RMSE ·         │
                         │   RMSLE · R²  +  SHAP Analysis     │
                         └────────────────────────────────────┘
```

### 1. Preprocessing

- **Imputation:** July 2024 gap filled with division-wise monthly averages (respects temporal order)
- **Short gaps:** Forward-fill only on past values — no look-ahead leakage
- **Scaling:** `StandardScaler` fit exclusively on training data; applied to test data
- **Target transform:** `y' = log(1 + y)` to handle zero-inflation and right skew

### 2. Feature Engineering

| Category | Feature(s) | Description |
|----------|-----------|-------------|
| **Autoregressive Lags** | `lag3`, `lag7`, `lag14` | Past case counts at 3, 7, 14-day offsets |
| **Rolling Statistics** | `rolling_7d_mean`, `rolling_14d_mean` | Smoothed outbreak momentum |
| **Seasonal Encoding** | `fourier_sin`, `fourier_cos` | Harmonic annual seasonality terms |
| **Season Labels** | One-hot | Winter / Spring / Pre-Monsoon / Monsoon / Post-Monsoon |
| **Climate Indices** | Heat Index (HI), RTI | Non-linear composite climate effects |
| **Meteorological Lags** | `avg_temp_lag14`, `rain_lag*` | Lagged temperature and rainfall |
| **Spatial Features** | Division one-hot (×8) | Region-specific effects in a global model |
| **Temporal** | `day_of_year` | Smooth within-year position |

**Formulas:**

$$y_t^{(\text{lag}_n)} = y_{t-n}, \quad n \in \{3, 7, 14\}$$

$$\text{mean}_t^{(w)} = \frac{1}{w} \sum_{i=t-w+1}^{t} y_i, \quad w \in \{7, 14\}$$

$$S_t = \sum_{k=1}^{K} \left[ A_k \cos\!\left(\frac{2\pi k t}{T}\right) + B_k \sin\!\left(\frac{2\pi k t}{T}\right) \right]$$

$$\text{RTI} = \text{Rainfall} \times T$$

### 3. Models Evaluated

| Model | Category |
|-------|----------|
| Ridge Regression | Linear baseline |
| Random Forest | Bagging ensemble |
| Support Vector Regression (SVR) | Kernel-based |
| XGBoost | Gradient boosting |
| CatBoost | Gradient boosting |
| **LightGBM ⭐** | **Gradient boosting (best)** |

### 4. Hyperparameter Optimization

- `RandomizedSearchCV` · **100 iterations** · **5-fold time-series CV**
- **Optimization objective:** RMSLE (log domain)
- **Early stopping:** 50–100 rounds patience on a held-out 10% validation split
- Final models re-fitted on the full training set before test evaluation

---

## 📈 Results

### Baseline Performance (Default Hyperparameters)

| Model | R² | MAE | RMSE | RMSLE |
|-------|:--:|:---:|:----:|:-----:|
| LightGBM | 0.899 | 9.148 | 20.236 | 0.773 |
| Random Forest | 0.896 | 9.173 | 20.476 | 0.778 |
| CatBoost | 0.872 | 9.852 | 22.739 | 0.777 |
| XGBoost | 0.849 | 10.759 | 24.708 | 0.812 |
| SVR | 0.689 | 14.389 | 35.478 | 0.957 |
| Ridge Regression | 0.640 | 17.206 | 38.203 | 1.097 |

### After Hyperparameter Tuning

| Model | R² | MAE | RMSE | RMSLE |
|-------|:--:|:---:|:----:|:-----:|
| **LightGBM ⭐** | **0.900** | **8.927** | **20.094** | **0.765** |
| CatBoost | 0.898 | 9.026 | 20.310 | 0.768 |
| XGBoost | 0.891 | 9.366 | 20.998 | 0.772 |
| Random Forest | 0.888 | 8.984 | 21.234 | 0.769 |
| Ridge Regression | 0.637 | 17.275 | 38.322 | 1.097 |
| SVR | 0.383 | 18.702 | 49.997 | 0.910 |

> **Key finding:** Hyperparameter tuning substantially improves gradient-boosting models. Ridge and SVR show limited or negative gains, confirming their unsuitability for this high-dimensional, non-linear problem. LightGBM achieves the highest R² and the lowest MAE, RMSE and RMSLE across all candidates.

### Temporal Forecasting Performance (2025 Test Period)

The LightGBM model closely tracks outbreak dynamics throughout 2025, capturing both seasonal peaks and troughs. The largest absolute errors occur during extreme monsoon and post-monsoon surges — where rapid incidence fluctuations push the limits of any data-driven model.

### Division × Season Error Analysis

Prediction errors vary significantly by region and season:

- **Highest MAE:** Dhaka during Monsoon (30.3) and Post-Monsoon (72.0) — driven by extreme, rapidly evolving urban transmission
- **Lowest MAE:** Sylhet and Rangpur throughout all seasons — low and stable incidence makes forecasting tractable
- **Barishal:** Notable spike in Pre-Monsoon (27.4) reflecting early seasonal surges

---

## 🔍 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values were computed for the optimized LightGBM model to provide both global feature importance and local prediction explanations.

### Top Predictors (Ranked by Mean |SHAP|)

```
Rank  Feature                      Interpretation
────  ───────────────────────────  ─────────────────────────────────────────
 1    rolling_Patients_7d_mean     7-day rolling avg of past cases (strongest)
 2    Patients_lag7                Dengue case count 7 days prior
 3    rolling_Patients_14d_mean    14-day rolling average of past cases
 4    Patients_lag14               Dengue case count 14 days prior
 5    fourier_cos                  Seasonal cosine encoding
 6    avg_temp_lag14               14-day lagged average temperature
 7    rolling_avg_temp_14d_mean    14-day rolling mean temperature
 8    rolling_avg_temp_7d_mean     7-day rolling mean temperature
 9    fourier_sin                  Seasonal sine encoding
10    Patients_lag3                Dengue case count 3 days prior
```

### Key Epidemiological Findings

1. **Past case counts dominate** — autoregressive lags and rolling averages are by far the strongest predictors, consistent with the self-sustaining dynamics of dengue transmission
2. **Seasonal structure matters** — Fourier encodings capture annual outbreak timing, reflecting the well-known monsoon/post-monsoon peak pattern
3. **Temperature plays a secondary but meaningful role** — lagged and rolling temperature features confirm the known association between warm conditions and Aedes mosquito activity
4. **Epidemiological plausibility confirmed** — SHAP explanations align with established dengue transmission science, validating the model as a trustworthy decision-support tool

---

## 📁 Repository Structure

```
dengue-forecasting-bangladesh/
│
├── 📂 data/
│   ├── raw/
│   │   ├── dghs_dengue_cases.csv           # Daily division-wise dengue cases (DGHS)
│   │   └── BD_Daily_Weather_2022-2025.csv  # Meteorological data (Open-Meteo/ERA5)
│   └── processed/
│       └── integrated_dataset.csv          # Cleaned & feature-engineered dataset
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb                        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb              # Data cleaning & preprocessing
│   ├── 03_feature_engineering.ipynb        # Lag features, Fourier encoding, climate indices
│   ├── 04_model_training.ipynb             # Baseline & tuned model training
│   ├── 05_evaluation.ipynb                 # Performance metrics & comparison
│   └── 06_explainability.ipynb             # SHAP analysis & visualization
│
├── 📦 src/
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
├── 📊 outputs/
│   ├── figures/                            # All plots and visualizations
│   └── models/                             # Saved model files (.pkl / .json)
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` or `conda`

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/dengue-forecasting-bangladesh.git
cd dengue-forecasting-bangladesh
```

### 2. Set Up Environment

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using conda:**
```bash
conda env create -f environment.yml
conda activate dengue-env
```

### 3. Fetch Meteorological Data

```bash
python src/data_collection/fetch_weather.py
```

Retrieves daily temperature, rainfall, and humidity data from Open-Meteo for all 8 divisions using centroid coordinates.

### 4. Run the Full Pipeline

```bash
# ── Step 1: Preprocess & engineer features ──────────────────────
python src/preprocessing/cleaning.py
python src/features/lag_features.py

# ── Step 2: Train and tune models ───────────────────────────────
python src/models/train_boosting.py
python src/models/hyperparameter_tuning.py

# ── Step 3: Evaluate and explain ────────────────────────────────
python src/evaluation/metrics.py
python src/evaluation/explainability.py
```

> **Alternative:** Run the Jupyter notebooks in order under `notebooks/` for an interactive, step-by-step walkthrough.

---

## 📦 Requirements

```
lightgbm      >= 4.0
catboost      >= 1.2
xgboost       >= 2.0
scikit-learn  >= 1.3
pandas        >= 2.0
numpy         >= 1.24
shap          >= 0.44
matplotlib    >= 3.7
seaborn       >= 0.12
geopandas     >= 0.14
requests      >= 2.31
jupyter
```

---

## 📤 Key Outputs

| Output File | Description |
|-------------|-------------|
| `outputs/figures/shap_beeswarm.png` | SHAP global feature importance (beeswarm) |
| `outputs/figures/observed_vs_predicted.png` | Temporal fit comparison — 2025 test period |
| `outputs/figures/division_mae_heatmap.png` | Division × Season MAE heatmap |
| `outputs/figures/choropleth_2022_2025.png` | Spatial burden map across Bangladesh |
| `outputs/figures/climate_trends.png` | Stacked dengue, temperature & rainfall trends |
| `outputs/models/lightgbm_tuned.pkl` | Saved best LightGBM model (tuned) |

---

## 🌍 Divisions Covered

| | | | |
|--|--|--|--|
| 🟦 Barishal | 🟦 Chattogram | 🟦 Dhaka | 🟦 Khulna |
| 🟦 Mymensingh | 🟦 Rajshahi | 🟦 Rangpur | 🟦 Sylhet |

---

## ⚠️ Limitations

- **Spatial resolution:** Division-level aggregation suppresses urban-rural heterogeneity and sub-district variation
- **Surveillance bias:** Official DGHS data may contain underreporting or reporting delays
- **Missing entomological data:** No mosquito vector density or serotype information included
- **Data gap:** July 2024 blackout period was imputed using monthly averages — a necessary approximation
- **Correlational explanations:** SHAP quantifies feature contributions, not causal relationships; causal inference requires further modeling

---

## 🔮 Future Work

- [ ] **Sub-district forecasting** — upazila-level spatial granularity for targeted interventions
- [ ] **Real-time data integration** — live DGHS surveillance and operational meteorological feeds
- [ ] **Human mobility features** — mobile network or anonymized movement data for urban spread modeling
- [ ] **Climate change scenarios** — long-horizon projections under CMIP6 warming pathways
- [ ] **Causal inference** — structural models to disentangle climate vs. epidemiological drivers
- [ ] **Cross-regional adaptation** — transfer learning to Southeast Asia and Latin America contexts

---

## 📄 Citation

If you use this code, dataset, or methodology in your research, please cite:

```bibtex
@thesis{ashik_sakib_2026_dengue,
  title     = {Explainable Ensemble Models for Spatio-Temporal Dengue Outbreak
               Forecasting in Bangladesh},
  author    = {Ashikur Rahman Ashik and Md. Sadman Sakib},
  school    = {Patuakhali Science and Technology University},
  year      = {2026},
  type      = {B.Sc. Thesis},
  address   = {Patuakhali, Bangladesh},
  department= {Computer Science and Engineering}
}
```

---

## 🙏 Acknowledgements

Sincere gratitude to:

- **Prof. Dr. Md. Abdul Masud** — Professor, Department of CSIT, PSTU, for his invaluable supervision and guidance throughout this research
- **Directorate General of Health Services (DGHS), Bangladesh** — for providing the official dengue surveillance data
- **Open-Meteo** and **ERA5 / ECMWF** — for open-access, high-resolution meteorological datasets that made this study possible

---

<div align="center">

<br/>

**Faculty of Computer Science and Engineering**  
Patuakhali Science and Technology University (PSTU)  
Patuakhali, Bangladesh · 2026

<br/>

---

<sub>Developed by <strong>Ashikur Rahman Ashik</strong> · Student ID: 2002026 · PSTU</sub>

<br/>

</div>
