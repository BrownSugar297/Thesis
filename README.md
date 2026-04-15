<div align="center">

# 🦟 Dengue Outbreak Forecasting — Bangladesh

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Best_Model-2ECC71?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6B35?style=for-the-badge)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-FFCC00?style=for-the-badge&logoColor=black)](https://catboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-E74C3C?style=for-the-badge)](https://shap.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-HPO-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Thesis](https://img.shields.io/badge/B.Sc._Thesis-PSTU_2026-8E44AD?style=for-the-badge)](#)

<br/>

> **Explainable Ensemble Models for Spatio-Temporal Dengue Outbreak Forecasting in Bangladesh**
> A leakage-aware ML pipeline combining daily epidemiological surveillance with climate reanalysis data
> across all 8 administrative divisions — powered by LightGBM + SHAP.

<br/>

| 🎯 R² | 📉 MAE | 📊 RMSE | 📐 RMSLE |
|:-----:|:------:|:-------:|:--------:|
| **0.900** | **8.927** | **20.094** | **0.765** |

<sub>Best model: LightGBM (tuned) · Test period: January – October 2025</sub>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture & Workflow](#-architecture--workflow)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Explainability (SHAP)](#-explainability-shap)
- [Technologies Used](#-technologies-used)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🔍 Overview

Dengue fever is a rapidly escalating public health crisis in Bangladesh, driven by Aedes mosquito proliferation under monsoon climate conditions. Traditional compartmental models (SIR/SEIR) and linear approaches fail to capture the **non-linear, time-lagged, and spatially heterogeneous** dynamics of dengue at national scale.

This project delivers a fully integrated, production-ready ML pipeline that fuses **daily epidemiological surveillance** (DGHS) with **high-resolution meteorological reanalysis** (Open-Meteo & ERA5) to forecast daily dengue case counts across all 8 divisions of Bangladesh — with transparent SHAP-based explanations aligned to established epidemiological science.

**Why it matters:** Early, explainable dengue forecasts enable public health authorities to pre-position medical resources, issue timely advisories, and target vector control interventions — potentially saving lives and reducing healthcare system strain.

---

## 🚀 Features

- **Leakage-free pipeline** — strict chronological train/test split with training-only `StandardScaler` fitting
- **Advanced feature engineering** — autoregressive lags, rolling statistics, Fourier seasonality, and composite climate indices (Heat Index, Rain-Temp Index)
- **6-model benchmark** — Ridge, Random Forest, SVR, XGBoost, CatBoost, LightGBM with baseline → tuned comparison
- **Rigorous HPO** — `RandomizedSearchCV` (100 iterations) with 5-fold time-series CV and early stopping
- **Full SHAP explainability** — global beeswarm plots and local prediction explanations
- **Division-level spatial analysis** — per-division, per-season error breakdown across all 8 administrative regions
- **Reproducible** — modular `src/` pipeline with saved model artifacts (`.pkl`, `.json`)

---

## 🧠 Architecture & Workflow

### Visual Overview

```mermaid
flowchart TD
    A(["Data Acquisition and Initial Cleaning"])
    --> B["Feature Engineering and Leakage Control"]
    --> C{"Chronological Time Series Split"}

    C -->|"Jan 2022 to Dec 2024"| D["Training Set"]
    C -->|"Jan 2025 to Oct 2025"| E["Test Set"]

    D --> F
    E --> H

    subgraph DATA_SPLIT ["DATA SPLIT"]
        D
        E
    end

    subgraph MODEL_TRAINING ["MODEL TRAINING"]
        F["Baseline Model Training"]
        --> G["Hyperparameter Optimization"]
    end

    G --> H["Best Optimized Model"]
    H --> I(["Evaluation"])
```

### Step-by-Step Pipeline

```
[Data Collection] ──▶ [Preprocessing] ──▶ [Feature Engineering]
      │                     │                      │
  DGHS cases          Imputation             Lag-3/7/14
  Open-Meteo          FWD-fill only          Rolling 7d/14d
  ERA5 reanalysis     log(1+y) target        Fourier sin/cos
                      Train-only scale       Heat Index, RTI
                                             Division one-hot
                              │
                    [Chronological Split]
                    Train: 2022 – 2024
                    Test:  2025
                              │
             ┌────────────────┴───────────────┐
             │    RandomizedSearchCV HPO      │
             │  5-fold TS-CV · 100 iter       │
             │  Early stopping (50–100 rounds)│
             └────────────────────────────────┘
                              │
              ┌───────────────┴──────────────┐
              │                              │
        [Baseline Models]            [Boosting Models]
        Ridge · RF · SVR             XGBoost · CatBoost
                                      LightGBM ⭐
              └───────────────┬──────────────┘
                              │
                     [Evaluation & SHAP]
                     MAE · RMSE · RMSLE · R²
                     Beeswarm · Division heatmap
```

---

## 📂 Project Structure

```
Regg Thesis/
│
├── 📂 Dataset/
│   ├── Dataset.csv                          # Raw integrated dataset
│   ├── Final_Dataset.csv                    # Cleaned & feature-engineered dataset
│   ├── split.py                             # Train/test split script
│   ├── correlation.py                       # Correlation analysis
│   ├── dataset_behavior.py                  # Dataset behavior analysis
│   └── Map_full.py                          # Spatial mapping script
│
├── 📂 Model_Train/
│   ├── 01_baseline_train_save.py            # Baseline model training & saving
│   ├── re_time_hpo_models.py                # Retraining HPO models
│   ├── hpo_ridge.py                         # Ridge HPO script
│   ├── hpo_randomforest.py                  # Random Forest HPO script
│   ├── hpo_svr.py                           # SVR HPO script
│   ├── hpo_xgboost.py                       # XGBoost HPO script
│   ├── hpo_catboost.py                      # CatBoost HPO script
│   ├── hpo_lightgbm.py                      # LightGBM HPO script
│   ├── scaler_for_X_cont.pkl                # Fitted StandardScaler
│   ├── X_train_scaled.csv                   # Scaled training features
│   └── y_train_log.csv                      # Log-transformed training target
│
├── 📂 Model_Trained/
│   ├── ridge_default.pkl                    # Trained baseline Ridge model
│   ├── randomforest_default.pkl             # Trained baseline Random Forest model
│   ├── svr_default.pkl                      # Trained baseline SVR model
│   ├── xgboost_default.pkl                  # Trained baseline XGBoost model
│   ├── catboost_default.pkl                 # Trained baseline CatBoost model
│   ├── lightgbm_default.pkl                 # Trained baseline LightGBM model
│   └── baseline_training_times_v4.csv       # Baseline model training times
│
├── � Model_Tuned/
│   ├── ridge_tuned.pkl                      # Tuned Ridge model
│   ├── randomforest_tuned.pkl               # Tuned Random Forest model
│   ├── svr_tuned.pkl                        # Tuned SVR model
│   ├── xgboost_tuned_full.pkl               # Tuned XGBoost model
│   ├── catboost_tuned.pkl                   # Tuned CatBoost model
│   └── lightgbm_tuned.pkl                   # Tuned LightGBM model (⭐ Best)
│
├── 📂 Model_Test/
│   ├── X_test_scaled.csv                    # Scaled test features
│   ├── y_test_log.csv                       # Log-transformed test target
│   ├── y_test_original_patients.csv         # Original (non-transformed) test target
│   ├── final_evaluation_table2.py           # Final evaluation script
│   └── python 01b_baseline_evaluate.py      # Baseline evaluation script
│
├── 📂 HPO_Results/
│   ├── ridge_hpo_results.csv                # Ridge HPO search results
│   ├── randomforest_hpo_results.csv         # Random Forest HPO results
│   ├── svr_hpo_results.csv                  # SVR HPO results
│   ├── xgboost_hpo_results.csv              # XGBoost HPO results
│   ├── catboost_hpo_results.csv             # CatBoost HPO results
│   └── lightgbm_hpo_results.csv             # LightGBM HPO results
│
├── 📂 Results/
│   ├── baseline_results_table1_final_v4.csv # Baseline model performance metrics
│   └── Table_Tuned_Models_Performance.csv   # Tuned model performance metrics
│
├── 📂 Fig/
│   ├── SHAP_Beeswarm.pdf                    # SHAP global feature importance
│   ├── Figure7_Choropleth_Map_Overall_YlOrRd_2022_2025.pdf  # Spatial map
│   ├── 2025_SpatioTemporal_Prediction.pdf   # Temporal prediction (2025)
│   ├── _TimeSeries_Fit_and_Error_FINAL_V_FixedScaling.pdf   # Fit & error plot
│   ├── _Seasonal_Error_YlOrRd.pdf           # Seasonal error heatmap
│   ├── Correlation_Heatmap_l.pdf            # Feature correlation heatmap
│   ├── Stacked_Climate_Trend_.pdf           # Climate trend visualization
│   ├── dataset behaviour_Raw_Distribution_LogScaled.pdf     # Data distribution
│   ├── flowchart.png                        # Pipeline flowchart
│   └── impact_on_performance_after_tuning_final_adjusted.pdf # Tuning impact
│
├── 📂 Paper/
│   ├── research_paper.pdf                   # Main research paper
│   ├── Predefence report.pdf                # Pre-defense report
│   ├── mahabub.pdf                          # Reference paper
│   └── nsu.pdf                              # Reference paper
│
├── 📂 Reference/
│   ├── final methodology.txt                # Methodology notes
│   ├── result defence.txt                   # Result defense notes
│   ├── table.txt                            # Table specifications
│   ├── tuning.txt                           # HPO tuning notes
│   ├── all column.txt                       # Column specifications
│   ├── dataset feature.txt                  # Feature descriptions
│   ├── dataset final ref.txt                # Dataset reference
│   ├── evaluation er somoy mone rakhte hobe.txt  # Evaluation reminders
│   ├── flowchart.txt                        # Flowchart notes
│   ├── inverse transformtxt.txt             # Inverse transform notes
│   ├── no data leak.txt                     # Data leak prevention notes
│   └── reference for data engg.txt          # Data engineering references
│
├── 📂 catboost_info/                        # CatBoost training information
│
├── 📂 Thesis_env/                           # Python virtual environment
│
├── 📂 time series and spatio/               # Time series & spatial analysis
│
├── bd_weather_extract.py                    # Bangladesh weather data extraction
├── metrics_helper.py                        # Metrics computation helper
├── LightGBM-architecture.png                # LightGBM architecture diagram
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore rules
└── readme.md                                # Project documentation
```

---

## 📁 Dataset

| Source | Description | Resolution | Coverage |
|--------|-------------|------------|----------|
| [DGHS Bangladesh](https://dghs.gov.bd) | Daily division-wise dengue case counts | Division-level | Jan 2022 – Oct 2025 |
| [Open-Meteo API](https://open-meteo.com) | Temperature, rainfall, relative humidity | Daily (gridded) | Jan 2022 – Oct 2025 |
| [ERA5 / ECMWF](https://cds.climate.copernicus.eu) | Meteorological reanalysis | Daily (gridded) | Jan 2022 – Oct 2025 |

**Dataset Statistics**

```
Total Observations  :  11,088   (8 divisions × ~1,386 days)
Zero Inflation      :  42.29%   daily records with zero cases
Target Variable     :  Daily dengue case count per division
Transformation      :  log(1 + y) for training; exp(ŷ) − 1 for evaluation
```

**Division-Wise Case Summary (2022–2025)**

| Division | Mean Cases/Day | Max Cases/Day |
|----------|:--------------:|:-------------:|
| 🔴 Dhaka | 217.07 | 1,703 |
| 🟠 Chattogram | 57.34 | 547 |
| 🟠 Barishal | 48.29 | 477 |
| 🟡 Khulna | 37.24 | 436 |
| 🟡 Rajshahi | 21.49 | 246 |
| 🟢 Mymensingh | 10.49 | 105 |
| 🟢 Rangpur | 5.79 | 88 |
| 🟢 Sylhet | 1.53 | 30 |

> **Note:** A data gap in July 2024 (nationwide network outage) was imputed using division-wise monthly averages to preserve regional variation without introducing look-ahead bias.

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
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

---

## ▶️ Usage

### Step 1 — Fetch Meteorological Data

```bash
python src/data_collection/fetch_weather.py
```

Retrieves daily temperature, rainfall, and humidity from Open-Meteo for all 8 divisions using centroid coordinates.

### Step 2 — Preprocess & Engineer Features

```bash
python src/preprocessing/cleaning.py
python src/features/lag_features.py
```

### Step 3 — Train & Tune Models

```bash
python src/models/train_boosting.py
python src/models/hyperparameter_tuning.py
```

### Step 4 — Evaluate & Explain

```bash
python src/evaluation/metrics.py
python src/evaluation/explainability.py
```

> **Interactive option:** Run notebooks `01` through `06` under `notebooks/` for a guided, step-by-step walkthrough with inline visualizations.

---

## 📊 Results

### Baseline Model Performance

| Model | R² | MAE | RMSE | RMSLE | MAPE (Non-Zero %) | Train Time (s) |
|-------|:--:|:---:|:----:|:-----:|:-----------------:|:--------------:|
| **LightGBM** | **0.899** | **9.149** | **20.236** | **0.773** | 59.77% | 0.20 |
| Random Forest | 0.897 | 9.174 | 20.476 | 0.779 | 57.95% | 3.46 |
| CatBoost | 0.872 | 9.853 | 22.739 | 0.778 | 60.94% | 9.22 |
| XGBoost | 0.849 | 10.756 | 24.709 | 0.813 | 62.61% | 2.24 |
| SVR | 0.690 | 14.389 | 35.479 | 0.957 | 64.83% | 4.02 |
| Ridge Regression | 0.640 | 17.206 | 38.204 | 1.098 | 66.04% | 0.04 |

### Tuned Model Performance

| Model | R² | MAE | RMSE | RMSLE | MAPE (Non-Zero %) | Train Time (s) |
|-------|:--:|:---:|:----:|:-----:|:-----------------:|:--------------:|
| **LightGBM ⭐** | **0.900** | **8.927** | **20.095** | **0.766** | 57.31% | 3.59 |
| CatBoost | 0.898 | 9.026 | 20.310 | 0.769 | 57.27% | 5.80 |
| XGBoost | 0.891 | 9.367 | 20.998 | 0.772 | 58.71% | 3.61 |
| Random Forest | 0.889 | 8.985 | 21.234 | 0.769 | 56.97% | 9.68 |
| Ridge Regression | 0.638 | 17.276 | 38.323 | 1.098 | 66.04% | 0.03 |
| SVR | 0.383 | 18.702 | 49.998 | 0.911 | 80.72% | 48.52 |

**Key Finding:** Hyperparameter tuning substantially improves all gradient-boosting models. Ridge and SVR show limited or negative gains, confirming their unsuitability for this high-dimensional, non-linear problem. LightGBM achieves the best score across all  metrics.


### Key Output Files

| File | Description |
|------|-------------|
| `outputs/figures/shap_beeswarm.png` | SHAP global feature importance |
| `outputs/figures/observed_vs_predicted.png` | Temporal fit — 2025 test period |
| `outputs/figures/division_mae_heatmap.png` | Division × Season MAE heatmap |
| `outputs/figures/choropleth_2022_2025.png` | Spatial burden map across Bangladesh |
| `outputs/models/lightgbm_tuned.pkl` | Saved best LightGBM model |

---

## 🔍 Explainability (SHAP)

SHAP values computed for the optimized LightGBM model provide transparent, epidemiologically grounded explanations.

**Top 10 Predictors (by Mean |SHAP|)**

```
Rank  Feature                      Interpretation
────  ───────────────────────────  ──────────────────────────────────────
 1    rolling_Patients_7d_mean     7-day rolling avg of past cases
 2    Patients_lag7                Case count 7 days prior
 3    rolling_Patients_14d_mean    14-day rolling average
 4    Patients_lag14               Case count 14 days prior
 5    fourier_cos                  Seasonal cosine encoding
 6    avg_temp_lag14               14-day lagged temperature
 7    rolling_avg_temp_14d_mean    14-day rolling mean temperature
 8    rolling_avg_temp_7d_mean     7-day rolling mean temperature
 9    fourier_sin                  Seasonal sine encoding
10    Patients_lag3                Case count 3 days prior
```

**Epidemiological Takeaways:**
- Autoregressive lags dominate, consistent with dengue's self-sustaining transmission dynamics
- Fourier terms capture the annual monsoon/post-monsoon outbreak timing
- Lagged temperature features confirm the known link between warm conditions and Aedes activity
- SHAP explanations align with established dengue science — validating the model as a trustworthy decision-support tool

---

## 🧪 Technologies Used

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.9+ |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, LightGBM, XGBoost, CatBoost |
| **HPO** | `RandomizedSearchCV` + early stopping callbacks |
| **Explainability** | SHAP |
| **Geospatial** | geopandas |
| **Visualization** | matplotlib, seaborn |
| **Weather Data** | Open-Meteo Archive API, ERA5 / ECMWF |
| **Notebooks** | Jupyter |
| **Serialization** | pickle (`.pkl`), JSON (`.json`) |

---

## 🔧 Configuration

No `.env` file is required. All configuration is handled directly in the source files.

Key settings to adjust in `src/`:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `LAG_DAYS` | `lag_features.py` | `[3, 7, 14]` | Autoregressive lag offsets |
| `ROLLING_WINDOWS` | `rolling_stats.py` | `[7, 14]` | Rolling window sizes (days) |
| `N_ITER` | `hyperparameter_tuning.py` | `100` | RandomizedSearchCV iterations |
| `CV_FOLDS` | `hyperparameter_tuning.py` | `5` | Time-series CV folds |
| `EARLY_STOPPING` | `train_boosting.py` | `50` rounds | Early stopping patience |
| `TEST_YEAR` | `transforms.py` | `2025` | Holdout test period start year |

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate docstrings.

---

## ⚠️ Limitations

- **Spatial resolution:** Division-level aggregation suppresses sub-district heterogeneity
- **Surveillance bias:** DGHS data may contain underreporting or reporting delays
- **Missing entomological data:** No mosquito vector density or serotype data included
- **Correlational explanations:** SHAP quantifies feature contributions, not causal relationships

---

## 🔮 Future Work

- [ ] Sub-district (upazila-level) spatial granularity
- [ ] Real-time DGHS + operational weather feed integration
- [ ] Human mobility features via anonymized mobile data
- [ ] Long-horizon projections under CMIP6 climate change scenarios
- [ ] Causal inference to disentangle climate vs. epidemiological drivers

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 👤 Author

**Ashikur Rahman Ashik**
B.Sc. Thesis · Faculty of Computer Science and Engineering
Patuakhali Science and Technology University (PSTU), Bangladesh · 2026

<div align="center">

---

<sub>Built with ❤️ · LightGBM · SHAP · Open-Meteo · ERA5</sub>

</div>

