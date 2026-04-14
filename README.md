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
dengue-forecasting-bangladesh/
│
├── 📂 data/
│   ├── raw/
│   │   ├── dghs_dengue_cases.csv           # Daily division-wise dengue cases (DGHS)
│   │   └── BD_Daily_Weather_2022-2025.csv  # Meteorological data (Open-Meteo / ERA5)
│   └── processed/
│       └── integrated_dataset.csv          # Cleaned & feature-engineered dataset
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb                        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb              # Cleaning & preprocessing
│   ├── 03_feature_engineering.ipynb        # Lags, Fourier, climate indices
│   ├── 04_model_training.ipynb             # Baseline & tuned model training
│   ├── 05_evaluation.ipynb                 # Metrics & model comparison
│   └── 06_explainability.ipynb             # SHAP analysis & visualization
│
├── 📦 src/
│   ├── data_collection/
│   │   └── fetch_weather.py                # Open-Meteo API retrieval script
│   ├── preprocessing/
│   │   ├── cleaning.py                     # Imputation & gap handling
│   │   └── transforms.py                   # Log transform, StandardScaler
│   ├── features/
│   │   ├── lag_features.py                 # Autoregressive lag generation
│   │   ├── rolling_stats.py                # Rolling window statistics
│   │   ├── fourier_encoding.py             # Seasonal Fourier features
│   │   └── climate_indices.py              # Heat Index, Rain-Temp Index
│   ├── models/
│   │   ├── train_baseline.py               # Ridge, RF, SVR
│   │   ├── train_boosting.py               # LightGBM, CatBoost, XGBoost
│   │   └── hyperparameter_tuning.py        # RandomizedSearchCV + early stopping
│   └── evaluation/
│       ├── metrics.py                      # MAE, RMSE, RMSLE, R²
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

### Baseline vs. Tuned Performance

| Model | R² | MAE | RMSE | RMSLE |
|-------|:--:|:---:|:----:|:-----:|
| **LightGBM ⭐ (tuned)** | **0.900** | **8.927** | **20.094** | **0.765** |
| CatBoost (tuned) | 0.898 | 9.026 | 20.310 | 0.768 |
| XGBoost (tuned) | 0.891 | 9.366 | 20.998 | 0.772 |
| Random Forest (tuned) | 0.888 | 8.984 | 21.234 | 0.769 |
| Ridge Regression | 0.637 | 17.275 | 38.322 | 1.097 |
| SVR | 0.383 | 18.702 | 49.997 | 0.910 |

**Key Finding:** Hyperparameter tuning substantially improves all gradient-boosting models. Ridge and SVR show limited or negative gains, confirming their unsuitability for this high-dimensional, non-linear problem. LightGBM achieves the best score across all four metrics.

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

