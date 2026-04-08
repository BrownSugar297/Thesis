#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib

# ------------------ USER CONFIG ------------------
X_TEST_SCALED = r"E:\Regg Thesis\time series and spatio\X_test_scaled.csv"
X_TEST_WITH_DATES = r"E:\Regg Thesis\time series and spatio\X_test_scaled_for_plotting.csv"
Y_TEST_ORIGINAL = r"E:\Regg Thesis\time series and spatio\y_test_original_patients.csv"
MODEL_PATH = r"E:\Regg Thesis\time series and spatio\lightgbm_tuned.pkl"
SAVE_FOLDER = r"E:\Regg Thesis\time series and spatio\Fig"

FIG_WIDTH = 7.0
FIG_DPI = 300

# ------------------ SCALING PARAMS (REAL VALUES) ------------------
# These are the mathematically verified, correct values.
TEMP_MU, TEMP_SIGMA = 30.3464, 3.6450
RAIN_MU, RAIN_SIGMA = 6.4082, 13.1971

# ------------------ IEEE STYLE ------------------
PALETTE = {
    "primary": "#4C72B0",
    "secondary": "#DD8452",
    "tertiary": "#55A868",
    "quaternary": "#C44E52",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.25,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})

# ------------------ SAVE HELPER ------------------
def save_figure(fname_base: str) -> None:
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    pdf_path = os.path.join(SAVE_FOLDER, f"{fname_base}.pdf")
    png_path = os.path.join(SAVE_FOLDER, f"{fname_base}.png")
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf")
    plt.savefig(png_path, bbox_inches="tight", format="png", dpi=FIG_DPI)
    print(f"Saved: {png_path}")

# ------------------ UTIL ------------------
def inverse_transform(y_log_array: np.ndarray) -> np.ndarray:
    y = np.exp(y_log_array)
    return np.round(np.maximum(0, y)).astype(int)

# ------------------ CUSTOM CLASS ------------------
class LGBMRegressorWithES:
    def __init__(self, **kwargs):
        pass

    def predict(self, X):
        if hasattr(self, "_Booster") and hasattr(self._Booster, "predict"):
            return self._Booster.predict(X)
        raise NotImplementedError("Underlying booster predict not available.")

# ------------------ LOAD DATA ------------------
time_agg = False
try:
    X_test_scaled = pd.read_csv(X_TEST_SCALED)
    df_dates_and_info = pd.read_csv(X_TEST_WITH_DATES)
    X_test = X_test_scaled.copy()

    # align column names
    df_dates_and_info.rename(columns={"date": "Date", "rainfall": "total_rainfall"}, inplace=True)

    INFO_COLS_TO_ADD = ["Date", "max_temp", "total_rainfall"]

    if all(c in df_dates_and_info.columns for c in INFO_COLS_TO_ADD) and len(df_dates_and_info) == len(X_test):
        for col in INFO_COLS_TO_ADD:
            X_test[col] = df_dates_and_info[col].values
        time_agg = True

    if time_agg:
        X_test["Date"] = pd.to_datetime(X_test["Date"])
        X_test["Year"] = X_test["Date"].dt.year
        X_test["Month"] = X_test["Date"].dt.month

        y_test_original = pd.read_csv(Y_TEST_ORIGINAL).iloc[:, 0]
        X_test["y_actual"] = y_test_original.values

    best_model = joblib.load(MODEL_PATH)

    NON_FEATURE_COLS = ["Date", "Division", "Year", "Month", "Season", "max_temp", "total_rainfall"]
    X_test_features = X_test.drop(columns=[c for c in NON_FEATURE_COLS if c in X_test.columns], errors="ignore")

    # If LightGBM model (has _Booster), ensure columns match expected features
    if hasattr(best_model, "_Booster") and getattr(best_model, "_Booster") is not None:
        expected = best_model._Booster.feature_name()
        for f in expected:
            if f not in X_test_features.columns:
                X_test_features[f] = 0.0
        X_test_features = X_test_features.reindex(columns=expected, fill_value=0.0)

    y_pred_log = best_model.predict(X_test_features)
    X_test["y_pred"] = inverse_transform(y_pred_log)
    X_test["abs_error"] = np.abs(X_test["y_actual"] - X_test["y_pred"])
    print("Data loaded successfully.")

except Exception as e:
    raise SystemExit(f"Error: {e}")

# ------------------ FIGURE 3 ------------------
def figure3_climate_trend():
    df_plot = df_dates_and_info[["Date", "max_temp", "total_rainfall"]].copy()

    # Correct inverse scaling
    df_plot["max_temp"] = (df_plot["max_temp"] * TEMP_SIGMA) + TEMP_MU
    df_plot["total_rainfall"] = (df_plot["total_rainfall"] * RAIN_SIGMA) + RAIN_MU
    df_plot["total_rainfall"] = df_plot["total_rainfall"].clip(lower=0)

    df_plot["y_actual"] = X_test["y_actual"]
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    df_plot["Year"] = df_plot["Date"].dt.year
    df_plot["Month"] = df_plot["Date"].dt.month

    df_climate = (
        df_plot.groupby(["Year", "Month"])[["y_actual", "max_temp", "total_rainfall"]]
        .agg({"y_actual": "sum", "max_temp": "mean", "total_rainfall": "sum"})
        .reset_index()
    )
    df_climate["Time"] = df_climate["Year"].astype(str) + "-" + df_climate["Month"].astype(str).str.zfill(2)

    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 1.1), sharex=True, gridspec_kw={"hspace": 0.5})
    n = len(df_climate)
    step = max(1, n // 12)
    time_ticks = df_climate["Time"][::step]

    axes[0].plot(df_climate["Time"], df_climate["y_actual"], color=PALETTE["primary"], linewidth=2)
    axes[0].set_ylabel("Cases (Count)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].set_title("(a) Dengue Cases", loc="left", pad=10, fontsize=10)

    axes[1].plot(df_climate["Time"], df_climate["max_temp"], color=PALETTE["secondary"], linewidth=1.7)
    axes[1].set_ylabel("Max Temp ($^\\circ$C)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].set_title("(b) Avg Max Temperature", loc="left", pad=10, fontsize=10)

    axes[2].plot(df_climate["Time"], df_climate["total_rainfall"], color=PALETTE["tertiary"], linewidth=1.7)
    axes[2].set_ylabel("Rainfall (mm)")
    axes[2].set_xlabel("Time (Year-Month)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.45)
    axes[2].set_xticks(time_ticks)
    axes[2].set_xticklabels(time_ticks, rotation=45, ha="right")
    axes[2].set_title("(c) Total Rainfall", loc="left", pad=10, fontsize=10)

    fig.tight_layout()
    save_figure("Figure3_Stacked_Climate_Trend_FINAL_V_FixedScaling")
    plt.close(fig)

# ------------------ FIGURE 4 ------------------
def figure4_timefit_and_error():
    df_fit = X_test.groupby(["Year", "Month"])[["y_actual", "y_pred", "abs_error"]].sum().reset_index()
    df_fit["Time"] = df_fit["Year"].astype(str) + "-" + df_fit["Month"].astype(str).str.zfill(2)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 0.7), sharex=True, gridspec_kw={"hspace": 0.28})
    n = len(df_fit)
    step = max(1, n // 12)
    time_ticks = df_fit["Time"][::step]

    axes[0].plot(df_fit["Time"], df_fit["y_actual"], label="Observed", color=PALETTE["primary"], linewidth=2)
    axes[0].plot(df_fit["Time"], df_fit["y_pred"], label="Predicted", color=PALETTE["secondary"], linewidth=1.7)
    axes[0].set_ylabel("Total Cases")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("(a) Observed vs Predicted", loc="left", pad=10, fontsize=10)

    axes[1].plot(df_fit["Time"], df_fit["abs_error"], label="Abs Error", color=PALETTE["quaternary"], linewidth=1.7)
    axes[1].set_ylabel("Abs Error")
    axes[1].set_xlabel("Time (Year-Month)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].set_xticks(time_ticks)
    axes[1].set_xticklabels(time_ticks, rotation=45, ha="right")
    axes[1].set_title("(b) Absolute Error", loc="left", pad=10, fontsize=10)

    fig.tight_layout()
    save_figure("Figure4_TimeSeries_Fit_and_Error_FINAL_V_FixedScaling")
    plt.close(fig)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    if time_agg:
        figure3_climate_trend()
        figure4_timefit_and_error()
        print("Both figures saved successfully.")
    else:
        print("Date/Climate loading failed — cannot plot.")
