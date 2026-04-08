import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# === USER CONFIGURATION: CHANGE THESE ON YOUR MACHINE ===
# =========================================================

# 💡 Set your desired output folder path here.
# The 'r' prefix handles backslashes correctly in Python.
SAVE_FOLDER = r"E:\Regg Thesis\time series and spatio\Fig"

# File names used in the current working directory
X_TEST_SCALED = r"E:\Regg Thesis\time series and spatio\X_test_scaled.csv"
X_TEST_WITH_DATES = r"E:\Regg Thesis\time series and spatio\X_test_scaled_for_plotting.csv"
Y_TEST_ORIGINAL = r"E:\Regg Thesis\time series and spatio\y_test_original_patients.csv"

# Figure parameters
FIG_WIDTH = 7.0
FIG_DPI = 300

# IEEE-Friendly Color Palette (8 distinct colors for divisions)
IEEE_PALETTE_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8800C4", "#FFC400", "#7A7A7A", "#00A9A5"
]

# ------------------ MODIFIED UTILITY FUNCTIONS ------------------

def save_figure(fname_base: str) -> None:
    """Saves the current Matplotlib figure to the predefined folder in both PNG and PDF formats."""
    global SAVE_FOLDER, FIG_DPI

    # Create folder if it doesn't exist (CRITICAL for user path)
    if SAVE_FOLDER and not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    # Prefix the filename with the folder path
    path_prefix = os.path.join(SAVE_FOLDER, fname_base)

    # 1. Save as PNG
    png_path = f"{path_prefix}.png"
    plt.savefig(png_path, bbox_inches="tight", format="png", dpi=FIG_DPI)
    print(f"Saved PNG: {png_path}")
    
    # 2. Save as PDF (Recommended for publication quality)
    pdf_path = f"{path_prefix}.pdf"
    # Note: PDF saves typically don't require DPI unless bitmap content is included.
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf") 
    print(f"Saved PDF: {pdf_path}")

def derive_season(month):
    """Manually assigns season based on month (including Spring)."""
    if month in [3, 4]:
        return 'Spring'
    elif month == 5:
        return 'Pre-Monsoon'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'Post-Monsoon'
    elif month in [12, 1, 2]:
        return 'Winter'
    else:
        return 'Unknown'

# ------------------ DATA LOADING AND PREPARATION ------------------

time_agg = False
try:
    X_test_scaled = pd.read_csv(X_TEST_SCALED)
    df_dates_and_info = pd.read_csv(X_TEST_WITH_DATES)
    X_test = X_test_scaled.copy()

    if len(df_dates_and_info) == len(X_test):
        # 1. Date and Season Derivation
        X_test["Date"] = pd.to_datetime(df_dates_and_info["date"])
        X_test["Month"] = X_test["Date"].dt.month
        X_test['Season'] = X_test['Month'].apply(derive_season)

        # 2. Add Actual Cases (y_test)
        y_test_original = pd.read_csv(Y_TEST_ORIGINAL).iloc[:, 0]
        X_test["y_actual"] = y_test_original.values
        
        # 3. Predicted Cases ('y_pred') - Placeholder
        X_test["y_pred"] = X_test["y_actual"].copy()
            
        # 4. Derive Division from one-hot columns
        division_cols = [c for c in df_dates_and_info.columns if c.startswith('division_')]
        if division_cols:
            X_test['Division'] = df_dates_and_info[division_cols].idxmax(axis=1).str.replace('division_', '')
        else:
            X_test['Division'] = 'Unknown'

        time_agg = True
    
    if not time_agg or 'Division' not in X_test.columns or 'Season' not in X_test.columns:
        raise ValueError("Critical columns could not be derived.")
    
    print("Data loading and preparation successful.")

except Exception as e:
    print(f"Error during data loading or preparation. Please check CSV file paths: {e}")
    
# ------------------ FIGURE 2: Total Predicted Cases (SUM) with SE of the Sum ------------------

def figure2_total_predicted_cases_with_errorbars():
    
    # Filter data for known divisions/seasons
    df_plot = X_test[
        ~X_test['Season'].isin(['Unknown']) & 
        ~X_test['Division'].isin(['Unknown'])
    ].copy()

    # 1. Aggregate Data and Calculate Error Bar Value
    df_aggregated = df_plot.groupby(['Division', 'Season'])['y_pred'].agg(
        Total_Predicted_Cases='sum',
        Daily_Std_Dev='std',
        Count='count'
    ).reset_index()

    # Calculate 95% CI of the Total Sum (1.96 * SE_Sum)
    df_aggregated['SE_Sum'] = df_aggregated['Daily_Std_Dev'] * np.sqrt(df_aggregated['Count'])
    df_aggregated['Error_Bar'] = 1.96 * df_aggregated['SE_Sum'] 

    # 2. Define Order and Styling
    season_order = ['Winter', 'Spring', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']
    df_aggregated['Season'] = pd.Categorical(df_aggregated['Season'], categories=season_order, ordered=True)
    df_aggregated = df_aggregated.sort_values('Season')

    division_list = sorted(df_aggregated['Division'].unique())
    division_color_map = {div: IEEE_PALETTE_COLORS[i] for i, div in enumerate(division_list)}
    
    # 3. Plotting using custom bar function
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH * 0.6))

    N_divisions = len(division_list)
    N_seasons = len(season_order)
    bar_width = 0.8 / N_divisions
    x_season_center = np.arange(N_seasons)

    for i, div in enumerate(division_list):
        df_div = df_aggregated[df_aggregated['Division'] == div].sort_values('Season')
        
        # Calculate x positions for this division's bars
        x_div = x_season_center + (i - (N_divisions - 1) / 2) * bar_width
        
        # Plot the bars and error bars
        ax.bar(
            x_div,
            df_div['Total_Predicted_Cases'],
            yerr=df_div['Error_Bar'], 
            width=bar_width,
            color=division_color_map[div],
            label=div,
            capsize=3, 
            ecolor='k', 
            linewidth=0.5
        )

    # 4. Styling and Annotations
    ax.set_title(
        "Total Predicted Cases by Division and Season (with 95% CI of Sum)", 
        loc='left', 
        pad=10, 
        fontsize=10, 
        weight='bold'
    )
    
    ax.set_xlabel("Season", fontsize=10)
    ax.set_ylabel("Total Predicted Cases (Count)", fontsize=10) 
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    ax.set_xticks(x_season_center)
    ax.set_xticklabels(season_order)

    ax.legend(
        title='Division',
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        fontsize=9,
        title_fontsize=9,
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 0.95, 1])

    # Save figure (saves both PNG and PDF to the SAVE_FOLDER)
    save_figure('Figure_Total_Predicted_Cases_with_SE_Sum_Errorbars') 
    plt.close(fig)
    
    # Save the aggregated data to CSV
    csv_path = os.path.join(SAVE_FOLDER, "Figure_Total_Predicted_Cases_SE_Sum_Data.csv")
    df_aggregated.to_csv(csv_path, index=False)
    print(f"Data saved to CSV: {csv_path}")


# ------------------ EXECUTION ------------------

if time_agg:
    figure2_total_predicted_cases_with_errorbars()
    print("\nProcessing complete. Check your designated SAVE_FOLDER for the output files.")
else:
    print("Execution skipped. Please ensure your input CSV files are correct.")