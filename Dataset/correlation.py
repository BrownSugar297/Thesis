import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ------------------ CONFIGURATION ------------------

# Use your local path here:
DATASET_PATH = "E:\Regg Thesis\Dataset\Final_Dataset.csv"
SAVE_FOLDER = 'Figures'
FIG_WIDTH = 10 

# ------------------ HELPER FUNCTION ------------------
def save_figure(name):
    """Save figure in IEEE-standard PDF format."""
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    
    full_path_pdf = os.path.join(SAVE_FOLDER, name + ".pdf")
    try:
        plt.savefig(
            full_path_pdf,
            bbox_inches="tight",
            format="pdf",
            dpi=300 # High resolution for thesis
        )
        print(f"Figure saved successfully to: {full_path_pdf}")
    except Exception as e:
        print(f"Could not save figure: {e}")
    plt.close()

# ------------------ PLOTTING FUNCTION ------------------

def plot_correlation_heatmap_ieee():
    
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset file not found at {DATASET_PATH}. Please ensure the file is accessible.")
        return

    # --- 1. Select Key Features for Correlation ---
    feature_cols = [
        'Patients', # Target variable
        'max_temp', 'min_temp', 'rainfall', 'humidity', # Raw weather
        'Patients_lag7', 'Patients_lag14', # Key Lagging Features
        'rolling_Patients_7d_mean', 'rolling_Patients_14d_mean', # Key Rolling Features
        'avg_temp_lag7', 'rolling_avg_temp_7d_mean', # Lagged/Rolling Weather
        'temp_humidity_index', # Composite feature
        'day_of_year', # Time Feature
    ]
    
    # Ensure all selected columns exist in the DataFrame
    selected_df = df[[col for col in feature_cols if col in df.columns]]
    
    if len(selected_df.columns) < 2:
        print("ERROR: Not enough relevant columns found for correlation plot.")
        return

    # --- 2. Calculate Correlation Matrix ---
    corr_matrix = selected_df.corr()

    # --- 3. Plotting Setup ---
    plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.9))
    
    # Use a divergent colormap (coolwarm/vlag)
    sns.heatmap(
        corr_matrix,
        annot=True, # Show correlation values
        fmt=".2f", # Format to 2 decimal places
        cmap='coolwarm', # Standard divergent colormap
        linewidths=.5, # Thin lines to separate cells
        cbar_kws={'label': 'Pearson Correlation Coefficient (r)'},
        # Implemented change: Lighter font weight for annotations
        annot_kws={"weight": "normal"} 
    )

    # --- 4. Aesthetics (IEEE Standard) ---
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    # --- 5. Save the Figure ---
    save_figure('Figure_Correlation_Heatmap_IEEE_Final')


# ------------------ EXECUTION ------------------

if __name__ == '__main__':
    plot_correlation_heatmap_ieee()