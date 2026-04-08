import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ CONFIG ------------------
# CRITICAL NOTE: The local path is kept here as requested, 
RAW_DATASET_PATH = r'E:\Regg Thesis\Dataset\Dataset.csv'
TARGET_COLUMN_NAME = 'Patients'
SAVE_FOLDER = 'Figures'
FIG_WIDTH = 6.0

# Lighter IEEE-appropriate academic blue (Changed from #4C72B0)
IEEE_BLUE = "#5B9AD9"  # Lighter, professional blue

def save_figure(name):
    """Save figure in IEEE-standard PDF format."""
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    
    # Attempt to save PDF
    try:
        plt.savefig(
            os.path.join(SAVE_FOLDER, name + ".pdf"),
            bbox_inches="tight",
            format="pdf"
        )
        print(f"Figure saved successfully to: {os.path.join(SAVE_FOLDER, name + '.pdf')}")
    except Exception as e:
        print(f"Could not save figure locally. Error: {e}")
    plt.close()

def plot_raw_distribution_log_frequency_ieee():
    
    try:
        # Load dataset
        df = pd.read_csv(RAW_DATASET_PATH)
        patients = df[TARGET_COLUMN_NAME].dropna()
    except Exception as e:
        print(f"FATAL ERROR: Could not load data. Check RAW_DATASET_PATH and file accessibility: {e}")
        return

    # Zero-inflation calculation
    zero_perc = (patients == 0).sum() * 100 / len(patients)

    # Create figure
    plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.7))

    # Lighter IEEE Blue Histogram (no title)
    plt.hist(
        patients,
        bins='auto',
        color=IEEE_BLUE,         # <-- NEW LIGHTER BLUE
        edgecolor="black",
        linewidth=0.5
    )

    # Log frequency scale
    plt.yscale("log")

    # Aesthetics
    plt.xlabel("Dengue Cases (Daily Count)", fontsize=11)
    plt.ylabel("Frequency (Log Scale)", fontsize=11)

    # Subtle grid (IEEE standard)
    plt.grid(True, linestyle='--', alpha=0.4, which='both')

    # Zero inflation annotation (important insight)
    plt.text(
        0.98, 0.95,
        f"Zero-Inflation: {zero_perc:.2f}%",
        transform=plt.gca().transAxes,
        ha='right',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            alpha=0.9
        )
    )

    plt.tight_layout()

    # Save IEEE standard figure
    save_figure("Figure_Raw_Distribution_LogScaled_IEEE_Final_V2")


# ------------------ RUN ------------------
if __name__ == "__main__":
    plot_raw_distribution_log_frequency_ieee()