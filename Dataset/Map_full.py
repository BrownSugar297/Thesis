import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ CONFIGURATION ------------------

# Use the absolute path provided in your previous query:
FULL_DATASET_PATH = r"E:\Regg Thesis\Dataset\Dataset.csv"
GEOJSON_PATH = r'E:\Regg Thesis\Figures\bgd_admbnda_adm1_bbs_20201113.shp' # <<< VERIFY THIS PATH
SAVE_FOLDER = r'E:\Regg Thesis\Figures' # Where to save the figure

# IEEE Style Globals
FIG_WIDTH = 7.0 
mpl.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42})
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ---------------------------------------------------

# Helper to save both PDF (primary) and 300dpi PNG (backup)
def save_figure(fname_base):
    """Saves the current Matplotlib figure to PDF and PNG."""
    pdf_path = os.path.join(SAVE_FOLDER, f"{fname_base}.pdf")
    png_path = os.path.join(SAVE_FOLDER, f"{fname_base}.png")
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', format='png', dpi=300)
    print(f"Saved: {pdf_path} and {png_path}")

# ---------------------------------------------------

def figure7_choropleth_map_overall():

    print(f"Starting Overall Choropleth Map generation using: {FULL_DATASET_PATH}...")
    
    # --- 1. Load and Prepare the Full Dataset ---
    try:
        df_full = pd.read_csv(FULL_DATASET_PATH)
        
        # --- KEY RENAME using uploaded column names ---
        df_full.rename(columns={
            'date': 'Date',          # Original date column
            'Patients': 'y_actual',  # Observed cases
            'division': 'Division'   # Division name is already here
        }, inplace=True)
        
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {FULL_DATASET_PATH}. Cannot proceed.")
        return
    except KeyError as e:
        print(f"ERROR: Missing expected column in dataset: {e}. Check your column names.")
        return
    
    # 1.1 Date Filtering (To ensure only 2022-2025 is included)
    if 'Date' in df_full.columns:
        try:
            df_full['Date'] = pd.to_datetime(df_full['Date'])
            df_full = df_full[
                (df_full['Date'].dt.year >= 2022) & 
                (df_full['Date'].dt.year <= 2025)
            ].copy()
            
        except Exception as e:
            print(f"Warning: Error processing 'Date' column: {e}. Skipping map.")
            return
    else:
        print("Warning: 'Date' column is missing. Skipping map.")
        return

    # 1.2 Division Check 
    if 'Division' not in df_full.columns:
        print("Warning: 'Division' column is missing. Skipping map.")
        return

    try:
        # 2. Load Shapefile
        gdf = gpd.read_file(GEOJSON_PATH)

        # 3. Prepare Case Data (AGGREGATE THE TOTAL for the entire filtered period)
        df_total = df_full.groupby('Division')['y_actual'].sum().reset_index()

        if df_total.empty:
            print(f"No observed cases found in the filtered dataset (2022-2025). Skipping Figure.")
            return

        # 4. Merge geospatial data with case data
        
        merge_column = None
        for col in ['NAME_ENGL', 'ADM1_EN', 'NAME_1']:
            if col in gdf.columns:
                merge_column = col
                break
        
        if not merge_column:
             merge_column = gdf.columns[0]
             print(f"Warning: Defaulting to first SHP column '{merge_column}' for merge.")

        # Clean Division Names for merging
        gdf['Division_Clean'] = (
            gdf[merge_column].astype(str).str.replace(' Division', '', regex=False)
            .str.replace('Barisal', 'Barishal', regex=False)
            .str.replace('Chittagong', 'Chattogram', regex=False) 
        )
        
        # Merge the GeoDataFrame with the case data
        gdf = gdf.merge(df_total, left_on='Division_Clean', right_on='Division', how='left')
        gdf['y_actual'] = gdf['y_actual'].fillna(0) 

        # 5. Plotting the Choropleth Map
        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 1.5))
        cax = fig.add_axes([0.1, 0.05, 0.8, 0.03]) 
        
        # Plot using 'y_actual' (Total Cases)
        gdf.plot(column='y_actual', 
                 cmap='YlGnBu', 
                 linewidth=0.8, 
                 ax=ax, 
                 edgecolor='0.5', 
                 legend=True,
                 cax=cax, 
                 legend_kwds={'label': "Total Observed Dengue Cases (2022–2025)", 
                              'orientation': "horizontal", 
                              'shrink': 1.0})

        # Add Division Names as labels
        for idx, row in gdf.iterrows():
            if row['geometry'].is_valid and not row['geometry'].is_empty:
                center_x, center_y = row['geometry'].centroid.x, row['geometry'].centroid.y
                division_name = row['Division_Clean']
                
                # --- MODIFIED LOGIC HERE: ONLY DHAKA REMAINS WHITE ---
                # This ensures Chattogram will now be black, as it's not 'Dhaka'.
                label_color = 'white' if division_name == 'Dhaka' else 'black' 
                
                ax.text(center_x, center_y, division_name,
                        fontsize=7, ha='center', color=label_color, alpha=0.9, weight='bold')

        ax.set_axis_off() 
        
        # 6. Save and Close
        save_figure('Figure7_Choropleth_Map_Overall_2022_2025')
        plt.close()
        
    except FileNotFoundError:
        print(f"ERROR: Shapefile not found at {GEOJSON_PATH}. Skipping Figure 7.")
    except Exception as e:
        print(f"CRITICAL ERROR plotting Choropleth Map: {e}")

if __name__ == "__main__":
    figure7_choropleth_map_overall()
    print(f"\nScript finished. Verify the file paths and run the script on your local machine.")