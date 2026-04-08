# Save as produce_bd_division_monthly.py and run with python3
import os
import xarray as xr
import rioxarray            # for raster reprojection support
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import numpy as np

# ---- USER: paths / inputs ----
ERA5_DIR = r"D:\Python\ashik\ERA5"     # <-- folder containing the 4 .nc files
DIV_SHAPE = r"D:\Python\ashik\data\gadm41_BGD_1.shp"
OUT_CSV = r"D:\Python\ashik\bd_division_monthly_2024_2025.csv"

# ---- load divisions ----
divs = gpd.read_file(DIV_SHAPE)
divs = divs.to_crs("EPSG:4326")  # ERA5 lat/lon

# ---- helper: aggregate single variable file (monthly) ----
def aggregate_monthly_var(nc_path, varname, stat='mean', to_celsius=False, precip_mm=False):
    ds = xr.open_dataset(nc_path)
    da = ds[varname]

    # Detect time coordinate
    time_coord = None
    for possible in ['time', 'month', 'valid_time', 'step']:
        if possible in da.coords:
            time_coord = possible
            break

    if time_coord is not None:
        times = pd.to_datetime(da[time_coord].values)
    else:
        # If no time coordinate, generate 12 monthly dates
        year = int(os.path.basename(nc_path).split('_')[-1].split('.')[0])
        times = pd.date_range(start=f'{year}-01', periods=12, freq='M')

    results = []
    for i, t in enumerate(times):
        if time_coord:
            arr = da.isel({time_coord: i}).load()
        else:
            arr = da.isel(time=i).load() if 'time' in da.dims else da.load()

        tmp = arr.rio.write_crs("EPSG:4326", inplace=False)
        zs = zonal_stats(divs.geometry, tmp.values, affine=tmp.rio.transform(),
                         stats=[stat], nodata=np.nan, geojson_out=False)
        row = {'year': t.year, 'month': t.month}
        for idx, z in enumerate(zs):
            name = divs.iloc[idx]['division'] if 'division' in divs.columns else divs.iloc[idx].get('NAME_1', f"div_{idx}")
            row[name] = z.get(stat, None)
        results.append(row)

    df = pd.DataFrame.from_records(results)
    if df.ndim == 1:  # single row converted to DataFrame
        df = pd.DataFrame([df])

    if to_celsius:
        for c in df.columns:
            if c not in ['year', 'month']:
                df[c] = df[c] - 273.15

    if precip_mm:
        for c in df.columns:
            if c not in ['year', 'month']:
                df[c] = df[c] * 1000.0  # convert m -> mm

    return df

# ---- Aggregate temperature ----
t2m_csv_parts = []
for year in [2024, 2025]:
    temp_file = os.path.join(ERA5_DIR, f"era5_t2m_monthly_{year}.nc")
    if os.path.exists(temp_file):
        df_temp = aggregate_monthly_var(temp_file, varname='t2m', stat='mean', to_celsius=True)
        t2m_csv_parts.append(df_temp)
temp_df = pd.concat(t2m_csv_parts, ignore_index=True) if t2m_csv_parts else pd.DataFrame()

# ---- Aggregate precipitation ----
precip_csv_parts = []
for year in [2024, 2025]:
    prec_file = os.path.join(ERA5_DIR, f"era5_tp_monthly_{year}.nc")
    if os.path.exists(prec_file):
        df_prec = aggregate_monthly_var(prec_file, varname='tp', stat='sum', precip_mm=True)
        precip_csv_parts.append(df_prec)
prec_df = pd.concat(precip_csv_parts, ignore_index=True) if precip_csv_parts else pd.DataFrame()

# ---- Merge and reshape ----
if not temp_df.empty and not prec_df.empty:
    merged = pd.merge(temp_df, prec_df, on=['year', 'month'], suffixes=('_t2m', '_precip'))

    records = []
    for _, r in merged.iterrows():
        for i, drow in divs.iterrows():
            name = drow['division'] if 'division' in drow else drow.get('NAME_1', f"div_{i}")
            mean_t = r.get(name + "_t2m", np.nan)
            total_p = r.get(name + "_precip", np.nan)
            records.append({
                'division': name,
                'year': int(r['year']),
                'month': int(r['month']),
                'mean_t2m_C': mean_t,
                'total_precip_mm': total_p
            })

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(OUT_CSV, index=False)
    print("Saved", OUT_CSV)
else:
    print("Temperature or precipitation NetCDF files not found — check ERA5_DIR and file names.")
