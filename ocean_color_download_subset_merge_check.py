# NASA OCEAN COLOR MONTHLY DATA PROCESSING TEMPLATE
# Download → Subset → Check monthly files → Low-RAM merge → Check merged file

# GitHub/Colab-ready template
# Product examples: POC, PIC, CHL, PAR, KD490



# 0) Mount Google Drive
from google.colab import drive

drive.mount('/content/drive')



# 1) USER SETTINGS — EDIT THIS SECTION ONLY

# Product settings

# Write your NASA product code here.
# Examples:
# POC   → PRODUCT_CODE = "POC", VARIABLE_NAME = "poc",     OUTPUT_VARIABLE = "POC"
# PIC   → PRODUCT_CODE = "PIC", VARIABLE_NAME = "pic",     OUTPUT_VARIABLE = "PIC"
# CHL   → PRODUCT_CODE = "CHL", VARIABLE_NAME = "chlor_a", OUTPUT_VARIABLE = "CHL"
# PAR   → PRODUCT_CODE = "PAR", VARIABLE_NAME = "par",     OUTPUT_VARIABLE = "PAR"
# KD490 → PRODUCT_CODE = "KD",  VARIABLE_NAME = "Kd_490",  OUTPUT_VARIABLE = "KD490"

PRODUCT_CODE = "X"          # <-- write your product code here
VARIABLE_NAME = "x"         # <-- write your original variable name here
OUTPUT_VARIABLE = "x"       # <-- write your final standardized variable name here


# Time settings
# Write time as year range. NASA monthly files use YYYYMMDD_YYYYMMDD format.


START_YEAR = yyyy              # <-- write start year YYYY
END_YEAR   = yyyy              # <-- write end year YYYY


# Region settings
# Write latitude and longitude range here.

LAT_MIN, LAT_MAX = xx, yy    # <-- write latitude range XX, YY
LON_MIN, LON_MAX = xx, yy   # <-- write longitude range XX, YY


# Path settings
# BASE_DIR = your main data folder
# RAW_DIR  = raw downloaded monthly NASA files
# SUB_DIR  = monthly subset files for your study region
# final_dir = processed output folder

BASE_DIR = "/content/drive/MyDrive/your main data folder"       # <-- write your base path
final_dir = "/content/drive/MyDrive/processed output folder                 # <-- write your processed path

RAW_DIR = f"{BASE_DIR}/{OUTPUT_VARIABLE}_monthly_raw"
SUB_DIR = f"{BASE_DIR}/{OUTPUT_VARIABLE}_monthly_monthly subset files for your study region"
temp_dir = f"{final_dir}/{OUTPUT_VARIABLE}_MONTHLY_STANDARDIZED_TEMP"

# Final merged output file
final_out = f"{final_dir}/{OUTPUT_VARIABLE}_monthly_{START_YEAR}_{END_YEAR}_monthly subset files for your study region.nc"

# Download log path
log_csv = f"{BASE_DIR}/{OUTPUT_VARIABLE}_download_log_{START_YEAR}_{END_YEAR}.csv"

# Skipped-file report path
skipped_csv = f"{final_dir}/{OUTPUT_VARIABLE}_skipped_files_LOW_RAM.csv"



# NASA Ocean Color settings

BASE_URL = "https://oceandata.sci.gsfc.nasa.gov/getfile"
RESOLUTION = "4km"

# Aqua is used first for consistency.
# Other sensors are backup if Aqua file is missing or failed.
SENSORS = [
    "AQUA_MODIS",
    "TERRA_MODIS",
    "SNPP_VIIRS",
    "NOAA20_VIIRS",
    "S3A_OLCI",
    "S3B_OLCI"
]



# 2) INSTALL AND IMPORT PACKAGES

!pip -q install xarray netCDF4 h5netcdf dask tqdm pandas numpy

import os
import gc
import re
import glob
import calendar
import subprocess
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

print("Product:", PRODUCT_CODE)
print("Original variable:", VARIABLE_NAME)
print("Final variable:", OUTPUT_VARIABLE)
print("Raw folder:", RAW_DIR)
print("Subset folder:", SUB_DIR)
print("Temporary standardized folder:", temp_dir)
print("Processed folder:", final_dir)
print("Final output:", final_out)




# 3) HELPER FUNCTIONS FOR DOWNLOAD AND SUBSET

def month_start_end(year, month):
    """Return NASA monthly start and end date in YYYYMMDD format."""
    start = f"{year}{month:02d}01"
    last_day = calendar.monthrange(year, month)[1]
    end = f"{year}{month:02d}{last_day:02d}"
    return start, end


def build_filename(sensor, year, month):
    """
    Build NASA Ocean Color monthly L3m filename.
    """
    start, end = month_start_end(year, month)
    return f"{sensor}.{start}_{end}.L3m.MO.{PRODUCT_CODE}.{VARIABLE_NAME}.{RESOLUTION}.nc"


def is_real_netcdf(path):
    """Check whether downloaded file is a real NetCDF file, not an HTML error page."""
    if not os.path.exists(path):
        return False

    if os.path.getsize(path) < 10000:
        return False

    with open(path, "rb") as f:
        head = f.read(500).lower()

    if b"<!doctype html" in head or b"<html" in head:
        return False

    return True


def wget_download(url, out_path):
    """Download a file from NASA Ocean Color using wget and Earthdata cookies."""
    cmd = [
        "wget",
        "--load-cookies", "/root/.urs_cookies",
        "--save-cookies", "/root/.urs_cookies",
        "--keep-session-cookies",
        "--auth-no-challenge=on",
        "-q",
        "-O", out_path,
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if is_real_netcdf(out_path):
        return True, "OK"

    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except:
            pass

    return False, result.stderr[-500:]


def subset_antarctic(infile, outfile):
    """Open monthly NASA file, keep target variable, rename coordinates, and subset region."""
    ds = xr.open_dataset(infile)

    if VARIABLE_NAME not in ds.data_vars:
        raise ValueError(
            f"Variable '{VARIABLE_NAME}' not found. Available variables: {list(ds.data_vars)}"
        )

    # Keep only selected variable and remove palette/extra variables
    ds = ds[[VARIABLE_NAME]]

    # Standard coordinate names
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})

    # Convert longitude from 0–360 to -180–180 if needed
    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.sortby("lon")

    # Select latitude and longitude range
    # NASA latitude is often descending: 89.98 to -89.98
    if float(ds.lat[0]) > float(ds.lat[-1]):
        ds_sub = ds.sel(
            lat=slice(LAT_MAX, LAT_MIN),
            lon=slice(LON_MIN, LON_MAX)
        )
    else:
        ds_sub = ds.sel(
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX)
        )

    ds_sub[VARIABLE_NAME] = ds_sub[VARIABLE_NAME].astype("float32")

    ds_sub[VARIABLE_NAME].attrs["source"] = f"NASA Ocean Color monthly L3m {PRODUCT_CODE}"
    ds_sub.attrs["processing"] = (
        f"Monthly {OUTPUT_VARIABLE} file downloaded, palette removed, "
        f"subset to latitude {LAT_MIN} to {LAT_MAX}, longitude {LON_MIN} to {LON_MAX}."
    )

    encoding = {
        VARIABLE_NAME: {
            "zlib": True,
            "complevel": 5,
            "shuffle": True,
            "_FillValue": np.float32(-9999.0),
            "dtype": "float32"
        }
    }

    ds_sub.to_netcdf(outfile, encoding=encoding)

    ds.close()
    ds_sub.close()
    gc.collect()



# 4) FULL DOWNLOAD LOOP: Monthly data START_YEAR–END_YEAR
#    Output: monthly subset files only, no merge here

records = []

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):

        ym = f"{year}-{month:02d}"
        success = False

        for sensor in SENSORS:

            filename = build_filename(sensor, year, month)
            url = f"{BASE_URL}/{filename}"

            raw_path = os.path.join(RAW_DIR, filename)
            sub_path = os.path.join(SUB_DIR, f"{OUTPUT_VARIABLE}_SO_{ym}_{sensor}.nc")

            # If subset already exists, skip
            if os.path.exists(sub_path) and os.path.getsize(sub_path) > 10000:
                print(f"Already exists {ym}: {sensor}")

                records.append({
                    "month": ym,
                    "sensor_used": sensor,
                    "status": "already_subset",
                    "raw_file": raw_path,
                    "subset_file": sub_path
                })

                success = True
                break

            # Download raw file if needed
            if not is_real_netcdf(raw_path):
                ok, msg = wget_download(url, raw_path)
            else:
                ok, msg = True, "already_raw"

            # Subset if download successful
            if ok:
                try:
                    subset_antarctic(raw_path, sub_path)

                    print(f"OK {ym}: {sensor}")

                    records.append({
                        "month": ym,
                        "sensor_used": sensor,
                        "status": "downloaded_and_subset",
                        "raw_file": raw_path,
                        "subset_file": sub_path
                    })

                    success = True
                    break

                except Exception as e:
                    print(f"Subset failed {ym} {sensor}: {e}")

                    records.append({
                        "month": ym,
                        "sensor_used": sensor,
                        "status": f"subset_failed: {e}",
                        "raw_file": raw_path,
                        "subset_file": ""
                    })

            else:
                records.append({
                    "month": ym,
                    "sensor_used": sensor,
                    "status": f"download_failed: {msg}",
                    "raw_file": filename,
                    "subset_file": ""
                })

        if not success:
            print(f"MISSING {ym}: no valid sensor file downloaded")

            records.append({
                "month": ym,
                "sensor_used": "NONE",
                "status": "missing_all_sensors",
                "raw_file": "",
                "subset_file": ""
            })



# Save download log

log_df = pd.DataFrame(records)
log_df.to_csv(log_csv, index=False)

print("\nDownload log saved:")
print(log_csv)

print("\nSummary:")
print(log_df["status"].value_counts())



# 5) CHECK DOWNLOADED MONTHLY SUBSET FILES


monthly_files = sorted(glob.glob(os.path.join(SUB_DIR, f"{OUTPUT_VARIABLE}_SO_*.nc")))

print(f"Total {OUTPUT_VARIABLE} monthly subset files:", len(monthly_files))
print("Expected monthly files:", (END_YEAR - START_YEAR + 1) * 12)

print("\nFirst 5:")
print(monthly_files[:5])

print("\nLast 5:")
print(monthly_files[-5:])

if os.path.exists(log_csv):
    log_df = pd.read_csv(log_csv)

    print("\nDownload summary:")
    print(log_df["status"].value_counts())

    missing = log_df[log_df["status"] == "missing_all_sensors"]

    print("\nMissing months:")
    if len(missing) == 0:
        print("No missing months.")
    else:
        print(missing[["month", "sensor_used", "status"]])



# 6) LOW-RAM MERGE: Monthly subset files into one NetCDF


def get_time_from_filename(file_path):
    """Extract YYYY-MM from monthly subset filename."""
    base = os.path.basename(file_path)
    match = re.search(r"(\d{4})-(\d{2})", base)

    if match is None:
        raise ValueError(f"Could not find YYYY-MM in filename: {base}")

    year = int(match.group(1))
    month = int(match.group(2))

    return pd.Timestamp(year=year, month=month, day=1)


def safe_open_dataset(file_path):
    """Try netcdf4 first, then h5netcdf."""
    last_error = None

    for engine in ["netcdf4", "h5netcdf"]:
        try:
            ds = xr.open_dataset(file_path, engine=engine)
            return ds
        except Exception as e:
            last_error = str(e)

    raise OSError(last_error)


files = sorted(glob.glob(os.path.join(SUB_DIR, "*.nc")))

print(f"{OUTPUT_VARIABLE} monthly files found:", len(files))
print("First 5 files:")
print(files[:5])

if len(files) == 0:
    raise FileNotFoundError(f"No {OUTPUT_VARIABLE} monthly files found. Check SUB_DIR path.")

standardized_files = []
skipped_files = []

for i, f in enumerate(files, start=1):
    base = os.path.basename(f)
    file_time = get_time_from_filename(f)
    ym = file_time.strftime("%Y-%m")

    out_temp = os.path.join(temp_dir, f"{OUTPUT_VARIABLE}_standardized_{ym}.nc")

    if os.path.exists(out_temp):
        print(f"[{i}/{len(files)}] Already standardized:", os.path.basename(out_temp))
        standardized_files.append(out_temp)
        continue

    print(f"\n[{i}/{len(files)}] Processing:", base)

    ds = None

    try:
        # Open one file only
        ds = safe_open_dataset(f)

        # Rename coordinates if needed
        rename_dict = {}

        if "latitude" in ds.coords:
            rename_dict["latitude"] = "lat"

        if "longitude" in ds.coords:
            rename_dict["longitude"] = "lon"

        if rename_dict:
            ds = ds.rename(rename_dict)

        # Check coordinates
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(f"lat/lon coordinates not found. Coordinates: {list(ds.coords)}")

        # Detect variable
        data_var = None

        for v in ds.data_vars:
            if v.lower() == VARIABLE_NAME.lower() or v.lower() == OUTPUT_VARIABLE.lower():
                data_var = v
                break

        if data_var is None:
            raise ValueError(f"{OUTPUT_VARIABLE} variable not found. Variables: {list(ds.data_vars)}")

        # Keep only selected variable
        ds = ds[[data_var]]

        # Rename variable to standard name
        if data_var != OUTPUT_VARIABLE:
            ds = ds.rename({data_var: OUTPUT_VARIABLE})

        # Latitude subset
        if ds.lat.values[0] < ds.lat.values[-1]:
            ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX))
        else:
            ds = ds.sel(lat=slice(LAT_MAX, LAT_MIN))

        # Longitude subset
        ds = ds.sel(lon=slice(LON_MIN, LON_MAX))

        # Remove existing time if present
        if "time" in ds.dims:
            ds = ds.squeeze("time", drop=True)

        if "time" in ds.coords:
            ds = ds.drop_vars("time")

        # Add correct monthly time
        ds = ds.expand_dims(time=[file_time])

        # Convert to float32
        ds[OUTPUT_VARIABLE] = ds[OUTPUT_VARIABLE].astype("float32")

        # Chunk for output
        ds = ds.chunk({"time": 1, "lat": 360, "lon": 720})

        encoding = {
            OUTPUT_VARIABLE: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": -9999.0
            }
        }

        # Save one standardized monthly file
        with ProgressBar():
            ds.to_netcdf(
                out_temp,
                format="NETCDF4",
                engine="netcdf4",
                encoding=encoding
            )

        ds.close()
        gc.collect()

        standardized_files.append(out_temp)
        print("✅ Saved:", out_temp)

    except Exception as e:
        skipped_files.append({
            "file": base,
            "path": f,
            "reason": str(e)
        })

        print("❌ Skipped:", base)
        print("Reason:", str(e))

        try:
            if ds is not None:
                ds.close()
        except:
            pass

        gc.collect()



# Save skipped-file report
if len(skipped_files) > 0:
    pd.DataFrame(skipped_files).to_csv(skipped_csv, index=False)
    print("\nSkipped file report saved:")
    print(skipped_csv)
else:
    print("\nNo files skipped.")



# Open standardized files lazily

standardized_files = sorted(glob.glob(os.path.join(temp_dir, f"{OUTPUT_VARIABLE}_standardized_*.nc")))

print("\nStandardized monthly files found:", len(standardized_files))

if len(standardized_files) == 0:
    raise RuntimeError("No standardized files found. Cannot merge.")

ds_final = xr.open_mfdataset(
    standardized_files,
    combine="by_coords",
    chunks={"time": 12, "lat": 360, "lon": 720},
    engine="netcdf4",
    parallel=False
)

ds_final = ds_final[[OUTPUT_VARIABLE]]
ds_final = ds_final.sortby("time")

# Remove duplicate months if any
_, unique_index = np.unique(ds_final["time"].values, return_index=True)
unique_index = sorted(unique_index)
ds_final = ds_final.isel(time=unique_index)

ds_final[OUTPUT_VARIABLE] = ds_final[OUTPUT_VARIABLE].astype("float32")



# Check time completeness

print("\nFinal merged dataset:")
print(ds_final)

expected_time = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-01", freq="MS")
actual_time = pd.to_datetime(ds_final.time.values)

missing_months = expected_time.difference(actual_time)

print("\nExpected months:", len(expected_time))
print("Available months:", len(actual_time))
print("Missing months:", len(missing_months))

if len(missing_months) > 0:
    print("\nMissing months:")
    print(missing_months.strftime("%Y-%m").tolist())
else:
    print(f"\nNo missing months. All {START_YEAR}–{END_YEAR} months are available.")



# Save final merged NetCDF

encoding = {
    OUTPUT_VARIABLE: {
        "zlib": True,
        "complevel": 5,
        "shuffle": True,
        "dtype": "float32",
        "_FillValue": -9999.0
    }
}

print("\nSaving final merged NetCDF...")

with ProgressBar():
    ds_final.to_netcdf(
        final_out,
        format="NETCDF4",
        engine="netcdf4",
        encoding=encoding
    )

ds_final.close()
gc.collect()

print(f"\n✅ Final LOW-RAM {OUTPUT_VARIABLE} file saved:")
print(final_out)
print("\nDone.")



# 7) SAFE LOW-RAM CHECK FOR FINAL MERGED FILE


merged_file = final_out

if not os.path.exists(merged_file):
    raise FileNotFoundError("Merged file not found. Check the path.")

print("✅ Merged file exists:")
print(merged_file)

size_gb = os.path.getsize(merged_file) / (1024**3)
print(f"File size: {size_gb:.2f} GB")

ds = xr.open_dataset(
    merged_file,
    engine="netcdf4",
    chunks={"time": 1, "lat": 300, "lon": 600}
)

print("\nDataset:")
print(ds)

print("\nVariables:", list(ds.data_vars))
print("Dimensions:", ds.dims)

if OUTPUT_VARIABLE in ds.data_vars:
    print(f"✅ {OUTPUT_VARIABLE} variable found.")
else:
    raise ValueError(f"❌ {OUTPUT_VARIABLE} variable not found.")

for d in ["time", "lat", "lon"]:
    if d in ds.dims:
        print(f"✅ {d} dimension found.")
    else:
        raise ValueError(f"❌ {d} dimension missing.")

time_values = pd.to_datetime(ds.time.values)

print("\nTime range:")
print(time_values.min(), "to", time_values.max())
print("Total months:", len(time_values))

expected_time = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-01", freq="MS")
missing_months = expected_time.difference(time_values)

print("Expected months:", len(expected_time))
print("Available months:", len(time_values))
print("Missing months:", len(missing_months))

if len(missing_months) > 0:
    print("❌ Missing months:")
    print(missing_months.strftime("%Y-%m").tolist())
else:
    print("✅ No missing months. Time is complete.")

lat_min = float(ds.lat.min().values)
lat_max = float(ds.lat.max().values)
lon_min = float(ds.lon.min().values)
lon_max = float(ds.lon.max().values)

print("\nLatitude range:", lat_min, "to", lat_max)
print("Longitude range:", lon_min, "to", lon_max)

if lat_min >= LAT_MIN and lat_max <= LAT_MAX:
    print("✅ Latitude range is correct.")
else:
    print("⚠️ Latitude range needs checking.")

if lon_min >= LON_MIN and lon_max <= LON_MAX:
    print("✅ Longitude range is correct.")
else:
    print("⚠️ Longitude range needs checking.")

sample_months = [
    f"{START_YEAR}-01-01",
    "2012-01-01",
    "2016-01-01",
    "2020-01-01",
    f"{END_YEAR}-12-01"
]

print("\nSample month data check:")

for m in sample_months:
    if pd.Timestamp(m) in time_values:
        arr = ds[OUTPUT_VARIABLE].sel(time=m)

        # Coarsen before statistics to avoid RAM crash
        arr_small = arr.coarsen(lat=10, lon=10, boundary="trim").mean(skipna=True)

        mn = float(arr_small.min(skipna=True).compute().values)
        mx = float(arr_small.max(skipna=True).compute().values)
        mean = float(arr_small.mean(skipna=True).compute().values)
        nan_pct = float(arr_small.isnull().mean().compute().values) * 100

        print(f"\n{m}")
        print("Min:", mn)
        print("Max:", mx)
        print("Mean:", mean)
        print(f"NaN %: {nan_pct:.2f}")

        if nan_pct < 100:
            print("✅ Valid data found.")
        else:
            print("❌ Fully NaN month.")

        gc.collect()
    else:
        print(f"\n{m} not available.")

ds.close()
gc.collect()

print(f"\n✅ LOW-RAM {OUTPUT_VARIABLE} checking finished.")
