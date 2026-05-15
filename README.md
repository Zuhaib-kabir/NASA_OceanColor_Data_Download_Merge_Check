# NASA_OceanColor_Data_Download_Merge_Check
Python workflow for downloading, merging, standardizing, and validating NASA Ocean Color NetCDF datasets with low RAM usage.


This repository provides a Python / Google Colab workflow for downloading, subsetting, merging, and checking monthly NASA Ocean Color data for marine science, oceanography, biogeochemistry, ecosystem, and climate studies.

The workflow is designed to make long-term NASA Ocean Color data processing easier, especially when working with large monthly NetCDF datasets.

---

## Why this workflow is useful

NASA Ocean Color data are very important for studying marine ecosystems, ocean biogeochemistry, phytoplankton dynamics, carbon cycling, light availability, water clarity, and climate-related ocean changes.

However, downloading NASA Ocean Color data manually can be time-consuming. A common problem is that data from one sensor may be unavailable for some months or regions. For example, AQUA_MODIS is often used as the main sensor, but sometimes data may be missing. In that case, users may need to manually search other sensors.

This workflow solves that problem by automatically trying multiple sensors.

Sensor order used in the workflow:

```python
AQUA_MODIS
TERRA_MODIS
SNPP_VIIRS
NOAA20_VIIRS
S3A_OLCI
S3B_OLCI
