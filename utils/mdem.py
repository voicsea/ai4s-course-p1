import numpy as np
import os
import rasterio

dem_dir = r"D:\rs_image\ImageToDEM\singleRGBNormalization\DEM"

global_dem_max = -np.inf
global_dem_min = np.inf

for fname in os.listdir(dem_dir):
    if fname.endswith('.tif'):
        dem_path = os.path.join(dem_dir, fname)

        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            dem_data_clean = np.nan_to_num(dem_data, nan=0.0)

        current_dem_max = dem_data_clean.max()
        current_dem_min = dem_data_clean.min()

        global_dem_max = max(current_dem_max, global_dem_max)
        global_dem_min = min(current_dem_min, global_dem_min)

print("\n" + "="*50)
print(dem_data_clean)
print(f"DEM数据集全局最大值：{global_dem_max:.3f}")
print(f"DEM数据集全局最小值：{global_dem_min:.3f}")
print("="*50)