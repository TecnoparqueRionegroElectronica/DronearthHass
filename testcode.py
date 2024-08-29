# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:10:27 2024

@author: public work
"""
import os
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import Window
from shapely.geometry import box, Point

DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")
PATCH_SIZE = (128, 128)

image_file = [f for f in os.listdir(IMAGES_DIR) 
              if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))][0] 
image_path = os.path.join(IMAGES_DIR, image_file)

# Open image and GeoJSON
with rasterio.open(image_path) as src:
    gdf = gpd.read_file(LABELS_PATH)
    polygons = gdf.geometry.to_list()

    # Extract a few patches and print labels
    for y in range(0, src.height - PATCH_SIZE[1] + 1, PATCH_SIZE[1]):
        for x in range(0, src.width - PATCH_SIZE[0] + 1, PATCH_SIZE[0]):
            window = Window(x, y, PATCH_SIZE[0], PATCH_SIZE[1])
            image = src.read(window=window)
            image = np.moveaxis(image, 0, -1)  

            center_point = Point(x + PATCH_SIZE[0] / 2, y + PATCH_SIZE[1] / 2)
            label = 1 if any(polygon.contains(center_point) for polygon in polygons) else 0
            print(f"Patch at ({x}, {y}): Shape - {image.shape}, Label - {label}") 
