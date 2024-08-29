# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:21:37 2024

@author: public
"""

# annotate.py

import os
import geopandas as gpd
import rasterio
import cv2
import numpy as np
from shapely.geometry import Polygon
from urllib.parse import urlparse

# --- Configuration ---
DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")

def get_local_file_uri(file_path):
    """Gets the URI for a local file."""
    file_path = os.path.abspath(file_path)
    return urlparse(f"file://{file_path}").geturl()

def create_geojson_from_polygons(polygons, output_path, crs="EPSG:4326"):
    """Creates a GeoJSON file from a list of polygons."""
    gdf = gpd.GeoDataFrame({'geometry': [Polygon(poly) for poly in polygons]})
    gdf.crs = crs
    gdf.to_file(output_path, driver="GeoJSON")
    
def annotate_image(image_path, output_geojson_path):
    """Allows manual annotation of images with polygons."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Annotating: {image_path}")

    cv2.namedWindow("Image Annotation", cv2.WINDOW_NORMAL)
    polygons = []
    current_polygon = []

    def on_mouse(event, x, y, flags, param):
        nonlocal current_polygon, polygons
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_polygon) > 2:
                polygons.append(current_polygon.copy())
                current_polygon = []
                cv2.polylines(img, [np.array(polygons[-1], np.int32)],
                              True, (0, 255, 0), 2)
                cv2.imshow("Image Annotation", img)

    cv2.setMouseCallback("Image Annotation", on_mouse)

    print("Instructions:")
    print("- Left-click to add points to the polygon.")
    print("- Right-click to complete the current polygon.")
    print("- Press 'q' to finish annotation and save the GeoJSON.")

    while True:
        cv2.imshow("Image Annotation", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    create_geojson_from_polygons(polygons, output_geojson_path)

if __name__ == "__main__":
    # --- Find and Annotate Images ---
    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))
    ]
    if not image_files:
        raise FileNotFoundError("No images found in the dataset directory.")

    if not os.path.exists(LABELS_PATH):
        print(f"Labels file not found. Annotating images...")
        for img_file in image_files:
            image_path = os.path.join(IMAGES_DIR, img_file)
            annotate_image(image_path, LABELS_PATH)
    else:
        print(f"Using existing labels: {LABELS_PATH}")