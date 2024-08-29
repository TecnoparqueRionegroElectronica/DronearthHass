# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:04:02 2024

@author: public
"""

# raster_vision_plant_detection.py

import os
import geopandas as gpd
import rasterio
import cv2
from shapely.geometry import Polygon
from urllib.parse import urlparse
from config import get_config
import subprocess
import numpy as np
from rastervision import *
from rastervision.core.data import ClassConfig
from rastervision.pytorch_backend.examples.utils import get_scene_info 
from rastervision.pytorch_backend.examples.utils import *

# --- Configuration ---
DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")
TRAIN_SCENES_CSV = os.path.join(DATA_PATH, "train-scenes.csv")
VAL_SCENES_CSV = os.path.join(DATA_PATH, "val-scenes.csv")

OUTPUT_DIR = "plant_detection_output"
SAVE_VISUALIZATIONS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utility Functions ---
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

# --- Function to write image URIs to CSV ---
def write_image_uris_to_csv(images_dir, output_csv, labels_uri=None):
    """Writes image URIs to a CSV file, optionally with corresponding label URIs."""
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_uri", "label_uri"])  # Header row
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff')):
                image_uri = get_local_file_uri(os.path.join(images_dir, img_file))
                # Always write both image URI and labels URI
                writer.writerow([image_uri, labels_uri]) 
# --- Main Execution Block ---
if __name__ == "__main__":
    from rastervision.core.data import ClassConfig

    class_config = ClassConfig(names=['planta', 'background'], colors=['green', 'black'])

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

    # --- Write Image URIs to CSV ---
    write_image_uris_to_csv(IMAGES_DIR, TRAIN_SCENES_CSV, labels_uri=LABELS_PATH)
    write_image_uris_to_csv(IMAGES_DIR, VAL_SCENES_CSV, labels_uri=LABELS_PATH) 

    # --- Run the Training Pipeline (using subprocess) ---
    print("Running training using 'rastervision run local'")
    subprocess.run(["rastervision", "run", "local", "config.py", "predict"])

    print("Finished processing. Check the output directory for prediction results.")