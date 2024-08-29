# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:51:47 2024

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
import os
import geopandas as gpd
import rasterio
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.rv_pipeline import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
import cv2
from shapely.geometry import Polygon
from urllib.parse import urlparse
from typing import List, Dict, Tuple
from config import get_config
from rastervision import * 
import subprocess   


# --- Configuration ---
DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH  # Use the current directory for images
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")

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
# --- Configuration ---
DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH  
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")

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

# --- Prediction and Visualization Function ---
def predict_and_visualize(image_uri, output_path):
    """Predicts and visualizes the results."""

    scene = Scene.from_geojson(
        id=os.path.basename(image_uri).split('.')[0],
        geojson_uri=LABELS_PATH,
        raster_source=RasterioSourceConfig(uris=[image_uri]),
        class_config=class_config
    )

    # Predict using the globally loaded model
    predictions = model.predict(scene) 
    
    if SAVE_VISUALIZATIONS:
      img_filename = scene.id + os.path.splitext(scene.raster_source.get_uris()[0])[1]
      output_image_path = os.path.join(output_path, f"predicted_{img_filename}")

      # Load the original image
      img = cv2.imread(scene.raster_source.get_uris()[0].replace("file://", ""))

      # Draw predicted plant locations as red dots
      for poly in predictions.get_geojson()['features']:
          if poly['properties']['class_id'] == 1:
              coords = np.array(poly['geometry']['coordinates'][0])
              for coord in coords:
                  x, y = scene.raster_source.raster_transformer.world_to_pixel(coord[0], coord[1])
                  cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

      cv2.imwrite(output_image_path, img)
      print(f"Prediction saved to: {output_image_path}")
      
# --- Main Execution Block ---
if __name__ == "__main__":
    class_config = ClassConfig(names=['planta', 'background'], colors=['green', 'black'])

    # --- Find and Annotate Image for Training ---
    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff'))
    ]
    if not image_files:
        raise FileNotFoundError("No images found in the dataset directory.")

    first_image_path = os.path.join(IMAGES_DIR, image_files[0])
    if not os.path.exists(LABELS_PATH):
        print(f"Labels file not found. Annotating the first image: {first_image_path}")
        annotate_image(first_image_path, LABELS_PATH)
    else:
        print(f"Using existing labels: {LABELS_PATH}")

    # --- Run the Training Pipeline (using subprocess) ---
    print("Running training using 'rastervision run local'")
    subprocess.run(["rastervision", "run", "local", "config.py"])

    print("Finished processing. Check the output directory for prediction results.")