# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:51:23 2024

@author: public
"""

import os
import os
import geopandas as gpd
import rasterio
import cv2
from shapely.geometry import Polygon
from urllib.parse import urlparse
import csv
import numpy as np
import geopandas as gpd
import subprocess
from shapely.geometry import Polygon
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# --- Configuration ---
DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")
OUTPUT_DIR = "plant_detection_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_local_file_uri(file_path):
    file_path = os.path.abspath(file_path)
    return urlparse(f"file://{file_path}").geturl()

def create_geojson_from_polygons(polygons, output_path, crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame({'geometry': [Polygon(poly) for poly in polygons]})
    gdf.crs = crs
    gdf.to_file(output_path, driver="GeoJSON")

def annotate_image(image_path, output_geojson_path):
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


# --- Data Loading and Preprocessing ---
def load_data(images_dir, labels_path):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.tif', '.tiff'))]
    images = []
    labels = []
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.moveaxis(image, 0, -1)  # Reorder dimensions to (height, width, channels)
            if image.shape[0] * image.shape[1] > 1024 * 1024:  # Adjust this threshold as needed
                image = cv2.resize(image, (1024, 1024))  # Example: Resize to 1024x1024
            images.append(image)

        # Extract corresponding label from GeoJSON
        gdf = gpd.read_file(labels_path)
        # Assuming each polygon in the GeoJSON corresponds to a plant ROI
        # and you want a binary label (1 for plant, 0 for no plant)
        label = 1 if any(gdf.geometry.intersects(Polygon())) else 0
        labels.append(label)

    return np.array(images), np.array(labels)

# --- Model Creation ---
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (plant/no plant)
    return model

# --- Training ---
if __name__ == "__main__":
    images = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))
    ]
    if not images:
        raise FileNotFoundError("No images found in the dataset directory.")
    first_image_path = os.path.join(DATA_PATH, images[0])
    if not os.path.exists(LABELS_PATH):
        annotate_image(first_image_path, LABELS_PATH)
        
    images, labels = load_data(DATA_PATH, LABELS_PATH)
    
    input_shape = images[0].shape
    model = create_model(input_shape)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=10, batch_size=8)  # Adjust epochs as needed

    # --- Save the Model ---
    model.save(os.path.join(OUTPUT_DIR, 'plant_detection_model.h5'))
    print("Model saved to:", os.path.join(OUTPUT_DIR, 'plant_detection_model.h5'))