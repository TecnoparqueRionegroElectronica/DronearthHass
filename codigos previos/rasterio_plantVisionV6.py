# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:24:56 2024
@author: public
"""
import os
import geopandas as gpd
import rasterio
import cv2
from shapely.geometry import Polygon
from urllib.parse import urlparse
import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50

# --- Configuration ---
DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")
OUTPUT_DIR = "plant_detection_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 4
EPOCHS = 10
USE_PRETRAINED = True  # Set to True to use ResNet50

# Utility functions 
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
def load_data(images_dir, labels_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, use_pretrained=USE_PRETRAINED):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.tif', '.tiff'))]
    images = []
    labels = []

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        # Open image with rasterio
        with rasterio.open(image_path) as src:
            image = src.read(
                out_shape=(src.count, img_height, img_width),
                resampling=rasterio.enums.Resampling.bilinear
            )
            image = np.moveaxis(image, 0, -1)
            image = image.astype(np.float32)

            # Repeat grayscale to 3 channels if using pretrained model
            if use_pretrained:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

            images.append(image)
        # Extract corresponding label from GeoJSON
        gdf = gpd.read_file(labels_path)
        label = 1 if any(gdf.geometry.intersects(Polygon())) else 0
        labels.append(label)
    return np.array(images), np.array(labels)

# --- Model Creation ---
def create_model(input_shape, num_classes=1, use_pretrained=USE_PRETRAINED, trainable_layers=10):
    if use_pretrained:
        # Ensure input shape has 3 channels for ResNet
        if input_shape[-1] != 3:
            input_shape = (*input_shape[:-1], 3)
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
    else:
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        # Build the model up to Flatten to calculate output shape
        model.build(input_shape=(None, *input_shape))  
        dense_input_size = model.layers[-1].output_shape[1]
        model.add(Dense(dense_input_size // 2, activation='relu'))  
        model.add(Dense(num_classes, activation='sigmoid'))
    return model

def classify_patches(model, image, patch_size=(64, 64)):
    height, width, channels = image.shape
    predictions = []

    for y in range(0, height - patch_size[1] + 1, patch_size[1]):
        for x in range(0, width - patch_size[0] + 1, patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            patch = np.expand_dims(patch, axis=0) # Add batch dimension
            prediction = model.predict(patch)
            predictions.append(((x, y), prediction[0][0])) # Store position and prediction

    return predictions

def visualize_predictions(image, predictions, threshold=0.5):
    for (x, y), prediction in predictions:
        if prediction >= threshold: # If prediction is above threshold, it's considered a plant
            cv2.rectangle(image, (x, y), (x + 64, y + 64), (0, 255, 0), 2) # Draw green rectangle around detected plants
    return image

# --- Training ---
if __name__ == "__main__":
    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))
    ]
    if not image_files:
        raise FileNotFoundError("No images found in the dataset directory.")
    first_image_path = os.path.join(DATA_PATH, image_files[0])
    if not os.path.exists(LABELS_PATH):
        annotate_image(first_image_path, LABELS_PATH)
        
    # Load data, passing the USE_PRETRAINED flag
    images, labels = load_data(DATA_PATH, LABELS_PATH, use_pretrained=USE_PRETRAINED)
    input_shape = images[0].shape
    model = create_model(input_shape, use_pretrained=USE_PRETRAINED)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # --- Save the Model ---
    model.save(os.path.join(OUTPUT_DIR, 'plant_detection_model.h5'))
    print("Model saved to:", os.path.join(OUTPUT_DIR, 'plant_detection_model.h5'))
    # --- Prediction on Unseen Images ---
    for img_file in image_files: # Iterate over image filenames 
        print(f"Analyzing: {img_file}")
        img_path = os.path.join(IMAGES_DIR, img_file)

        # Load and preprocess image
        with rasterio.open(img_path) as src:
            image = src.read(
                out_shape=(src.count, IMG_HEIGHT, IMG_WIDTH),
                resampling=rasterio.enums.Resampling.bilinear
            )
            image = np.moveaxis(image, 0, -1) 
            image = image.astype(np.float32) 

        # Predict on patches
        predictions = classify_patches(model, image)
        # Visualize and save
        output_image = visualize_predictions(image, predictions)
        output_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file}")
        cv2.imwrite(output_path, output_image)