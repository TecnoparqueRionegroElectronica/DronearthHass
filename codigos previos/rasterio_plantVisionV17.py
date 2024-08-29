# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:44:11 2024

@author: public
"""

import os
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import cv2
from shapely.geometry import *
from urllib.parse import urlparse
import csv
import numpy as np
import tensorflow as tf
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


IMG_HEIGHT = 128  # Reduced patch size
IMG_WIDTH = 128   # Reduced patch size
BATCH_SIZE = 2  # Decreased batch size
EPOCHS = 15     # Increased epochs
USE_PRETRAINED = True  # Set to True to use ResNet50
PATCH_SIZE = (IMG_HEIGHT, IMG_WIDTH)  # Match input size for pretrained model

# --- GPU Configuration (Attempt within Spyder) ---
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    print("Using GPU for processing.")
else:
    print("No GPU detected. Using CPU for processing.")

# Utility functions
def get_local_file_uri(file_path):
    file_path = os.path.abspath(file_path)
    return urlparse(f"file://{file_path}").geturl()

def create_geojson_from_polygons(polygons, output_path, crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame({'geometry': [Polygon(poly) for poly in polygons]})
    gdf.crs = crs
    gdf.to_file(output_path, driver="GeoJSON")

def annotate_image(image_path, output_geojson_path):
    with rasterio.open(image_path) as src:
        # Calculate width and height of the quarter section
        width = src.width // 2
        height = src.height // 2

        # Read the lower-left quarter, handling different channel counts
        window = Window(0, src.height - height, width, height)
        if src.count == 1:
            img_data = src.read(1, window=window)  # Read single band
            img_data = np.expand_dims(img_data, axis=2)  # Add channel dimension
        else:
            img_data = src.read([1, 2, 3], window=window)
            img_data = np.moveaxis(img_data, 0, -1)

        # Convert to 8-bit for OpenCV compatibility
        img_data = (img_data * 255.0 / img_data.max()).astype(np.uint8)

        # Create a compatible OpenCV image (ensure contiguous data)
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Annotating: {image_path} (Lower-left quarter)")

    cv2.namedWindow("Image Annotation", cv2.WINDOW_NORMAL)
    polygons = []
    current_polygon = []

    def on_mouse(event, x, y, flags, param):
        nonlocal current_polygon, polygons
        if event == cv2.EVENT_LBUTTONDOWN:
            # Adjust coordinates to full image scale
            current_polygon.append([x + width, y + height])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_polygon) > 2:
                polygons.append(current_polygon.copy())
                current_polygon = []
                # Adjust coordinates for drawing on the quarter image
                cv2.polylines(img, [np.array([[pt[0] - width, pt[1] - height] for pt in polygons[-1]], np.int32)],
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
def load_data(images_dir, labels_path, patch_size=PATCH_SIZE, use_pretrained=USE_PRETRAINED):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.tif', '.tiff'))]
    images = []
    labels = []

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)

        # Open image with rasterio
        with rasterio.open(image_path) as src:
            for y in range(0, src.height - patch_size[1] + 1, patch_size[1]):
                for x in range(0, src.width - patch_size[0] + 1, patch_size[0]):
                    window = Window(x, y, patch_size[0], patch_size[1])
                    image = src.read(window=window)
                    image = np.moveaxis(image, 0, -1)  # Reorder to (height, width, channels)
                    image = image.astype(np.float32)

                  
                    # Handle 4-channel images (remove alpha channel)
                    if image.shape[-1] == 4:
                        image = image[:,:,:3] 

                    # Convert to 3 channels if using pretrained model (ResNet expects 3 channels)
                    if use_pretrained and image.shape[-1] != 3:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                    # Normalize data
                    image = image / 255.0

                    images.append(image)

                    # Extract corresponding label from GeoJSON
                    gdf = gpd.read_file(labels_path)
                    patch_polygon = box(x, y, x + patch_size[0], y + patch_size[1])
                    label = 1 if any(gdf.geometry.intersects(patch_polygon)) else 0
                    labels.append(label)

    return np.array(images), np.array(labels)

# --- Model Creation ---
def create_model(input_shape, num_classes=1, use_pretrained=USE_PRETRAINED, trainable_layers=10):
    if use_pretrained:
        # Ensure input_shape has 3 channels
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
        return model # Added return statement for pretrained branch

    else:
        model = Sequential()
        # Correct input_shape for the first Conv2D layer (without batch dimension)
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))  
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())

        # Calculate Dense layer input size after Flatten - No need to build the model here
        dense_input_size = np.prod(model.output_shape[1:])
        model.add(Dense(dense_input_size // 2, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
        return model # Added return statement for non-pretrained branch

def classify_patches(model, image, patch_size=PATCH_SIZE, use_pretrained=USE_PRETRAINED):  # Add use_pretrained argument
    height, width, channels = image.shape
    predictions = []

    for y in range(0, height - patch_size[1] + 1, patch_size[1]):
        for x in range(0, width - patch_size[0] + 1, patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]

            # Handle 4-channel images (remove alpha channel)
            if patch.shape[-1] == 4:
                patch = patch[:, :, :3]

            # Convert to 3 channels if using pretrained model and patch is grayscale
            if use_pretrained and patch.shape[-1] != 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)

            patch = np.expand_dims(patch, axis=0)
            prediction = model.predict(patch)

            # Save patch with prediction for debugging
            patch_filename = f"patch_{y}_{x}_pred_{prediction[0][0]:.2f}.png"
            patch_path = os.path.join(OUTPUT_DIR, patch_filename)
            cv2.imwrite(patch_path, patch * 255.0)

            predictions.append(((x, y), prediction[0][0]))

    return predictions

def visualize_predictions(image, predictions, threshold=0.7): # Adjust threshold
    output_image = image.copy() # Create a copy to draw on 
    for (x, y), prediction in predictions:
        if prediction >= threshold:
            # Correct rectangle coordinates for visualization
            cv2.rectangle(output_image, (x, y), (x + PATCH_SIZE[0], y + PATCH_SIZE[1]), (0, 255, 0), 2)  
    return output_image

# --- Function to save predicted ROIs as GeoJSON ---
def save_predicted_rois(predictions, image_path, output_geojson_path, threshold=0.3):
    with rasterio.open(image_path) as src:
        transform = src.transform
    predicted_polygons = []
    for (x, y), prediction in predictions:
        if prediction >= threshold:
            polygon = box(x, y, x + PATCH_SIZE[0], y + PATCH_SIZE[1])
            # Transform polygon to image coordinates
            polygon = transform * polygon
            predicted_polygons.append(polygon)

    create_geojson_from_polygons(predicted_polygons, output_geojson_path)
# --- Enhanced save_image Function ---
def save_image(image, output_path):
    # Try saving as TIFF using rasterio
    try:
        with rasterio.open(output_path[:-4] + ".tif", 'w', **profile) as dst:
            dst.write(np.moveaxis(image, -1, 0))  # Reorder channels for rasterio
        print(f"Image saved as: {output_path[:-4]}.tif")
        return

    except Exception as e:
        print(f"Error saving as TIFF: {e}")

    # Try saving as PNG using OpenCV
    try:
        cv2.imwrite(output_path[:-4] + ".png", image)
        print(f"Image saved as: {output_path[:-4]}.png")
        return

    except Exception as e:
        print(f"Error saving as PNG: {e}")

    # Try saving as JPG using OpenCV
    try:
        cv2.imwrite(output_path[:-4] + ".jpg", image)
        print(f"Image saved as: {output_path[:-4]}.jpg")
        return

    except Exception as e:
        print(f"Error saving as JPG: {e}")

    print(f"Could not save image to any format: {output_path}")


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

    # --- Load Pretrained Model if Available ---
    model_path_h5 = os.path.join(OUTPUT_DIR, 'plant_detection_model.h5')
    model_path_keras = os.path.join(OUTPUT_DIR, 'plant_detection_model.keras')
    
    if os.path.exists(model_path_h5):
        print("Loading pretrained model from:", model_path_h5)
        model = keras.models.load_model(model_path_h5)
    elif os.path.exists(model_path_keras):
        print("Loading pretrained model from:", model_path_keras)
        model = keras.models.load_model(model_path_keras)
    else:
        # Load data, passing the USE_PRETRAINED flag
        images, labels = load_data(DATA_PATH, LABELS_PATH, patch_size=PATCH_SIZE, use_pretrained=USE_PRETRAINED)
        input_shape = images[0].shape

        model = create_model(input_shape, use_pretrained=USE_PRETRAINED)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(images, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
        # --- Save the Model ---
        model.save(os.path.join(OUTPUT_DIR, 'plant_detection_model.h5'))
        model.save(os.path.join(OUTPUT_DIR, 'plant_detection_model.keras'))
        print("Model saved to:", os.path.join(OUTPUT_DIR, 'plant_detection_model.h5'))

    # --- Prediction on Unseen Images ---
    for img_file in image_files:
        print(f"Analyzing: {img_file}")
        img_path = os.path.join(IMAGES_DIR, img_file)

        # Load and preprocess image
        with rasterio.open(img_path) as src:
            height = src.height
            width = src.width
            profile = src.profile  # Get rasterio profile for saving
            image = src.read()
            image = np.moveaxis(image, 0, -1)
            image = image.astype(np.float32)

        # Predict on patches (no channel conversion here)
        predictions = classify_patches(model, image, patch_size=PATCH_SIZE, use_pretrained=USE_PRETRAINED)

        # Visualize and save
        output_image = visualize_predictions(image, predictions, threshold=0.2)  # Adjusted threshold
        output_image_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file}")
        save_image(output_image, output_image_path)  # Use enhanced save_image function

        # Save predicted ROIs as GeoJSON
        output_geojson_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file[:-4]}.geojson")
        save_predicted_rois(predictions, img_path, output_geojson_path, threshold=0.1)  # Adjusted threshold
