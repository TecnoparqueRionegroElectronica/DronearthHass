# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:42:15 2024

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

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 2  # Reduced batch size
EPOCHS = 10
USE_PRETRAINED = False  # Set to True to use ResNet50
PATCH_SIZE = (128, 128)  # Match input size for pretrained model

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

        # Read the lower-left quarter
        window = Window(0, src.height - height, width, height) 
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
                cv2.polylines(img, [np.array([[pt[0]-width, pt[1]-height] for pt in polygons[-1]], np.int32)],
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
                    image = np.moveaxis(image, 0, -1) 
                    image = image.astype(np.float32) 

                  
                    # Handle 4-channel images (remove alpha channel)
                    if image.shape[-1] == 4:
                        image = image[:,:,:3] 

                    # Convert to 3 channels if using pretrained model 
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
                patch = patch[:, :, :3]  # Select only the first 3 channels (RGB)

            # Convert to 3 channels if using pretrained model and patch is grayscale 
            if use_pretrained and patch.shape[-1] == 1:  # Check for single-channel grayscale
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)

            patch = np.expand_dims(patch, axis=0)  # Add batch dimension
            prediction = model.predict(patch)
            predictions.append(((x, y), prediction[0][0]))

    return predictions
def visualize_predictions(image, predictions, threshold=0.8): # Adjust threshold
    output_image = image.copy() # Create a copy to draw on 
    for (x, y), prediction in predictions:
        if prediction >= threshold:
            # Correct rectangle coordinates for visualization
            cv2.rectangle(output_image, (x, y), (x + PATCH_SIZE[0], y + PATCH_SIZE[1]), (0, 255, 0), 2)  
    return output_image

# --- Function to save predicted ROIs as GeoJSON ---
def save_predicted_rois(predictions, image_path, output_geojson_path, threshold=0.7):
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
        keras.saving.save_model(model, os.path.join(OUTPUT_DIR, 'plant_detection_model.keras'))
        print("Model saved to:", os.path.join(OUTPUT_DIR, 'plant_detection_model.h5'))

    # --- Prediction on Unseen Images ---
    for img_file in image_files: # Iterate over image filenames 
        print(f"Analyzing: {img_file}")
        img_path = os.path.join(IMAGES_DIR, img_file)

        # Load and preprocess image
        with rasterio.open(img_path) as src:
            height = src.height
            width = src.width
            image = src.read()
            image = np.moveaxis(image, 0, -1) 
            image = image.astype(np.float32) 

        # Predict on patches (no channel conversion here)
        predictions = classify_patches(model, image, patch_size=PATCH_SIZE)

        # Visualize and save
        output_image = visualize_predictions(image, predictions)
        output_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file}")
        cv2.imwrite(output_path, output_image)

        # Save predicted ROIs as GeoJSON
        output_geojson_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file[:-4]}.geojson")
        save_predicted_rois(predictions, img_path, output_geojson_path)