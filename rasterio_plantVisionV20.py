# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:49:07 2024

@author: public work
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
from sklearn.model_selection import train_test_split, ParameterGrid

# --- Configuration ---
DATA_PATH = os.getcwd()
IMAGES_DIR = DATA_PATH
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")
OUTPUT_DIR = "plant_detection_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_HEIGHT = 32  # Reduced patch size
IMG_WIDTH = 32   # Reduced patch size
BATCH_SIZE = 1  # Decreased batch size
EPOCHS = 5     # Reduced epochs for faster testing
USE_PRETRAINED = True  # Set to True to use ResNet50
PATCH_SIZE = (IMG_HEIGHT, IMG_WIDTH)  # Match input size for pretrained model
VALIDATION_SPLIT = 0.2  # Proportion of data for validation

# --- Utility functions ---
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
def load_data(images_dir, labels_path, patch_size=PATCH_SIZE, use_pretrained=USE_PRETRAINED, max_images=5):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    images = []
    labels = []

    # Read GeoJSON once
    gdf = gpd.read_file(labels_path)
    polygons = gdf.geometry.to_list()

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

                    # Convert to 3 channels if using pretrained model (ResNet expects 3 channels)
                    if use_pretrained and image.shape[-1] != 3:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                    # Normalize data
                    image = image / 255.0

                    images.append(image)
                    center_point = Point(x + patch_size[0] / 2, y + patch_size[1] / 2)
                    label = 1 if any(polygon.contains(center_point) for polygon in polygons) else 0
                    labels.append(label)

    return np.array(images), np.array(labels)
# --- Model Creation ---
def create_model(input_shape, num_classes=1, use_pretrained=USE_PRETRAINED, trainable_layers=25):
    if use_pretrained:
        # Ensure input_shape has 3 channels
        if input_shape[-1] != 3:
            input_shape = (*input_shape[:-1], 3)
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = True

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
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

# --- Function to classify patches of an image using TensorFlow Lite ---
def classify_patches_tflite(interpreter, image, patch_size=PATCH_SIZE):
    height, width, _ = image.shape
    predictions = []

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']

    for y in range(0, height - patch_size[1] + 1, patch_size[1]):
        for x in range(0, width - patch_size[0] + 1, patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            patch = patch / 255.0  # Normalize patch
            patch = np.expand_dims(patch, axis=0).astype(np.float32) # Add batch dimension

            # Set tensor and invoke the interpreter
            interpreter.set_tensor(input_details[0]['index'], patch)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = output_data[0][0]

            # --- DEBUG: Save patch with prediction value ---
            patch_filename = f"patch_{y}_{x}_pred_{prediction:.2f}.png"
            patch_path = os.path.join(OUTPUT_DIR, patch_filename)
            cv2.imwrite(patch_path, patch * 255.0)

            predictions.append(((x, y), prediction))

    return predictions

def visualize_predictions(image, predictions, threshold=0.5): # Adjust threshold
    output_image = image.copy() # Create a copy to draw on 
    for (x, y), prediction in predictions:
        if prediction >= threshold:
            # Correct rectangle coordinates for visualization
            cv2.rectangle(output_image, (x, y), (x + PATCH_SIZE[0], y + PATCH_SIZE[1]), (0, 255, 0), 2)  
    return output_image

# --- Function to save predicted ROIs as GeoJSON ---
def save_predicted_rois(predictions, image_path, output_geojson_path, threshold=0.5):
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

# --- Model Training Function ---
def train_model(input_shape, images, labels, epochs, batch_size, use_pretrained=USE_PRETRAINED, validation_split=VALIDATION_SPLIT):
    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=validation_split, random_state=42
    )

    model = create_model(input_shape, use_pretrained=use_pretrained)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

    # Evaluate the model on the validation set
    _, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return model, val_accuracy

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

    # --- Parameter Grid for Exploration ---
    param_grid = {
        'patch_size': [(32, 32)],  # Start with a single patch size
        'batch_size': [1],  # Start with a single batch size
        'epochs': [5],  # Start with a small number of epochs
    }

    best_accuracy = 0.0
    best_model_path = None # Store path to best model

    # --- Training Loop (Choose ResNet or Sequential) ---
    for model_type in ["resnet", "sequential"]:
        print(f"Training {model_type} model...")
        for params in ParameterGrid(param_grid):
            print(f"Training with parameters: {params}")
            patch_size = params['patch_size']
            batch_size = params['batch_size']
            epochs = params['epochs']

            # Load a limited number of images
            images, labels = load_data(DATA_PATH, LABELS_PATH, patch_size=patch_size, 
                                       use_pretrained=(model_type == "resnet"), max_images=5)
            input_shape = images[0].shape

            model, val_accuracy = train_model(
                input_shape, images, labels, epochs, batch_size, use_pretrained=(model_type == "resnet")
            )

            # --- Convert to TensorFlow Lite ---
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            # --- Save TensorFlow Lite Model ---
            tflite_model_filename = f"{model_type}_model_patch{patch_size[0]}_batch{batch_size}_epochs{epochs}.tflite"
            tflite_model_path = os.path.join(OUTPUT_DIR, tflite_model_filename)
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved to: {tflite_model_path}")

            # --- Update Best Model ---
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = tflite_model_path # Store the path to best TFLite model 

    # --- Prediction on Unseen Images (using best TFLite model) ---
    if best_model_path is not None:
        print(f"Loading best model from: {best_model_path}")
        interpreter = tf.lite.Interpreter(model_path=best_model_path)
        interpreter.allocate_tensors()

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

            # Predict on patches
            predictions = classify_patches_tflite(interpreter, image, patch_size=PATCH_SIZE)

            # --- DEBUGGING OUTPUTS ---
            # 1. Save predictions to a text file:
            with open(os.path.join(OUTPUT_DIR, f"predictions_{img_file[:-4]}.txt"), "w") as f:
                for (x, y), pred in predictions:
                    f.write(f"Patch at ({x}, {y}): Prediction = {pred:.4f}\n")

            # 2. Visualize patches with high and low predictions:
            for (x, y), pred in predictions:
                if pred > 0.7:  
                    patch = image[y:y + PATCH_SIZE[1], x:x + PATCH_SIZE[0]]
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"high_pred_patch_{y}_{x}.png"), patch * 255.0)
                if pred < 0.3:
                    patch = image[y:y + PATCH_SIZE[1], x:x + PATCH_SIZE[0]]
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"low_pred_patch_{y}_{x}.png"), patch * 255.0)

            # 3. Print summary statistics of predictions:
            preds_array = np.array([pred for (_, _), pred in predictions])
            print(f"Prediction Statistics (Mean: {preds_array.mean():.4f}, "
                  f"Std: {preds_array.std():.4f}, Max: {preds_array.max():.4f})")
            # --- END OF DEBUGGING OUTPUTS ---

            # Visualize and save image
            output_image = visualize_predictions(image, predictions, threshold=0.2)
            output_image_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file}")
            save_image(output_image, output_image_path)

            # Save predicted ROIs as GeoJSON
            output_geojson_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file[:-4]}.geojson")
            save_predicted_rois(predictions, img_path, output_geojson_path, threshold=0.1)