# -*- coding: utf-8 -*-
    
# config.py
    
import os
import csv
from os import *
from rastervision.core.rv_pipeline import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
import numpy as np
from rastervision import *
from rastervision.core.data import ClassConfig
from rastervision.pytorch_backend.examples.utils import get_scene_info 
from rastervision.pytorch_backend.examples.utils import *
    
    
    # Update paths for your data
TRAIN_SCENES_CSV = "train-scenes.csv"
VAL_SCENES_CSV = "val-scenes.csv"
AOI_PATH = "aoi.geojson"  # If using AOIs
DATA_PATH = os.getcwd()
LABELS_PATH = os.path.join(DATA_PATH, "labels.geojson")
TRAIN_SCENES_CSV = os.path.join(DATA_PATH, "train-scenes.csv")
VAL_SCENES_CSV = os.path.join(DATA_PATH, "val-scenes.csv")
OUTPUT_DIR = "plant_detection_output"
SAVE_VISUALIZATIONS = True
    
    # --- Custom get_scene_info Function ---
def get_scene_info(csv_path):
        """Reads scene information (image and label URI) from a CSV file."""
        scene_info = []
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                scene_info.append((row[0], row[1])) # Image URI in column 0, label URI in column 1
        return scene_info
    
def get_config(runner):
        
        class_config = ClassConfig(names=['planta', 'background'])
        chip_sz = 256
        # --- Load Scene Information from CSVs ---
        def make_scene(scene_info: Tuple[str, str]) -> SceneConfig:
         (raster_uri, label_uri) = scene_info
         id = os.path.splitext(os.path.basename(raster_uri))[0]

        raster_source = RasterioSourceConfig(
            uris=[raster_uri],  # Pass as a list of URIs
            channel_order=[0, 1, 2]
        )

        label_source = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uris=[label_uri],  # Pass as a list of URIs
                transformers=[
                    ClassInferenceTransformerConfig(default_class_id=1)
                ]
            ),
            ioa_thresh=0.5
        )

        # Return the correctly constructed SceneConfig object
        return SceneConfig(
            id=id, 
            raster_source=raster_source, 
            label_source=label_source, 
            aoi_uris=[AOI_PATH] if AOI_PATH else None  # Add AOI only if defined
        )
        train_scenes = [make_scene(info) for info in get_scene_info(os.path.join(DATA_PATH, TRAIN_SCENES_CSV))]
        val_scenes = [make_scene(info) for info in get_scene_info(os.path.join(DATA_PATH, VAL_SCENES_CSV))]        
               
        scene_dataset = DatasetConfig(
            class_config=class_config,
            train_scenes=train_scenes,
            validation_scenes=val_scenes
        )
    
        backend = PyTorchSemanticSegmentationConfig(
            data=SemanticSegmentationGeoDataConfig(
                scene_dataset=dataset,
                sampling=WindowSamplingConfig(
                    method=WindowSamplingMethod.random,
                    size=chip_sz,
                    max_windows=10
                )
            ),
            model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
            solver=SolverConfig(lr=1e-4, num_epochs=3, batch_sz=2),
            gpu_ids=[0],
            log_tensorboard=True,
            run_tensorboaxrd=True
        )
        # --- Main Configuration ---
        return ChipClassificationConfig(
            root_uri=OUTPUT_DIR,
            dataset=scene_dataset,
            backend=backend,
            chip_options=chip_options,
            predict_options=PredictOptions(chip_sz=chip_sz)
        ) 