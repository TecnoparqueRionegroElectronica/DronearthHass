# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:21:03 2024

@author: public
"""

# config.py

from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.rv_pipeline import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
import subprocess
import numpy as np
from rastervision import *


def get_config(runner):
    class_config = ClassConfig(names=['planta', 'background'], colors=['green', 'black'])
    chip_sz = 256

    # --- Dataset Configuration ---
    train_scenes = []
    validation_scenes = []
    for img_file in os.listdir(IMAGES_DIR):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff')):
            img_path = os.path.join(IMAGES_DIR, img_file)
            image_uri = f"file://{img_path}" 

            train_scenes.append(
                SceneConfig(
                    id=f"train_{img_file}",
                    raster_source=RasterioSourceConfig(uris=[image_uri]),
                    label_source=SemanticSegmentationLabelSourceConfig(
                        raster_source=RasterizedSourceConfig(
                            vector_source=GeoJSONVectorSourceConfig(uri=LABELS_PATH)
                        )
                    )
                )
            )
            validation_scenes.append(
                SceneConfig(
                    id=f"val_{img_file}",
                    raster_source=RasterioSourceConfig(uris=[image_uri]),
                    label_source=SemanticSegmentationLabelSourceConfig(
                        raster_source=RasterizedSourceConfig(
                            vector_source=GeoJSONVectorSourceConfig(uri=LABELS_PATH)
                        )
                    )
                )
            )

    dataset = SemanticSegmentationDataConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=validation_scenes
    )

    # --- Backend Configuration ---
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
        run_tensorboard=True,
        predict_chip_sz=chip_sz
    )

    # --- Main Configuration ---
    return SemanticSegmentationConfig(
        root_uri=OUTPUT_DIR,
        dataset=dataset,
        backend=backend,
        predict_options=SemanticSegmentationPredictOptions(
            chip_sz=chip_sz,
            output_strategy=SemanticSegmentationOutputStrategy.separate_files
        )
    )