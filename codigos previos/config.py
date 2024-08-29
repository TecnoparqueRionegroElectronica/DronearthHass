# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:52:42 2024

@author: public
"""

# config.py

from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.rv_pipeline import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *

def get_config(runner):
    class_config = ClassConfig(names=['planta', 'background'], colors=['green', 'black'])
    chip_sz = 256

    # --- Dataset Configuration ---
    train_image_uri = "/pexels-tomfisk-1859828.jpg"  # Replace with your training image URI
    val_image_uri = "/pexels-tomfisk-1859828.jpg"   # Replace with your validation image URI

    dataset = SemanticSegmentationDatasetConfig(
        class_config=class_config,
        train_scenes=[SceneConfig(id='train',
                                 raster_source=RasterioSourceConfig(uris=[train_image_uri]),
                                 label_source=SemanticSegmentationLabelSourceConfig(
                                     raster_source=RasterizedSourceConfig(
                                         vector_source=GeoJSONVectorSourceConfig(uri=LABELS_PATH)
                                     )
                                 ))],
        validation_scenes=[SceneConfig(id='val',
                                      raster_source=RasterioSourceConfig(uris=[val_image_uri]),
                                      label_source=SemanticSegmentationLabelSourceConfig(
                                          raster_source=RasterizedSourceConfig(
                                              vector_source=GeoJSONVectorSourceConfig(uri=LABELS_PATH)
                                          )
                                      ))]
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
        run_tensorboard=True
    )

    return SemanticSegmentationConfig(
        root_uri=OUTPUT_DIR,
        dataset=dataset,
        backend=backend,
        predict_options=SemanticSegmentationPredictOptions(chip_sz=chip_sz)
    )