"""Mitochondria detection and ML-based tracking in microscopy videos."""

__version__ = "1.0.0"
__author__ = "Viacheslav I. Pasko"
__email__ = "paskoslava2001@yandex.ru"
__license__ = "MIT"

from .detector import MitochondriaDetector
from .features import TrackingFeatureExtractor, TrackingFeatures
from .tracker import MLTracker, TrackingAnnotationGenerator, Track
from .training import TrackingDataLoader, TrackingModelTrainer
from .visualization import TrackVisualizer, visualize_tracking_results
from .batch import (
    BatchProcessor,
    process_videos_batch,
    export_training_data,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "MitochondriaDetector",
    "TrackingFeatureExtractor",
    "TrackingFeatures",
    "Track",
    "TrackingAnnotationGenerator",
    "MLTracker",
    "TrackingDataLoader",
    "TrackingModelTrainer",
    "TrackVisualizer",
    "BatchProcessor",
    "visualize_tracking_results",
    "process_videos_batch",
    "export_training_data",
]
