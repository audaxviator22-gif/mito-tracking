"""Pairwise features for track association.

This module defines the feature vector used by the association classifier.
Given two detections (typically from consecutive frames), it produces five scalars that describe how 
likely they are to be the same mitochondria:

    1. position_distance     - how far apart the centers are, the distance is normalized
    2. area_ratio            - how similar the areas are (0 means different, 1 means same)
    3. intensity_similarity  - how similar the mean intensities of two objects are
    4. shape_similarity      - how similar their aspect ratios are
    5. iou                   - bounding-box intersection over union

All features are in [0, 1] except ``position_distance``, which is capped at
``max_position_distance`` (2.0 by default), to set the input range.

The extractor is stateless, the same instance can safely be used across threads and videos, 
provided the ``pixels_per_micrometer`` calibration is appropriate for the video's acquisition settings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np


# Feature record

@dataclass
class TrackingFeatures:
    """Five features describing how similar two detections are.

    Attributes are:
    
    position_distance : float
        Euclidean distance between centers, normalized by
        ``reference_distance_um * ppm`` and clipped to [0, max_position_distance].
        Smaller means more likely same object.
    area_ratio : float
        ``min(area1, area2) / max(area1, area2)``. In [0, 1].
        1.0 = identical area; 0.0 = very different.
    intensity_similarity : float
        ``1 - |i1 - i2| / max(i1, i2)``. In [0, 1].
        1.0 = identical intensity.
    shape_similarity : float
        Same formula applied to aspect ratio (max(w, h) / min(w, h)).
        1.0 = identical shape elongation.
    iou : float
        Bounding-box Intersection over Union. In [0, 1].
        1.0 = perfectly overlapping boxes; 0.0 = no overlap.
    """

    position_distance: float
    area_ratio: float
    intensity_similarity: float
    shape_similarity: float
    iou: float

    # Names in the same order as ``to_vector`` — used for plots and reports.
    FEATURE_NAMES: Tuple[str, ...] = (
        "position_distance",
        "area_ratio",
        "intensity_similarity",
        "shape_similarity",
        "iou",
    )

    def to_vector(self) -> np.ndarray:
        """Return a fixed-order float32 vector for the classifier."""
        return np.array(
            [
                self.position_distance,
                self.area_ratio,
                self.intensity_similarity,
                self.shape_similarity,
                self.iou,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-friendly dict (useful for debugging)."""
        return asdict(self)


# Making sure the FEATURE_NAMES is not treated as a dataclass field
TrackingFeatures.FEATURE_NAMES = (
    "position_distance",
    "area_ratio",
    "intensity_similarity",
    "shape_similarity",
    "iou",
)


# Feature extractor

class TrackingFeatureExtractor:
    """Compute pairwise features between detections.

    Parameters:
    
    pixels_per_micrometer : float, default 10.0
        Spatial calibration used to convert pixel distances to micrometers.
        Should match the values used when generating annotations and when training the classifier.
    reference_distance_um : float, default 5.0
        Distance (in µm) used to normalize ``position_distance``. A detection
        pair separated by this distance will have ``position_distance = 1.0``.
        Larger values make the distance feature less sensitive.
    max_position_distance : float, default 2.0
        Hard upper limit on the normalized position distance. Prevents outliers from dominating the classifier
        when two detections are far apart.
    default_intensity : float, default 50.0
        Fallback mean intensity for when the detection doesn't report one. Used to avoid NaNs in the similarity computation.
    default_aspect_ratio : float, default 1.5
        Fallback aspect ratio, for cases when the detection doesn't report one.
    """

    def __init__(
        self,
        pixels_per_micrometer: float = 10.0,
        reference_distance_um: float = 5.0,
        max_position_distance: float = 2.0,
        default_intensity: float = 50.0,
        default_aspect_ratio: float = 1.5,
    ) -> None:
        if pixels_per_micrometer <= 0:
            raise ValueError("pixels_per_micrometer must be positive")
        if reference_distance_um <= 0:
            raise ValueError("reference_distance_um must be positive")

        self.ppm = float(pixels_per_micrometer)
        self.reference_distance_um = float(reference_distance_um)
        self.max_position_distance = float(max_position_distance)
        self.default_intensity = float(default_intensity)
        self.default_aspect_ratio = float(default_aspect_ratio)

    # Public API

    def compute_features(self, det1: Dict, det2: Dict) -> TrackingFeatures:
        """Compute the full feature vector between two detections.

        Parameters for this part:

        det1, det2 : dict
            Detection dicts with at least ``bbox``. Other keys are used when
            present; sensible defaults are substituted otherwise.

        Returns

        TrackingFeatures
        """
        c1 = np.asarray(self._get_center(det1), dtype=float)
        c2 = np.asarray(self._get_center(det2), dtype=float)

        # Position distance (normalized, clipped)
        distance_px = float(np.linalg.norm(c1 - c2))
        distance_um = distance_px / self.ppm
        position_distance = min(
            distance_um / self.reference_distance_um,
            self.max_position_distance,
        )

        # Area ratio (0 = different, 1 = identical)
        area1 = self._get_area_um2(det1)
        area2 = self._get_area_um2(det2)
        area_ratio = _safe_ratio(area1, area2)

        # Intensity similarity
        intensity1 = float(det1.get("mean_intensity", self.default_intensity))
        intensity2 = float(det2.get("mean_intensity", self.default_intensity))
        intensity_similarity = _similarity(intensity1, intensity2)

        # Shape similarity (aspect ratio)
        ar1 = self._get_aspect_ratio(det1)
        ar2 = self._get_aspect_ratio(det2)
        shape_similarity = _similarity(ar1, ar2)

        # IoU
        iou = self._calculate_iou(det1["bbox"], det2["bbox"])

        return TrackingFeatures(
            position_distance=float(position_distance),
            area_ratio=float(area_ratio),
            intensity_similarity=float(intensity_similarity),
            shape_similarity=float(shape_similarity),
            iou=float(iou),
        )

    def feature_vector(self, features: TrackingFeatures) -> np.ndarray:
        """Return the numpy vector representation of ``features``."""
        return features.to_vector()

    def feature_names(self) -> Tuple[str, ...]:
        """Return the ordered feature names."""
        return TrackingFeatures.FEATURE_NAMES

    # Internal helpers

    @staticmethod
    def _get_center(det: Dict) -> List[float]:
        """Return the center as [cx, cy], computing it from bbox if absent."""
        center = det.get("center")
        if center is not None:
            return [float(center[0]), float(center[1])]
        x, y, w, h = det["bbox"]
        return [x + w / 2.0, y + h / 2.0]

    def _get_area_um2(self, det: Dict) -> float:
        """Return the area in µm², computing it from bbox if absent."""
        area = det.get("area_um2")
        if area is not None and area > 0:
            return float(area)
        x, y, w, h = det["bbox"]
        return float(w * h) / (self.ppm ** 2)

    def _get_aspect_ratio(self, det: Dict) -> float:
        """Return the aspect ratio, computing it from bbox if absent."""
        ar = det.get("aspect_ratio")
        if ar is not None and ar > 0:
            return float(ar)
        x, y, w, h = det["bbox"]
        return float(max(w, h) / max(min(w, h), 1))

    @staticmethod
    def _calculate_iou(bbox1: Tuple[float, float, float, float],
                       bbox2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union of two [x, y, w, h] boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        inter_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = w1 * h1 + w2 * h2 - inter_area

        return float(inter_area / union_area) if union_area > 0 else 0.0


# Numeric helpers (module-level, easy to unit-test)

def _safe_ratio(a: float, b: float) -> float:
    """Return min(a, b) / max(a, b), safely handling zeros.

    Returns 0.0 if both are zero or if the maximum is zero.
    """
    if a <= 0 and b <= 0:
        return 0.0
    if a <= 0 or b <= 0:
        return 0.0
    return float(min(a, b) / max(a, b))


def _similarity(a: float, b: float) -> float:
    """Return 1 - |a - b| / max(a, b), clipped to [0, 1].

    A value of 1.0 means identical; 0.0 means maximally different.
    Returns 0.5 when both are zero (undefined case → neutral).
    """
    m = max(a, b)
    if m <= 0:
        return 0.5
    return float(max(0.0, 1.0 - abs(a - b) / m))
