"""THius detector is based on a classical computer-vision for mitochondria in single frames. 
The pipeline: grayscale > background subtraction > CLAHE > threshold (adaptive + Otsu) > morphological cleanup > contours > filter by size/shape/intensity

It was designed as dependency light (only OpenCV and NumPy) and CPU-only, so it can run on any device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import joblib
import numpy as np


# Detection record

@dataclass
class Detection:
    """Detected mitochondria in one frame.

    Attributes
    bbox: [x, y, w, h]
        Bounding box in pixels, top-left corner convention.
    center: [cx, cy]
        Centroid in pixels (integer-rounded).
    area_um2: float
        Contour area in square micrometers.
    area_px: float
        Contour area in pixels (this data is kept for diagnostics).
    circularity: float
        Perimeter-based circularity in [0, 1]; 1.0 = perfect circle.
    aspect_ratio: float
        max(w, h) / min(w, h); ≥ 1.0.
    mean_intensity: float
        Mean grayscale intensity inside the contour (0–255).
    confidence: float
        Either heuristic or model-based confidence in [0, 1].
    frame: int
        Frame index at which this detection was found (it equals -1 if unknown).
    """

    bbox: List[int]
    center: List[int]
    area_um2: float
    area_px: float
    circularity: float
    aspect_ratio: float
    mean_intensity: float
    confidence: float
    frame: int = -1

    def to_dict(self) -> Dict:
        """Return a plain dict (JSON-friendly, backward-compatible)."""
        return {
            "bbox": self.bbox,
            "center": self.center,
            "area_um2": self.area_um2,
            "area_pixels": self.area_px,
            "circularity": self.circularity,
            "aspect_ratio": self.aspect_ratio,
            "mean_intensity": self.mean_intensity,
            "confidence": self.confidence,
            "frame": self.frame,
        }


# Detector part

class MitochondriaDetector:
    """Detect mitochondria in single frames.

    Parameters include:
    pixels_per_micrometer : float, default 10.0
        Spatial calibration used to convert pixel areas to µm².
    min_area_um2 : float, default 0.2
        Minimum area for a candidate, smaller objects are excluded.
    max_area_um2 : float, default 50.0
        Maximum area for a candidate. Larger objects (clumps, debris) are excluded from the analysis.
    max_circularity : float, default 0.8
        Upper bound on circularity. Mitochondria are elongated, and therefore perfectly round objects are typically artifacts, which means they have to be excluded.
    min_intensity : float, default 30.0
        Minimum mean grayscale intensity inside the contour (0–255).
    min_confidence : float, default 0.6
        Minimum confidence for a detection to be returned. Only used when a detection model is loaded, otherwise, the heuristic confidence is compared against this value.
    model_path : str or Path, optional
    """

    def __init__(
        self,
        pixels_per_micrometer: float = 10.0,
        min_area_um2: float = 0.2,
        max_area_um2: float = 50.0,
        max_circularity: float = 0.8,
        min_intensity: float = 30.0,
        min_confidence: float = 0.6,
        model_path: Optional[str] = None,
    ) -> None:
        self.ppm = float(pixels_per_micrometer)
        self.min_area_um2 = float(min_area_um2)
        self.max_area_um2 = float(max_area_um2)
        self.max_circularity = float(max_circularity)
        self.min_intensity = float(min_intensity)
        self.min_confidence = float(min_confidence)

        self.model = None
        self.scaler = None

        if model_path is not None and Path(model_path).exists():
            self.load_model(model_path)

    # Model I/O

    def load_model(self, model_path: str) -> bool:
        """Load a trained detection classifier from a joblib file.

        Returns True when successful, and False if not. In case of failure, the detector returns to the heuristic confidence scoring.
        """
        try:
            data = joblib.load(model_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Could not load detection model: {exc}")
            return False

        self.model = data.get("classifier")
        self.scaler = data.get("scaler")
        print(f"✓ Detection model loaded from {model_path}")
        return True

    # Public API

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Detect mitochondria in a single frame.

        Parameters include:
        frame : np.ndarray
            BGR (from OpenCV) or single-channel grayscale image.
        confidence_threshold : float, optional
            Overrides ``self.min_confidence`` for this call.

        It returns list of dicts, with one dict per detection.
        """
        if frame is None or frame.size == 0:
            return []

        threshold = self.min_confidence if confidence_threshold is None else confidence_threshold

        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # Background subtraction through large Gaussian blur
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        background_subtracted = cv2.subtract(gray, blurred)

        # CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(background_subtracted)

        # Adaptive threshold + Otsu (combined application of both)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 2,
        )
        _, otsu = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = cv2.bitwise_or(adaptive, otsu)

        # Morphology cleanup, first opening (removing noise), then closing (fill holes)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # Contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and build Detection objects
        detections: List[Detection] = []
        for contour in contours:
            det = self._contour_to_detection(contour, gray)
            if det is None:
                continue
            det.confidence = self._compute_confidence(det)
            if det.confidence < threshold:
                continue
            detections.append(det)

        return [d.to_dict() for d in detections]

    # Internal helpers

    def _contour_to_detection(
        self,
        contour: np.ndarray,
        gray: np.ndarray,
    ) -> Optional[Detection]:
        """Convert an OpenCV contour into a Detection, or return None."""
        area_px = float(cv2.contourArea(contour))
        if area_px <= 0:
            return None

        area_um2 = area_px / (self.ppm ** 2)
        if not (self.min_area_um2 <= area_um2 <= self.max_area_um2):
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return None

        circularity = float(4.0 * np.pi * area_px / (perimeter ** 2))
        if circularity > self.max_circularity:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return None

        aspect_ratio = float(max(w, h) / max(min(w, h), 1))

        # Mean intensity inside the contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_intensity = float(cv2.mean(gray, mask=mask)[0])
        if mean_intensity < self.min_intensity:
            return None

        # Centroid from image moments (falls back to bbox center)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        return Detection(
            bbox=[int(x), int(y), int(w), int(h)],
            center=[cx, cy],
            area_um2=area_um2,
            area_px=area_px,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            mean_intensity=mean_intensity,
            confidence=0.0,  # filled in by caller
        )

    def _compute_confidence(self, det: Detection) -> float:
        """Scoring a detection.

        If a trained classifier is loaded, its probability is used.
        Otherwise, a heuristic score is returned based on how close to the mitochondria the candidate is.
        """
        if self.model is not None and self.scaler is not None:
            return self._confidence_from_model(det)
        return self._heuristic_confidence(det)

    def _confidence_from_model(self, det: Detection) -> float:
        """Scoring using a trained model
        """
        features = np.array(
            [[
                det.area_um2,
                det.circularity,
                det.aspect_ratio,
                det.mean_intensity,
            ]],
            dtype=np.float32,
        )
        features_scaled = self.scaler.transform(features)
        return float(self.model.predict_proba(features_scaled)[0, 1])

    @staticmethod
    def _heuristic_confidence(det: Detection) -> float:
        """Heuristic confidence based on prior knowledge about mitochondria.

        Starts at 0.5 and adds small bonuses for typical mitochondria characteristics:
        - area between 1 and 20 µm² (the chosen, biologically plausible range)
        - has elongated shape (circularity < 0.5, or aspect_ratio > 1.5)
        - reasonably bright (mean_intensity > 50)
        """
        conf = 0.5
        if 1.0 < det.area_um2 < 20.0:
            conf += 0.2
        if det.circularity < 0.5:
            conf += 0.2
        elif det.circularity < 0.7:
            conf += 0.1
        if det.aspect_ratio > 1.5:
            conf += 0.1
        if det.mean_intensity > 50.0:
            conf += 0.1
        return min(conf, 1.0)

    # Introspection

    def __repr__(self) -> str:
        return (
            f"MitochondriaDetector(ppm={self.ppm}, "
            f"area=[{self.min_area_um2}, {self.max_area_um2}] µm², "
            f"max_circularity={self.max_circularity}, "
            f"min_intensity={self.min_intensity}, "
            f"min_confidence={self.min_confidence}, "
            f"model={'yes' if self.model is not None else 'no'})"
        )
