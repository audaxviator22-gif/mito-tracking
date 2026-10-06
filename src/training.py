"""Loading annotations and training the association classifier.

The training set is derived automatically from track annotations:

- Positive examples are consecutive detections within the same track.
- Negative examples: detections in the same frame from different tracks.

No manual pair-labeling is required.
"""

from __future__ import annotations

import json
import random
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .features import TrackingFeatureExtractor

warnings.filterwarnings("ignore")


# Data loader

class TrackingDataLoader:
    """Load annotation JSON files and build a labeled training set.

    Parameters:

    annotations_folder : str or Path
        Folder containing ``*_tracks.json`` files (produced by
        :class:`~mito_tracking.tracker.TrackingAnnotationGenerator`).
    pixels_per_micrometer : float, default 10.0
        Spatial calibration. Must match the value that is used when generating the annotations.
    reference_distance_um : float, default 5.0
        Passed to :class:`~mito_tracking.features.TrackingFeatureExtractor`.
    negative_ratio : float, default 1.0
        Number of negative examples per positive. 1.0 = balanced.
    max_negatives : int, default 5000
        Strict limit on the number of negative examples.
    seed : int, default 42
        Random seed for negative-pair sampling.
    """

    def __init__(
        self,
        annotations_folder: str,
        pixels_per_micrometer: float = 10.0,
        reference_distance_um: float = 5.0,
        negative_ratio: float = 1.0,
        max_negatives: int = 5000,
        seed: int = 42,
    ) -> None:
        self.folder = Path(annotations_folder)
        self.ppm = float(pixels_per_micrometer)
        self.reference_distance_um = float(reference_distance_um)
        self.negative_ratio = float(negative_ratio)
        self.max_negatives = int(max_negatives)
        self.seed = int(seed)

        self.annotations: List[Dict] = []
        self.tracks: List[Dict] = []
        self.frame_detections: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)

    # Loading


    def load_annotations(self) -> bool:
        """Read all annotation JSON files from the folder."""
        files = sorted(self.folder.glob("*_tracks.json"))
        if not files:
            files = sorted(self.folder.glob("*.json"))
        if not files:
            print(f"No annotation JSON files found in {self.folder}")
            return False

        print(f"Found {len(files)} annotation file(s)")
        for fp in files:
            try:
                with open(fp, "r") as f:
                    self.annotations.append(json.load(f))
                print(f"   {fp.name}")
            except Exception as exc:
                print(f"   {fp.name}: {exc}")

        return bool(self.annotations)

    # Parsing

    def extract_tracks_and_detections(self) -> List[Dict]:
        """Parse annotations into flat lists of tracks and detections."""
        self.tracks = []
        self.frame_detections = defaultdict(list)

        for annotation in self.annotations:
            video_name = annotation.get("video_name", "video")

            # The preferred format has frame_detections with track_id.
            for frame_str, dets in annotation.get("frame_detections", {}).items():
                frame = int(frame_str)
                for det in dets:
                    track_id = det.get("track_id", -1)
                    if track_id < 0:
                        continue

                    record = {
                        "track_id": int(track_id),
                        "bbox": list(det["bbox"]),
                        "center": det.get("center") or [
                            det["bbox"][0] + det["bbox"][2] / 2,
                            det["bbox"][1] + det["bbox"][3] / 2,
                        ],
                        "area_um2": float(det.get("area_um2", 0.0)),
                        "confidence": float(det.get("confidence", 0.7)),
                        "circularity": float(det.get("circularity", 0.5)),
                        "aspect_ratio": float(det.get("aspect_ratio", 1.5)),
                        "mean_intensity": float(det.get("mean_intensity", 50.0)),
                    }
                    self.frame_detections[(video_name, frame)].append(record)

        # Reconstructing per-track sequences from frame_detections
        per_track: Dict[Tuple[str, int], List[Tuple[int, Dict]]] = defaultdict(list)
        for (video, frame), dets in self.frame_detections.items():
            for det in dets:
                per_track[(video, det["track_id"])].append((frame, det))

        for (video, track_id), sequence in per_track.items():
            sequence.sort(key=lambda x: x[0])  # sort by frame index
            frames = [f for f, _ in sequence]
            detections = [d for _, d in sequence]

            if len(frames) < 2:
                continue  # a track needs at least two frames to be useful

            areas = [d["area_um2"] for d in detections if d["area_um2"] > 0]
            intensities = [d["mean_intensity"] for d in detections if d["mean_intensity"] > 0]

            self.tracks.append({
                "video": video,
                "track_id": track_id,
                "frames": frames,
                "detections": detections,
                "length": len(frames),
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "avg_area_um2": float(np.mean(areas)) if areas else 0.0,
                "avg_intensity": float(np.mean(intensities)) if intensities else 0.0,
            })

        print(
            f"\nExtracted: {len(self.tracks)} valid tracks, "
            f"{sum(len(v) for v in self.frame_detections.values())} detections"
        )
        return self.tracks


    # Training set construction


    def create_training_dataset(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict]]:
        """Build the (X, y) training set.

        Returns thw following:
    
        X : np.ndarray of shape (n_samples, 5) or None in case it's empty
        y : np.ndarray of shape (n_samples,) with values in {0, 1} or None
        metadata : list of dicts (one per sample) with information about provenance
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        extractor = TrackingFeatureExtractor(
            pixels_per_micrometer=self.ppm,
            reference_distance_um=self.reference_distance_um,
        )

        X: List[np.ndarray] = []
        y: List[int] = []
        metadata: List[Dict] = []

        # Positive examples: consecutive frames within one track
        print("\nBuilding positive examples...")
        for track in self.tracks:
            dets = track["detections"]
            frames = track["frames"]
            for i in range(len(dets) - 1):
                features = extractor.compute_features(dets[i], dets[i + 1])
                X.append(extractor.feature_vector(features))
                y.append(1)
                metadata.append({
                    "type": "positive",
                    "video": track["video"],
                    "track_id": track["track_id"],
                    "frame_pair": (frames[i], frames[i + 1]),
                })

        n_positives = len(X)
        print(f"  Positive examples: {n_positives}")

        if n_positives == 0:
            print("No positive examples found. Check that tracks have ≥ 2 frames.")
            return None, None, []

        max_negatives = min(int(n_positives * self.negative_ratio), self.max_negatives)

        # ---- Negative examples: same frame, different tracks ----
        print("Building negative examples (same-frame, different track)...")
        n_negatives = 0
        same_frame_candidates: List[Tuple[str, int, List[Dict]]] = [
            (video, frame, dets)
            for (video, frame), dets in self.frame_detections.items()
            if len(dets) >= 2
        ]
        random.shuffle(same_frame_candidates)

        for video, frame, dets in same_frame_candidates:
            if n_negatives >= max_negatives:
                break
            # Subsample pairs to avoid quadratic blow-up when many detections
            n_pairs = min(len(dets), 10)
            for i in range(n_pairs):
                if n_negatives >= max_negatives:
                    break
                for j in range(i + 1, min(i + 5, len(dets))):
                    if dets[i]["track_id"] == dets[j]["track_id"]:
                        continue
                    features = extractor.compute_features(dets[i], dets[j])
                    X.append(extractor.feature_vector(features))
                    y.append(0)
                    n_negatives += 1
                    metadata.append({
                        "type": "negative_same_frame",
                        "video": video,
                        "frame": frame,
                    })
                    if n_negatives >= max_negatives:
                        break

        print(f"  Negative examples: {n_negatives}")
        print(f"  Total: {len(X)}")

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.int32),
            metadata,
        )


# Model trainer


class TrackingModelTrainer:
    """Train a Random Forest classifier for track association.

    Parameters:

    pixels_per_micrometer : float, default 10.0
        Spatial calibration; must match the value used at inference time.
    reference_distance_um : float, default 5.0
        Passed to the feature extractor.
    seed : int, default 42
        Random seed for reproducibility across splits and training.
    test_size : float, default 0.2
        Fraction of examples held out for validation.
    n_estimators : int, default 100
        Number of trees in the forest.
    max_depth : int, default 10
        Maximum tree depth.
    class_weight : str or None, default 'balanced'
        Passed to RandomForestClassifier; handles class imbalance.
    negative_ratio : float, default 1.0
        Passed to the data loader.
    max_negatives : int, default 5000
        Passed to the data loader.
    """

    def __init__(
        self,
        pixels_per_micrometer: float = 10.0,
        reference_distance_um: float = 5.0,
        seed: int = 42,
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 10,
        class_weight: Optional[str] = "balanced",
        negative_ratio: float = 1.0,
        max_negatives: int = 5000,
    ) -> None:
        self.ppm = float(pixels_per_micrometer)
        self.reference_distance_um = float(reference_distance_um)
        self.seed = int(seed)
        self.test_size = float(test_size)

        self.classifier = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=self.seed,
            class_weight=class_weight,
        )
        self.scaler = StandardScaler()
        self.feature_extractor = TrackingFeatureExtractor(
            pixels_per_micrometer=self.ppm,
            reference_distance_um=self.reference_distance_um,
        )

        self.negative_ratio = float(negative_ratio)
        self.max_negatives = int(max_negatives)

        self.is_trained = False
        self.training_date: Optional[str] = None

    # Training


    def train(self, annotations_folder: str, verbose: bool = True) -> bool:
        """Loading annotations, building the dataset, and then training the classifier.

        Returns True on success, False if not.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        if verbose:
            print("=" * 60)
            print("TRAINING TRACKING MODEL")
            print("=" * 60)

        # 1. Load data
        loader = TrackingDataLoader(
            annotations_folder=annotations_folder,
            pixels_per_micrometer=self.ppm,
            reference_distance_um=self.reference_distance_um,
            negative_ratio=self.negative_ratio,
            max_negatives=self.max_negatives,
            seed=self.seed,
        )
        if not loader.load_annotations():
            return False
        if not loader.extract_tracks_and_detections():
            return False

        # 2. Build feature matrix
        X, y, _ = loader.create_training_dataset()
        if X is None or len(X) == 0:
            print("No training examples generated.")
            return False

        # 3. Train/validation split (stratified to preserve class balance)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y,
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 4. Train
        if verbose:
            print("\nTraining Random Forest...")
        self.classifier.fit(X_train_scaled, y_train)

        # 5. Evaluate
        train_acc = self.classifier.score(X_train_scaled, y_train)
        val_acc = self.classifier.score(X_val_scaled, y_val)

        if verbose:
            print(f"  Train accuracy: {train_acc:.4f}")
            print(f"  Val   accuracy: {val_acc:.4f}")

            cv_scores = cross_val_score(
                self.classifier, X_train_scaled, y_train, cv=5
            )
            print(f"  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            y_pred = self.classifier.predict(X_val_scaled)
            print("\nClassification report:")
            print(classification_report(
                y_val, y_pred,
                target_names=["different track", "same track"],
                digits=4,
            ))

            print("Confusion matrix (rows=true, cols=predicted):")
            print(confusion_matrix(y_val, y_pred))

            print("\nFeature importances:")
            for name, imp in zip(
                self.feature_extractor.feature_names(),
                self.classifier.feature_importances_,
            ):
                print(f"  {name}: {imp:.4f}")

        self.is_trained = True
        self.training_date = datetime.now().isoformat()
        return True

    # Persistence


    def save(self, path: str = "tracking_model.pkl") -> bool:
        """Saving the trained classifier, scaler, and metadata."""
        if not self.is_trained:
            print("Model is not trained yet — call .train() first.")
            return False

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "classifier": self.classifier,
            "scaler": self.scaler,
            "feature_extractor": self.feature_extractor,
            "training_date": self.training_date,
            "pixels_per_micrometer": self.ppm,
            "reference_distance_um": self.reference_distance_um,
            "seed": self.seed,
            "feature_names": list(self.feature_extractor.feature_names()),
        }
        joblib.dump(payload, path)
        print(f" Model saved to: {path}")
        return True

    def load(self, path: str = "tracking_model.pkl") -> bool:
        """Loading a previously saved model."""
        try:
            payload = joblib.load(path)
        except Exception as exc:
            print(f"Could not load model: {exc}")
            return False

        self.classifier = payload["classifier"]
        self.scaler = payload["scaler"]
        self.feature_extractor = payload.get(
            "feature_extractor",
            TrackingFeatureExtractor(
                pixels_per_micrometer=payload.get("pixels_per_micrometer", self.ppm),
                reference_distance_um=payload.get("reference_distance_um", self.reference_distance_um),
            ),
        )
        self.ppm = float(payload.get("pixels_per_micrometer", self.ppm))
        self.reference_distance_um = float(payload.get("reference_distance_um", self.reference_distance_um))
        self.training_date = payload.get("training_date")
        self.is_trained = True

        print(f" Model loaded from: {path}")
        if self.training_date:
            print(f"  Trained: {self.training_date}")
        return True

    # Inference-time convenience

    def predict_probability(self, det1: Dict, det2: Dict) -> float:
        """Return P(same track | det1, det2) for a single pair."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained; call .train() or .load() first.")

        features = self.feature_extractor.compute_features(det1, det2)
        vector = self.feature_extractor.feature_vector(features).reshape(1, -1)
        vector_scaled = self.scaler.transform(vector)
        return float(self.classifier.predict_proba(vector_scaled)[0, 1])

    # Introspection


    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "not trained"
        return (
            f"TrackingModelTrainer(ppm={self.ppm}, "
            f"n_estimators={self.classifier.n_estimators}, "
            f"status={status})"
        )
