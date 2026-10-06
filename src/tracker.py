"""Detection-to-track association for mitochondria.

Two tracker components are included:

- :class:`TrackingAnnotationGenerator`
    Uses a geometric cost matrix (position + area) with a Hungarian method assignment. 
    It's used to build ground-truth tracks from raw videos, which in turn become the training set for the ML tracker.

- :class:`MLTracker`
    Uses a trained random forest classifier to score candidate pairs.

Both classes share the :class:`Track` dataclass and follow the same active tracks/ completed tracks state machine.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import MitochondriaDetector
from .features import TrackingFeatureExtractor


# Track record

@dataclass
class Track:
    """A single mitochondria track (a sequence of detections across frames).

    Attributes:

    track_id : int
        Unique identifier within a video.
    detections : list of dict
        Detection dicts (as returned by the detector), in temporal order.
    frames : list of int
        Frame indices corresponding to the detections.
    start_frame, end_frame : int
        First and last frame indices.
    avg_area_um2 : float
        Mean area over the track (µm²).
    avg_intensity : float
        Mean grayscale intensity over the track.
    displacement_um : float
        Total path length (sum of step lengths), in µm.
    speed_um_per_sec : float
        Average speed = displacement / duration.
    linearity : float
        Net displacement / path length; 1.0 = straight line.
    """

    track_id: int
    detections: List[Dict]
    frames: List[int]
    start_frame: int
    end_frame: int
    avg_area_um2: float
    avg_intensity: float
    displacement_um: float
    speed_um_per_sec: float
    linearity: float

    @property
    def length_frames(self) -> int:
        """Number of frames spanned by the track."""
        return len(self.frames)

    def to_dict(self, include_detections: bool = True) -> Dict:
        """Serialize the track to a JSON-friendly dict."""
        out = {
            "track_id": self.track_id,
            "frames": list(self.frames),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "length_frames": self.length_frames,
            "avg_area_um2": self.avg_area_um2,
            "avg_intensity": self.avg_intensity,
            "displacement_um": self.displacement_um,
            "speed_um_per_sec": self.speed_um_per_sec,
            "linearity": self.linearity,
        }
        if include_detections:
            out["detections"] = self.detections
        return out


# Internal state for an active track (not yet completed)

@dataclass
class _ActiveTrack:
    """Keeping records while a track is still being extended."""

    track_id: int
    detections: List[Dict] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    last_frame: int = 0

    def add(self, det: Dict, frame: int) -> None:
        self.detections.append(det)
        self.frames.append(frame)
        self.last_frame = frame


# Utility: finalize a track (compute statistics)

def _finalize_track(
    track_id: int,
    active: _ActiveTrack,
    pixels_per_micrometer: float,
    fps: float,
) -> Track:
    """Computing all properties for a completed track."""
    dets = active.detections
    frames = active.frames

    areas = [d.get("area_um2", 0.0) for d in dets]
    intensities = [d.get("mean_intensity", 0.0) for d in dets]

    centers = [
        np.asarray(
            d.get("center")
            or [d["bbox"][0] + d["bbox"][2] / 2.0, d["bbox"][1] + d["bbox"][3] / 2.0]
        )
        for d in dets
    ]

    if len(centers) > 1:
        # Sum of step lengths in pixels → µm
        total_px = float(
            np.sum(
                [
                    np.linalg.norm(centers[i + 1] - centers[i])
                    for i in range(len(centers) - 1)
                ]
            )
        )
        total_um = total_px / pixels_per_micrometer

        # Straight-line displacement start→end
        straight_um = float(np.linalg.norm(centers[-1] - centers[0])) / pixels_per_micrometer

        linearity = straight_um / total_um if total_um > 0 else 1.0

        duration_s = (frames[-1] - frames[0]) / fps if fps > 0 else 0.0
        speed = total_um / duration_s if duration_s > 0 else 0.0
    else:
        total_um = 0.0
        linearity = 1.0
        speed = 0.0

    return Track(
        track_id=track_id,
        detections=dets,
        frames=frames,
        start_frame=frames[0],
        end_frame=frames[-1],
        avg_area_um2=float(np.mean(areas)) if areas else 0.0,
        avg_intensity=float(np.mean(intensities)) if intensities else 0.0,
        displacement_um=total_um,
        speed_um_per_sec=speed,
        linearity=linearity,
    )


# Shared association primitives

def _hungarian_association(
    cost_matrix: np.ndarray,
    max_cost: float,
) -> Dict[int, int]:
    """Solve the assignment problem and return a dict {row_idx: col_idx}.

    Pairs whose cost exceeds ``max_cost`` are dropped.
    """
    if cost_matrix.size == 0:
        return {}

    rows, cols = linear_sum_assignment(cost_matrix)
    return {
        int(r): int(c)
        for r, c in zip(rows, cols)
        if cost_matrix[r, c] <= max_cost
    }


# Tracker 1 — geometric association, it is used for building training data.

class TrackingAnnotationGenerator:
    """Build tracks by associating detections with the geometric cost matrix.

    Here, it is used to derive ground-truth annotations from raw videos, which are later used to train
    :class:`MLTracker`. Geometric association is deliberate: no model is needed to produce training data, 
    only physical constraints.

    Parameters are the following:
    
    pixels_per_micrometer : float, default 10.0
        Spatial calibration.
    fps : float, default 30.0
        Temporal calibration.
    detector : MitochondriaDetector, optional
        Insert a custom detector. If None, it will be created with default
        parameters.
    max_distance_um : float, default 2.0
        Maximum allowed movement between consecutive frames.
    max_area_change : float, default 0.5
        Maximum allowed relative area change between consecutive frames.
    distance_weight : float, default 0.7
        Weight of the distance term in the cost matrix. Area weight is
        ``1 - distance_weight``.
    """

    def __init__(
        self,
        pixels_per_micrometer: float = 10.0,
        fps: float = 30.0,
        detector: Optional[MitochondriaDetector] = None,
        max_distance_um: float = 2.0,
        max_area_change: float = 0.5,
        distance_weight: float = 0.7,
    ) -> None:
        self.ppm = float(pixels_per_micrometer)
        self.fps = float(fps)
        self.detector = detector or MitochondriaDetector(pixels_per_micrometer=self.ppm)
        self.max_distance_um = float(max_distance_um)
        self.max_area_change = float(max_area_change)
        self.distance_weight = float(distance_weight)
        self.next_track_id = 0


    # Association

    def _cost_matrix(self, prev_dets: List[Dict], curr_dets: List[Dict]) -> np.ndarray:
        """Build the geometric cost matrix."""
        n_prev, n_curr = len(prev_dets), len(curr_dets)
        cost = np.zeros((n_prev, n_curr), dtype=np.float32)

        for i, prev in enumerate(prev_dets):
            pc = np.asarray(prev.get("center", [0, 0]))
            for j, curr in enumerate(curr_dets):
                cc = np.asarray(curr.get("center", [0, 0]))

                # Position cost (normalized)
                dist_um = float(np.linalg.norm(pc - cc)) / self.ppm
                d_cost = min(dist_um / self.max_distance_um, 1.0)

                # Area change cost
                a1 = prev.get("area_um2", 0.0)
                a2 = curr.get("area_um2", 0.0)
                denom = max(a1, a2, 1e-9)
                area_change = abs(a1 - a2) / denom
                a_cost = min(area_change / self.max_area_change, 1.0)

                total = self.distance_weight * d_cost + (1 - self.distance_weight) * a_cost

                # Hard constraints as large penalties
                if dist_um > self.max_distance_um:
                    total += 2.0
                if area_change > self.max_area_change:
                    total += 1.0

                cost[i, j] = total

        return cost

    def associate(self, prev_dets: List[Dict], curr_dets: List[Dict]) -> Dict[int, int]:
        """Return a dict mapping previous index, then current index for associated pairs."""
        if not prev_dets or not curr_dets:
            return {}
        cost = self._cost_matrix(prev_dets, curr_dets)
        # Accept only associations with cost < 1.5 (same threshold as the
        # original notebook for reproducibility)
        return {
            i: j for i, j in _hungarian_association(cost, max_cost=1.5).items()
        }

    # Tracking

    def track_video(
        self,
        video_path: str,
        max_gap_frames: int = 5,
        min_track_length: int = 10,
        verbose: bool = True,
    ) -> List[Track]:
        """\nTracking mitochondria in a single video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: cannot open {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.fps = fps

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if verbose:
            print(f"\nTracking mitochondria in: {Path(video_path).name}")
            print(f"Total frames: {total_frames}")

        active: Dict[int, _ActiveTrack] = {}
        completed: List[Track] = []
        frame_idx = 0
        last_progress = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            for det in detections:
                det["frame"] = frame_idx

            self._step(detections, active, completed, frame_idx, max_gap_frames, min_track_length)

            frame_idx += 1
            if verbose and frame_idx - last_progress >= 100:
                print(f"  {frame_idx}/{total_frames} frames, active: {len(active)}")
                last_progress = frame_idx

        cap.release()

        # Finalize any tracks still active at the end of the video
        for tid, t in list(active.items()):
            if t.length := len(t.frames) >= min_track_length:  # noqa: F841
                pass
            if len(t.frames) >= min_track_length:
                completed.append(_finalize_track(tid, t, self.ppm, self.fps))
            del active[tid]

        if verbose:
            print(f"\ Tracking complete: {len(completed)} tracks found")
        return completed

    def _step(
        self,
        detections: List[Dict],
        active: Dict[int, _ActiveTrack],
        completed: List[Track],
        frame_idx: int,
        max_gap_frames: int,
        min_track_length: int,
    ) -> None:
        """One frame of tracking logic."""
        if active and detections:
            prev_dets = [t.detections[-1] for t in active.values() if t.detections]
            track_ids = [tid for tid, t in active.items() if t.detections]

            assoc = self.associate(prev_dets, detections)
            matched = set()
            for pi, ci in assoc.items():
                tid = track_ids[pi]
                active[tid].add(detections[ci], frame_idx)
                matched.add(ci)

            for i, det in enumerate(detections):
                if i not in matched:
                    self.next_track_id += 1
                    t = _ActiveTrack(track_id=self.next_track_id, last_frame=frame_idx)
                    t.add(det, frame_idx)
                    active[self.next_track_id] = t

            self._retire_lost(active, completed, frame_idx, max_gap_frames, min_track_length)

        elif detections:
            for det in detections:
                self.next_track_id += 1
                t = _ActiveTrack(track_id=self.next_track_id, last_frame=frame_idx)
                t.add(det, frame_idx)
                active[self.next_track_id] = t

    def _retire_lost(
        self,
        active: Dict[int, _ActiveTrack],
        completed: List[Track],
        frame_idx: int,
        max_gap_frames: int,
        min_track_length: int,
    ) -> None:
        lost = []
        for tid, t in active.items():
            if frame_idx - t.last_frame > max_gap_frames:
                if len(t.frames) >= min_track_length:
                    completed.append(_finalize_track(tid, t, self.ppm, self.fps))
                lost.append(tid)
        for tid in lost:
            del active[tid]

    # Save annotations

    def save_annotation_format(
        self,
        tracks: List[Track],
        video_path: str,
        output_file: Optional[str] = None,
    ) -> Dict:
        """Save tracks as JSON, for training later."""
        if output_file is None:
            output_file = Path(video_path).parent / f"{Path(video_path).stem}_tracks.json"
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        frame_dets: Dict[int, List[Dict]] = defaultdict(list)
        for tr in tracks:
            for det, frame in zip(tr.detections, tr.frames):
                frame_dets[frame].append({
                    "track_id": tr.track_id,
                    "bbox": det["bbox"],
                    "center": det["center"],
                    "area_um2": det["area_um2"],
                    "confidence": det.get("confidence", 0.7),
                    "circularity": det.get("circularity", 0.5),
                    "aspect_ratio": det.get("aspect_ratio", 1.5),
                    "mean_intensity": det.get("mean_intensity", 50.0),
                })

        payload = {
            "video_name": Path(video_path).stem,
            "video_path": str(Path(video_path).resolve()),
            "fps": self.fps,
            "pixels_per_um": self.ppm,
            "analysis_date": datetime.now().isoformat(),
            "tracks": [t.to_dict(include_detections=False) for t in tracks],
            "frame_detections": {str(k): v for k, v in frame_dets.items()},
            "statistics": {
                "total_tracks": len(tracks),
                "avg_track_length": float(np.mean([t.length_frames for t in tracks])) if tracks else 0.0,
                "max_track_length": int(max((t.length_frames for t in tracks), default=0)),
                "avg_speed_um_s": float(np.mean([t.speed_um_per_sec for t in tracks])) if tracks else 0.0,
                "total_detections": int(sum(t.length_frames for t in tracks)),
            },
        }

        with open(output_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f" Annotations saved to: {output_file}")
        return payload


# Tracker 2 — ML association (inference)

class MLTracker:
    """Inference-time tracker using a trained association classifier.

    Parameters:
    model_path : str
        Path to a ``.pkl`` file containing the trained classifier and scaler
        (as saved by :class:`mito_tracking.training.TrackingModelTrainer`).
    pixels_per_micrometer, fps : float
        Spatial and temporal calibration.
    min_confidence : float, default 0.6
        Detection threshold; detections below this are excluded from the analysis.
    max_gap_frames : int, default 5
        How many consecutive missing frames are tolerated before a track is
        finalized.
    min_track_length : int, default 5
        Minimum number of frames a track must have to be included in the analysis.
    association_threshold : float, default 0.5
        Classifier probability threshold, for accepting an association.
    detector : MitochondriaDetector, optional
        Inject a custom detector.
    """

    def __init__(
        self,
        model_path: str,
        pixels_per_micrometer: float = 10.0,
        fps: float = 30.0,
        min_confidence: float = 0.6,
        max_gap_frames: int = 5,
        min_track_length: int = 5,
        association_threshold: float = 0.5,
        detector: Optional[MitochondriaDetector] = None,
    ) -> None:
        print(f"\nLoading tracking model from {model_path}...")
        data = joblib.load(model_path)

        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.feature_extractor = data.get(
            "feature_extractor",
            TrackingFeatureExtractor(pixels_per_micrometer),
        )

        self.ppm = float(pixels_per_micrometer)
        self.fps = float(fps)
        self.min_confidence = float(min_confidence)
        self.max_gap_frames = int(max_gap_frames)
        self.min_track_length = int(min_track_length)
        self.association_threshold = float(association_threshold)
        self.detector = detector or MitochondriaDetector(pixels_per_micrometer=self.ppm)
        self.next_track_id = 0

        print(f" Model loaded (trained: {data.get('training_date', 'unknown')})")

    # Association

    def predict_association(self, det1: Dict, det2: Dict) -> float:
        """Return P(same track | features)."""
        feats = self.feature_extractor.compute_features(det1, det2)
        vec = self.feature_extractor.feature_vector(feats).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        return float(self.classifier.predict_proba(vec_scaled)[0, 1])

    def associate(self, prev_dets: List[Dict], curr_dets: List[Dict]) -> Dict[int, int]:
        """Associate using the trained classifier + Hungarian assignment."""
        if not prev_dets or not curr_dets:
            return {}

        cost = np.zeros((len(prev_dets), len(curr_dets)), dtype=np.float32)
        for i, prev in enumerate(prev_dets):
            for j, curr in enumerate(curr_dets):
                cost[i, j] = 1.0 - self.predict_association(prev, curr)

        rows, cols = linear_sum_assignment(cost)
        return {
            int(r): int(c)
            for r, c in zip(rows, cols)
            if (1.0 - cost[r, c]) >= self.association_threshold
        }

    # Video-level tracking

    def track_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_visualization: bool = True,
    ) -> Optional[Dict]:
        """Track a single video and save results to JSON."""
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return None

        output_dir = Path(output_dir) if output_dir else video_path.parent / f"{video_path.stem}_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: cannot open {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'=' * 60}\nTRACKING VIDEO: {video_path.name}\n{'=' * 60}")
        print(f"Resolution: {width}x{height}, frames: {total_frames}, fps: {self.fps:.2f}")

        vis_writer = None
        vis_path = None
        if save_visualization:
            vis_path = output_dir / f"{video_path.stem}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vis_writer = cv2.VideoWriter(str(vis_path), fourcc, self.fps, (width, height))

        active: Dict[int, _ActiveTrack] = {}
        completed: List[Track] = []
        frame_idx = 0
        last_progress = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame, confidence_threshold=self.min_confidence)
            for det in detections:
                det["frame"] = frame_idx

            self._step(detections, active, completed, frame_idx)

            if vis_writer is not None:
                vis_writer.write(self._render(frame, active, detections, frame_idx))

            frame_idx += 1
            progress = int(frame_idx / total_frames * 100)
            if progress >= last_progress + 10:
                print(f"  {frame_idx}/{total_frames} ({progress}%)")
                last_progress = progress

        cap.release()
        if vis_writer is not None:
            vis_writer.release()

        for tid, t in list(active.items()):
            if len(t.frames) >= self.min_track_length:
                completed.append(_finalize_track(tid, t, self.ppm, self.fps))
            del active[tid]

        payload = {
            "video_name": video_path.stem,
            "video_path": str(video_path.resolve()),
            "analysis_date": datetime.now().isoformat(),
            "fps": self.fps,
            "pixels_per_um": self.ppm,
            "total_frames": total_frames,
            "total_tracks": len(completed),
            "tracks": [t.to_dict(include_detections=True) for t in completed],
            "statistics": self._compute_statistics(completed),
        }

        results_file = output_dir / f"{video_path.stem}_tracking_results.json"
        with open(results_file, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        print(f"\nTotal tracks: {len(completed)}")
        print(f"Results: {results_file}")
        if save_visualization:
            print(f"Visualization: {vis_path}")

        return payload

    def _step(
        self,
        detections: List[Dict],
        active: Dict[int, _ActiveTrack],
        completed: List[Track],
        frame_idx: int,
    ) -> None:
        """One frame of ML-based tracking."""
        if active and detections:
            prev_dets = [t.detections[-1] for t in active.values() if t.detections]
            track_ids = [tid for tid, t in active.items() if t.detections]

            assoc = self.associate(prev_dets, detections)
            matched = set()
            for pi, ci in assoc.items():
                tid = track_ids[pi]
                active[tid].add(detections[ci], frame_idx)
                matched.add(ci)

            for i, det in enumerate(detections):
                if i not in matched:
                    self.next_track_id += 1
                    t = _ActiveTrack(track_id=self.next_track_id, last_frame=frame_idx)
                    t.add(det, frame_idx)
                    active[self.next_track_id] = t

            lost = []
            for tid, t in active.items():
                if frame_idx - t.last_frame > self.max_gap_frames:
                    if len(t.frames) >= self.min_track_length:
                        completed.append(_finalize_track(tid, t, self.ppm, self.fps))
                    lost.append(tid)
            for tid in lost:
                del active[tid]

        elif detections:
            for det in detections:
                self.next_track_id += 1
                t = _ActiveTrack(track_id=self.next_track_id, last_frame=frame_idx)
                t.add(det, frame_idx)
                active[self.next_track_id] = t

    # Visualization

    @staticmethod
    def _render(
        frame: np.ndarray,
        active: Dict[int, _ActiveTrack],
        detections: List[Dict],
        frame_idx: int,
    ) -> np.ndarray:
        """Draw boxes, IDs, and trails on a copy of the frame."""
        vis = frame.copy()

        # Deterministic colors per track ID
        colors: Dict[int, Tuple[int, int, int]] = {}
        for tid in active:
            np.random.seed(tid)
            colors[tid] = tuple(int(c) for c in np.random.randint(0, 255, 3))

        for tid, t in active.items():
            if not t.detections:
                continue
            det = t.detections[-1]
            x, y, w, h = det["bbox"]
            color = colors[tid]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"ID:{tid}", (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if len(t.detections) > 1:
                pts = [np.asarray(d["center"]) for d in t.detections[-10:]]
                for i in range(len(pts) - 1):
                    cv2.line(vis, tuple(pts[i].astype(int)),
                             tuple(pts[i + 1].astype(int)), color, 2)

        # Unmatched detections are drawn in yellow
        matched_ids = {id(t.detections[-1]) for t in active.values() if t.detections}
        for det in detections:
            if id(det) not in matched_ids:
                x, y, w, h = det["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)

        cv2.putText(
            vis,
            f"Frame {frame_idx}  Active: {len(active)}  Detections: {len(detections)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        return vis

    # Statistics

    @staticmethod
    def _compute_statistics(tracks: List[Track]) -> Dict:
        if not tracks:
            return {
                "total_tracks": 0,
                "avg_track_length": 0.0,
                "avg_speed_um_s": 0.0,
                "avg_area_um2": 0.0,
                "avg_linearity": 0.0,
            }
        lengths = [t.length_frames for t in tracks]
        speeds = [t.speed_um_per_sec for t in tracks if t.speed_um_per_sec > 0]
        areas = [t.avg_area_um2 for t in tracks if t.avg_area_um2 > 0]
        lin = [t.linearity for t in tracks]
        return {
            "total_tracks": len(tracks),
            "avg_track_length": float(np.mean(lengths)),
            "max_track_length": int(max(lengths)),
            "min_track_length": int(min(lengths)),
            "avg_speed_um_s": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed_um_s": float(max(speeds)) if speeds else 0.0,
            "avg_area_um2": float(np.mean(areas)) if areas else 0.0,
            "avg_linearity": float(np.mean(lin)) if lin else 0.0,
        }
