"""Batch processing of many microscopy videos.

This module runs the classical (geometric) tracker over every video in a
folder and produces:

- One ``<video>_tracks.json`` file per video (used for training)
- ``batch_summary.json`` + ``batch_summary.csv`` (used for reports)
- Optionally, one ``<video>_tracking.mp4`` per video (for visual QC)
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .detector import MitochondriaDetector
from .tracker import TrackingAnnotationGenerator


# BatchProcessor

class BatchProcessor:
    """Process a folder of videos with the geometric tracker.

    Parameters:
    pixels_per_micrometer : float, default 10.0
        Spatial calibration. Must match the value used at inference time.
    fps : float, default 30.0
        Temporal calibration.
    detector : MitochondriaDetector, optional
        Inject a custom detector. If None, a fresh one is created per video
        with default parameters and the configured ppm.
    max_gap_frames : int, default 5
        Frames a track can be missing before being closed.
    min_track_length : int, default 10
        Minimum number of frames per track. Shorter tracks are discarded.
    verbose : bool, default True
        Print per-video progress.
    """

    def __init__(
        self,
        pixels_per_micrometer: float = 10.0,
        fps: float = 30.0,
        detector: Optional[MitochondriaDetector] = None,
        max_gap_frames: int = 5,
        min_track_length: int = 10,
        verbose: bool = True,
    ) -> None:
        self.ppm = float(pixels_per_micrometer)
        self.fps = float(fps)
        self.detector = detector
        self.max_gap_frames = int(max_gap_frames)
        self.min_track_length = int(min_track_length)
        self.verbose = bool(verbose)

        self.results: Dict[str, Dict] = {}

    # Public API — folder processing

    def process_folder(
        self,
        input_folder: str,
        output_folder: Optional[str] = None,
        video_pattern: str = "*.mp4",
        save_visualizations: bool = False,
    ) -> Dict[str, Dict]:
        """Process all matching videos in a folder.

        Parameters:
        input_folder : str or Path
            Folder containing source videos.
        output_folder : str or Path, optional
            Where to write per-video .json and summary files. Defaults to
            ``<input_folder>/tracking_data``.
        video_pattern : str, default '*.mp4'
            Video pattern selection (e.g., ``"*.avi"`` or ``"mito_*.mp4"``).
        save_visualizations : bool, default False
            If True, render one annotated MP4 per video.

        Returns

        dict
            ``{video_stem: result_dict}``.
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_path}")

        output_path = Path(output_folder) if output_folder else input_path / "tracking_data"
        output_path.mkdir(parents=True, exist_ok=True)

        videos = sorted(input_path.glob(video_pattern))
        if not videos:
            print(f"No videos matching '{video_pattern}' in {input_path}")
            return {}

        if self.verbose:
            print("=" * 60)
            print("BATCH TRACKING")
            print("=" * 60)
            print(f"Found {len(videos)} video(s)")
            print(f"Output: {output_path}")
            print("=" * 60)

        for i, video in enumerate(videos, 1):
            if self.verbose:
                print(f"\n[{i}/{len(videos)}] {video.name}")
            try:
                self.results[video.stem] = self._process_one(
                    video=video,
                    output_dir=output_path,
                    save_visualization=save_visualizations,
                )
            except Exception as exc:
                print(f"\n Error processing {video.name}: {exc}")
                traceback.print_exc()
                self.results[video.stem] = {"error": str(exc)}

        self._create_summary(output_path)
        return self.results

    # Single-video processing


    def _process_one(
        self,
        video: Path,
        output_dir: Path,
        save_visualization: bool,
    ) -> Dict:
        """Run the geometric tracker on one video, and then save its annotations."""
        # Fresh tracker per video (avoids leaking `next_track_id` across videos)
        tracker = TrackingAnnotationGenerator(
            pixels_per_micrometer=self.ppm,
            fps=self.fps,
            detector=self.detector,
        )

        tracks = tracker.track_video(
            str(video),
            max_gap_frames=self.max_gap_frames,
            min_track_length=self.min_track_length,
            verbose=self.verbose,
        )

        if not tracks:
            return {
                "video_name": video.stem,
                "total_tracks": 0,
                "analysis": {},
                "output_file": None,
            }

        # Save annotations
        json_out = output_dir / f"{video.stem}_tracks.json"
        tracker.save_annotation_format(tracks, str(video), str(json_out))

        # Optional visualization
        if save_visualization:
            vis_out = output_dir / f"{video.stem}_tracking.mp4"
            from .visualization import TrackVisualizer
            TrackVisualizer._render_video(str(video), tracks, str(vis_out))

        return {
            "video_name": video.stem,
            "total_tracks": len(tracks),
            "analysis": self._analyze(tracks),
            "output_file": str(json_out),
        }

    # Track statistics

    @staticmethod
    def _analyze(tracks: List) -> Dict:
        """Compute summary statistics for a list of tracks.

        Classification thresholds:
        - Non-linear: speed < 0.1 µm/s
        - Directed:   linearity > 0.7
        - Diffusive:  the rest
        """
        if not tracks:
            return {}

        speeds = [t.speed_um_per_sec for t in tracks]
        lengths = [t.length_frames for t in tracks]
        linearities = [t.linearity for t in tracks]
        areas = [t.avg_area_um2 for t in tracks if t.avg_area_um2 > 0]

        Non-linear = sum(1 for s in speeds if s < 0.1)
        directed = sum(1 for l in linearities if l > 0.7)
        diffusive = len(tracks) - Non-linear - directed

        return {
            "total_tracks": len(tracks),
            "avg_speed_um_s": float(np.mean(speeds)) if speeds else 0.0,
            "std_speed_um_s": float(np.std(speeds)) if speeds else 0.0,
            "max_speed_um_s": float(max(speeds)) if speeds else 0.0,
            "avg_track_length_frames": float(np.mean(lengths)) if lengths else 0.0,
            "max_track_length_frames": int(max(lengths)) if lengths else 0,
            "avg_linearity": float(np.mean(linearities)) if linearities else 0.0,
            "avg_area_um2": float(np.mean(areas)) if areas else 0.0,
            "movement_classification": {
                "Non-linear": Non-linear,
                "directed": directed,
                "diffusive": diffusive,
            },
        }

    # Summary report

    def _create_summary(self, output_folder: Path) -> Dict:
        """Write batch_summary.json and batch_summary.csv."""
        summary = {
            "processing_date": datetime.now().isoformat(),
            "config": {
                "pixels_per_micrometer": self.ppm,
                "fps": self.fps,
                "max_gap_frames": self.max_gap_frames,
                "min_track_length": self.min_track_length,
            },
            "total_videos": len(self.results),
            "successful": sum(1 for r in self.results.values() if "error" not in r),
            "failed": sum(1 for r in self.results.values() if "error" in r),
            "total_tracks_all_videos": 0,
            "videos": [],
        }

        rows: List[Dict] = []
        total_tracks = 0

        for name, result in self.results.items():
            if "error" not in result:
                n = result.get("total_tracks", 0)
                total_tracks += n
                analysis = result.get("analysis", {})

                summary["videos"].append({
                    "name": name,
                    "status": "success",
                    "tracks": n,
                    "analysis": analysis,
                    "output_file": result.get("output_file"),
                })

                mc = analysis.get("movement_classification", {})
                rows.append({
                    "Video": name,
                    "Status": "Success",
                    "Tracks": n,
                    "Avg_Track_Length_frames": round(analysis.get("avg_track_length_frames", 0.0), 1),
                    "Avg_Speed_um_s": round(analysis.get("avg_speed_um_s", 0.0), 3),
                    "Max_Speed_um_s": round(analysis.get("max_speed_um_s", 0.0), 3),
                    "Avg_Area_um2": round(analysis.get("avg_area_um2", 0.0), 3),
                    "Avg_Linearity": round(analysis.get("avg_linearity", 0.0), 3),
                    "Non-linear": mc.get("Non-linear", 0),
                    "Directed": mc.get("directed", 0),
                    "Diffusive": mc.get("diffusive", 0),
                })
            else:
                summary["videos"].append({
                    "name": name,
                    "status": "failed",
                    "error": result["error"],
                })
                rows.append({
                    "Video": name,
                    "Status": "Failed",
                    "Tracks": 0,
                    "Avg_Track_Length_frames": 0,
                    "Avg_Speed_um_s": 0.0,
                    "Max_Speed_um_s": 0.0,
                    "Avg_Area_um2": 0.0,
                    "Avg_Linearity": 0.0,
                    "Non-linear": 0,
                    "Directed": 0,
                    "Diffusive": 0,
                })

        summary["total_tracks_all_videos"] = total_tracks

        # Write JSON
        json_file = output_folder / "batch_summary.json"
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Write CSV
        df = pd.DataFrame(rows)
        csv_file = output_folder / "batch_summary.csv"
        df.to_csv(csv_file, index=False)

        # Print a short report
        if self.verbose:
            print(f"\n{'=' * 60}\nBATCH PROCESSING SUMMARY\n{'=' * 60}")
            print(f"Videos: {summary['total_videos']}, "
                  f"success: {summary['successful']}, "
                  f"failed: {summary['failed']}")
            print(f"Total tracks: {summary['total_tracks_all_videos']}")
            print(f"\nSaved:\n  {json_file}\n  {csv_file}")
            print(f"\n{df.to_string(index=False)}")

        return summary

    # Export all per-video JSONs into one flat file

    @staticmethod
    def export_training_data(
        output_folder: str,
        fmt: str = "csv",
    ) -> List[Dict]:
        """Flatten per-video annotation JSONs into a single file.

        Each row is one detection, with its own video name, frame index, track id, bounding box, and detection features.

        Parameters:

        output_folder : str or Path
            Folder containing the ``*_tracks.json`` files.
        fmt : {'csv', 'json', 'mot'}
            Output format. ``'mot'`` produces a MOTChallenge-style text file
            (``frame,id,x,y,w,h,conf,-1,-1,-1``).

        Returns

        list of dict
            All detection rows (empty list if no files were found).
        """
        folder = Path(output_folder)
        json_files = sorted(folder.glob("*_tracks.json"))
        if not json_files:
            print(f"No '*_tracks.json' files found in {folder}")
            return []

        rows: List[Dict] = []
        for jf in json_files:
            with open(jf, "r") as f:
                data = json.load(f)

            video_name = data.get("video_name", jf.stem)
            for frame_str, detections in data.get("frame_detections", {}).items():
                frame = int(frame_str)
                for det in detections:
                    rows.append({
                        "video": video_name,
                        "frame": frame,
                        "track_id": int(det["track_id"]),
                        "bbox_x": float(det["bbox"][0]),
                        "bbox_y": float(det["bbox"][1]),
                        "bbox_w": float(det["bbox"][2]),
                        "bbox_h": float(det["bbox"][3]),
                        "center_x": float(det.get("center", [0, 0])[0]),
                        "center_y": float(det.get("center", [0, 0])[1]),
                        "area_um2": float(det.get("area_um2", 0.0)),
                        "confidence": float(det.get("confidence", 0.0)),
                        "circularity": float(det.get("circularity", 0.0)),
                        "aspect_ratio": float(det.get("aspect_ratio", 0.0)),
                        "mean_intensity": float(det.get("mean_intensity", 0.0)),
                    })

        if not rows:
            print("No detections to export.")
            return []

        if fmt == "csv":
            path = folder / "training_data_all.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
        elif fmt == "json":
            path = folder / "training_data_all.json"
            with open(path, "w") as f:
                json.dump(rows, f, indent=2)
        elif fmt == "mot":
            path = folder / "training_data_mot.txt"
            with open(path, "w") as f:
                for r in rows:
                    # MOTChallenge: frame, id, x, y, w, h, conf, -1, -1, -1
                    f.write(
                        f"{r['frame'] + 1},{r['track_id']},"
                        f"{r['bbox_x']:.2f},{r['bbox_y']:.2f},"
                        f"{r['bbox_w']:.2f},{r['bbox_h']:.2f},"
                        f"{r['confidence']:.4f},-1,-1,-1\n"
                    )
        else:
            raise ValueError(f"Unknown format: {fmt} (use 'csv', 'json', or 'mot')")

        print(f"✓ Exported {len(rows)} detections to: {path}")
        return rows


# Convenience functions

def process_videos_batch(
    input_folder: str,
    output_folder: Optional[str] = None,
    ppm: float = 10.0,
    fps: float = 30.0,
    max_gap: int = 5,
    min_length: int = 10,
    save_viz: bool = False,
    video_pattern: str = "*.mp4",
) -> Dict[str, Dict]:
    """One-call wrapper: create a BatchProcessor and run it."""
    processor = BatchProcessor(
        pixels_per_micrometer=ppm,
        fps=fps,
        max_gap_frames=max_gap,
        min_track_length=min_length,
    )
    return processor.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        video_pattern=video_pattern,
        save_visualizations=save_viz,
    )


def export_training_data(annotation_folder: str, fmt: str = "csv") -> List[Dict]:
    """One-call wrapper for exporting flat training data."""
    return BatchProcessor.export_training_data(annotation_folder, fmt=fmt)
