#!/usr/bin/env python
"""Run mitochondria tracking on every video in a folder.

Examples

    # Minimal usage — uses defaults from configs/default.yaml
    python scripts/track_folder.py \
        --input data/sample \
        --output outputs/tracking \
        --model models/tracker.pkl

    # Skip visualization to save time
    python scripts/track_folder.py \
        --input data/raw \
        --output outputs/tracking \
        --model models/tracker.pkl \
        --no-viz

    # Process only .avi files, custom thresholds
    python scripts/track_folder.py \
        --input data/raw \
        --output outputs/tracking \
        --model models/tracker.pkl \
        --pattern "*.avi" \
        --min-confidence 0.7 \
        --association-threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from mito_tracking.config import Config
from mito_tracking.detector import MitochondriaDetector
from mito_tracking.tracker import MLTracker
from mito_tracking.utils import set_seed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-track mitochondria in all videos of a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder containing input videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Folder where results (JSON + optional videos) are written.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained tracking model (*.pkl).",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern for input video files.",
    )
    parser.add_argument(
        "--ppm",
        type=float,
        default=None,
        help="Pixels per micrometer (overrides config value).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second (overrides config value).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum detection confidence (overrides config value).",
    )
    parser.add_argument(
        "--association-threshold",
        type=float,
        default=None,
        help="ML association confidence threshold (overrides config value).",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=None,
        help="Frames a track can be lost before closing (overrides config value).",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip generation of annotated videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config value).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostic information.",
    )

    return parser.parse_args(argv)


def _resolve(cli_value, cfg_value, name: str, verbose: bool):
    """Return CLI value if provided, else config value."""
    if cli_value is not None:
        if verbose and cfg_value is not None:
            print(f"  {name}: {cli_value} (overrides config {cfg_value})")
        return cli_value
    return cfg_value


def main(argv=None) -> int:
    args = parse_args(argv)

    # Load configuration
    if args.config.exists():
        cfg = Config.from_yaml(args.config)
        if args.verbose:
            print(f"Loaded config: {args.config}")
    else:
        print(f"Warning: config file {args.config} not found, using built-in defaults.")
        cfg = Config()

    # Resolve parameters (CLI overrides config)
    ppm = _resolve(args.ppm, cfg.scale.pixels_per_micrometer, "ppm", args.verbose)
    fps = _resolve(args.fps, cfg.scale.fps, "fps", args.verbose)
    min_conf = _resolve(
        args.min_confidence, cfg.detection.min_confidence, "min_confidence", args.verbose
    )
    assoc_thr = _resolve(
        args.association_threshold,
        cfg.tracking.association_threshold,
        "association_threshold",
        args.verbose,
    )
    max_gap = _resolve(
        args.max_gap_frames, cfg.tracking.max_gap_frames, "max_gap_frames", args.verbose
    )
    seed = _resolve(args.seed, cfg.training.seed, "seed", args.verbose)

    # Validate inputs
    if not args.input.exists():
        print(f"Error: input folder not found: {args.input}", file=sys.stderr)
        return 1

    if not args.model.exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        print("Hint: run scripts/train.py first to create a model.")
        return 1

    videos = sorted(args.input.glob(args.pattern))
    if not videos:
        print(
            f"Error: no files matching '{args.pattern}' in {args.input}",
            file=sys.stderr,
        )
        return 1

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Fix seed for reproducibility
    set_seed(seed)

    # Print run header
    print("=" * 70)
    print("MITOCHONDRIA TRACKING")
    print("=" * 70)
    print(f"Input:                  {args.input.resolve()}")
    print(f"Output:                 {args.output.resolve()}")
    print(f"Model:                  {args.model.resolve()}")
    print(f"Videos found:           {len(videos)}")
    print(f"Pattern:                {args.pattern}")
    print(f"Pixels per micrometer:  {ppm}")
    print(f"FPS:                    {fps}")
    print(f"Min detection conf:     {min_conf}")
    print(f"Association threshold:  {assoc_thr}")
    print(f"Max gap frames:         {max_gap}")
    print(f"Save visualization:     {not args.no_viz}")
    print(f"Seed:                   {seed}")
    print("=" * 70)

    # Initialize detector and tracker
    detector = MitochondriaDetector(
        pixels_per_micrometer=ppm,
        min_confidence=min_conf,
    )

    try:
        tracker = MLTracker(
            model_path=str(args.model),
            pixels_per_micrometer=ppm,
            fps=fps,
            min_confidence=min_conf,
            max_gap_frames=max_gap,
            association_threshold=assoc_thr,
            detector=detector,
        )
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        return 2

    # Process each video
    successes = 0
    failures = 0
    failed_videos: list[tuple[str, str]] = []

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {video.name}")
        try:
            result = tracker.track_video(
                str(video),
                output_dir=str(args.output),
                save_visualization=not args.no_viz,
            )
            if result is None:
                failures += 1
                failed_videos.append((video.name, "returned None"))
                continue
            successes += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping.")
            break
        except Exception as exc:
            failures += 1
            failed_videos.append((video.name, str(exc)))
            print(f"\nError processing {video.name}: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total videos:  {len(videos)}")
    print(f"Successful:    {successes}")
    print(f"Failed:        {failures}")

    if failed_videos:
        print("\nFailed videos:")
        for name, err in failed_videos:
            print(f"  - {name}: {err}")

    print(f"\nResults in: {args.output.resolve()}")

    # Exit code reflects whether any failures occurred
    if failures == 0:
        return 0
    if successes == 0:
        return 3
    return 4  # partial success


if __name__ == "__main__":
    sys.exit(main())
