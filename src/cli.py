"""Command-line interface for the mito-tracking package.

It has four commands, all installed as shell entry points after
``pip install -e .``:

- ``mito-train`` - Trains the association model from annotations
- ``mito-track`` - Runs inference on a single video or a folder
- ``mito-viz``   - Generates figures from a tracking-results JSON
- ``mito-batch`` - Runs the geometric tracker over a folder to build
                  training annotations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .batch import BatchProcessor, export_training_data
from .config import Config
from .detector import MitochondriaDetector
from .tracker import MLTracker
from .training import TrackingModelTrainer
from .utils import set_seed
from .visualization import visualize_tracking_results


# Shared argument helpers

def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a YAML config file. Defaults to configs/default.yaml if present.",
    )


def _add_scale_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ppm", type=float, default=None,
        help="Pixels per micrometer. Overrides config.",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Frames per second. Overrides config.",
    )


def _add_seed_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42.",
    )


def _load_config(args: argparse.Namespace) -> Config:
    """Load config from --config if given, else from configs/default.yaml, else defaults."""
    if args.config is not None:
        if not args.config.exists():
            print(f"Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        return Config.from_yaml(args.config)

    default = Path("configs/default.yaml")
    if default.exists():
        return Config.from_yaml(default)

    return Config()


def _resolve_scale(args: argparse.Namespace, cfg: Config) -> tuple[float, float]:
    """Pick ppm/fps from CLI args or config, with CLI taking precedence."""
    ppm = args.ppm if args.ppm is not None else cfg.scale.pixels_per_micrometer
    fps = args.fps if args.fps is not None else cfg.scale.fps
    return float(ppm), float(fps)


# mito-train

def train_main(argv: Optional[list] = None) -> int:
    """Entry point for the ``mito-train`` command."""
    parser = argparse.ArgumentParser(
        prog="mito-train",
        description="Train the mitochondria tracking-association model.",
    )
    parser.add_argument(
        "--annotations", type=Path, required=True,
        help="Folder containing *_tracks.json annotation files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/tracker.pkl"),
        help="Path to save the trained model (.pkl).",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of trees in the Random Forest. Default: 100.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Maximum tree depth. Default: 10.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of examples used for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--negative-ratio", type=float, default=1.0,
        help="Negatives per positive. 1.0 = balanced. Default: 1.0.",
    )
    _add_scale_args(parser)
    _add_seed_arg(parser)
    _add_config_arg(parser)

    args = parser.parse_args(argv)
    set_seed(args.seed)

    cfg = _load_config(args)
    ppm, _ = _resolve_scale(args, cfg)

    if not args.annotations.exists():
        print(f"Annotations folder not found: {args.annotations}", file=sys.stderr)
        return 1

    trainer = TrackingModelTrainer(
        pixels_per_micrometer=ppm,
        reference_distance_um=cfg.features.reference_distance_um,
        seed=args.seed,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        negative_ratio=args.negative_ratio,
        max_negatives=cfg.training.max_negatives,
    )

    if not trainer.train(str(args.annotations)):
        print("Training failed.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not trainer.save(str(args.output)):
        return 1

    return 0


# mito-track

def track_main(argv: Optional[list] = None) -> int:
    """Entry point for the ``mito-track`` command."""
    parser = argparse.ArgumentParser(
        prog="mito-track",
        description="Track mitochondria in one video or a folder of videos.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Single video file.")
    source.add_argument("--input-folder", type=Path, help="Folder of videos.")

    parser.add_argument(
        "--output", type=Path, required=True,
        help="Directory where results (JSON + optional MP4) are written.",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to a trained .pkl model.",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.mp4",
        help="Glob pattern when processing a folder. Default: *.mp4.",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.6,
        help="Detection confidence threshold. Default: 0.6.",
    )
    parser.add_argument(
        "--association-threshold", type=float, default=0.5,
        help="ML association probability threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--max-gap", type=int, default=5,
        help="Maximum consecutive missing frames. Default: 5.",
    )
    parser.add_argument(
        "--min-track-length", type=int, default=5,
        help="Minimum frames per track. Default: 5.",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization video rendering.",
    )
    _add_scale_args(parser)
    _add_seed_arg(parser)
    _add_config_arg(parser)

    args = parser.parse_args(argv)
    set_seed(args.seed)

    cfg = _load_config(args)
    ppm, fps = _resolve_scale(args, cfg)

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    # Building a new detector for an invocation
    detector = MitochondriaDetector(
        pixels_per_micrometer=ppm,
        min_area_um2=cfg.detection.min_area_um2,
        max_area_um2=cfg.detection.max_area_um2,
        max_circularity=cfg.detection.max_circularity,
        min_intensity=cfg.detection.min_intensity,
        min_confidence=args.min_confidence,
    )

    tracker = MLTracker(
        model_path=str(args.model),
        pixels_per_micrometer=ppm,
        fps=fps,
        min_confidence=args.min_confidence,
        max_gap_frames=args.max_gap,
        min_track_length=args.min_track_length,
        association_threshold=args.association_threshold,
        detector=detector,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        if not args.input.exists():
            print(f"Video not found: {args.input}", file=sys.stderr)
            return 1
        result = tracker.track_video(
            str(args.input),
            output_dir=str(args.output),
            save_visualization=not args.no_viz,
        )
        if result is None:
            return 1
    else:
        videos = sorted(args.input_folder.glob(args.pattern))
        if not videos:
            print(f"No videos matching {args.pattern} in {args.input_folder}",
                  file=sys.stderr)
            return 1
        for v in videos:
            print(f"\n[{v.name}]")
            tracker.track_video(
                str(v),
                output_dir=str(args.output),
                save_visualization=not args.no_viz,
            )

    return 0


# mito-viz

def visualize_main(argv: Optional[list] = None) -> int:
    """Entry point for the ``mito-viz`` command."""
    parser = argparse.ArgumentParser(
        prog="mito-viz",
        description="Generate figures from a tracking-results JSON file.",
    )
    parser.add_argument(
        "--results", type=Path, required=True,
        help="Path to *_tracking_results.json.",
    )
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Optional source video for background in trajectory plot.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to save figures. Default: <results_folder>/visualizations.",
    )

    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"Results file not found: {args.results}", file=sys.stderr)
        return 1

    viz = visualize_tracking_results(
        str(args.results),
        video_path=str(args.video) if args.video else None,
        output_dir=str(args.output) if args.output else None,
    )

    return 0 if viz is not None else 1

# mito-batch

def batch_main(argv: Optional[list] = None) -> int:
    """Entry point for the ``mito-batch`` command.

    Runs the geometric tracker over a folder to generate training annotations.
    """
    parser = argparse.ArgumentParser(
        prog="mito-batch",
        description=(
            "Run the geometric tracker over a folder of videos to produce "
            "training annotations."
        ),
    )
    parser.add_argument(
        "input", type=Path, required=True,
        help="Folder containing source videos.",
    )
    parser.add_argument(
        "output", type=Path, default=None,
        help="Where to write per-video JSONs and summary. Default: <input>/tracking_data.",
    )
    parser.add_argument(
        "pattern", type=str, default="*.mp4",
        help="Glob pattern for video. Default: *.mp4.",
    )
    parser.add_argument(
        "max-gap", type=int, default=5,
        help="Maximum consecutive missing frames. Default: 5.",
    )
    parser.add_argument(
        "min-track-length", type=int, default=10,
        help="Minimum frames per track. Default: 10.",
    )
    parser.add_argument(
        "export", choices=["csv", "json", "mot"], default=None,
        help="Optional: export all detections to a flat file in this format.",
    )
    parser.add_argument(
        "viz", action="store_true",
        help="Also render one MP4 per video (slower).",
    )
    parser.add_argument(
        "quiet", action="store_true",
        help="Suppress per-video progress output.",
    )
    _add_scale_args(parser)
    _add_config_arg(parser)

    args = parser.parse_args(argv)

    cfg = _load_config(args)
    ppm, fps = _resolve_scale(args, cfg)

    if not args.input.exists():
        print(f"Input folder not found: {args.input}", file=sys.stderr)
        return 1

    processor = BatchProcessor(
        pixels_per_micrometer=ppm,
        fps=fps,
        max_gap_frames=args.max_gap,
        min_track_length=args.min_track_length,
        verbose=not args.quiet,
    )

    results = processor.process_folder(
        input_folder=str(args.input),
        output_folder=str(args.output) if args.output else None,
        video_pattern=args.pattern,
        save_visualizations=args.viz,
    )

    if not results:
        return 1

    if args.export:
        out_dir = args.output if args.output else args.input / "tracking_data"
        export_training_data(str(out_dir), fmt=args.export)

    return 0


# Top-level dispatcher (optional)

def main(argv: Optional[list] = None) -> int:
    """Optional top-level dispatcher.

    Usage:
        python -m mito_tracking train ...
        python -m mito_tracking track ...
        python -m mito_tracking viz ...
        python -m mito_tracking batch ...
    """
    parser = argparse.ArgumentParser(
        prog="mito-tracking",
        description="Unified CLI. Run `mito-tracking <command> --help` for details.",
    )
    parser.add_argument(
        "command",
        choices=["train", "track", "viz", "batch"],
        help="Subcommand to run.",
    )
    args, remaining = parser.parse_known_args(argv)

    dispatch = {
        "train": train_main,
        "track": track_main,
        "viz":   visualize_main,
        "batch": batch_main,
    }
    return dispatch[args.command](remaining)


if __name__ == "__main__":
    sys.exit(main())
