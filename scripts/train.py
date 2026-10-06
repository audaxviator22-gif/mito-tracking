#!/usr/bin/env python
"""Train the tracking-association model from annotation files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mito_tracking.config import Config
from mito_tracking.training import TrackingModelTrainer
from mito_tracking.utils import set_seed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest association model for mitochondria tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Folder containing *_tracks.json annotation files.",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tracker.pkl"),
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--ppm",
        type=float,
        default=None,
        help="Pixels per micrometer (overrides config value).",
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

    # --- Resolve parameters (CLI overrides config) ---
    ppm = args.ppm if args.ppm is not None else cfg.scale.pixels_per_micrometer
    seed = args.seed if args.seed is not None else cfg.training.seed

    # --- Validate input ---
    if not args.annotations.exists():
        print(f"Error: annotations folder not found: {args.annotations}", file=sys.stderr)
        return 1

    json_files = list(args.annotations.glob("*_tracks.json"))
    if not json_files:
        print(f"Error: no *_tracks.json files found in {args.annotations}", file=sys.stderr)
        print("Hint: run scripts/track_folder.py first to generate annotations.")
        return 1

    if args.verbose:
        print(f"Found {len(json_files)} annotation files.")
        print(f"Pixels per micrometer: {ppm}")
        print(f"Seed: {seed}")

    # Fix seed for reproducibility
    set_seed(seed)

    # Train
    trainer = TrackingModelTrainer(
        pixels_per_micrometer=ppm,
        seed=seed,
    )

    success = trainer.train(str(args.annotations))
    if not success:
        print("Training failed.", file=sys.stderr)
        return 2

    # Save model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not trainer.save(str(args.output)):
        print("Failed to save model.", file=sys.stderr)
        return 3

    print(f"\nModel saved to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
