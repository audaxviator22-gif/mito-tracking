#!/usr/bin/env python
"""Generating static and interactive plots from a tracking-results JSON file.

Produces:
  - Trajectory plot (PNG image)
  - Velocity distribution plot (PNG)
  - Directionality analysis plot (PNG)
  - Interactive Plotly dashboard (HTML)

Examples

    # Minimal usage — video is optional
    python scripts/visualize.py \
        --results outputs/tracking/mito_10_tracking_results.json

    # Use the source video as background for the trajectory plot
    python scripts/visualize.py \
        --results outputs/tracking/mito_10_tracking_results.json \
        --video data/sample/mito_10.mp4

    # Custom output directory
    python scripts/visualize.py \
        --results outputs/tracking/mito_10_tracking_results.json \
        --output figures/paper_fig1

    # Generate only selected plots
    python scripts/visualize.py \
        --results outputs/tracking/mito_10_tracking_results.json \
        --only trajectories velocity
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from mito_tracking.visualization import TrackVisualizer, visualize_tracking_results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualizations from a tracking-results JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to a *_tracking_results.json file.",
    )

    # Optional
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional source video used as background for trajectory plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to <results_folder>/visualizations.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["trajectories", "velocity", "directionality", "dashboard"],
        default=["trajectories", "velocity", "directionality", "dashboard"],
        help="Subset of plots to generate (default: all).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (dots per inch) for saved PNG figures.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostic information.",
    )

    return parser.parse_args(argv)


def _load_results(results_path: Path) -> dict:
    """Load and validate a tracking-results JSON file."""
    import json

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not results_path.is_file():
        raise ValueError(f"Expected a file, got: {results_path}")

    with open(results_path, "r") as f:
        data = json.load(f)

    if "tracks" not in data:
        raise ValueError(
            f"Results file {results_path.name} has no 'tracks' key. "
            "Expected the output of MLTracker.track_video() or BatchProcessor."
        )

    return data


def main(argv=None) -> int:
    args = parse_args(argv)

    # Validate inputs
    if not args.results.exists():
        print(f"Error: results file not found: {args.results}", file=sys.stderr)
        print("Hint: run scripts/track_folder.py first to generate results.", file=sys.stderr)
        return 1

    if args.video is not None and not args.video.exists():
        print(f"Warning: video file not found: {args.video}", file=sys.stderr)
        print("         Trajectory plot will be produced without a background.", file=sys.stderr)
        args.video = None

    # Load results
    try:
        data = _load_results(args.results)
    except Exception as exc:
        print(f"Error loading results: {exc}", file=sys.stderr)
        return 2

    tracks = data.get("tracks", [])
    if not tracks:
        print(f"No tracks in {args.results}. Nothing to visualize.", file=sys.stderr)
        return 3

    # Resolve output directory
    if args.output is None:
        output_dir = args.results.parent / "visualizations"
    else:
        output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print run header
    name = args.results.stem.replace("_tracking_results", "")
    print("=" * 70)
    print("TRACKING VISUALIZATION")
    print("=" * 70)
    print(f"Results file:     {args.results.resolve()}")
    print(f"Video:            {args.video.resolve() if args.video else '(none)'}")
    print(f"Video name:       {name}")
    print(f"Tracks loaded:    {len(tracks)}")
    print(f"Pixels per µm:    {data.get('pixels_per_um', 10)}")
    print(f"FPS:              {data.get('fps', 30)}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Plots requested:  {', '.join(args.only)}")
    print(f"Figure DPI:       {args.dpi}")
    print("=" * 70)

    # Load background frame if video provided
    background_img = None
    if args.video is not None:
        try:
            import cv2
            cap = cv2.VideoCapture(str(args.video))
            ret, frame = cap.read()
            cap.release()
            if ret:
                background_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if args.verbose:
                    print(f"Loaded background frame from {args.video.name}")
            else:
                print("Warning: could not read first frame from video.", file=sys.stderr)
        except Exception as exc:
            print(f"Warning: could not load video background: {exc}", file=sys.stderr)

    # Build visualizer
    ppm = data.get("pixels_per_um", 10)
    fps = data.get("fps", 30)
    viz = TrackVisualizer(tracks, pixels_per_um=ppm, fps=fps)

    # Generate requested plots
    produced_files: list[Path] = []

    import matplotlib
    matplotlib.rcParams["savefig.dpi"] = args.dpi

    try:
        if "trajectories" in args.only:
            out = output_dir / f"{name}_trajectories.png"
            viz.plot_trajectories(
                background_img=background_img,
                output_file=str(out),
            )
            produced_files.append(out)

        if "velocity" in args.only:
            out = output_dir / f"{name}_velocity.png"
            viz.plot_velocity(output_file=str(out))
            produced_files.append(out)

        if "directionality" in args.only:
            out = output_dir / f"{name}_directionality.png"
            viz.plot_directionality(output_file=str(out))
            produced_files.append(out)

        if "dashboard" in args.only:
            out = output_dir / f"{name}_dashboard.html"
            viz.interactive_dashboard(str(out))
            produced_files.append(out)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError during plotting: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 4

    # Final summary
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Generated {len(produced_files)} file(s):")
    for f in produced_files:
        size_kb = f.stat().st_size / 1024 if f.exists() else 0
        print(f"  {f.name:<50} {size_kb:>8.1f} KB")
    print(f"\nOutput directory: {output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
