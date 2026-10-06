"""Static and interactive visualization of mitochondria tracking results.

This module turns a list of tracks (as loaded from a
``*_tracking_results.json`` file) into the figures:

- ``plot_trajectories`` - 2D overview of every track's path
- ``plot_velocity`` - violin/box/histogram of speeds
- ``plot_directionality`` - linearity, rose plot, and classification
- ``interactive_dashboard`` - an HTML Plotly dashboard for exploration
- ``_render_video`` - writes an annotated MP4 (used by the tracker)

All figures are saved as PNG by default..
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# TrackVisualizer


class TrackVisualizer:
    """Comprehensive visualization of mitochondria tracks.

    Parameters
    tracks : list of dict
        Track dictionaries, each with at least ``track_id``, ``frames``,
        ``length_frames``, ``speed_um_per_sec``, ``linearity``, and (usually) ``detections`` 
        containing per-frame ``bbox``/``center``/``area_um2``.
    pixels_per_um : float, default 10.0
        Spatial calibration, used for scaling distance-based plots.
    fps : float, default 30.0
        Temporal calibration, used for scaling velocity-based plots.
    dpi : int, default 300
        DPI for saved PNG figures.
    """

    def __init__(
        self,
        tracks: List[Dict],
        pixels_per_um: float = 10.0,
        fps: float = 30.0,
        dpi: int = 300,
    ) -> None:
        self.tracks = list(tracks)
        self.ppm = float(pixels_per_um)
        self.fps = float(fps)
        self.dpi = int(dpi)
        self.colors = self._generate_colors(len(self.tracks))

    # Shared helpers

    @staticmethod
    def _generate_colors(n: int) -> List:
        """Return a list of distinguishable colors."""
        if n <= 0:
            return []
        cmap = plt.cm.get_cmap("tab20", max(n, 20))
        return [cmap(i % 20) for i in range(n)]

    @staticmethod
    def _centers(track: Dict) -> np.ndarray:
        """Return an (N, 2) array of per-frame centers."""
        out = []
        for det in track.get("detections", []):
            center = det.get("center")
            if center is None:
                x, y, w, h = det["bbox"]
                center = [x + w / 2, y + h / 2]
            out.append(center)
        return np.asarray(out, dtype=float) if out else np.zeros((0, 2))

    @staticmethod
    def _save_figure(fig, output_file: Optional[str], dpi: int = 300) -> None:
        """Save and close a matplotlib figure."""
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {output_file}")
        plt.close(fig)

    #  Trajectories

    def plot_trajectories(
        self,
        figsize: Tuple[int, int] = (12, 10),
        background_img: Optional[np.ndarray] = None,
        output_file: Optional[str] = None,
        show_arrows: bool = True,
    ) -> Optional[Tuple]:
        """Plot the 2D trajectory of every track.

        Parameters
        figsize : (width, height) in inches.
        background_img : optional
            A first-frame image to render behind the trajectories.
        output_file : optional
            Path to save the figure. If None, the figure is discarded.
        show_arrows : bool
            If True, it draws small arrows along each trajectory to indicate
            the direction of motion.
        """
        if not self.tracks:
            print("No tracks to plot")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        if background_img is not None:
            ax.imshow(background_img, cmap="gray", alpha=0.5)

        for i, track in enumerate(self.tracks):
            centers = self._centers(track)
            if len(centers) < 2:
                continue

            color = self.colors[i % len(self.colors)]
            ax.plot(
                centers[:, 0], centers[:, 1],
                "-", color=color, linewidth=1.5, alpha=0.8,
            )

            # Start (green) and end (red) markers
            ax.plot(centers[0, 0], centers[0, 1], "o",
                    color="green", markersize=7,
                    markeredgecolor="white", markeredgewidth=1)
            ax.plot(centers[-1, 0], centers[-1, 1], "s",
                    color="red", markersize=7,
                    markeredgecolor="white", markeredgewidth=1)

            # Direction arrows (a few, evenly spaced along the track)
            if show_arrows and len(centers) > 10:
                n_arrows = min(5, len(centers) // 10)
                if n_arrows > 0:
                    idxs = np.linspace(0, len(centers) - 1, n_arrows, dtype=int)
                    for k in idxs[1:-1]:
                        dx = centers[k + 1, 0] - centers[k, 0]
                        dy = centers[k + 1, 1] - centers[k, 1]
                        if dx * dx + dy * dy > 0:
                            ax.arrow(
                                centers[k, 0], centers[k, 1],
                                dx * 0.3, dy * 0.3,
                                head_width=5, head_length=5,
                                fc=color, ec=color, alpha=0.6,
                            )

            # Track ID label at midpoint
            mid = len(centers) // 2
            ax.text(
                centers[mid, 0], centers[mid, 1] + 8,
                f"ID:{track['track_id']}",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        # Stats box in the top left corner
        stats_text = self._stats_text()
        ax.text(
            0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("X position (pixels)", fontsize=12)
        ax.set_ylabel("Y position (pixels)", fontsize=12)
        ax.set_title("Mitochondria track trajectories",
                     fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        if len(self.tracks) <= 20:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()
        self._save_figure(fig, output_file, self.dpi)
        return fig, ax

    # Velocity distributions

    def plot_velocity(
        self,
        figsize: Tuple[int, int] = (14, 6),
        output_file: Optional[str] = None,
    ) -> Optional[Tuple]:
        """Violin plot + instantaneous velocity histogram + length-stratified view.

        Three panels:

        1. Violin + box of per-track average velocities.
        2. Histogram of instantaneous (frame-to-frame) velocities
        3. Velocity grouped by track length (short / medium / long).
        """
        # Extract velocities
        track_velocities = [
            t.get("speed_um_per_sec", 0.0)
            for t in self.tracks
            if t.get("speed_um_per_sec", 0.0) > 0
        ]
        if not track_velocities:
            print("No velocity data available")
            return None

        # Instantaneous velocities (from per-frame displacements)
        inst_velocities: List[float] = []
        for track in self.tracks:
            centers = self._centers(track)
            if len(centers) < 2:
                continue
            for i in range(len(centers) - 1):
                dist_um = float(np.linalg.norm(centers[i + 1] - centers[i])) / self.ppm
                dt = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
                inst_velocities.append(dist_um / dt if dt > 0 else 0.0)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: violin + box of per-track velocities
        axes[0].violinplot(
            track_velocities, positions=[1],
            showmeans=True, showmedians=True, widths=0.7,
        )
        axes[0].boxplot(track_velocities, positions=[1], widths=0.3,
                        patch_artist=True, showfliers=True)
        axes[0].set_xticks([1])
        axes[0].set_xticklabels(["All tracks"])
        axes[0].set_ylabel("Velocity (μm/s)", fontsize=12)
        axes[0].set_title("Average velocity per track",
                          fontsize=12, fontweight="bold")
        mean_v = float(np.mean(track_velocities))
        axes[0].axhline(mean_v, color="red", linestyle="--",
                        label=f"Mean = {mean_v:.2f}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Panel 2: instantaneous velocity histogram + KDE
        if inst_velocities:
            axes[1].hist(inst_velocities, bins=50, color="steelblue",
                         alpha=0.7, edgecolor="black", density=True)
            if len(inst_velocities) > 1:
                try:
                    kde = stats.gaussian_kde(inst_velocities)
                    xs = np.linspace(0, max(inst_velocities), 100)
                    axes[1].plot(xs, kde(xs), "r-", linewidth=2, label="KDE")
                    axes[1].legend()
                except Exception:
                    pass
            axes[1].set_xlabel("Instantaneous velocity (μm/s)", fontsize=12)
            axes[1].set_ylabel("Density", fontsize=12)
            axes[1].set_title("Instantaneous velocity",
                              fontsize=12, fontweight="bold")
            axes[1].grid(alpha=0.3)

        # Panel 3: velocity by track length category
        short = [t["speed_um_per_sec"] for t in self.tracks
                 if t.get("length_frames", 0) < 30 and t.get("speed_um_per_sec", 0) > 0]
        medium = [t["speed_um_per_sec"] for t in self.tracks
                  if 30 <= t.get("length_frames", 0) < 100 and t.get("speed_um_per_sec", 0) > 0]
        long_ = [t["speed_um_per_sec"] for t in self.tracks
                 if t.get("length_frames", 0) >= 100 and t.get("speed_um_per_sec", 0) > 0]

        groups = [g for g in (short, medium, long_) if g]
        labels = [l for l, g in zip(["Short\n(<30)", "Medium\n(30-100)", "Long\n(>100)"],
                                     (short, medium, long_)) if g]

        if groups:
            parts = axes[2].violinplot(groups, positions=range(1, len(groups) + 1),
                                       showmeans=True, showmedians=True)
            palette = ["lightcoral", "lightblue", "lightgreen"]
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(palette[i % len(palette)])
                body.set_alpha(0.7)
            axes[2].set_xticks(range(1, len(groups) + 1))
            axes[2].set_xticklabels(labels)
            axes[2].set_ylabel("Velocity (μm/s)", fontsize=12)
            axes[2].set_title("Velocity by track length",
                              fontsize=12, fontweight="bold")
            axes[2].grid(alpha=0.3)

        plt.suptitle("Velocity distribution analysis",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_file, self.dpi)
        return fig, axes

    # Directionality analysis

    def plot_directionality(
        self,
        figsize: Tuple[int, int] = (14, 10),
        output_file: Optional[str] = None,
    ) -> Optional[Tuple]:
        """Six-panel directionality analysis.

        Panels:

        1. Linearity histogram.
        2. Displacement vs path length (scatter colored by linearity).
        3. Direction rose plot (polar histogram of net movement direction).
        4. Linearity grouped by speed category (box plot).
        5. Linearity vs speed (scatter).
        6. Classification pie chart (directed / diffusive / confined).
        """
        # Extract metrics
        linearities, displacements, path_lengths, directions = [], [], [], []

        for track in self.tracks:
            centers = self._centers(track)
            if len(centers) < 2:
                continue

            lin = float(track.get("linearity", 1.0))
            disp = float(track.get("displacement_um", 0.0))

            path_um = float(np.sum([
                np.linalg.norm(centers[i + 1] - centers[i])
                for i in range(len(centers) - 1)
            ])) / self.ppm

            direction_vec = centers[-1] - centers[0]
            angle = float(np.arctan2(direction_vec[1], direction_vec[0]))

            linearities.append(lin)
            displacements.append(disp)
            path_lengths.append(path_um)
            directions.append(angle)

        if not linearities:
            print("No directionality data available")
            return None

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Panel 1: linearity histogram
        axes[0, 0].hist(linearities, bins=20, color="purple",
                        alpha=0.7, range=(0, 1), edgecolor="black")
        mean_lin = float(np.mean(linearities))
        axes[0, 0].axvline(mean_lin, color="red", linestyle="--",
                           label=f"Mean = {mean_lin:.3f}")
        axes[0, 0].set_xlabel("Linearity (net / path)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Linearity distribution", fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Panel 2: displacement vs path length
        sc = axes[0, 1].scatter(
            path_lengths, displacements, c=linearities,
            cmap="viridis", alpha=0.6, s=50,
        )
        lim = max(max(path_lengths, default=1), max(displacements, default=1))
        axes[0, 1].plot([0, lim], [0, lim], "r--", alpha=0.5, label="y = x")
        axes[0, 1].set_xlabel("Path length (μm)")
        axes[0, 1].set_ylabel("Displacement (μm)")
        axes[0, 1].set_title("Displacement vs path length", fontweight="bold")
        axes[0, 1].legend()
        plt.colorbar(sc, ax=axes[0, 1], label="Linearity")

        # Panel 3: direction rose plot
        ax_polar = fig.add_subplot(2, 3, 3, projection="polar")
        n_bins = 36
        theta = np.linspace(0, 2 * np.pi, n_bins + 1)
        hist, _ = np.histogram(directions, bins=theta)
        hist = hist.astype(float)
        if hist.max() > 0:
            hist /= hist.max()
        ax_polar.bar(theta[:-1], hist, width=2 * np.pi / n_bins,
                     color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax_polar.set_title("Direction rose", fontweight="bold", pad=20)
        ax_polar.set_theta_zero_location("E")
        ax_polar.set_theta_direction(-1)

        if directions:
            mean_angle = float(np.mean(directions))
            ax_polar.arrow(mean_angle, 0, 0, 0.8,
                           width=0.1, fc="red", ec="red", alpha=0.8)
            ax_polar.text(mean_angle, 0.9,
                          f"Mean: {np.rad2deg(mean_angle):.0f}°",
                          ha="center", fontsize=8, color="red")

        # Panel 4: linearity by speed category (box plot)
        speeds = [t.get("speed_um_per_sec", 0.0) for t in self.tracks
                  if t.get("speed_um_per_sec", 0.0) >= 0]
        buckets = {"Slow\n(<0.2)": [], "Medium\n(0.2–0.5)": [], "Fast\n(>0.5)": []}

        for s, l in zip(speeds, linearities):
            if s < 0.2:
                buckets["Slow\n(<0.2)"].append(l)
            elif s < 0.5:
                buckets["Medium\n(0.2–0.5)"].append(l)
            else:
                buckets["Fast\n(>0.5)"].append(l)

        valid_buckets = {k: v for k, v in buckets.items() if v}
        if valid_buckets:
            bp = axes[1, 0].boxplot(
                list(valid_buckets.values()),
                patch_artist=True,
                showmeans=True,
            )
            palette = ["lightcoral", "lightblue", "lightgreen"]
            for patch, color in zip(bp["boxes"], palette[:len(valid_buckets)]):
                patch.set_facecolor(color)
            axes[1, 0].set_xticklabels(valid_buckets.keys())
            axes[1, 0].set_ylabel("Linearity")
            axes[1, 0].set_title("Linearity by speed", fontweight="bold")
            axes[1, 0].grid(alpha=0.3)

        # Panel 5: linearity vs speed scatter
        axes[1, 1].scatter(speeds, linearities, alpha=0.6, c="darkgreen", s=40)
        axes[1, 1].set_xlabel("Speed (μm/s)")
        axes[1, 1].set_ylabel("Linearity")
        axes[1, 1].set_title("Linearity vs speed", fontweight="bold")
        axes[1, 1].grid(alpha=0.3)

        # Addштп linear regression if we have enough points
        non_zero = [(s, l) for s, l in zip(speeds, linearities) if s > 0]
        if len(non_zero) > 5:
            s_vals, l_vals = zip(*non_zero)
            z = np.polyfit(s_vals, l_vals, 1)
            xs = np.linspace(min(s_vals), max(s_vals), 100)
            axes[1, 1].plot(xs, np.poly1d(z)(xs), "r--", alpha=0.8,
                            label=f"Slope = {z[0]:.3f}")
            axes[1, 1].legend()

        # Panel 6: classification pie chart
        directed = sum(1 for l in linearities if l > 0.7)
        diffusive = sum(1 for l in linearities if 0.3 <= l <= 0.7)
        confined = sum(1 for l in linearities if l < 0.3)

        if directed + diffusive + confined > 0:
            axes[1, 2].pie(
                [directed, diffusive, confined],
                labels=[f"Directed\n({directed})",
                        f"Diffusive\n({diffusive})",
                        f"Confined\n({confined})"],
                colors=["lightgreen", "gold", "lightcoral"],
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[1, 2].set_title("Classification", fontweight="bold")

        plt.suptitle("Directionality analysis",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_file, self.dpi)
        return fig, axes


    # Interactive Plotly dashboard

    def interactive_dashboard(
        self,
        output_file: str = "tracking_dashboard.html",
    ) -> Optional[go.Figure]:
        """Build an HTML dashboard with interactive Plotly figures.

        Panels:
        - Trajectories
        - Velocity violin
        - Speed vs linearity
        - Track-length histogram
        """
        if not self.tracks:
            print("No tracks to visualize")
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Trajectories", "Velocity distribution",
                            "Speed vs linearity", "Track length distribution"),
        )

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                   "#bcbd22", "#17becf"]

        # Trajectories
        for i, track in enumerate(self.tracks):
            centers = self._centers(track)
            if len(centers) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=centers[:, 0], y=centers[:, 1],
                    mode="lines+markers",
                    name=f"Track {track['track_id']}",
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=4),
                    hovertemplate=(
                        f"Track {track['track_id']}<br>"
                        "X: %{x:.0f}<br>Y: %{y:.0f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # Velocity violin
        speeds = [t.get("speed_um_per_sec", 0.0) for t in self.tracks
                  if t.get("speed_um_per_sec", 0.0) > 0]
        if speeds:
            fig.add_trace(
                go.Violin(
                    y=speeds, box_visible=True, meanline_visible=True,
                    name="Velocity", line_color="steelblue",
                    fillcolor="lightblue", opacity=0.6, showlegend=False,
                ),
                row=1, col=2,
            )

        # Speed vs linearity
        scatter_x = [t.get("speed_um_per_sec", 0.0) for t in self.tracks]
        scatter_y = [t.get("linearity", 1.0) for t in self.tracks]
        fig.add_trace(
            go.Scatter(
                x=scatter_x, y=scatter_y,
                mode="markers",
                marker=dict(
                    size=10, color=scatter_y, colorscale="Viridis",
                    showscale=False,
                ),
                hovertemplate="Speed: %{x:.2f} μm/s<br>Linearity: %{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=2, col=1,
        )

        # Track length histogram
        lengths = [t.get("length_frames", 0) for t in self.tracks]
        if lengths:
            fig.add_trace(
                go.Histogram(
                    x=lengths, nbinsx=20,
                    marker_color="lightgreen", opacity=0.7,
                    showlegend=False,
                ),
                row=2, col=2,
            )

        fig.update_layout(
            title_text="Mitochondria tracking dashboard",
            title_font_size=20,
            height=900,
            template="plotly_white",
        )
        fig.update_xaxes(title_text="X (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Y (pixels)", row=1, col=1)
        fig.update_xaxes(title_text="Velocity (μm/s)", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=2)
        fig.update_xaxes(title_text="Speed (μm/s)", row=2, col=1)
        fig.update_yaxes(title_text="Linearity", row=2, col=1)
        fig.update_xaxes(title_text="Track length (frames)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        fig.write_html(output_file)
        print(f" Dashboard: {output_file}")
        return fig

    # Video rendering

    @staticmethod
    def _render_video(
        video_path: str,
        tracks: List,
        output_path: str,
    ) -> None:
        """Render an annotated MP4 with boxes and trails.

        ``tracks`` can be a list of :class:`~mito_tracking.tracker.Track`
        objects *or* dictionaries.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Build frame → list of (track_id, detection)
        per_frame: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)
        for track in tracks:
            # Support both Track objects and dicts
            tid = getattr(track, "track_id", None) or track.get("track_id", -1)
            frames = getattr(track, "frames", None) or track.get("frames", [])
            dets = getattr(track, "detections", None) or track.get("detections", [])
            for f, d in zip(frames, dets):
                per_frame[f].append((tid, d))

        # Deterministic colors
        colors: Dict[int, Tuple[int, int, int]] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for tid, det in per_frame.get(frame_idx, []):
                if tid not in colors:
                    np.random.seed(tid)
                    colors[tid] = tuple(int(c) for c in np.random.randint(0, 255, 3))
                x, y, w, h = det["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[tid], 2)
                cv2.putText(frame, f"ID:{tid}", (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[tid], 2)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        print(f"Video rendered: {output_path}")


    # Stats summary

    def _stats_text(self) -> str:
        """Return a short text summary used as overlay in plots."""
        if not self.tracks:
            return "No tracks"

        speeds = [t.get("speed_um_per_sec", 0.0) for t in self.tracks
                  if t.get("speed_um_per_sec", 0.0) > 0]
        linearities = [t.get("linearity", 1.0) for t in self.tracks]
        lengths = [t.get("length_frames", 0) for t in self.tracks]

        areas: List[float] = []
        for t in self.tracks:
            for det in t.get("detections", []):
                areas.append(det.get("area_um2", 0.0))

        lines = [f"Total tracks: {len(self.tracks)}"]
        if speeds:
            lines.append(f"Avg speed: {np.mean(speeds):.2f} ± {np.std(speeds):.2f} μm/s")
        lines.append(f"Avg linearity: {np.mean(linearities):.3f}")
        lines.append(f"Avg length: {np.mean(lengths):.1f} frames")
        if areas:
            lines.append(f"Avg area: {np.mean(areas):.2f} μm²")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TrackVisualizer(n_tracks={len(self.tracks)}, "
            f"ppm={self.ppm}, fps={self.fps}, dpi={self.dpi})"
        )


# Convenience function — generate everything from one JSON file

def visualize_tracking_results(
    results_file: str,
    video_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Optional[TrackVisualizer]:
    """Loading a results .json and generating all figures.

    Parameters:
    
    results_file : str
        Path to a ``*_tracking_results.json`` file.
    video_path : str, optional
        Path to the source video; used to extract the background frame for
        the trajectory plot.
    output_dir : str, optional
        A directory where to save figures. Defaults to ``<results_folder>/visualizations``.

    Returns
    TrackVisualizer or None
    """
    results_file = Path(results_file)
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        print("No tracks found in results")
        return None

    ppm = float(data.get("pixels_per_um", 10.0))
    fps = float(data.get("fps", 30.0))

    output_dir = Path(output_dir) if output_dir else results_file.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = results_file.stem.replace("_tracking_results", "")

    # Optional background image from the source video
    background: Optional[np.ndarray] = None
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            background = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(f"\n{'=' * 60}\nGENERATING VISUALIZATIONS\n{'=' * 60}")

    viz = TrackVisualizer(tracks, pixels_per_um=ppm, fps=fps)

    viz.plot_trajectories(
        background_img=background,
        output_file=str(output_dir / f"{video_name}_trajectories.png"),
    )
    viz.plot_velocity(
        output_file=str(output_dir / f"{video_name}_velocity.png"),
    )
    viz.plot_directionality(
        output_file=str(output_dir / f"{video_name}_directionality.png"),
    )
    viz.interactive_dashboard(
        output_file=str(output_dir / f"{video_name}_dashboard.html"),
    )

    print(f"\n All visualizations saved to: {output_dir}")
    return viz
