#!/usr/bin/env python3
"""消融实验数据可视化：8组数据 × 4种通道组合。

生成图1-4 + 量化汇总表，用于博士开题报告的偏振通道消融实验部分。
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────

COMBOS = {
    "S0":          {"label": "Only $S_0$",        "color": "#E41A1C", "order": 0},
    "S0-DoP":      {"label": "$S_0$+DoP",         "color": "#FF7F00", "order": 1},
    "S0-AoP":      {"label": "$S_0$+AoP",         "color": "#4DAF4A", "order": 2},
    "S0-AoP-DoP":  {"label": "$S_0$+DoP+AoP",     "color": "#377EB8", "order": 3},
}
COMBO_KEYS = sorted(COMBOS, key=lambda k: COMBOS[k]["order"])

DATASETS = [
    ("2026-04-16-13-27-22", "[1]",  94, 205,   99),
    ("2026-04-16-13-33-43", "[2]",  9.6, 54.3, 111),
    ("2026-04-16-13-39-45", "[3]",  6.7, 40.4, 117),
    ("2026-04-16-13-49-09", "[4]",  3.2, 30.6, 107),
    ("2026-04-16-13-54-30", "[5]",  2.6, 18.7, 108),
    ("2026-04-16-13-59-22", "[6]",  1.4,  9.3, 113),
    ("2026-04-16-14-04-13", "[7]",  1.1,  5.5, 120),
    ("2026-04-16-14-09-36", "[8]",  0.9,  4.7, 110),
]

TARGET_SPAN = 5.0
DIVERGENCE_SPEED_FACTOR = 5.0
DIVERGENCE_DISP_FACTOR = 3.0
DIVERGENCE_PREFIX_RATIO = 0.8
MIN_VALID_FRAMES = 10

ROOM_LENGTH = 5.0  # meters, used for TPE scaling

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_traj(filepath):
    """Load trajectory CSV. Returns (timestamps, positions_xyz) or (None, None)."""
    if not os.path.exists(filepath):
        return None, None
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    except ValueError:
        return None, None
    if data.ndim == 1 or len(data) == 0:
        return None, None
    timestamps = data[:, 0]
    positions = data[:, 1:4]  # tx, ty, tz
    return timestamps, positions


def load_stats(filepath):
    """Load stats CSV. Returns structured dict or empty dict."""
    if not os.path.exists(filepath):
        return {}
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    except ValueError:
        return {}
    if data.ndim == 1 or len(data) == 0:
        return {}
    return {
        "num_matched": data[:, 2],
        "num_total": data[:, 3],
        "avg_track": data[:, 4],
    }


def find_files(base_dir, combo_key, timestamp):
    """Find traj and stats CSVs for a given combo + timestamp.

    Traj files use exact timestamp match. Stats files may have 1s offset,
    so we match by closest HHMMSS (all files in the same combo dir).
    """
    combo_dir = base_dir / combo_key
    if not combo_dir.is_dir():
        return None, None

    # Traj: exact timestamp match
    traj_files = list(combo_dir.glob(f"dark*{timestamp}*.csv"))
    traj_path = traj_files[0] if traj_files else None

    # Stats: extract HHMMSS from each file, find closest to target
    parts = timestamp.split("-")
    target_hms = int(parts[3] + parts[4] + parts[5])

    stats_files = list(combo_dir.glob("polarfp_stats_*.csv"))
    best_path = None
    best_dist = 999
    for f in stats_files:
        m = re.search(r'_(\d{6})_', f.name)
        if m:
            file_hms = int(m.group(1))
            dist = abs(file_hms - target_hms)
            if dist < best_dist:
                best_dist = dist
                best_path = f

    stats_path = best_path if best_dist < 60 else None
    return traj_path, stats_path


# ─── Divergence Detection (from polarfp_vis_traj.py) ─────────────────────────

def detect_divergence(positions, timestamps):
    """Detect trajectory divergence point. Returns valid end index (exclusive)."""
    n = len(positions)
    if n < 5:
        return n

    disp = np.linalg.norm(positions - positions[0], axis=1)
    prefix_len = int(n * DIVERGENCE_PREFIX_RATIO)
    max_prefix_disp = np.max(disp[:prefix_len])

    diffs = np.diff(positions, axis=0)
    dt = np.diff(timestamps)
    dt = np.clip(dt, 1e-6, None)
    speeds = np.linalg.norm(diffs, axis=1) / dt

    speed_prefix_len = int((n - 1) * DIVERGENCE_PREFIX_RATIO)
    v_median = np.median(speeds[:speed_prefix_len])
    speed_threshold = v_median * DIVERGENCE_SPEED_FACTOR

    speed_cut = n
    for i in range(n - 2, -1, -1):
        if speeds[i] > speed_threshold:
            speed_cut = i + 1
        else:
            break

    disp_cut = n
    if max_prefix_disp > 1e-6:
        disp_threshold = max_prefix_disp * DIVERGENCE_DISP_FACTOR
        for i in range(prefix_len, n):
            if disp[i] > disp_threshold:
                disp_cut = i
                break

    return min(speed_cut, disp_cut)


def align_scale(positions, target_span=TARGET_SPAN):
    """Scale trajectory to target span using PCA longest axis."""
    span = np.max(np.ptp(positions, axis=0))
    if span < 1e-6:
        return positions, 1.0

    centered = positions - positions.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_dir = eigenvectors[:, np.argmax(eigenvalues)]

    projections = centered @ principal_dir
    longest_span = np.ptp(projections)

    if longest_span < 1e-6:
        return positions, 1.0

    scale = target_span / longest_span
    scaled = positions * scale
    scaled -= scaled[0]
    return scaled, scale


# ─── Process All Data ────────────────────────────────────────────────────────

def process_all():
    """Load, detect divergence, scale-align all combos × datasets.

    Returns:
        results[ds_idx][combo_key] = dict with traj + stats info
    """
    results = []
    for ds_idx, (timestamp, label, min_lux, max_lux, duration) in enumerate(DATASETS):
        row = {}
        for combo_key in COMBO_KEYS:
            traj_path, stats_path = find_files(BASE_DIR, combo_key, timestamp)

            ts, pos = load_traj(traj_path) if traj_path else (None, None)
            stats = load_stats(stats_path) if stats_path else {}

            if ts is None or pos is None or len(pos) < MIN_VALID_FRAMES:
                row[combo_key] = {
                    "valid": False,
                    "reason": "file_missing" if ts is None else "too_short",
                    "stats": stats,
                }
                continue

            original_len = len(pos)
            valid_end = detect_divergence(pos, ts)
            diverged = valid_end < original_len

            pos_valid = pos[:valid_end]
            ts_valid = ts[:valid_end]

            scaled, scale_factor = align_scale(pos_valid)

            # TPE: distance from scaled start to scaled end (both in meters after scaling)
            tpe = np.linalg.norm(scaled[-1, :2] - scaled[0, :2])

            # For diverged trajectories, also compute endpoint from original full data
            original_endpoint = pos[-1]
            original_start = pos[0]

            # Stats summary — only 4 core metrics
            if stats.get("num_matched") is not None and len(stats["num_matched"]) > 0:
                match_ratio = stats["num_matched"] / np.clip(stats["num_total"], 1, None)
                effective_obs = stats["num_matched"] * stats["avg_track"]
                stats_summary = {
                    "num_matched_mean": np.mean(stats["num_matched"]),
                    "num_matched_std": np.std(stats["num_matched"]),
                    "match_ratio_mean": np.mean(match_ratio),
                    "match_ratio_std": np.std(match_ratio),
                    "effective_obs_mean": np.mean(effective_obs),
                    "effective_obs_per_frame": effective_obs,
                }
            else:
                stats_summary = {}

            row[combo_key] = {
                "valid": True,
                "positions_xy": scaled[:, :2],
                "positions_z": scaled[:, 2],
                "scale_factor": scale_factor,
                "original_len": original_len,
                "truncated_len": valid_end,
                "diverged": diverged,
                "tpe": tpe,
                "min_lux": min_lux,
                "max_lux": max_lux,
                "stats": stats_summary,
            }
        results.append(row)
    return results


# ─── Figure 1: TPE Bar Chart ─────────────────────────────────────────────────

def plot_tpe_bar(results, output_dir):
    """Grouped bar chart: TPE for each dataset × combo."""
    fig, ax = plt.subplots(figsize=(14, 5))

    n_datasets = len(DATASETS)
    n_combos = len(COMBO_KEYS)
    bar_width = 0.2
    x = np.arange(n_datasets)

    for ci, combo_key in enumerate(COMBO_KEYS):
        cfg = COMBOS[combo_key]
        vals = []
        diverged_flags = []
        for row in results:
            r = row[combo_key]
            if r.get("valid"):
                vals.append(r["tpe"])
                diverged_flags.append(r["diverged"])
            else:
                vals.append(0)
                diverged_flags.append(False)

        offsets = x + (ci - n_combos / 2 + 0.5) * bar_width
        bars = ax.bar(offsets, vals, bar_width, label=cfg["label"], color=cfg["color"])

        # Mark diverged bars
        for i, div in enumerate(diverged_flags):
            if div:
                ax.text(offsets[i], vals[i] if vals[i] > 0 else 0.01, "DIV",
                        ha="center", va="bottom", fontsize=7, color=cfg["color"],
                        fontweight="bold", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in DATASETS], fontsize=10)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("TPE [m]", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add lux labels below dataset numbers
    lux_labels = []
    for d in DATASETS:
        if d[2] >= 50:
            lux_labels.append(f"{d[2]:.0f}--{d[3]:.0f} lux")
        else:
            lux_labels.append(f"{d[2]}--{d[3]} lux")

    ax.set_title("Terminal Position Error by Dataset and Channel Combination",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    path = output_dir / "fig1_tpe_bar.png"
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
    plt.close(fig)


# ─── Figure 2: Feature Stats Line Charts ─────────────────────────────────────

def _plot_feature_stats_group(results, output_dir, ds_indices, figure_label):
    """2-row × N-col per-figure: num matched + effective observations.
    Each column = one dataset.
    """
    subset = [i for i in range(8) if i in ds_indices]
    n = len(subset)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for local_idx, ds_idx in enumerate(subset):
        ds_label = DATASETS[ds_idx][1]
        for combo_key in COMBO_KEYS:
            cfg = COMBOS[combo_key]
            r = results[ds_idx][combo_key]
            stats = r.get("stats", {})
            if not stats:
                continue

            timestamp = DATASETS[ds_idx][0]
            traj_path, stats_path = find_files(BASE_DIR, combo_key, timestamp)
            if stats_path is None:
                continue
            raw_stats = load_stats(stats_path)
            if not raw_stats:
                continue

            num_matched = raw_stats["num_matched"]
            avg_track = raw_stats["avg_track"]
            effective_obs = num_matched * avg_track

            frames = np.arange(len(num_matched))

            axes[0, local_idx].plot(frames, num_matched, color=cfg["color"],
                                    linewidth=1.2)
            axes[1, local_idx].plot(frames, effective_obs, color=cfg["color"],
                                    linewidth=1.2)

        axes[0, local_idx].set_title(f"Dataset {ds_label}", fontsize=10, fontweight="bold")
        axes[0, local_idx].set_ylabel("Number of Matched Features", fontsize=9)
        axes[0, local_idx].set_xlabel("Frame Number", fontsize=9)
        axes[1, local_idx].set_ylabel("Effective Observations", fontsize=9)
        axes[1, local_idx].set_xlabel("Frame Number", fontsize=9)
        axes[0, local_idx].grid(True, alpha=0.2)
        axes[1, local_idx].grid(True, alpha=0.2)

    # Per-subplot legend below each column
    for local_idx in range(n):
        legend_elements = [
            plt.Line2D([0], [0], color=COMBOS[k]["color"], linewidth=2, label=COMBOS[k]["label"])
            for k in COMBO_KEYS
        ]
        axes[1, local_idx].legend(
            handles=legend_elements, loc="upper center", ncol=4,
            fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.6)
        )

    # Overall figure title
    if figure_label == "part1":
        fig.suptitle("Channel Ablation: Matched Features & Effective Observations (Datasets [1]–[4])",
                     fontsize=12, fontweight="bold", y=0.98)
    else:
        fig.suptitle("Channel Ablation: Matched Features & Effective Observations (Datasets [5]–[8])",
                     fontsize=12, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    path = output_dir / f"fig2_feature_stats_{figure_label}.png"
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_feature_stats(results, output_dir):
    """Split into two figures: datasets 1-4 and 5-8, each column is one dataset."""
    _plot_feature_stats_group(results, output_dir, set(range(0, 4)), "part1")
    _plot_feature_stats_group(results, output_dir, set(range(4, 8)), "part2")


# ─── Figure 3: Trajectory Comparison ─────────────────────────────────────────

def _plot_trajectories_group(results, output_dir, ds_indices, figure_label):
    """2-row × 2-col: XY trajectory comparison for a subset of datasets."""
    subset = [i for i in ds_indices if i < len(DATASETS)]
    if len(subset) != 4:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for local_idx, ds_idx in enumerate(subset):
        timestamp, label, min_lux, max_lux, duration = DATASETS[ds_idx]
        row = local_idx // 2
        col = local_idx % 2
        ax = axes[row, col]

        for combo_key in COMBO_KEYS:
            cfg = COMBOS[combo_key]
            r = results[ds_idx][combo_key]
            if not r.get("valid"):
                ax.text(0.5, 0.5, f"{cfg['label']}: failed",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color=cfg["color"], alpha=0.6)
                continue

            xy = r["positions_xy"]
            sf = r["scale_factor"]

            ax.plot(xy[:, 0], xy[:, 1],
                    color=cfg["color"], linewidth=2,
                    label=f"{cfg['label']} (×{sf:.1f})")
            ax.plot(xy[0, 0], xy[0, 1], "o", color="blue", markersize=6, zorder=10)
            ax.plot(xy[-1, 0], xy[-1, 1], "*", color=cfg["color"], markersize=9, zorder=10)

        lux_label = f"{min_lux}--{max_lux} lux" if min_lux < 50 else f"{min_lux:.0f}--{max_lux:.0f} lux"
        ax.set_title(f"Trail {label}\n({lux_label})", fontsize=10, fontweight="bold")
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel("Y [m]", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        # Trails 1/2: legend in upper right to avoid blocking; others: auto
        if ds_idx in (0, 1):
            ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
        else:
            ax.legend(fontsize=6, loc="best", framealpha=0.7)

    fig.text(0.5, 0.005, r"$\bullet$ start, $\star$ end",
             ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = output_dir / f"fig3_trajectory_{figure_label}.png"
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_trajectories(results, output_dir):
    """Split into two figures: datasets 1-4 and 5-8."""
    _plot_trajectories_group(results, output_dir, set(range(0, 4)), "part1")
    _plot_trajectories_group(results, output_dir, set(range(4, 8)), "part2")


# ─── Figure 4: Ablation Summary Trends ───────────────────────────────────────

def plot_ablation_summary(results, output_dir):
    """1-row × 3-col: Matched features, match ratio, effective observations."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    max_luxes = np.array([d[3] for d in DATASETS])

    for ci, combo_key in enumerate(COMBO_KEYS):
        cfg = COMBOS[combo_key]
        match_ratios = []
        matched_features = []
        eff_obs_vals = []
        valid_mask = []

        for ds_idx, row in enumerate(results):
            r = row[combo_key]
            if r.get("valid"):
                stats = r.get("stats", {})
                match_ratios.append(stats.get("match_ratio_mean", 0) * 100)
                matched_features.append(stats.get("num_matched_mean", 0))
                eff_obs_vals.append(stats.get("effective_obs_mean", 0))
                valid_mask.append(True)
            else:
                match_ratios.append(0)
                matched_features.append(0)
                eff_obs_vals.append(0)
                valid_mask.append(False)

        match_ratios = np.array(match_ratios)
        matched_features = np.array(matched_features)
        eff_obs_vals = np.array(eff_obs_vals)
        valid_mask = np.array(valid_mask)

        axes[0].scatter(max_luxes[valid_mask], matched_features[valid_mask], color=cfg["color"],
                        s=80, label=cfg["label"], zorder=3, edgecolors="white", linewidths=0.5)
        axes[0].plot(max_luxes[valid_mask], matched_features[valid_mask], color=cfg["color"],
                     linewidth=1.5, alpha=0.5, zorder=2)
        axes[1].scatter(max_luxes[valid_mask], match_ratios[valid_mask], color=cfg["color"],
                        s=80, zorder=3, edgecolors="white", linewidths=0.5)
        axes[1].plot(max_luxes[valid_mask], match_ratios[valid_mask], color=cfg["color"],
                     linewidth=1.5, alpha=0.5, zorder=2)
        axes[2].scatter(max_luxes[valid_mask], eff_obs_vals[valid_mask], color=cfg["color"],
                        s=80, zorder=3, edgecolors="white", linewidths=0.5)
        axes[2].plot(max_luxes[valid_mask], eff_obs_vals[valid_mask], color=cfg["color"],
                     linewidth=1.5, alpha=0.5, zorder=2)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Num Matched", fontsize=10)
    axes[0].set_title("Matched Features vs Max Illuminance", fontsize=10, fontweight="bold")

    axes[1].set_ylabel("Match Ratio [%]", fontsize=10)
    axes[1].set_title("Match Ratio vs Max Illuminance", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 95)

    axes[2].set_ylabel("Effective Observations", fontsize=10)
    axes[2].set_title("Effective Observations vs Max Illuminance", fontsize=10, fontweight="bold")

    legend_elements = [
        plt.Line2D([0], [0], color=COMBOS[k]["color"], marker="o", linestyle="",
                   markersize=8, label=COMBOS[k]["label"])
        for k in COMBO_KEYS
    ]
    for ax in axes:
        ax.set_xlabel("Max Illuminance [lux]", fontsize=9)
        ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    fig.tight_layout()
    path = output_dir / "fig4_ablation_summary.png"
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
    plt.close(fig)


# ─── Table 1: LaTeX Table ────────────────────────────────────────────────────

def generate_table(results, output_dir):
    """Generate LaTeX table with all metrics."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{消融实验：不同通道组合在不同光照条件下的量化指标}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"光照条件 & 通道组合 & 匹配数 & 匹配率/\% & 有效观测 & TPE/m & 发散 \\")
    lines.append(r"\midrule")

    for ds_idx, (timestamp, label, min_lux, max_lux, duration) in enumerate(DATASETS):
        lux_str = f"{min_lux}--{max_lux}~lux"

        for combo_key in COMBO_KEYS:
            cfg = COMBOS[combo_key]
            r = results[ds_idx][combo_key]

            if not r.get("valid"):
                lines.append(
                    f"{lux_str} & {cfg['label']} & -- & -- & -- & -- & -- \\"
                )
                continue

            stats = r.get("stats", {})
            matched = stats.get("num_matched_mean", 0)
            matched_std = stats.get("num_matched_std", 0)
            match_ratio = stats.get("match_ratio_mean", 0) * 100
            match_ratio_std = stats.get("match_ratio_std", 0)
            eff_obs = stats.get("effective_obs_mean", 0)
            tpe = r["tpe"]
            diverged = "✓" if r["diverged"] else "--"

            lines.append(
                f"{lux_str} & {cfg['label']} & "
                f"{matched:.0f}$\\pm${matched_std:.0f} & "
                f"{match_ratio:.1f}$\\pm${match_ratio_std:.1f} & "
                f"{eff_obs:.0f} & "
                f"{tpe:.3f} & {diverged} \\"
            )

        # Add separator between light groups
        if ds_idx < len(DATASETS) - 1:
            next_min = DATASETS[ds_idx + 1][2]
            if next_min < 50 and min_lux >= 50:
                lines.append(r"\cmidrule{1-7}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = output_dir / "table_ablation.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


# ─── Print Summary ───────────────────────────────────────────────────────────

def print_summary(results):
    """Print console summary of all results."""
    print(f"\n{'='*80}")
    print(f"{'Dataset':<8} {'Combo':<20} {'Frames':>8} {'TPE[m]':>8} {'Match#':>7} {'MRatio':>7} {'EffObs':>8} {'Status'}")
    print(f"{'='*80}")

    for ds_idx, (timestamp, label, min_lux, max_lux, duration) in enumerate(DATASETS):
        for combo_key in COMBO_KEYS:
            cfg = COMBOS[combo_key]
            r = results[ds_idx][combo_key]
            if not r.get("valid"):
                print(f"{label:<8} {cfg['label']:<20} {'--':>8} {'--':>8} {'--':>7} {'--':>7} {'--':>8} {'FAILED'}")
                continue
            stats = r.get("stats", {})
            frames = f"{r['truncated_len']}/{r['original_len']}"
            tpe = f"{r['tpe']:.3f}"
            matched = f"{stats.get('num_matched_mean', 0):.0f}"
            mr = f"{stats.get('match_ratio_mean', 0)*100:.1f}"
            eo = f"{stats.get('effective_obs_mean', 0):.0f}"
            status = "DIV" if r["diverged"] else "OK"
            print(f"{label:<8} {cfg['label']:<20} {frames:>8} {tpe:>8} {matched:>7} {mr:>7} {eo:>8} {status}")
    print(f"{'='*80}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="消融实验数据可视化")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory for figures and tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process all data
    print("Loading and processing all data...")
    results = process_all()

    # Console summary
    print_summary(results)

    # Generate figures
    print("Generating Figure 1: TPE Bar Chart...")
    plot_tpe_bar(results, output_dir)

    print("Generating Figure 2: Feature Stats Lines...")
    plot_feature_stats(results, output_dir)

    print("Generating Figure 3: Trajectory Comparison...")
    plot_trajectories(results, output_dir)

    print("Generating Figure 4: Ablation Summary...")
    plot_ablation_summary(results, output_dir)

    # Generate table
    print("Generating LaTeX Table...")
    generate_table(results, output_dir)

    print("\nDone! All outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
