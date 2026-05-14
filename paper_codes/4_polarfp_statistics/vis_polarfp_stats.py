#!/usr/bin/env python3
"""
PolarFP-VINS: Visualize global feature pool statistics across illumination conditions.

Produces:
  - fig_bar_stats.png            Bar chart of per-trajectory mean metrics
  - fig_line_{dataset}.png       Line charts by dataset (3 metrics × 3 datasets)
  - stats_table.tex              LaTeX booktabs table of mean values

Usage:
  cd paper_codes/4_polarfp_statistics
  python vis_polarfp_stats.py
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "fig"

# Time-stamp prefix → (dataset_key, illumination_label)
DATA_SETS = {
    "13272":  ("bright", "94–205 lux"),
    "135430": ("medium", "2.6–18.6 lux"),
    "140936": ("dark",   "0.9–4.8 lux"),
}

# CSV channel suffix → (combo_key, display_label)
CHANNEL_COMBOS = {
    "s0_dop_aopsin_aopcos": ("all",     "S0+DoP+AoP"),
    "s0_dop":               ("s0+dop",  "S0+DoP"),
    "s0_aopsin_aopcos":     ("s0+aop",  "S0+AoP"),
    "s0":                   ("s0",      "S0 only"),
}

# Sorting order (x-axis / legend)
COMBO_ORDER = ["all", "s0+dop", "s0+aop", "s0"]
DATASET_ORDER = ["bright", "medium", "dark"]

# Colors for bar chart groups (by combo)
BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]   # all, s0+dop, s0+aop, s0

# Metric definitions:  (display_name, y_label, fmt_code)
METRICS = {
    "num_matched":    ("Num Matched",        "Count",               ".0f"),
    "match_ratio":    ("Match Ratio",        "Ratio",               ".3f"),
    "avg_track":      ("Avg Track Length",   "Frames",              ".2f"),
    "effective_obs":  ("Effective Obs",      "Count",               ".0f"),
}

DATASET_LABELS = {
    "bright": "94–205 lux",
    "medium": "2.6–18.6 lux",
    "dark":   "0.9–4.8 lux",
}
COMBO_DISPLAY = {"all": "S0+DoP+AoP", "s0+dop": "S0+DoP", "s0+aop": "S0+AoP", "s0": "S0 only"}


# ────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────
def parse_filename(filename: str):
    """Return (dataset_key, combo_key) or (None, None) if unparseable."""
    # Pattern: polarpfp_stats_YYYYMMDD_HHMMSS_ch_names.csv
    m = re.match(r"polarfp_stats_\d{8}_(\d+)_(.+)\.csv$", filename)
    if not m:
        return None, None
    ts_prefix, chan_part = m.group(1), m.group(2)

    # Match dataset by longest prefix
    ds_key = None
    for prefix, (key, _) in DATA_SETS.items():
        if ts_prefix.startswith(prefix):
            ds_key = key
            break

    combo_key = None
    for suffix, (key, _) in CHANNEL_COMBOS.items():
        if chan_part == suffix:
            combo_key = key
            break

    return ds_key, combo_key


def load_data(data_dir: str):
    """
    Return nested dict:
      data[(dataset_key, combo_key)] = pd.DataFrame
    """
    all_data = {}
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath) or not fname.endswith(".csv"):
            continue
        ds, combo = parse_filename(fname)
        if ds is None or combo is None:
            warnings.warn(f"Skipping unrecognised file: {fname}")
            continue
        df = pd.read_csv(fpath)
        all_data[(ds, combo)] = df
    return all_data


# ────────────────────────────────────────────────────────────
# Metric computation
# ────────────────────────────────────────────────────────────
def add_derived_columns(df: pd.DataFrame):
    """Add match_ratio and effective_obs columns in-place."""
    df["match_ratio"] = df["num_global_matched"] / df["num_global_total"].replace(0, np.nan)
    df["effective_obs"] = df["num_global_matched"] * df["avg_global_track"]
    return df


def compute_means(all_data: dict):
    """
    Return dict:  (dataset, combo) -> {metric: mean_value}
    """
    means = {}
    for key, df in all_data.items():
        df = add_derived_columns(df)
        m = {
            "num_matched":   df["num_global_matched"].mean(),
            "match_ratio":   df["match_ratio"].mean(),
            "avg_track":     df["avg_global_track"].mean(),
            "effective_obs": df["effective_obs"].mean(),
        }
        means[key] = m
    return means


# ────────────────────────────────────────────────────────────
# Bar chart (all 4 metrics, 2×2 subplots)
# ────────────────────────────────────────────────────────────
def plot_bar_charts(all_means: dict, output_path: str):
    """4 subplots (2×2), one per metric; x-axis = 3 datasets, grouped bars per combo."""
    n_datasets = len(DATASET_ORDER)
    n_combos   = len(COMBO_ORDER)
    x = np.arange(n_datasets)                     # dataset positions
    width = 0.8 / n_combos                        # bar width per combo

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes_flat = axes.flatten()

    for ax_idx, (metric, (disp_name, y_label, fmt)) in enumerate(METRICS.items()):
        ax = axes_flat[ax_idx]

        for ci, combo in enumerate(COMBO_ORDER):
            vals = []
            for di, ds in enumerate(DATASET_ORDER):
                key = (ds, combo)
                v = all_means.get(key, {}).get(metric, 0)
                vals.append(v)
            offset = (ci - (n_combos - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=COMBO_DISPLAY[combo],
                          color=BAR_COLORS[ci],
                          edgecolor="white", linewidth=0.5)

        ax.set_title(disp_name, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        ax.grid(axis="y", alpha=0.3)
        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend below all subplots
    handles = [Patch(facecolor=BAR_COLORS[i],
                     label=COMBO_DISPLAY[COMBO_ORDER[i]])
               for i in range(n_combos)]
    fig.legend(handles=handles, loc="lower center",
               ncol=n_combos, fontsize=12, frameon=False)
    plt.subplots_adjust(bottom=0.12, wspace=0.30, hspace=0.35)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ────────────────────────────────────────────────────────────
# Line charts (one PNG per dataset, 1×3 subplots)
# ────────────────────────────────────────────────────────────
def plot_line_charts_by_dataset(all_data: dict, output_dir: str):
    """
    One PNG per dataset: 1×3 subplots (num_matched, match_ratio, effective_obs).
    Different combos → different colours, all solid lines.
    """
    metrics_to_plot = ["num_matched", "match_ratio", "effective_obs"]
    metric_labels = {
        "num_matched": "Num Matched",
        "match_ratio": "Match Ratio",
        "effective_obs": "Effective Obs",
    }
    metric_ylabels = {
        "num_matched": "Count",
        "match_ratio": "Ratio",
        "effective_obs": "Count",
    }

    for ds in DATASET_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            for ci, combo in enumerate(COMBO_ORDER):
                key = (ds, combo)
                if key not in all_data:
                    continue
                df = all_data[key]
                if "match_ratio" not in df.columns or "effective_obs" not in df.columns:
                    df = add_derived_columns(df)

                # Skip first 5 frames (initialization transient)
                skip = 5
                x = df["frame_idx"].values[skip:]
                if metric == "num_matched":
                    y = df["num_global_matched"].values[skip:]
                elif metric == "match_ratio":
                    y = df["match_ratio"].values[skip:]
                elif metric == "effective_obs":
                    y = df["effective_obs"].values[skip:]

                ax.plot(x, y,
                        label=COMBO_DISPLAY[combo],
                        color=BAR_COLORS[ci],
                        linewidth=1.8,
                        alpha=0.85)

            ax.set_xlabel("Frame", fontsize=11)
            ax.set_ylabel(metric_ylabels[metric], fontsize=11)
            ax.set_title(metric_labels[metric], fontsize=13, fontweight="bold")
            ax.grid(alpha=0.25)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Single legend below figure
        handles = [Patch(facecolor=BAR_COLORS[i],
                         label=COMBO_DISPLAY[COMBO_ORDER[i]])
                   for i in range(len(COMBO_ORDER))]
        fig.legend(handles=handles, loc="lower center",
                   ncol=len(COMBO_ORDER), fontsize=12, frameon=False)

        plt.subplots_adjust(bottom=0.15, wspace=0.30)
        output_path = os.path.join(output_dir, f"fig_line_{ds}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {output_path}")


# ────────────────────────────────────────────────────────────
# LaTeX table
# ────────────────────────────────────────────────────────────
def export_latex_table(all_means: dict, output_path: str):
    """Write a booktabs table of per-trajectory mean metrics."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean per-frame metrics across different illumination "
                 r"conditions and channel combinations.}")
    lines.append(r"\label{tab:polarfp_stats}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Illumination & Channels & Num Matched & Match Ratio & "
                 r"Avg Track (fr) & Effective Obs \\")
    lines.append(r"\midrule")

    row_template = r"{} & {} & {} & {} & {} & {} \\"

    for ds in DATASET_ORDER:
        for combo in COMBO_ORDER:
            key = (ds, combo)
            m = all_means.get(key, {})
            if not m:
                continue
            num_m   = f"{m['num_matched']:.0f}"
            ratio   = f"{m['match_ratio']:.3f}"
            avg_tr  = f"{m['avg_track']:.2f}"
            eff_obs = f"{m['effective_obs']:.0f}"
            lines.append(row_template.format(
                DATASET_LABELS[ds], COMBO_DISPLAY[combo],
                num_m, ratio, avg_tr, eff_obs))
        lines.append(r"\cmidrule{1-6}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {output_path}")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    all_data = load_data(DATA_DIR)

    if not all_data:
        print("ERROR: no CSV files found — check DATA_DIR path.")
        return

    n_files = len(all_data)
    print(f"  Found {n_files} data files.")

    # Verify all expected (dataset, combo) pairs exist
    expected = [(ds, combo) for ds in DATASET_ORDER for combo in COMBO_ORDER]
    missing = [k for k in expected if k not in all_data]
    if missing:
        warnings.warn(f"Missing expected combos: {missing}")

    print("Computing per-trajectory means ...")
    all_means = compute_means(all_data)

    print("Plotting bar chart (4 metrics × 3 datasets × 4 combos) ...")
    plot_bar_charts(all_means, os.path.join(OUTPUT_DIR, "fig_bar_stats.png"))

    print("Plotting line charts by dataset (3 metrics × 3 datasets) ...")
    plot_line_charts_by_dataset(all_data, OUTPUT_DIR)

    print("Exporting LaTeX table ...")
    export_latex_table(all_means, os.path.join(OUTPUT_DIR, "stats_table.tex"))

    print("Done.")


if __name__ == "__main__":
    main()
