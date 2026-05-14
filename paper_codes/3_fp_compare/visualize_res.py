#!/usr/bin/env python3
"""
visualize_res.py — PolarFP-VINS Feature Detector Comparison Visualization
Generates visualization from fp_compare_table3 CSV outputs.

Plan A: Grouped bar chart grid (3 lighting × 3 metrics)
Supplement: Comprehensive bar chart (1 figure, 3 subplots for 3 lighting)
Markdown summary table

Usage:
    python visualize_res.py
"""

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR / "output"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lighting metadata
LIGHTING_INFO = {
    '13-27-22': {'label': 'Bright (94–205 lux)',    'short': 'Bright', 'order': 0},
    '13-54-30': {'label': 'Medium (2.6–18.6 lux)',  'short': 'Medium', 'order': 1},
    '14-09-36': {'label': 'Dark (0.9–4.8 lux)',     'short': 'Dark',   'order': 2},
}

# Channel color mapping
CH_COLOR = {'DoP': '#3274A1', 'sinAoP': '#E1812C', 'cosAoP': '#3A923A'}
CH_ORDER = ['DoP', 'sinAoP', 'cosAoP']
CH_LABEL = {'DoP': 'DoP (P0)', 'sinAoP': 'sinAoP (P1)', 'cosAoP': 'cosAoP (P2)'}

# Detector ordering & color
DET_ORDER = ['FAST', 'GFTT', 'SuperPoint']
DET_COLOR = {
    'FAST':       '#E69F00',
    'GFTT':       '#56B4E9',
    'SuperPoint': '#D55E00',
}
DET_HATCH = {'FAST': '', 'GFTT': '..', 'SuperPoint': '//'}

# Global matplotlib settings
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
sns.set_style('ticks')
sns.set_palette('colorblind')


# ============================================================
# Data loading
# ============================================================
def load_data():
    """Load 3 CSV files, merge, and compute derived metrics."""
    dfs = []
    for fname in ['table3_13-27-22.csv', 'table3_13-54-30.csv', 'table3_14-09-36.csv']:
        path = CSV_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        dfs.append(df)

    if not dfs:
        print("[ERROR] No CSV files found.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)

    df['lighting_label'] = df['dataset'].map(lambda d: LIGHTING_INFO[d]['label'])
    df['lighting_short'] = df['dataset'].map(lambda d: LIGHTING_INFO[d]['short'])
    df['lighting_order'] = df['dataset'].map(lambda d: LIGHTING_INFO[d]['order'])

    df['inlier_ratio'] = df['avg_inlier'] / df['avg_match']

    df['channel']  = pd.Categorical(df['channel'], CH_ORDER, ordered=True)
    df['detector'] = pd.Categorical(df['detector'], DET_ORDER, ordered=True)

    return df


# ============================================================
# Helper — grouped bar positions
# ============================================================
def grouped_x(n_groups, n_bars, bar_width=0.22, gap=0.06):
    total = n_bars * bar_width
    offsets = np.arange(n_bars) * bar_width - total / 2 + bar_width / 2
    return np.arange(n_groups), offsets


# ============================================================
# Plan A — Grouped Bar Chart Grid
# ============================================================
def plan_a(df):
    """
    3 rows (lighting) × 3 cols (avg_inlier | inlier_ratio | avg_time_ms).
    Within each subplot: x = 3 detectors, grouped bars = 3 channels.
    """
    print("[Plan A] Generating grouped bar chart grid...")

    datasets_ordered = sorted(df['dataset'].unique(),
                              key=lambda d: LIGHTING_INFO[d]['order'])
    metrics = [
        ('avg_inlier',   'Avg Inlier Count'),
        ('inlier_ratio', 'Inlier Ratio (inlier/match)'),
        ('avg_time_ms',  'Avg Time (ms)'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    bar_w = 0.22
    n_d = len(DET_ORDER)
    n_c = len(CH_ORDER)

    for row, dataset in enumerate(datasets_ordered):
        lx = LIGHTING_INFO[dataset]
        sub = df[df['dataset'] == dataset]

        for col, (metric, mlabel) in enumerate(metrics):
            ax = axes[row][col]
            gx, offsets = grouped_x(n_d, n_c, bar_w)

            for ci, ch in enumerate(CH_ORDER):
                vals = []
                for det in DET_ORDER:
                    v = sub[(sub['channel'] == ch) & (sub['detector'] == det)][metric].values
                    vals.append(v[0] if len(v) > 0 else 0)
                ax.bar(gx + offsets[ci], vals, bar_w,
                       color=CH_COLOR[ch], label=CH_LABEL[ch],
                       edgecolor='white', linewidth=0.3, zorder=2)

            ax.set_xticks(gx)
            ax.set_xticklabels(DET_ORDER, rotation=0, ha='center', fontsize=9)
            ax.set_title(f'{mlabel}\n{lx["label"]}', fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.35, zorder=0)

            if metric == 'avg_time_ms':
                ax.set_ylabel('ms', fontsize=9)
            elif metric == 'inlier_ratio':
                ax.set_ylabel('ratio', fontsize=9)
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
            else:
                ax.set_ylabel('count', fontsize=9)

    # Single legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle('Feature Detector Comparison: Inlier Quality & Time by Lighting Condition',
                 fontsize=16, fontweight='bold', y=1.04)
    fig.savefig(OUT_DIR / 'planA_fp_metrics_grid.png')
    plt.close(fig)
    print("  -> Saved planA_fp_metrics_grid.png")


# ============================================================
# Comprehensive bar chart (paper-friendly)
# ============================================================
def plan_comprehensive(df):
    """
    3 subplots (1 per lighting), each: 3 detectors × 3 channels = 9 bars.
    Shows avg_inlier as the primary metric.
    """
    print("[Comprehensive] Generating comprehensive bar chart...")

    datasets_ordered = sorted(df['dataset'].unique(),
                              key=lambda d: LIGHTING_INFO[d]['order'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.subplots_adjust(wspace=0.25)

    bar_w = 0.22
    n_d = len(DET_ORDER)
    n_c = len(CH_ORDER)

    for col, dataset in enumerate(datasets_ordered):
        ax = axes[col]
        sub = df[df['dataset'] == dataset]
        lx = LIGHTING_INFO[dataset]

        gx, offsets = grouped_x(n_d, n_c, bar_w)
        for ci, ch in enumerate(CH_ORDER):
            vals = []
            for det in DET_ORDER:
                v = sub[(sub['channel'] == ch) & (sub['detector'] == det)]['avg_inlier'].values
                vals.append(v[0] if len(v) > 0 else 0)
            ax.bar(gx + offsets[ci], vals, bar_w,
                   color=CH_COLOR[ch], label=CH_LABEL[ch],
                   edgecolor='white', linewidth=0.3, zorder=2)

        ax.set_xticks(gx)
        ax.set_xticklabels(DET_ORDER, rotation=0, ha='center', fontsize=10)
        ax.set_title(lx['label'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Avg Inlier Count', fontsize=9)
        ax.grid(axis='y', alpha=0.35, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 1.08))

    fig.suptitle('Feature Detector Comparison: Avg Inlier by Lighting and Channel',
                 fontsize=14, fontweight='bold', y=1.15)
    fig.savefig(OUT_DIR / 'fp_summary_bars.png')
    plt.close(fig)
    print("  -> Saved fp_summary_bars.png")


# ============================================================
# Markdown summary table
# ============================================================
def markdown_table(df):
    """Output a formatted markdown summary table."""
    md_path = OUT_DIR / 'fp_summary_table.md'

    datasets_ordered = sorted(df['dataset'].unique(),
                              key=lambda d: LIGHTING_INFO[d]['order'])

    lines = []
    lines.append("# Feature Detector Comparison Summary\n")
    lines.append("Each cell: **avg_inlier** / inlier_ratio / time(ms)\n")

    for dataset in datasets_ordered:
        lx = LIGHTING_INFO[dataset]
        sub = df[df['dataset'] == dataset]
        lines.append(f"## {lx['label']}\n")
        lines.append("| Channel | " + " | ".join(DET_ORDER) + " |")
        lines.append("|" + "---|" * (len(DET_ORDER) + 1))

        for ch in CH_ORDER:
            cells = []
            for det in DET_ORDER:
                row = sub[(sub['channel'] == ch) & (sub['detector'] == det)]
                if len(row) > 0:
                    r = row.iloc[0]
                    cells.append(
                        f"**{r['avg_inlier']:.1f}** / {r['inlier_ratio']:.2%} / {r['avg_time_ms']:.1f}ms")
                else:
                    cells.append("—")
            lines.append(f"| {CH_LABEL[ch]} | " + " | ".join(cells) + " |")
        lines.append("")

    # Best per lighting summary
    lines.append("## Best per Lighting\n")
    lines.append("| Lighting | Best Inlier | Best Ratio | Fastest |")
    lines.append("|---|---|---|---|")
    for dataset in datasets_ordered:
        sub = df[df['dataset'] == dataset]
        lx = LIGHTING_INFO[dataset]
        best_inlier = sub.loc[sub['avg_inlier'].idxmax()]
        best_ratio  = sub.loc[sub['inlier_ratio'].idxmax()]
        fastest     = sub.loc[sub['avg_time_ms'].idxmin()]

        lines.append(
            f"| {lx['short']} | "
            f"{best_inlier['channel']}+{best_inlier['detector']} "
            f"({best_inlier['avg_inlier']:.1f}) | "
            f"{best_ratio['channel']}+{best_ratio['detector']} "
            f"({best_ratio['inlier_ratio']:.2%}) | "
            f"{fastest['channel']}+{fastest['detector']} "
            f"({fastest['avg_time_ms']:.1f}ms) |"
        )

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  -> Saved fp_summary_table.md")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("PolarFP-VINS — Feature Detector Comparison Visualization")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df)} rows from {df['dataset'].nunique()} datasets.")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Channels: {list(df['channel'].unique())}")
    print(f"Detectors: {list(df['detector'].unique())}")
    print()

    plan_a(df)
    plan_comprehensive(df)
    markdown_table(df)

    print(f"\nDone! All figures saved to: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
