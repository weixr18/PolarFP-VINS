#!/usr/bin/env python3
"""
visualize_res.py — PolarFP-VINS Filter Comparison Visualization
Generates three visualization plans from filter_compare_table2 CSV outputs.

Plan A: Grouped bar chart grid (3 lighting × 3 metrics)
Plan B: Quality-time scatter + heatmaps
Plan C: Comprehensive dashboard + markdown table

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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR / "output"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lighting metadata — ordered from bright to dark
LIGHTING_INFO = {
    '13-27-22':   {'label': 'Bright lighting (94–205 lux)',    'short': 'Bright',   'order': 0},
    '13-54-30':   {'label': 'Medium lighting (2.6–18.6 lux)',  'short': 'Medium',   'order': 1},
    '14-09-36':   {'label': 'Dark lighting (0.9–4.8 lux)',     'short': 'Dark',     'order': 2},
}

# Channel → color mapping
CH_COLOR = {'DoP': '#3274A1', 'sinAoP': '#E1812C', 'cosAoP': '#3A923A'}
CH_ORDER = ['DoP', 'sinAoP', 'cosAoP']
CH_LABEL = {'DoP': 'DoP (P0)', 'sinAoP': 'sinAoP (P1)', 'cosAoP': 'cosAoP (P2)'}

# Filter → order & color
FLT_ORDER = ['Raw', 'Median', 'Guided', 'Bilateral', 'NLM']
FLT_COLOR = {
    'Raw':       '#555555',
    'Median':    '#E69F00',
    'Guided':    '#56B4E9',
    'Bilateral': '#009E73',
    'NLM':       '#D55E00',
}
FLT_HATCH = {
    'Raw': '', 'Median': '..', 'Guided': '//',
    'Bilateral': 'xx', 'NLM': '\\\\',
}

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
    for fname in ['table2_13-27-22.csv', 'table2_13-54-30.csv', 'table2_14-09-36.csv']:
        path = CSV_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Lighting metadata
    df['lighting_label'] = df['dataset'].map(lambda d: LIGHTING_INFO[d]['label'])
    df['lighting_short'] = df['dataset'].map(lambda d: LIGHTING_INFO[d]['short'])
    df['lighting_order'] = df['dataset'].map(lambda d: LIGHTING_INFO[d]['order'])

    # Derived metrics
    df['inlier_ratio'] = df['avg_inlier'] / df['avg_match']       # matching precision
    df['yield_ratio']  = df['avg_inlier'] / df['avg_detect']      # end-to-end yield

    # Categorical ordering
    df['channel'] = pd.Categorical(df['channel'], CH_ORDER, ordered=True)
    df['filter']  = pd.Categorical(df['filter'], FLT_ORDER, ordered=True)

    return df


# ============================================================
# Helper — grouped bar positions
# ============================================================
def grouped_x(n_groups, n_bars, bar_width=0.22, gap=0.06):
    """Return x-centers for grouped bars."""
    total = n_bars * bar_width
    offsets = np.arange(n_bars) * bar_width - total / 2 + bar_width / 2
    return np.arange(n_groups), offsets


# ============================================================
# Plan A — Grouped Bar Chart Grid
# ============================================================
def plan_a(df):
    """
    3 rows (lighting) × 3 cols (avg_inlier | inlier_ratio | time).
    Within each subplot: x = 5 filters, grouped bars = 3 channels.
    """
    print("[Plan A] Generating grouped bar chart grid...")

    datasets_ordered = sorted(df['dataset'].unique(), key=lambda d: LIGHTING_INFO[d]['order'])
    metrics = [
        ('avg_inlier',   'Avg Inlier Count'),
        ('inlier_ratio', 'Inlier Ratio (inlier/match)'),
        ('avg_time_ms',  'Avg Time (ms) — log scale'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    bar_w = 0.22
    n_f = len(FLT_ORDER)
    n_c = len(CH_ORDER)

    for row, dataset in enumerate(datasets_ordered):
        lx = LIGHTING_INFO[dataset]
        sub = df[df['dataset'] == dataset]

        for col, (metric, mlabel) in enumerate(metrics):
            ax = axes[row][col]
            gx, offsets = grouped_x(n_f, n_c, bar_w)

            for ci, ch in enumerate(CH_ORDER):
                vals = []
                for fl in FLT_ORDER:
                    v = sub[(sub['channel'] == ch) & (sub['filter'] == fl)][metric].values
                    vals.append(v[0] if len(v) > 0 else 0)
                bars = ax.bar(gx + offsets[ci], vals, bar_w,
                              color=CH_COLOR[ch], label=CH_LABEL[ch],
                              edgecolor='white', linewidth=0.3, zorder=2)

            ax.set_xticks(gx)
            ax.set_xticklabels(FLT_ORDER, rotation=15, ha='right', fontsize=8)
            ax.set_title(f'{mlabel}\n{lx["label"]}', fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.35, zorder=0)

            if metric == 'avg_time_ms':
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.set_ylabel('ms (log)', fontsize=9)
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

    fig.suptitle('Filtering Method Comparison: Inlier Quality & Time by Lighting Condition',
                 fontsize=18, fontweight='bold', y=1.03)
    fig.savefig(OUT_DIR / 'planA_metrics_grid.png')
    plt.close(fig)
    print("  -> Saved planA_metrics_grid.png")

    # ---- Plan A bonus: time bar chart (commented out, keep code) ----
    # plan_a_time_detail(df, datasets_ordered)


def plan_a_time_detail(df, datasets_ordered):
    """Separate time bar chart: 3 panels (lighting), log + linear inset."""
    n_d = len(datasets_ordered)
    fig, axes = plt.subplots(1, n_d, figsize=(16, 5.5))

    bar_w = 0.22
    n_f = len(FLT_ORDER)
    n_c = len(CH_ORDER)

    for col, dataset in enumerate(datasets_ordered):
        ax = axes[col]
        sub = df[df['dataset'] == dataset]
        lx = LIGHTING_INFO[dataset]

        gx, offsets = grouped_x(n_f, n_c, bar_w)
        for ci, ch in enumerate(CH_ORDER):
            vals = []
            for fl in FLT_ORDER:
                v = sub[(sub['channel'] == ch) & (sub['filter'] == fl)]['avg_time_ms'].values
                vals.append(v[0] if len(v) > 0 else 0)
            ax.bar(gx + offsets[ci], vals, bar_w, color=CH_COLOR[ch],
                   label=CH_LABEL[ch], edgecolor='white', linewidth=0.3, zorder=2)

        # Draw a break line to separate NLM visually
        ax.axhline(30, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(4, 32, 'NLM outlier ↑', fontsize=7, color='red', ha='right', va='bottom')

        ax.set_xticks(gx)
        ax.set_xticklabels(FLT_ORDER, rotation=15, ha='right', fontsize=8)
        ax.set_title(lx['label'].replace('\n', ' '), fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Time (ms)', fontsize=9)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(axis='y', alpha=0.35, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle('Plan A — Per-Frame Processing Time (log scale, NLM broken out)',
                 fontsize=13, fontweight='bold', y=1.12)
    fig.savefig(OUT_DIR / 'planA_time_detail.png')
    plt.close(fig)
    print("  -> Saved planA_time_detail.png")


# ============================================================
# Plan B — Scatter quality-time + Heatmaps
# ============================================================
def plan_b(df):
    """Scatter: avg_time vs avg_inlier, subplots by channel. Heatmaps: 3 lighting."""
    print("[Plan B] Generating scatter + heatmaps...")

    plan_b_scatter(df)
    plan_b_heatmaps(df)


def plan_b_scatter(df):
    """Quality-time trade-off scatter, 1 row × 3 channels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.subplots_adjust(wspace=0.22)

    markers = {'Bright': 'o', 'Medium': 's', 'Dark': '^'}
    lighting_order = ['Bright', 'Medium', 'Dark']

    for ci, ch in enumerate(CH_ORDER):
        ax = axes[ci]
        ch_data = df[df['channel'] == ch]

        for fl in FLT_ORDER:
            pts = ch_data[ch_data['filter'] == fl]
            # Sort by lighting order for consistent line
            pts = pts.set_index('lighting_short').reindex(lighting_order).reset_index()
            pts = pts.dropna(subset=['avg_inlier'])
            if len(pts) == 0:
                continue

            for _, row in pts.iterrows():
                ax.scatter(row['avg_time_ms'], row['avg_inlier'],
                          color=FLT_COLOR[fl], marker=markers.get(row['lighting_short'], 'o'),
                          s=90, edgecolors='white', linewidth=0.5, zorder=3)
            # Connect same filter across lighting
            ax.plot(pts['avg_time_ms'], pts['avg_inlier'],
                   color=FLT_COLOR[fl], alpha=0.4, linewidth=0.8, zorder=2)

        ax.set_xscale('log')
        ax.set_xlabel('Avg Time (ms, log)', fontsize=9)
        ax.set_ylabel('Avg Inlier Count', fontsize=9)
        ax.set_title(CH_LABEL[ch], fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, zorder=0)

        # Annotate NLM points
        nlm_pts = ch_data[ch_data['filter'] == 'NLM']
        for _, row in nlm_pts.iterrows():
            ax.annotate(row['lighting_short'][0], (row['avg_time_ms'], row['avg_inlier']),
                       fontsize=6, ha='left', va='bottom', color='#D55E00', alpha=0.8)

    # Legends
    from matplotlib.lines import Line2D
    flt_handles = [Line2D([0], [0], color=FLT_COLOR[fl], lw=2, label=fl) for fl in FLT_ORDER]
    mkr_handles = [Line2D([0], [0], marker=m, color='gray', markersize=8, lw=0,
                          markerfacecolor='gray', label=lt)
                   for lt, m in markers.items()]
    leg1 = axes[0].legend(handles=flt_handles, title='Filter', fontsize=7,
                          title_fontsize=8, loc='upper left')
    leg2 = axes[0].legend(handles=mkr_handles, title='Lighting', fontsize=7,
                          title_fontsize=8, loc='lower right')
    axes[0].add_artist(leg1)

    fig.suptitle('Plan B — Quality vs Time Trade-off by Channel',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.savefig(OUT_DIR / 'planB_scatter.png')
    plt.close(fig)
    print("  -> Saved planB_scatter.png")


def plan_b_heatmaps(df):
    """Heatmaps: 3 lighting × 1 metric (avg_inlier). Rows=channel, Cols=filter."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.subplots_adjust(wspace=0.15)

    datasets_ordered = sorted(df['dataset'].unique(), key=lambda d: LIGHTING_INFO[d]['order'])

    vmin = df['avg_inlier'].min()
    vmax = df['avg_inlier'].max()

    for col, dataset in enumerate(datasets_ordered):
        ax = axes[col]
        lx = LIGHTING_INFO[dataset]
        sub = df[df['dataset'] == dataset]

        # Pivot: rows=channel, cols=filter, value=avg_inlier
        mat = sub.pivot_table(index='channel', columns='filter',
                              values='avg_inlier', aggfunc='first')
        # Ensure ordering
        mat = mat.reindex(index=CH_ORDER, columns=FLT_ORDER)

        sns.heatmap(mat, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Avg Inlier', 'shrink': 0.8},
                    annot_kws={'fontsize': 9})
        ax.set_title(lx['label'].replace('\n', ' '), fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.suptitle('Plan B — Avg Inlier Heatmap by Lighting Condition',
                 fontsize=13, fontweight='bold', y=1.03)
    fig.savefig(OUT_DIR / 'planB_heatmaps.png')
    plt.close(fig)
    print("  -> Saved planB_heatmaps.png")


# ============================================================
# Plan C — Comprehensive Dashboard
# ============================================================
def plan_c(df):
    """A single comprehensive dashboard + markdown summary table."""
    print("[Plan C] Generating dashboard + table...")

    datasets_ordered = sorted(df['dataset'].unique(), key=lambda d: LIGHTING_INFO[d]['order'])

    # ---- C1: Multi-panel dashboard figure ----
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  height_ratios=[1, 1.1])

    # C1a: Top-left 3×1 — inlier bar comparison across lighting
    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.25)
    _plan_c_inlier_bars(fig, gs_top, df, datasets_ordered)

    # C1b: Bottom row — 3 heatmaps (inlier, inlier_ratio, time)
    gs_bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.28)
    _plan_c_heatmap_row(fig, gs_bot, df, datasets_ordered)

    fig.suptitle('Plan C — Comprehensive Filter Comparison Dashboard',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.savefig(OUT_DIR / 'planC_dashboard.png')
    plt.close(fig)
    print("  -> Saved planC_dashboard.png")

    # ---- C2: Markdown summary table ----
    _plan_c_markdown_table(df, datasets_ordered)


def _plan_c_inlier_bars(fig, gs, df, datasets_ordered):
    """Plan C: one big grouped bar — inlier across all lighting side by side."""
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    bar_w = 0.22
    n_f = len(FLT_ORDER)
    n_c = len(CH_ORDER)

    for col, dataset in enumerate(datasets_ordered):
        ax = axes[col]
        sub = df[df['dataset'] == dataset]
        lx = LIGHTING_INFO[dataset]

        gx, offsets = grouped_x(n_f, n_c, bar_w)
        for ci, ch in enumerate(CH_ORDER):
            vals = []
            for fl in FLT_ORDER:
                v = sub[(sub['channel'] == ch) & (sub['filter'] == fl)]['avg_inlier'].values
                vals.append(v[0] if len(v) > 0 else 0)
            ax.bar(gx + offsets[ci], vals, bar_w, color=CH_COLOR[ch],
                   edgecolor='white', linewidth=0.3, zorder=2)

        ax.set_xticks(gx)
        ax.set_xticklabels(FLT_ORDER, rotation=20, ha='right', fontsize=8)
        ax.set_title(lx['label'].replace('\n', ' '), fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Inlier Count', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 1.25))


def _plan_c_heatmap_row(fig, gs, df, datasets_ordered):
    """Three metric summary heatmaps (averaged across lighting)."""
    metrics = ['avg_inlier', 'inlier_ratio', 'avg_time_ms']
    mlabels = ['Avg Inlier Count', 'Inlier Ratio', 'Avg Time (ms)']
    cmaps = ['YlOrRd', 'YlGnBu', 'YlOrBr_r']

    for mi, (metric, mlabel, cmap) in enumerate(zip(metrics, mlabels, cmaps)):
        ax = fig.add_subplot(gs[0, mi])

        # Build a (channel × filter) matrix averaged/summarized across datasets
        rows = []
        for ch in CH_ORDER:
            row_vals = []
            for fl in FLT_ORDER:
                vals = df[(df['channel'] == ch) & (df['filter'] == fl)][metric].values
                row_vals.append(np.mean(vals) if len(vals) > 0 else np.nan)
            rows.append(row_vals)
        mat = pd.DataFrame(rows, index=[CH_LABEL[c] for c in CH_ORDER], columns=FLT_ORDER)

        annot_fmt = '.3f' if metric == 'inlier_ratio' else '.1f'
        fmt_override = '.0f' if metric == 'avg_time_ms' and mat.max().max() < 200 else annot_fmt

        sns.heatmap(mat, annot=True, fmt=fmt_override, cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': mlabel, 'shrink': 0.75},
                    annot_kws={'fontsize': 8})
        ax.set_title(mlabel, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')


def _plan_c_markdown_table(df, datasets_ordered):
    """Output a formatted markdown summary table."""
    md_path = OUT_DIR / 'planC_summary_table.md'

    lines = []
    lines.append("# Plan C — Filter Comparison Summary Table\n")
    lines.append("Each cell: **avg_inlier** / inlier_ratio / time(ms)\n")

    for dataset in datasets_ordered:
        lx = LIGHTING_INFO[dataset]
        sub = df[df['dataset'] == dataset]

        lines.append(f"## {lx['label'].replace(chr(10), ' ')}\n")

        # Header
        lines.append("| Channel | " + " | ".join(FLT_ORDER) + " |")
        lines.append("|" + "---|" * (len(FLT_ORDER) + 1))

        for ch in CH_ORDER:
            cells = []
            for fl in FLT_ORDER:
                row = sub[(sub['channel'] == ch) & (sub['filter'] == fl)]
                if len(row) > 0:
                    r = row.iloc[0]
                    cells.append(f"**{r['avg_inlier']:.1f}** / {r['inlier_ratio']:.2%} / {r['avg_time_ms']:.1f}ms")
                else:
                    cells.append("—")
            lines.append(f"| {CH_LABEL[ch]} | " + " | ".join(cells) + " |")

        lines.append("")

    # Add a "best per metric" summary
    lines.append("## Best per Lighting\n")
    lines.append("| Lighting | Best Inlier | Best Ratio | Fastest (excl. NLM) |")
    lines.append("|---|---|---|---|")
    for dataset in datasets_ordered:
        sub = df[df['dataset'] == dataset]
        lx = LIGHTING_INFO[dataset]
        best_inlier = sub.loc[sub['avg_inlier'].idxmax()]
        best_ratio = sub.loc[sub['inlier_ratio'].idxmax()]
        non_nlm = sub[sub['filter'] != 'NLM']
        fastest = non_nlm.loc[non_nlm['avg_time_ms'].idxmin()]

        lines.append(
            f"| {lx['short']} | "
            f"{best_inlier['channel']}+{best_inlier['filter']} "
            f"({best_inlier['avg_inlier']:.1f}) | "
            f"{best_ratio['channel']}+{best_ratio['filter']} "
            f"({best_ratio['inlier_ratio']:.2%}) | "
            f"{fastest['channel']}+{fastest['filter']} "
            f"({fastest['avg_time_ms']:.1f}ms) |"
        )

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  -> Saved planC_summary_table.md")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("PolarFP-VINS — Filter Comparison Visualization")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df)} rows from {df['dataset'].nunique()} datasets.")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Channels: {list(df['channel'].unique())}")
    print(f"Filters:  {list(df['filter'].unique())}")
    print()

    plan_a(df)
    # plan_b(df)   # commented out, keep code
    # plan_c(df)   # commented out, keep code

    print(f"\nDone! All figures saved to: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
