import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import ticker
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Rectangle

from .seg_utils import bp
from .seg_eval_utils import score_approaches_triadic_boundary_similarity_complete


"""def plot_approach_evaluation(data_dict, threshold):
    df = pd.DataFrame(data_dict)

    # Professional styling for paper publication
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks", {"grid.linestyle": "--"})

    plt.figure(figsize=(10, 6))

    # 1. Define the warping functions (Square Root Scaling)
    def forward(x):
        return np.power(x, 0.25)

    def inverse(x):
        return np.power(x, 4)

    # 2. Create the scatter plot
    # We plot this BEFORE setting the scale to ensure everything maps correctly
    plot = sns.scatterplot(
        data=df,
        x='time',
        y='score',
        hue='Labels:',
        style='Shapes:',
        s=80,
        alpha=0.6,
        palette='viridis',
        edgecolor='w',
        linewidth=0.5,
        zorder=3  # Ensure points are above the threshold line
    )

    # 3. Apply the Custom Square Root Scale
    plt.xscale('function', functions=(forward, inverse))

    # 4. Handle X-axis Ticks
    # We define specific steps to show detail in the 0-200 range
    ticks = [0, 1, 15, 80, 250, 600, 1000]
    plt.xticks(ticks)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Ensure plot starts at 0 and accommodates outliers
    plt.xlim(0, df['time'].max() * 1.05)

    # 5. Constrain Y-axis
    plt.ylim(0.2, 1.05)

    # 6. Threshold line
    plt.axhline(
        y=threshold,
        color='#d62728',
        linestyle='--',
        linewidth=1.5,
        label=f'Human ({round(threshold, 2):.2f})',
        zorder=2
    )

    # 7. Professional Labeling
    plt.xlabel(r"Time Complexity (s) [$\mathbf{\sqrt[4]{t}}$ scale]", fontweight='bold')
    plt.ylabel('Bounding Similarity', fontweight='bold')

    sns.despine()

    # 8. Legend Management


    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles, labels,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=True,
        fontsize='small'
    )


    plt.tight_layout()
    # Ensure bp() is defined in your environment or replace with a string path
    plt.savefig(f"{bp()}/data/plots/score_vs_time.pdf", format="pdf", bbox_inches='tight')
    plt.show()"""


def plot_approach_evaluation(data_dict, threshold):
    df = pd.DataFrame(data_dict)

    # --- 1. PRE-DEFINE MAPPINGS (Ensure legend and plot match) ---
    unique_labels = sorted(df['labels'].unique())
    colors = sns.color_palette('viridis', n_colors=len(unique_labels))
    label_color_map = dict(zip(unique_labels, colors))

    unique_shapes = sorted(df['shapes'].unique())
    marker_list = ['o', "*", 'P', 'X', '^', 's', 'd']
    shape_marker_map = dict(zip(unique_shapes, marker_list[:len(unique_shapes)]))

    # Setup Styling
    # 1. Update the rcParams to use STIX fonts
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
    })
    sns.set_context("paper", font_scale=2.3)
    sns.set_style("ticks", {"grid.linestyle": "--"})
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # 2. Warp functions
    def forward(x): return np.power(x, 0.25)

    def inverse(x): return np.power(x, 4)

    # 3. Plotting with EXPLICIT maps
    sns.scatterplot(
        data=df, x='time', y='score',
        hue='labels', palette=label_color_map,  # Use the map
        style='shapes', markers=shape_marker_map,  # Use the map
        s=120, alpha=0.9,
        edgecolor='w', linewidth=0.5, zorder=3,
        legend=False
    )

    # 4. Axes & Scale (Keeping your existing logic)
    plt.xscale('function', functions=(forward, inverse))
    ticks = [0, 1, 15, 80, 250, 600, 1000]
    plt.xticks(ticks)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlim(0, df['time'].max() * 1.05)
    plt.ylim(0.2, 1.05)

    # 5. Threshold Line
    plt.axhline(y=threshold, color='#d62728', linestyle='--', linewidth=1.5, zorder=2)

    # 6. Labels
    plt.xlabel(r"Time Complexity (s) [$\mathbf{\sqrt[4]{t}}$ scale]", fontweight='bold')
    plt.ylabel('Bounding Similarity', fontweight='bold')
    sns.despine()

    # ---------------------------------------------------------
    # 7. MANUAL LEGEND CONSTRUCTION (Now using the same maps)
    # ---------------------------------------------------------
    hue_handles = [mlines.Line2D([], [], color=label_color_map[lab], marker='o',
                                 linestyle='None', markersize=8, label=str(lab))
                   for lab in unique_labels]

    shape_handles = [mlines.Line2D([], [], color='gray', marker=shape_marker_map[shp],
                                   linestyle='None', markersize=8, label=str(shp))
                     for shp in unique_shapes]

    thresh_handle = [mlines.Line2D([], [], color='#d62728', linestyle='--',
                                   linewidth=1.5, label=f'Human ({threshold:.2f})')]

    # Create Legends
    leg1 = ax.legend(handles=hue_handles, title="Engine Comb.", loc='lower left',
                     bbox_to_anchor=(0.05, 0.7), ncol=3, frameon=False, fontsize='small',
                     borderpad=0.2,  # Increase for more "breathing room" inside the box
                     handletextpad=0.2,  # Decrease to pull text closer to the markers
                     columnspacing=0.8,
                     labelspacing=0.2
                     )
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=shape_handles, title="Late Fusion", loc='lower center',
                     bbox_to_anchor=(0.75, 0.82), ncol=2, frameon=False, fontsize='small',
                     borderpad=0.2,  # Increase for more "breathing room" inside the box
                     handletextpad=0.1,  # Decrease to pull text closer to the markers
                     columnspacing=0.8,
                     labelspacing=0.1
                     )
    ax.add_artist(leg2)

    # Using loc='lower right' for threshold as requested
    leg3 = ax.legend(handles=thresh_handle, title="Baseline", loc='lower center',
                     bbox_to_anchor=(0.75, 0.7), frameon=False, fontsize='small',
                     borderpad=0.2,  # Increase for more "breathing room" inside the box
                     handletextpad=0.1,  # Decrease to pull text closer to the markers
                     columnspacing=0.8,
                     labelspacing=0.1
                     )

    # --- 8. WRAPPER BOX (Refined) ---
    plt.tight_layout()
    fig.canvas.draw()

    inv = ax.transAxes.inverted()
    b1 = inv.transform_bbox(leg1.get_window_extent())
    b2 = inv.transform_bbox(leg2.get_window_extent())

    x0, y0 = min(b1.x0, b2.x0), min(b1.y0, b2.y0)
    x1, y1 = max(b1.x1, b2.x1), max(b1.y1, b2.y1)

    pad = 0.015
    outer_box = Rectangle(
        (x0 - pad, y0 - pad),
        (x1 - x0) + (2 * pad), (y1 - y0) + (2 * pad),
        fill=False, edgecolor='black', linewidth=1.5,
        transform=ax.transAxes, clip_on=False, zorder=0
    )
    ax.add_patch(outer_box)

    plt.tight_layout()
    plt.savefig(f"{bp()}/data/plots/score_vs_time.pdf", format="pdf", bbox_inches='tight')
    plt.show()

def plot_score_vs_time():
    data, threshold = score_approaches_triadic_boundary_similarity_complete()
    plot_approach_evaluation(data, threshold)