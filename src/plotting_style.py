"""Shared matplotlib style for all figures: print-quality, colorblind-safe,
consistent across the paper. Import and call `apply_style()` once per
script before creating any figure."""
from __future__ import annotations

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
PALETTE = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000", "grey": "#7F7F7F",
}
METHOD_COLORS = {
    "linear": PALETTE["grey"], "random_forest": PALETTE["sky"],
    "xgboost": PALETTE["blue"], "lightgbm": PALETTE["green"], "mlp": PALETTE["purple"],
}
OPT_COLORS = {
    "status_quo": PALETTE["grey"], "equal_split": PALETTE["sky"],
    "recenter_only": PALETTE["orange"], "optimized": PALETTE["blue"],
    "greedy": PALETTE["vermillion"], "ga": PALETTE["purple"], "oracle": PALETTE["black"],
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,   # embed fonts as TrueType, not Type-3 -- required by many journals
        "ps.fonttype": 42,
    })


def save(fig, path_no_ext):
    fig.savefig(f"{path_no_ext}.pdf")
    fig.savefig(f"{path_no_ext}.png")
