"""Figures for the report.

  python -m src.report.figures

The numbers behind each figure are in reports/estimates/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.paths import REPORTS

SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
LABELS = {
    ("japan_2019", "all treated"): "Japan 2019 hike (8→10%)",
    ("singapore_2023", "all treated"): "Singapore 2023 hike (7→8%)",
    ("singapore_2024", "all treated"): "Singapore 2024 hike (8→9%)",
    ("india_gst2", "all treated"): "India GST 2.0 cuts, all items",
    ("india_gst2", "food and beverages"): "   food and beverages",
    ("india_gst2", "durables"): "   durables",
}


def overall_forest() -> None:
    d = pd.read_csv(REPORTS / "estimates" / "overall.csv")
    d["key"] = list(zip(d.episode, d.group))
    d = d[d.key.isin(LABELS)].set_index("key").loc[list(LABELS)]
    y = list(range(len(d)))[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.vlines([0, 1], -0.6, len(d) - 0.45, color=INK_2, lw=1, zorder=1)
    ax.hlines(y, d.ci_low, d.ci_high, color=SERIES, lw=2, zorder=2, capstyle="round")
    ax.scatter(d.rho, y, s=70, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)

    ax.set_yticks(y, [LABELS[k] for k in d.index], color=INK, fontsize=9.5)
    ax.set_ylim(-0.6, len(d) - 0.2)
    ax.set_xlim(-2, 2)
    ax.set_xlabel("Share of the tax change reaching prices (ρ), with 95% interval", color=INK_2, fontsize=9)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    top = len(d) - 0.35
    ax.text(0, top, "none", ha="center", va="bottom", color=INK_2, fontsize=8.5)
    ax.text(1, top, "full", ha="center", va="bottom", color=INK_2, fontsize=8.5)
    fig.suptitle("Japan's 2019 hike reached prices; most of India's 2025 cut had not after three months",
                 x=0.01, ha="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "overall_passthrough.png", facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


if __name__ == "__main__":
    overall_forest()
