
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure


MM_PER_INCH = 25.4
DOUBLE_COLUMN_WIDTH_IN = 180.0 / MM_PER_INCH

COLORS = {
    "blue": "#4C78A8",
    "green": "#59A14F",
    "orange": "#F28E2B",
    "purple": "#8B6FB0",
    "gray": "#8A8A8A",
    "dark": "#262626",
    "grid": "#D9D9D9",
}


def apply_journal_style() -> None:

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 7.0,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def export_figure(figure: Figure, output_dir: Path, stem: str) -> tuple[Path, ...]:

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_dir / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png", "tiff"))
    figure.savefig(paths[0], bbox_inches="tight", pad_inches=0.04)
    figure.savefig(paths[1], bbox_inches="tight", pad_inches=0.04)
    figure.savefig(paths[2], dpi=600, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        paths[3],
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return paths
