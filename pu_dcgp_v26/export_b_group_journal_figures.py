"""Export independent publication figures for the B-group analysis."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from .b_group_auxiliary_analysis import build_b_group_auxiliary_analysis
from .journal_figure_style import COLORS, apply_journal_style, export_figure


OUTPUT_DIR = Path(__file__).with_name("figures") / "b_group_journal"
MANIFEST_PATH = OUTPUT_DIR / "figure_manifest.json"
ZIP_PATH = Path(__file__).with_name("figures") / "b_group_journal_figures.zip"

DISTANCES = np.asarray((80.0, 90.0, 100.0, 110.0, 120.0))


def _finish_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.55, linestyle=":")
    axis.set_axisbelow(True)
    axis.set_xticks(DISTANCES)


def _plot_cell_means(cells, outcome: str, ylabel: str, stem: str) -> dict:
    figure, axis = plt.subplots(figsize=(3.55, 2.45), constrained_layout=True)
    for position, color, marker in (
        (90.0, COLORS["blue"], "o"),
        (110.0, COLORS["orange"], "s"),
    ):
        selected = cells[cells["measurement_position_mm"] == position].sort_values(
            "nominal_spray_distance_mm"
        )
        axis.errorbar(
            selected["nominal_spray_distance_mm"],
            selected[f"{outcome}_mean"],
            yerr=selected[f"{outcome}_run_sd"],
            color=color,
            marker=marker,
            markersize=4.2,
            capsize=2.2,
            label=f"DPV plane {position:.0f} mm",
        )
    axis.set_xlabel("Executed nominal distance block (mm)")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, loc="best")
    _finish_axes(axis)
    paths = export_figure(figure, OUTPUT_DIR, stem)
    plt.close(figure)
    return {
        "stem": stem,
        "files": [path.name for path in paths],
        "caption": (
            f"Run-balanced B-group {ylabel.lower()} at the 90 and 110 mm DPV planes. "
            "Points are cell means and error bars are between-run standard deviations (n=3)."
        ),
    }


def _plot_position_difference(differences, outcome: str, ylabel: str, stem: str) -> dict:
    selected = differences[differences["outcome"] == outcome].sort_values(
        "nominal_spray_distance_mm"
    )
    values = selected["far_minus_near"].to_numpy(dtype=float)
    lower = selected["bootstrap_95_lower"].to_numpy(dtype=float)
    upper = selected["bootstrap_95_upper"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(3.55, 2.45), constrained_layout=True)
    axis.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axis.errorbar(
        DISTANCES,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        color=COLORS["green"],
        marker="o",
        markersize=4.4,
        capsize=2.4,
    )
    fit = np.polyval(np.polyfit(DISTANCES, values, 1), DISTANCES)
    axis.plot(DISTANCES, fit, color=COLORS["dark"], linewidth=1.0, linestyle="--")
    axis.set_xlabel("Executed nominal distance block (mm)")
    axis.set_ylabel(ylabel)
    _finish_axes(axis)
    paths = export_figure(figure, OUTPUT_DIR, stem)
    plt.close(figure)
    return {
        "stem": stem,
        "files": [path.name for path in paths],
        "caption": (
            f"Far-minus-near B-group {ylabel.lower()} (110 mm minus 90 mm). "
            "Error bars are 95% within-cell run-bootstrap intervals; the dashed line is a descriptive linear trend."
        ),
    }


def _plot_source_sensitivity(source_summary: dict) -> dict:
    figure, axis = plt.subplots(figsize=(3.55, 2.45), constrained_layout=True)
    series = (
        (
            "processed_strips_cell_mean",
            "STRIPS, cell mean",
            COLORS["green"],
            "o",
        ),
        ("raw_prt_cell_mean", "Raw PRT, cell mean", COLORS["orange"], "s"),
        ("raw_prt_cell_median", "Raw PRT, cell median", COLORS["blue"], "^"),
    )
    axis.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    for key, label, color, marker in series:
        values = source_summary["source_summaries"][key][
            "far_minus_near_velocity_m_s"
        ]
        axis.plot(
            DISTANCES,
            values,
            color=color,
            marker=marker,
            markersize=4.0,
            label=label,
        )
    axis.set_xlabel("Executed nominal distance block (mm)")
    axis.set_ylabel("Velocity difference, 110−90 mm (m/s)")
    axis.legend(frameon=False, loc="best")
    _finish_axes(axis)
    paths = export_figure(figure, OUTPUT_DIR, "b05_velocity_source_sensitivity")
    plt.close(figure)
    return {
        "stem": "b05_velocity_source_sensitivity",
        "files": [path.name for path in paths],
        "caption": (
            "Sensitivity of the B-group far-minus-near velocity pattern to the DPV export representation and cell aggregation. "
            "All 30 runs are retained."
        ),
    }


def build_figures() -> list[dict]:
    apply_journal_style()
    result, frames = build_b_group_auxiliary_analysis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = frames["cell_means"]
    differences = frames["position_differences"]
    records = [
        _plot_cell_means(
            cells,
            "velocity_m_s",
            "Particle velocity (m/s)",
            "b01_velocity_by_dpv_plane",
        ),
        _plot_position_difference(
            differences,
            "velocity_m_s",
            "Velocity difference, 110−90 mm (m/s)",
            "b02_velocity_position_difference",
        ),
        _plot_position_difference(
            differences,
            "temperature_c",
            "Temperature difference, 110−90 mm (°C)",
            "b03_temperature_position_difference",
        ),
        _plot_position_difference(
            differences,
            "particle_diameter_um",
            "Diameter difference, 110−90 mm (μm)",
            "b04_diameter_position_difference",
        ),
        _plot_source_sensitivity(result["raw_prt_velocity_sensitivity"]),
    ]
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "schema": "pu_dcgp_v26_b_group_journal_figures_v1",
                "independent_figure_count": len(records),
                "formats": ["pdf", "svg", "png_600dpi", "tiff_600dpi_lzw"],
                "figures": records,
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    archive = shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", OUTPUT_DIR)
    if Path(archive) != ZIP_PATH:
        raise AssertionError("Unexpected B figure archive path")
    return records


def main() -> None:
    records = build_figures()
    print(f"Exported {len(records)} independent B figures to {OUTPUT_DIR}")
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
