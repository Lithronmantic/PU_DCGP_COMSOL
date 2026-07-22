
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.pu_dcgp import (
    ManifestDataSource,
    PUDCGPConfig,
    a_group_doe_estimands,
    bootstrap_matched_distribution_effect,
)

from .a_group_network_point_reanalysis import _assert_corrected_mapping
from .journal_figure_style import (
    COLORS,
    DOUBLE_COLUMN_WIDTH_IN,
    apply_journal_style,
    export_figure,
)


DATA_DIR = Path(__file__).with_name("data")
EVIDENCE_PATH = DATA_DIR / "a_group_causal_network_evidence.json"
FIGURE_DATA_PATH = DATA_DIR / "a_group_quantile_figure_data.json"
OUTPUT_DIR = Path(__file__).with_name("figures") / "journal_a_group"

OUTCOME_ORDER = ("temperature_c", "velocity_m_s", "particle_diameter_um")
OUTCOME_LABELS = {
    "temperature_c": "Quantile effect on particle temperature (°C)",
    "velocity_m_s": "Quantile effect on particle velocity (m s$^{-1}$)",
    "particle_diameter_um": "Quantile effect on detected particle size (µm)",
}
OUTCOME_FILE_LABELS = {
    "temperature_c": "temperature",
    "velocity_m_s": "velocity",
    "particle_diameter_um": "particle_diameter",
}
ESTIMAND_LABELS = {
    "current_600_to_800": "Arc current: 600 to 800 A",
    "argon_80_to_120": "Argon flow: 80 to 120 scfh",
    "powder_10_to_30": "Powder feed: 10 to 30 g min$^{-1}$",
    "distance_80_to_120": "Spray distance: 80 to 120 mm",
}
ESTIMAND_FILE_LABELS = {
    "current_600_to_800": "current",
    "argon_80_to_120": "argon",
    "powder_10_to_30": "powder",
    "distance_80_to_120": "distance",
}
SOURCE_COLORS = {
    "current_a": COLORS["blue"],
    "argon_flow_scfh": COLORS["green"],
    "powder_feed_g_min": COLORS["orange"],
    "spray_distance_mm": COLORS["purple"],
}


def _figure_number(estimand_index: int, outcome_index: int) -> int:
    return 8 + 3 * estimand_index + outcome_index


def _plot_one(record: dict) -> tuple[Path, ...]:
    color = SOURCE_COLORS[record["treatment_name"]] if record["supported"] else COLORS["gray"]
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.25))
    grid = record["quantile_grid"]
    effect = record["point_effect"]
    lower = record["simultaneous_lower_bound"]
    upper = record["simultaneous_upper_bound"]
    axis.fill_between(
        grid,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0.0,
        label="95% simultaneous band",
    )
    axis.plot(
        grid,
        effect,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        markerfacecolor="white" if not record["supported"] else color,
        markeredgecolor=color,
        markeredgewidth=0.8,
        label="Quantile effect",
        zorder=3,
    )
    axis.axhline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)
    axis.set_xlim(0.045, 0.955)
    axis.set_xticks((0.05, 0.25, 0.50, 0.75, 0.95))
    axis.set_xlabel("Particle-distribution quantile")
    axis.set_ylabel(OUTCOME_LABELS[record["outcome"]])
    suffix = ""
    if record["status"] == "exploratory_admit":
        suffix = " (exploratory)"
    elif record["status"] == "abstain":
        suffix = " (not supported)"
    axis.set_title(ESTIMAND_LABELS[record["estimand_id"]] + suffix, loc="left", fontweight="normal")
    axis.grid(axis="both", color=COLORS["grid"], linewidth=0.55, linestyle=":")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 1.16), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.20, top=0.80)
    stem = (
        f"fig{record['figure_number']:02d}_"
        f"{ESTIMAND_FILE_LABELS[record['estimand_id']]}_"
        f"{OUTCOME_FILE_LABELS[record['outcome']]}_quantile_effect"
    )
    return export_figure(figure, OUTPUT_DIR, stem)


def build_figure_data() -> dict:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    evidence_by_key = {
        (edge["estimand_id"], edge["target"]): edge
        for edge in evidence["edges"]
    }
    runs = ManifestDataSource(groups=("A",)).load()
    _assert_corrected_mapping(runs)
    config = PUDCGPConfig(effect_bootstrap_replicates=2000)
    records = []
    for estimand_index, estimand in enumerate(a_group_doe_estimands()):
        for outcome_index, outcome in enumerate(OUTCOME_ORDER):
            result = bootstrap_matched_distribution_effect(runs, config, estimand, outcome)
            band = result.quantile_band
            edge = evidence_by_key[(estimand.estimand_id, outcome)]
            records.append(
                {
                    "figure_number": _figure_number(estimand_index, outcome_index),
                    "estimand_id": estimand.estimand_id,
                    "treatment_name": estimand.treatment_name,
                    "outcome": outcome,
                    "status": edge["status"],
                    "supported": edge["supported"],
                    "quantile_grid": band.quantile_grid.tolist(),
                    "point_effect": band.point_effect.tolist(),
                    "pointwise_lower_bound": band.pointwise_lower_bound.tolist(),
                    "pointwise_upper_bound": band.pointwise_upper_bound.tolist(),
                    "simultaneous_lower_bound": band.simultaneous_lower_bound.tolist(),
                    "simultaneous_upper_bound": band.simultaneous_upper_bound.tolist(),
                    "simultaneous_critical_value": band.simultaneous_critical_value,
                    "simultaneous_excludes_zero_everywhere": band.simultaneous_excludes_zero_everywhere,
                }
            )
    return {
        "schema": "pu_dcgp_v26_a_group_quantile_figure_data_v1",
        "bootstrap_replicates": config.effect_bootstrap_replicates,
        "interval_level": config.effect_interval_level,
        "mapping_rule": "NNN.csv = UCE-RNNN = design row NNN = execution order NNN",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Reuse the frozen JSON figure data without rerunning bootstrap.",
    )
    arguments = parser.parse_args()
    apply_journal_style()
    if arguments.render_only:
        data = json.loads(FIGURE_DATA_PATH.read_text(encoding="utf-8"))
    else:
        data = build_figure_data()
        FIGURE_DATA_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for record in data["records"]:
        _plot_one(record)
    plt.close("all")


if __name__ == "__main__":
    main()
