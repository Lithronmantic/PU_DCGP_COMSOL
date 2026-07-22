
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .journal_figure_style import (
    COLORS,
    DOUBLE_COLUMN_WIDTH_IN,
    apply_journal_style,
    export_figure,
)


DATA_DIR = Path(__file__).with_name("data")
EVIDENCE_PATH = DATA_DIR / "a_group_causal_network_evidence.json"
STABILITY_PATH = DATA_DIR / "a_group_network_stability_reanalysis.json"
SENSITIVITY_PATH = DATA_DIR / "a_group_network_count_sensitivity.json"
OUTPUT_DIR = Path(__file__).with_name("figures") / "journal_a_group"

SOURCE_ORDER = ("current_a", "argon_flow_scfh", "powder_feed_g_min", "spray_distance_mm")
SOURCE_LABELS = {
    "current_a": "Arc current\n600 to 800 A",
    "argon_flow_scfh": "Argon flow\n80 to 120 scfh",
    "powder_feed_g_min": "Powder feed\n10 to 30 g min$^{-1}$",
    "spray_distance_mm": "Spray distance\n80 to 120 mm",
}
SOURCE_COLORS = {
    "current_a": COLORS["blue"],
    "argon_flow_scfh": COLORS["green"],
    "powder_feed_g_min": COLORS["orange"],
    "spray_distance_mm": COLORS["purple"],
}
TARGET_ORDER = ("temperature_c", "velocity_m_s", "particle_diameter_um")
TARGET_LABELS = {
    "temperature_c": "Particle-temperature\ndistribution",
    "velocity_m_s": "Particle-velocity\ndistribution",
    "particle_diameter_um": "Detected particle-size\ndistribution",
}
FOREST_LABELS = {
    "current_a": "Arc current (600 to 800 A)",
    "argon_flow_scfh": "Argon flow (80 to 120 scfh)",
    "powder_feed_g_min": "Powder feed (10 to 30 g min$^{-1}$)",
    "spray_distance_mm": "Spray distance (80 to 120 mm)",
}
OUTCOME_AXIS_LABELS = {
    "temperature_c": "Mean effect on particle temperature (°C)",
    "velocity_m_s": "Mean effect on particle velocity (m s$^{-1}$)",
    "particle_diameter_um": "Mean effect on detected particle size (µm)",
}
OUTCOME_STEMS = {
    "temperature_c": "fig02_temperature_mean_effects",
    "velocity_m_s": "fig03_velocity_mean_effects",
    "particle_diameter_um": "fig04_particle_diameter_mean_effects",
}
ROBUSTNESS_STEMS = {
    "temperature_c": "fig05_temperature_robustness",
    "velocity_m_s": "fig06_velocity_robustness",
    "particle_diameter_um": "fig07_particle_diameter_robustness",
}


def _load_evidence() -> dict:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _load_robustness_data() -> tuple[dict, dict]:
    return (
        json.loads(STABILITY_PATH.read_text(encoding="utf-8")),
        json.loads(SENSITIVITY_PATH.read_text(encoding="utf-8")),
    )


def _effect_text(edge: dict) -> str:
    units = {"temperature_c": "°C", "velocity_m_s": "m/s", "particle_diameter_um": "µm"}
    short = {"temperature_c": "T", "velocity_m_s": "V", "particle_diameter_um": "D"}
    return f"{short[edge['target']]} {edge['mean_effect']:+.1f} {units[edge['target']]}"


def plot_supported_network(evidence: dict) -> tuple[Path, ...]:
    supported = [edge for edge in evidence["edges"] if edge["supported"]]
    by_source = {
        source: [edge for edge in supported if edge["source"] == source]
        for source in SOURCE_ORDER
    }
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.35))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    source_y = dict(zip(SOURCE_ORDER, (0.83, 0.62, 0.41, 0.20)))
    target_y = dict(zip(TARGET_ORDER, (0.76, 0.50, 0.24)))

    for source in SOURCE_ORDER:
        y = source_y[source]
        color = SOURCE_COLORS[source]
        axis.add_patch(
            FancyBboxPatch(
                (0.025, y - 0.072),
                0.31,
                0.144,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                linewidth=1.1,
                edgecolor=color,
                facecolor=color + "12",
                zorder=3,
            )
        )
        axis.text(0.043, y + 0.027, SOURCE_LABELS[source], va="center", color=COLORS["dark"], fontsize=7.7)
        text = "   ".join(_effect_text(edge) for edge in by_source[source])
        if source == "spray_distance_mm":
            text += "   V/D not supported"
        axis.text(0.043, y - 0.040, text, va="center", color="#555555", fontsize=6.7)

    for target in TARGET_ORDER:
        y = target_y[target]
        axis.add_patch(
            FancyBboxPatch(
                (0.79, y - 0.057),
                0.185,
                0.114,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                linewidth=0.9,
                edgecolor="#777777",
                facecolor="#F4F4F4",
                zorder=3,
            )
        )
        axis.text(0.8825, y, TARGET_LABELS[target], ha="center", va="center", color=COLORS["dark"], fontsize=7.4)

    curvature = {
        "current_a": (-0.10, -0.07, -0.04),
        "argon_flow_scfh": (-0.03, 0.00, 0.03),
        "powder_feed_g_min": (0.04, 0.07, 0.10),
        "spray_distance_mm": (0.14,),
    }
    for source in SOURCE_ORDER:
        for index, edge in enumerate(by_source[source]):
            axis.add_patch(
                FancyArrowPatch(
                    (0.335, source_y[source]),
                    (0.783, target_y[edge["target"]]),
                    connectionstyle=f"arc3,rad={curvature[source][index]}",
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=1.35,
                    linestyle="--" if edge["claim_role"] == "exploratory" else "-",
                    color=SOURCE_COLORS[source],
                    alpha=0.82,
                    zorder=1,
                )
            )
    axis.text(
        0.025,
        0.035,
        "Solid: conditionally supported confirmatory total effect    Dashed: exploratory total effect",
        fontsize=6.7,
        color="#555555",
        va="center",
    )
    return export_figure(figure, OUTPUT_DIR, "fig01_a_group_supported_effect_network")


def plot_mean_effect_forest(evidence: dict, outcome: str) -> tuple[Path, ...]:
    edges = [
        next(edge for edge in evidence["edges"] if edge["source"] == source and edge["target"] == outcome)
        for source in SOURCE_ORDER
    ]
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.15))
    y = np.arange(len(edges))[::-1]
    for position, edge in zip(y, edges):
        mean = edge["mean_effect"]
        lower, upper = edge["mean_interval"]
        if edge["status"] == "conditional_admit":
            color, marker, fill = COLORS["blue"], "o", COLORS["blue"]
        elif edge["status"] == "exploratory_admit":
            color, marker, fill = COLORS["orange"], "D", COLORS["orange"]
        else:
            color, marker, fill = COLORS["gray"], "o", "white"
        axis.errorbar(
            mean,
            position,
            xerr=np.array([[mean - lower], [upper - mean]]),
            fmt=marker,
            markersize=5.0,
            markerfacecolor=fill,
            markeredgecolor=color,
            markeredgewidth=1.0,
            ecolor=color,
            elinewidth=1.25,
            capsize=3,
            capthick=1.0,
            zorder=3,
        )
    axis.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)
    axis.set_yticks(y, [FOREST_LABELS[edge["source"]] for edge in edges])
    axis.set_xlabel(OUTCOME_AXIS_LABELS[outcome])
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6, linestyle=":")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    legend = (
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["blue"], label="Confirmatory supported"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["orange"], label="Exploratory supported"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], label="Not supported"),
    )
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.31, right=0.98, bottom=0.22, top=0.82)
    return export_figure(figure, OUTPUT_DIR, OUTCOME_STEMS[outcome])


def plot_robustness(
    evidence: dict,
    stability: dict,
    sensitivity: dict,
    outcome: str,
) -> tuple[Path, ...]:
    primary = {
        (edge["estimand_id"], edge["target"]): edge
        for edge in evidence["edges"]
    }
    sequence = {
        (contrast["estimand_id"], name): values
        for contrast in stability["contrasts"]
        for name, values in contrast["outcomes"].items()
    }
    count = {
        (row["estimand_id"], row["outcome"]): row
        for row in sensitivity["comparisons"]
    }
    estimands = [
        next(
            edge["estimand_id"]
            for edge in evidence["edges"]
            if edge["source"] == source and edge["target"] == outcome
        )
        for source in SOURCE_ORDER
    ]
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.25))
    base_y = np.arange(len(SOURCE_ORDER))[::-1]
    offsets = {"primary": 0.18, "sequence": 0.0, "count": -0.18}
    styles = {
        "primary": (COLORS["blue"], "o", "Primary matched"),
        "sequence": (COLORS["orange"], "^", "Sequence adjusted"),
        "count": (COLORS["green"], "s", "142-run count sensitivity"),
    }
    for y, estimand in zip(base_y, estimands):
        key = (estimand, outcome)
        values = {
            "primary": primary[key]["mean_effect"],
            "sequence": sequence[key]["sequence_adjusted_mean_effect"],
            "count": count[key]["sensitivity_mean_effect"],
        }
        axis.plot(
            [min(values.values()), max(values.values())],
            [y, y],
            color="#BDBDBD",
            linewidth=0.9,
            zorder=1,
        )
        lower, upper = primary[key]["mean_interval"]
        axis.errorbar(
            values["primary"],
            y + offsets["primary"],
            xerr=np.array(
                [[values["primary"] - lower], [upper - values["primary"]]]
            ),
            fmt="none",
            ecolor=COLORS["blue"],
            elinewidth=0.9,
            capsize=2.5,
            alpha=0.55,
            zorder=2,
        )
        for analysis in ("primary", "sequence", "count"):
            color, marker, _ = styles[analysis]
            axis.plot(
                values[analysis],
                y + offsets[analysis],
                marker=marker,
                markersize=4.8,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="none",
                zorder=3,
            )
    axis.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    axis.set_yticks(base_y, [FOREST_LABELS[source] for source in SOURCE_ORDER])
    axis.set_xlabel(OUTCOME_AXIS_LABELS[outcome])
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6, linestyle=":")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    handles = tuple(
        Line2D(
            [0],
            [0],
            marker=styles[name][1],
            color="none",
            markerfacecolor=styles[name][0],
            markeredgecolor=styles[name][0],
            label=styles[name][2],
        )
        for name in ("primary", "sequence", "count")
    )
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.31, right=0.98, bottom=0.22, top=0.82)
    return export_figure(figure, OUTPUT_DIR, ROBUSTNESS_STEMS[outcome])


def main() -> None:
    apply_journal_style()
    evidence = _load_evidence()
    stability, sensitivity = _load_robustness_data()
    plot_supported_network(evidence)
    for outcome in TARGET_ORDER:
        plot_mean_effect_forest(evidence, outcome)
        plot_robustness(evidence, stability, sensitivity, outcome)
    plt.close("all")


if __name__ == "__main__":
    main()
