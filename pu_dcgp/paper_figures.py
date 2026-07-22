"""Generate manuscript figures only from frozen benchmark and A-group artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
FIGURE_DIR = PACKAGE_DIR / "figures"


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    filename: str
    sources: tuple[str, ...]
    description: str


FIGURE_SPECS = (
    FigureSpec(
        "figure_1",
        "figure_1_workflow.png",
        ("PAPER_FRAMEWORK.md", "admission_gate.py"),
        "Particle records, distribution representation, contrast, and reporting gates.",
    ),
    FigureSpec(
        "figure_2",
        "figure_2_known_truth_scenarios.png",
        ("SYNTHETIC_BENCHMARK_CONTRACT.md", "benchmark_generator.py"),
        "Five known-truth scenarios and their intended method or gate stress.",
    ),
    FigureSpec(
        "figure_3",
        "figure_3_h1_shape_recovery.png",
        ("data/formal_benchmark.summary.json",),
        "H1 shape-effect IRMSE by scenario, sample size, and estimator.",
    ),
    FigureSpec(
        "figure_4",
        "figure_4_calibration_and_reporting.png",
        ("data/formal_benchmark.summary.json",),
        "H2 coverage and H3-H4 reporting safety-power summary.",
    ),
    FigureSpec(
        "figure_5",
        "figure_5_powder_diameter_effect.png",
        ("data/DISTRIBUTIONAL_EFFECT_UNCERTAINTY_AUDIT.md",),
        "A-group powder-feed effect on the detected-diameter quantile function.",
    ),
)


Builder = Callable[[Path], Path]


def figure_specs() -> tuple[FigureSpec, ...]:
    return FIGURE_SPECS


def _rounded_box(axis, x: float, y: float, width: float, height: float, text: str, facecolor: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor="#4B5563",
        linewidth=1.1,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.4)


def build_figure_1(output_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, axis = plt.subplots(figsize=(10.2, 3.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    y = 0.47
    height = 0.30
    width = 0.135
    positions = (0.015, 0.18, 0.345, 0.51, 0.675, 0.84)
    labels = (
        "DPV particle records\nper A-group run",
        "Empirical quantiles\n19 coordinates",
        "Quantile scores + GP\n(benchmark)\nor direct matched curves (A)",
        "Matched effect\nmean interval +\nsimultaneous band",
        "Seven fixed gates\nsupport, stability,\nand uncertainty",
        "Reporting decision\nconditional admit\nor abstain",
    )
    colors = ("#E5E7EB", "#DCECF7", "#DCECF7", "#E8F3ED", "#FCE7C7", "#E8F3ED")

    for x, label, color in zip(positions, labels, colors):
        _rounded_box(axis, x, y, width, height, label, color)
    for left, right in zip(positions[:-1], positions[1:]):
        axis.add_patch(
            FancyArrowPatch(
                (left + width + 0.004, y + height / 2),
                (right - 0.004, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color="#4B5563",
            )
        )

    annotation_x = 0.845
    annotation_y = 0.10
    annotation_w = 0.125
    annotation_h = 0.20
    _rounded_box(
        axis,
        annotation_x,
        annotation_y,
        annotation_w,
        annotation_h,
        "Physical consistency\nread-only annotation",
        "#F3F4F6",
    )
    axis.add_patch(
        FancyArrowPatch(
            (positions[-1] + width / 2, y - 0.01),
            (annotation_x + annotation_w / 2, annotation_y + annotation_h + 0.01),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            linestyle="--",
            color="#6B7280",
        )
    )
    axis.text(
        0.505,
        0.89,
        "Estimate first; authorize reporting only after prespecified evidence checks",
        ha="center",
        va="center",
        fontsize=10.2,
        weight="bold",
    )
    axis.text(
        0.64,
        0.17,
        "Annotation cannot change the estimate, band, or decision",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#4B5563",
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "figure_1_workflow.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def build_figure_2(output_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, axis = plt.subplots(figsize=(9.3, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.145, 0.94, "Known-truth scenario", ha="center", fontsize=10, weight="bold")
    axis.text(0.485, 0.94, "Controlled design or data change", ha="center", fontsize=10, weight="bold")
    axis.text(0.835, 0.94, "Prespecified evaluation target", ha="center", fontsize=10, weight="bold")

    rows = (
        (
            "Identified, balanced\nparticle counts",
            "Complete matched support; randomized order;\n80 detected particles per run",
            "H1 shape recovery\nH2 balanced non-inferiority\nH4 retained power at n=144",
            "#DCECF7",
        ),
        (
            "Identified, heterogeneous\nparticle counts",
            "Same design and truth; particle counts vary\nfrom 20 to 240 independently of outcomes",
            "H1 shape recovery\nH2 calibration improvement\nH4 retained power at n=144",
            "#DCECF7",
        ),
        (
            "Sequence-aligned drift",
            "High-current anchors occur later; opposing\ntemperature drift reverses the raw direction",
            "H3 sequence-sensitivity gate\nwithholds current-temperature",
            "#FCE7C7",
        ),
        (
            "Module sign reversal",
            "Powder-diameter effects are pointwise\nopposites in DOE-1 and DOE-2",
            "H3 module-direction gates\nwithhold powder-diameter",
            "#FCE7C7",
        ),
        (
            "Insufficient overlap",
            "Argon intervention retains four exact strata,\nbelow the frozen minimum of five",
            "H3 structural-support gate\nwithholds argon-velocity",
            "#FCE7C7",
        ),
    )
    y_positions = (0.78, 0.61, 0.44, 0.27, 0.10)
    for y, (scenario, change, target, color) in zip(y_positions, rows):
        _rounded_box(axis, 0.015, y, 0.26, 0.12, scenario, color)
        _rounded_box(axis, 0.31, y, 0.35, 0.12, change, "#F3F4F6")
        _rounded_box(axis, 0.70, y, 0.285, 0.12, target, color)
        axis.add_patch(
            FancyArrowPatch(
                (0.278, y + 0.06),
                (0.307, y + 0.06),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="#6B7280",
            )
        )
        axis.add_patch(
            FancyArrowPatch(
                (0.663, y + 0.06),
                (0.697, y + 0.06),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="#6B7280",
            )
        )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "figure_2_known_truth_scenarios.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def load_formal_summary() -> dict:
    path = DATA_DIR / "formal_benchmark.summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload["formal_complete"]:
        raise ValueError("formal benchmark is not complete")
    return payload


def build_figure_3(output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    summary = load_formal_summary()
    scenarios = (
        ("identified_balanced_particles", "Balanced particle counts"),
        ("identified_heterogeneous_particles", "Heterogeneous particle counts"),
    )
    methods = (
        ("mean_gp", "Mean GP", "#6B7280", "o"),
        ("distribution_gp_no_pu", "Distribution GP (no PU)", "#0072B2", "s"),
        ("pu_dcgp", "PU-DCGP", "#D55E00", "^"),
    )
    rows = {
        (row["scenario_id"], row["sample_size"], row["method_name"]): row
        for row in summary["aggregates"]
    }
    sample_sizes = (48, 96, 144)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharey=True)
    for axis, (scenario_id, title) in zip(axes, scenarios):
        for method_name, label, color, marker in methods:
            values = [
                rows[(scenario_id, size, method_name)]["median_shape_normalized_irmse"]
                for size in sample_sizes
            ]
            axis.plot(
                sample_sizes,
                values,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5.5,
                label=label,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Runs per dataset")
        axis.set_xticks(sample_sizes)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Median shape-effect normalized IRMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.10, right=0.98, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "figure_3_h1_shape_recovery.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def build_figure_4(output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    summary = load_formal_summary()
    rows = {
        (row["scenario_id"], row["sample_size"], row["method_name"]): row
        for row in summary["aggregates"]
    }
    hypotheses = {row["hypothesis_id"]: row for row in summary["hypotheses"]}
    sample_sizes = (48, 96, 144)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35))

    coverage_specs = (
        ("identified_balanced_particles", "distribution_gp_no_pu", "Balanced, no PU", "#0072B2", "-", "s"),
        ("identified_balanced_particles", "pu_dcgp", "Balanced, PU", "#D55E00", "-", "^"),
        ("identified_heterogeneous_particles", "distribution_gp_no_pu", "Heterogeneous, no PU", "#0072B2", "--", "s"),
        ("identified_heterogeneous_particles", "pu_dcgp", "Heterogeneous, PU", "#D55E00", "--", "^"),
    )
    for scenario, method, label, color, linestyle, marker in coverage_specs:
        values = [
            rows[(scenario, size, method)]["simultaneous_coverage_rate"]
            for size in sample_sizes
        ]
        axes[0].plot(
            sample_sizes,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.6,
            markersize=4.8,
        )
    axes[0].axhline(0.95, color="#6B7280", linestyle=":", linewidth=1.2)
    axes[0].text(143, 0.953, "Nominal 0.95", ha="right", va="bottom", fontsize=7.5)
    axes[0].set_ylim(0.80, 0.975)
    axes[0].set_ylabel("Simultaneous coverage")
    axes[0].set_title("(a) H2 calibration", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7.4, loc="lower right")

    for method, label, color, marker in (
        ("pu_dcgp", "Ungated PU", "#6B7280", "o"),
        ("support_gated_pu_dcgp", "Support-gated PU", "#009E73", "D"),
    ):
        values = []
        for size in sample_sizes:
            scenario_values = [
                rows[(scenario, size, method)]["target_unsupported_admission_rate"]
                for scenario in (
                    "sequence_aligned_drift",
                    "module_sign_reversal",
                    "insufficient_overlap",
                )
            ]
            values.append(sum(scenario_values) / len(scenario_values))
        axes[1].plot(
            sample_sizes,
            values,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5.0,
        )
    axes[1].axhline(0.05, color="#6B7280", linestyle=":", linewidth=1.2)
    axes[1].text(143, 0.057, "Maximum 0.05", ha="right", va="bottom", fontsize=7.5)
    axes[1].set_ylim(-0.02, 0.66)
    axes[1].set_ylabel("Unsupported-target admission")
    axes[1].set_title("(b) H3 reporting safety", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.8, loc="upper left")

    h4 = hypotheses["H4"]["evidence"]
    values = (
        h4["supported_active_admission_power"],
        h4["supported_null_false_admission"],
    )
    axes[2].bar((0, 1), values, width=0.58, color=("#D55E00", "#009E73"))
    axes[2].hlines(0.80, -0.36, 0.36, color="#111827", linestyle="--", linewidth=1.2)
    axes[2].hlines(0.05, 0.64, 1.36, color="#111827", linestyle="--", linewidth=1.2)
    for x, value in enumerate(values):
        axes[2].text(x, value + 0.025, f"{value:.4f}", ha="center", fontsize=8.5)
    axes[2].text(0, 0.815, "minimum 0.80", ha="center", va="bottom", fontsize=7.5)
    axes[2].text(1, 0.065, "maximum 0.05", ha="center", va="bottom", fontsize=7.5)
    axes[2].set_xticks((0, 1), ("Active\nadmission", "Null false\nadmission"))
    axes[2].set_ylim(0, 0.90)
    axes[2].set_ylabel("Admission rate")
    axes[2].set_title("(c) H4 retained power", fontsize=10)

    for axis in axes:
        axis.set_xticks(sample_sizes) if axis is not axes[2] else None
        axis.set_xlabel("Runs per dataset") if axis is not axes[2] else None
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(top=0.88, bottom=0.19, left=0.07, right=0.99, wspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "figure_4_calibration_and_reporting.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def load_powder_diameter_effect() -> list[dict[str, float]]:
    path = DATA_DIR / "DISTRIBUTIONAL_EFFECT_UNCERTAINTY_AUDIT.md"
    text = path.read_text(encoding="utf-8")
    section = text.split("## Powder-feed effect on particle diameter", 1)[1]
    section = section.split("## Final real-data boundary", 1)[0]
    rows: list[dict[str, float]] = []
    pattern = re.compile(
        r"^\|\s*(0\.\d+)\s*\|\s*([-+0-9.]+)\s*\|\s*"
        r"([-+0-9.]+)\s+to\s+([-+0-9.]+)\s*\|\s*"
        r"([-+0-9.]+)\s+to\s+([-+0-9.]+)\s*\|\s*"
        r"(0\.\d+)\s*\|$"
    )
    for line in section.splitlines():
        match = pattern.match(line)
        if match:
            values = [float(value) for value in match.groups()]
            rows.append(
                {
                    "quantile": values[0],
                    "effect": values[1],
                    "point_low": values[2],
                    "point_high": values[3],
                    "sim_low": values[4],
                    "sim_high": values[5],
                    "same_sign_probability": values[6],
                }
            )
    if len(rows) != 19:
        raise ValueError(f"expected 19 powder-diameter quantile rows, found {len(rows)}")
    return rows


def build_figure_5(output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = load_powder_diameter_effect()
    quantiles = [row["quantile"] for row in rows]
    effects = [row["effect"] for row in rows]
    point_low = [row["point_low"] for row in rows]
    point_high = [row["point_high"] for row in rows]
    sim_low = [row["sim_low"] for row in rows]
    sim_high = [row["sim_high"] for row in rows]

    fig, axis = plt.subplots(figsize=(6.6, 3.7))
    axis.fill_between(
        quantiles,
        sim_low,
        sim_high,
        color="#E69F00",
        alpha=0.18,
        linewidth=0,
        label="Simultaneous 95% band",
    )
    axis.fill_between(
        quantiles,
        point_low,
        point_high,
        color="#56B4E9",
        alpha=0.35,
        linewidth=0,
        label="Pointwise 95% interval",
    )
    axis.plot(
        quantiles,
        effects,
        color="#111827",
        linewidth=2.0,
        marker="o",
        markersize=3.7,
        label="Matched quantile effect",
    )
    axis.axhline(0, color="#6B7280", linestyle="--", linewidth=1.1)
    axis.set_xlim(0.04, 0.96)
    axis.set_xticks((0.05, 0.25, 0.50, 0.75, 0.95))
    axis.set_xlabel("Particle-diameter quantile")
    axis.set_ylabel("Detected-diameter change (um)")
    axis.set_title("Powder feed: 30 minus 10 g/min", fontsize=10.5)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.13, right=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "figure_5_powder_diameter_effect.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def build_registry() -> dict[str, Builder]:
    """Return implemented builders; filled one verified figure layer at a time."""
    return {
        "figure_1": build_figure_1,
        "figure_2": build_figure_2,
        "figure_3": build_figure_3,
        "figure_4": build_figure_4,
        "figure_5": build_figure_5,
    }


def generate_all(output_dir: Path = FIGURE_DIR) -> tuple[Path, ...]:
    registry = build_registry()
    return tuple(registry[spec.figure_id](output_dir) for spec in FIGURE_SPECS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the frozen figure plan.")
    parser.add_argument("--all", action="store_true", help="Generate all registered manuscript figures.")
    parser.add_argument("--figure", choices=[spec.figure_id for spec in FIGURE_SPECS])
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.all:
        for output in generate_all(args.output_dir):
            print(output)
        return
    if args.list or args.figure is None:
        for spec in FIGURE_SPECS:
            print(f"{spec.figure_id}\t{spec.filename}\t{', '.join(spec.sources)}")
        return

    registry = build_registry()
    if args.figure not in registry:
        raise SystemExit(f"{args.figure} is planned but not implemented yet")
    output = registry[args.figure](args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
