
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pu_dcgp import (
    ManifestDataSource,
    PUDCGPConfig,
    a_group_doe_estimands,
    audit_contrast_support,
    estimate_matched_distribution_effects,
)


OUTPUT_DIR = Path(__file__).with_name("data")
JSON_PATH = OUTPUT_DIR / "a_group_network_point_reanalysis.json"
REPORT_PATH = OUTPUT_DIR / "A_GROUP_NETWORK_POINT_REANALYSIS.md"

OUTCOME_LABELS = {
    "temperature_c": ("Temperature", "deg C"),
    "velocity_m_s": ("Velocity", "m/s"),
    "particle_diameter_um": ("Particle diameter", "um"),
}


def _assert_corrected_mapping(runs) -> None:
    expected = np.arange(1, 151, dtype=float)
    observed_ids = np.asarray([int(run_id.rsplit("R", 1)[1]) for run_id in runs.run_ids])
    execution = runs.context_values[:, runs.context_names.index("execution_order")]
    if len(runs.run_ids) != 150:
        raise AssertionError("The primary A-group analysis requires all 150 runs")
    np.testing.assert_array_equal(observed_ids, expected.astype(int))
    np.testing.assert_array_equal(execution, expected)


def build_point_reanalysis() -> dict:
    source = ManifestDataSource(groups=("A",))
    runs = source.load()
    _assert_corrected_mapping(runs)
    config = PUDCGPConfig()

    contrasts = []
    for estimand in a_group_doe_estimands():
        support = audit_contrast_support(runs, estimand)
        effect = estimate_matched_distribution_effects(runs, config, estimand)
        outcomes = {}
        for outcome, result in effect.aggregate_effects.items():
            outcomes[outcome] = {
                "mean_difference": result.mean_difference,
                "median_quantile_difference": result.median_quantile_difference,
                "wasserstein_norm": result.wasserstein_norm,
                "mean_stratum_wasserstein": result.mean_stratum_wasserstein,
                "leave_one_out_min": result.leave_one_out_min,
                "leave_one_out_max": result.leave_one_out_max,
                "leave_one_out_sign_stable": result.leave_one_out_sign_stable,
                "quantile_grid": effect.quantile_grid.tolist(),
                "quantile_difference": result.quantile_difference.tolist(),
            }
        contrasts.append(
            {
                "estimand_id": estimand.estimand_id,
                "treatment_name": estimand.treatment_name,
                "reference_value": estimand.reference_value,
                "intervention_value": estimand.intervention_value,
                "claim_role": estimand.claim_role,
                "matched_strata": len(support.strata),
                "matched_reference_runs": support.reference_runs,
                "matched_intervention_runs": support.intervention_runs,
                "modules_within_comparison": list(support.modules_within_comparison),
                "median_absolute_sequence_gap": support.median_absolute_sequence_gap,
                "normalized_median_sequence_gap": support.normalized_median_sequence_gap,
                "positive_sequence_gaps": support.positive_sequence_gaps,
                "negative_sequence_gaps": support.negative_sequence_gaps,
                "zero_sequence_gaps": support.zero_sequence_gaps,
                "support_level": support.support_level,
                "support_reasons": list(support.support_reasons),
                "outcomes": outcomes,
            }
        )

    manifest_bytes = source.manifest_path.read_bytes()
    return {
        "schema": "pu_dcgp_v26_a_group_network_point_reanalysis_v1",
        "analysis_scope": {
            "group": "A",
            "run_count": len(runs.run_ids),
            "unique_setting_count": int(np.unique(runs.treatment_values, axis=0).shape[0]),
            "particle_count": int(
                sum(len(sample) for sample in runs.particle_samples["temperature_c"])
            ),
            "mapping_rule": "NNN.csv = UCE-RNNN = design row NNN = execution order NNN",
            "outcome_blind_run_exclusions": 0,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "estimator": {
            "matching": "exact match on the other three varied DOE factors",
            "within_stratum": "run-balanced intervention minus reference",
            "aggregation": "equal weight across matched strata",
            "interpretation": "point-estimate layer only; not yet an admitted causal edge",
        },
        "contrasts": contrasts,
    }


def _format_signed(value: float) -> str:
    return f"{value:+.4f}"


def render_report(result: dict) -> str:
    scope = result["analysis_scope"]
    lines = [
        "# A-group pure-data causal-network point reanalysis",
        "",
        "## Frozen scope",
        "",
        f"- Mapping: `{scope['mapping_rule']}`.",
        f"- Data: {scope['run_count']} runs, {scope['unique_setting_count']} unique DOE settings, "
        f"{scope['particle_count']} jointly valid DPV particle rows.",
        "- Primary analysis retains every A-group run. No run was removed using an outcome or model result.",
        "- Each effect exactly matches the other three varied DOE factors, balances runs within a stratum, "
        "then gives each matched stratum equal weight.",
        "",
        "## Design support",
        "",
        "| Contrast | Matched strata | Matched runs (reference/intervention) | Shared DOE modules | Median sequence gap | Gap directions (+/-/0) | Support |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for contrast in result["contrasts"]:
        lines.append(
            f"| {contrast['estimand_id']} | {contrast['matched_strata']} | "
            f"{contrast['matched_reference_runs']}/{contrast['matched_intervention_runs']} | "
            f"{', '.join(contrast['modules_within_comparison'])} | "
            f"{contrast['median_absolute_sequence_gap']:.1f} | "
            f"{contrast['positive_sequence_gaps']}/{contrast['negative_sequence_gaps']}/{contrast['zero_sequence_gaps']} | "
            f"{contrast['support_level']} |"
        )
    lines.extend(
        [
            "",
            "## Matched point effects",
            "",
            "| Contrast | Outcome | Mean difference | Median-quantile difference | Wasserstein norm | Leave-one-stratum-out range | Direction stable |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for contrast in result["contrasts"]:
        for outcome, values in contrast["outcomes"].items():
            label, unit = OUTCOME_LABELS[outcome]
            lines.append(
                f"| {contrast['estimand_id']} | {label} | "
                f"{_format_signed(values['mean_difference'])} {unit} | "
                f"{_format_signed(values['median_quantile_difference'])} {unit} | "
                f"{values['wasserstein_norm']:.4f} {unit} | "
                f"{_format_signed(values['leave_one_out_min'])} to "
                f"{_format_signed(values['leave_one_out_max'])} {unit} | "
                f"{'Yes' if values['leave_one_out_sign_stable'] else 'No'} |"
            )
    stable_count = sum(
        int(values["leave_one_out_sign_stable"])
        for contrast in result["contrasts"]
        for values in contrast["outcomes"].values()
    )
    lines.extend(
        [
            "",
            "## Boundary of this layer",
            "",
            f"{stable_count} of 12 point-effect directions survive every leave-one-stratum-out deletion. "
            "This is evidence of descriptive robustness, not yet a causal conclusion. "
            "Sequence-adjusted, module-consistency, interval and simultaneous-band checks are required next.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = build_point_reanalysis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
