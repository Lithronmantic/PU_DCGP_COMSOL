
from __future__ import annotations

import json
from pathlib import Path

from experiments.pu_dcgp import (
    ManifestDataSource,
    PUDCGPConfig,
    a_group_doe_estimands,
    estimate_module_consistency,
    estimate_sequence_adjusted_effects,
)

from .a_group_network_point_reanalysis import _assert_corrected_mapping


OUTPUT_DIR = Path(__file__).with_name("data")
JSON_PATH = OUTPUT_DIR / "a_group_network_stability_reanalysis.json"
REPORT_PATH = OUTPUT_DIR / "A_GROUP_NETWORK_STABILITY_REANALYSIS.md"


def build_stability_reanalysis() -> dict:
    runs = ManifestDataSource(groups=("A",)).load()
    _assert_corrected_mapping(runs)
    config = PUDCGPConfig()
    contrasts = []
    for estimand in a_group_doe_estimands():
        sequence = estimate_sequence_adjusted_effects(runs, config, estimand)
        modules = estimate_module_consistency(runs, config, estimand)
        outcome_results = {}
        for outcome in config.outcome_columns:
            sequence_outcome = sequence.outcome_effects[outcome]
            module_outcome = modules.outcome_consistency[outcome]
            outcome_results[outcome] = {
                "unadjusted_mean_effect": sequence_outcome.unadjusted_mean_effect,
                "sequence_adjusted_mean_effect": sequence_outcome.adjusted_mean_effect,
                "sequence_slope_per_10_runs": sequence_outcome.sequence_slope_per_10_runs,
                "sequence_sign_retained": sequence_outcome.mean_sign_retained,
                "sequence_relative_magnitude": sequence_outcome.relative_magnitude,
                "sequence_adjusted_wasserstein_norm": sequence_outcome.adjusted_wasserstein_norm,
                "module_balanced_mean_effect": module_outcome.module_balanced_mean_effect,
                "module_mean_effects": module_outcome.module_mean_effects,
                "module_mean_direction_consistent": module_outcome.direction_consistent,
                "module_quantile_direction_consistent": module_outcome.quantile_direction_consistent,
                "module_balanced_sign_retained": module_outcome.module_balanced_sign_retained,
                "module_absolute_magnitude_ratio": module_outcome.absolute_magnitude_ratio,
            }
        contrasts.append(
            {
                "estimand_id": estimand.estimand_id,
                "claim_role": estimand.claim_role,
                "matched_strata": sequence.matched_strata,
                "matched_runs": sequence.matched_runs,
                "treatment_sequence_correlation": sequence.treatment_sequence_correlation,
                "design_condition_number": sequence.design_condition_number,
                "modules": [
                    {
                        "module_code": module.module_code,
                        "matched_strata": module.matched_strata,
                        "reference_runs": module.reference_runs,
                        "intervention_runs": module.intervention_runs,
                    }
                    for module in modules.modules
                ],
                "all_modules_have_multiple_strata": modules.all_modules_have_multiple_strata,
                "outcomes": outcome_results,
            }
        )
    return {
        "schema": "pu_dcgp_v26_a_group_network_stability_reanalysis_v1",
        "scope": "A group, all 150 corrected-mapping runs",
        "sequence_model": "matched-stratum fixed effects plus linear execution-order term",
        "module_model": "effects recomputed separately within each supporting DOE module",
        "contrasts": contrasts,
    }


def render_report(result: dict) -> str:
    lines = [
        "# A-group pure-data network stability reanalysis",
        "",
        "## Sequence identifiability",
        "",
        "The sequence sensitivity fits matched-stratum fixed effects with one linear execution-order term. "
        "The treatment-sequence correlation and design condition number are reported because a sign-retaining coefficient is not persuasive when treatment and order are nearly inseparable.",
        "",
        "| Contrast | Matched runs | Treatment-sequence correlation | Condition number |",
        "|---|---:|---:|---:|",
    ]
    for contrast in result["contrasts"]:
        lines.append(
            f"| {contrast['estimand_id']} | {contrast['matched_runs']} | "
            f"{contrast['treatment_sequence_correlation']:+.4f} | "
            f"{contrast['design_condition_number']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Outcome stability",
            "",
            "| Contrast | Outcome | Unadjusted | Sequence-adjusted | Sign retained | Sequence slope / 10 runs | Module effects | Module mean direction | Module quantile direction |",
            "|---|---|---:|---:|---|---:|---|---|---|",
        ]
    )
    for contrast in result["contrasts"]:
        for outcome, values in contrast["outcomes"].items():
            modules = ", ".join(
                f"{name}:{effect:+.4f}"
                for name, effect in values["module_mean_effects"].items()
            )
            lines.append(
                f"| {contrast['estimand_id']} | {outcome} | "
                f"{values['unadjusted_mean_effect']:+.4f} | "
                f"{values['sequence_adjusted_mean_effect']:+.4f} | "
                f"{'Yes' if values['sequence_sign_retained'] else 'No'} | "
                f"{values['sequence_slope_per_10_runs']:+.4f} | {modules} | "
                f"{'Yes' if values['module_mean_direction_consistent'] else 'No'} | "
                f"{'Yes' if values['module_quantile_direction_consistent'] else 'No'} |"
            )
    sequence_stable = sum(
        int(values["sequence_sign_retained"])
        for contrast in result["contrasts"]
        for values in contrast["outcomes"].values()
    )
    module_mean_stable = sum(
        int(values["module_mean_direction_consistent"])
        for contrast in result["contrasts"]
        for values in contrast["outcomes"].values()
    )
    module_quantile_stable = sum(
        int(values["module_quantile_direction_consistent"])
        for contrast in result["contrasts"]
        for values in contrast["outcomes"].values()
    )
    lines.extend(
        [
            "",
            "## Layer summary",
            "",
            f"Sequence-adjusted sign retained: {sequence_stable}/12. "
            f"Module mean direction consistent: {module_mean_stable}/12. "
            f"Module full-quantile direction consistent: {module_quantile_stable}/12.",
            "",
            "These checks still do not provide confidence intervals; hierarchical bootstrap is the next layer.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = build_stability_reanalysis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
