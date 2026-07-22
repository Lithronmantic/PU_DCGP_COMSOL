"""Outcome-blind low-particle-count sensitivity for corrected A-group edges."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.pu_dcgp import (
    ManifestDataSource,
    PUDCGPConfig,
    evaluate_all_effect_admissions,
    subset_run_batch,
)

from .a_group_network_point_reanalysis import _assert_corrected_mapping


OUTPUT_DIR = Path(__file__).with_name("data")
QC_FLAGS_PATH = OUTPUT_DIR / "a_group_qc_run_flags.csv"
PRIMARY_PATH = OUTPUT_DIR / "a_group_network_inference_reanalysis.json"
JSON_PATH = OUTPUT_DIR / "a_group_network_count_sensitivity.json"
REPORT_PATH = OUTPUT_DIR / "A_GROUP_NETWORK_COUNT_SENSITIVITY.md"


def _sensitivity_run_ids() -> set[str]:
    with QC_FLAGS_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 150:
        raise AssertionError("QC flag table must contain all 150 A-group runs")
    return {
        row["run_id"]
        for row in rows
        if row["count_support_sensitivity_include"].lower() == "true"
    }


def build_count_sensitivity() -> dict:
    runs = ManifestDataSource(groups=("A",)).load()
    _assert_corrected_mapping(runs)
    included_ids = _sensitivity_run_ids()
    selected = [index for index, run_id in enumerate(runs.run_ids) if run_id in included_ids]
    sensitivity_runs = subset_run_batch(runs, selected)
    if len(sensitivity_runs.run_ids) != 142:
        raise AssertionError("Frozen particle-count sensitivity must retain 142 runs")

    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    primary_by_key = {
        (row["estimand_id"], row["outcome"]): row
        for row in primary["decisions"]
    }
    decisions = evaluate_all_effect_admissions(sensitivity_runs, PUDCGPConfig())
    comparisons = []
    for decision in decisions:
        key = (decision.estimand.estimand_id, decision.outcome)
        baseline = primary_by_key[key]
        effect_ratio = (
            decision.point_mean_effect / baseline["point_mean_effect"]
            if baseline["point_mean_effect"] != 0
            else None
        )
        primary_supported = baseline["status"] in {
            "admit",
            "conditional_admit",
            "exploratory_admit",
        }
        sensitivity_supported = decision.status in {
            "admit",
            "conditional_admit",
            "exploratory_admit",
        }
        comparisons.append(
            {
                "estimand_id": key[0],
                "outcome": key[1],
                "primary_status": baseline["status"],
                "sensitivity_status": decision.status,
                "status_retained": baseline["status"] == decision.status,
                "primary_supported": primary_supported,
                "sensitivity_supported": sensitivity_supported,
                "support_decision_retained": primary_supported == sensitivity_supported,
                "primary_mean_effect": baseline["point_mean_effect"],
                "sensitivity_mean_effect": decision.point_mean_effect,
                "sign_retained": baseline["point_mean_effect"] * decision.point_mean_effect > 0,
                "effect_ratio": effect_ratio,
                "sensitivity_mean_lower_bound": decision.mean_lower_bound,
                "sensitivity_mean_upper_bound": decision.mean_upper_bound,
                "sensitivity_failed_gates": list(decision.failed_gates),
            }
        )
    excluded_ids = sorted(set(runs.run_ids) - included_ids)
    return {
        "schema": "pu_dcgp_v26_a_group_network_count_sensitivity_v1",
        "selection_rule": "pre-frozen outcome-blind jointly-valid particle count >= 20",
        "primary_run_count": len(runs.run_ids),
        "sensitivity_run_count": len(sensitivity_runs.run_ids),
        "excluded_run_ids": excluded_ids,
        "all_signs_retained": all(row["sign_retained"] for row in comparisons),
        "all_statuses_retained": all(row["status_retained"] for row in comparisons),
        "all_support_decisions_retained": all(
            row["support_decision_retained"] for row in comparisons
        ),
        "comparisons": comparisons,
    }


def render_report(result: dict) -> str:
    lines = [
        "# A-group low-particle-count sensitivity",
        "",
        f"The primary analysis contains {result['primary_run_count']} runs. The outcome-blind sensitivity retains "
        f"{result['sensitivity_run_count']} runs using the pre-frozen rule: {result['selection_rule']}.",
        "",
        f"Excluded only in sensitivity: {', '.join(result['excluded_run_ids'])}.",
        "",
        "| Contrast | Outcome | Primary effect | Sensitivity effect | Ratio | Sign retained | Primary status | Sensitivity status |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in result["comparisons"]:
        lines.append(
            f"| {row['estimand_id']} | {row['outcome']} | "
            f"{row['primary_mean_effect']:+.4f} | {row['sensitivity_mean_effect']:+.4f} | "
            f"{row['effect_ratio']:+.3f} | {'Yes' if row['sign_retained'] else 'No'} | "
            f"{row['primary_status']} | {row['sensitivity_status']} |"
        )
    lines.extend(
        [
            "",
            f"All 12 signs retained: {'Yes' if result['all_signs_retained'] else 'No'}. "
            f"All 12 supported/not-supported decisions retained: "
            f"{'Yes' if result['all_support_decisions_retained'] else 'No'}. "
            "Some exact status labels can strengthen from conditional to eligible because removing a pre-flagged run changes the sequence-support classification; the primary 150-run label remains authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = build_count_sensitivity()
    JSON_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
