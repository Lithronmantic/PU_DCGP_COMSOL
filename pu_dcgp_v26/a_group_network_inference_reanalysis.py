"""Finite-sample uncertainty and frozen edge decisions for corrected A data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from experiments.pu_dcgp import (
    ManifestDataSource,
    PUDCGPConfig,
    evaluate_all_effect_admissions,
)

from .a_group_network_point_reanalysis import _assert_corrected_mapping


OUTPUT_DIR = Path(__file__).with_name("data")
JSON_PATH = OUTPUT_DIR / "a_group_network_inference_reanalysis.json"
REPORT_PATH = OUTPUT_DIR / "A_GROUP_NETWORK_INFERENCE_REANALYSIS.md"


def build_inference_reanalysis() -> dict:
    runs = ManifestDataSource(groups=("A",)).load()
    _assert_corrected_mapping(runs)
    config = PUDCGPConfig()
    decisions = evaluate_all_effect_admissions(runs, config)
    rows = []
    for decision in decisions:
        rows.append(
            {
                "estimand_id": decision.estimand.estimand_id,
                "treatment_name": decision.estimand.treatment_name,
                "claim_role": decision.estimand.claim_role,
                "outcome": decision.outcome,
                "status": decision.status,
                "support_level": decision.support_level,
                "conditional_reasons": list(decision.conditional_reasons),
                "point_mean_effect": decision.point_mean_effect,
                "mean_lower_bound": decision.mean_lower_bound,
                "mean_upper_bound": decision.mean_upper_bound,
                "simultaneous_lower_min": decision.simultaneous_lower_min,
                "simultaneous_upper_max": decision.simultaneous_upper_max,
                "evidence": decision.evidence.as_dict(),
                "passed_gates": list(decision.passed_gates),
                "failed_gates": list(decision.failed_gates),
            }
        )
    counts = Counter(row["status"] for row in rows)
    return {
        "schema": "pu_dcgp_v26_a_group_network_inference_reanalysis_v1",
        "scope": "A group, all 150 corrected-mapping runs",
        "bootstrap_replicates": config.effect_bootstrap_replicates,
        "interval_level": config.effect_interval_level,
        "status_counts": dict(sorted(counts.items())),
        "decisions": rows,
    }


def render_report(result: dict) -> str:
    lines = [
        "# A-group pure-data network inference reanalysis",
        "",
        f"Uncertainty uses {result['bootstrap_replicates']} hierarchical bootstrap replicates "
        f"at the {100 * result['interval_level']:.0f}% level. Matched strata, runs within arms, "
        "and particles within runs are resampled.",
        "",
        "| Contrast | Outcome | Mean effect | 95% mean interval | Simultaneous band envelope | Status | Failed gates |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in result["decisions"]:
        failed = ", ".join(row["failed_gates"]) if row["failed_gates"] else "None"
        lines.append(
            f"| {row['estimand_id']} | {row['outcome']} | "
            f"{row['point_mean_effect']:+.4f} | "
            f"[{row['mean_lower_bound']:+.4f}, {row['mean_upper_bound']:+.4f}] | "
            f"[{row['simultaneous_lower_min']:+.4f}, {row['simultaneous_upper_max']:+.4f}] | "
            f"{row['status']} | {failed} |"
        )
    lines.extend(
        [
            "",
            "## Decision meaning",
            "",
            "`conditional_admit` means every frozen numerical gate passed, while the edge remains conditional on the executed ordered DOE design. "
            "`exploratory_admit` is an exploratory spray-distance edge passing all gates. "
            "`abstain` means at least one frozen gate failed and the edge is not drawn as supported.",
            "",
            f"Status counts: {json.dumps(result['status_counts'], ensure_ascii=False, sort_keys=True)}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = build_inference_reanalysis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
