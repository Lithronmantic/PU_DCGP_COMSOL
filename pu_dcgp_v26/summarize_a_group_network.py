
from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
INFERENCE_PATH = DATA_DIR / "a_group_network_inference_reanalysis.json"
STABILITY_PATH = DATA_DIR / "a_group_network_stability_reanalysis.json"
SENSITIVITY_PATH = DATA_DIR / "a_group_network_count_sensitivity.json"
JSON_PATH = DATA_DIR / "a_group_causal_network_evidence.json"
REPORT_PATH = DATA_DIR / "A_GROUP_NETWORK_REANALYSIS_SUMMARY.md"

SUPPORTED_STATUSES = {"admit", "conditional_admit", "exploratory_admit"}
UNITS = {
    "temperature_c": "deg C",
    "velocity_m_s": "m/s",
    "particle_diameter_um": "um",
}


def build_summary() -> dict:
    inference = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    stability = json.loads(STABILITY_PATH.read_text(encoding="utf-8"))
    sensitivity = json.loads(SENSITIVITY_PATH.read_text(encoding="utf-8"))
    stability_by_key = {
        (contrast["estimand_id"], outcome): values
        for contrast in stability["contrasts"]
        for outcome, values in contrast["outcomes"].items()
    }
    sensitivity_by_key = {
        (row["estimand_id"], row["outcome"]): row
        for row in sensitivity["comparisons"]
    }

    edges = []
    for row in inference["decisions"]:
        key = (row["estimand_id"], row["outcome"])
        stable = stability_by_key[key]
        count_check = sensitivity_by_key[key]
        supported = row["status"] in SUPPORTED_STATUSES
        edges.append(
            {
                "source": row["treatment_name"],
                "target": row["outcome"],
                "estimand_id": row["estimand_id"],
                "claim_role": row["claim_role"],
                "supported": supported,
                "status": row["status"],
                "mean_effect": row["point_mean_effect"],
                "mean_interval": [row["mean_lower_bound"], row["mean_upper_bound"]],
                "unit": UNITS[row["outcome"]],
                "sequence_adjusted_effect": stable["sequence_adjusted_mean_effect"],
                "sequence_sign_retained": stable["sequence_sign_retained"],
                "module_direction_consistent": stable["module_mean_direction_consistent"],
                "count_sensitivity_effect": count_check["sensitivity_mean_effect"],
                "count_sensitivity_support_decision_retained": count_check[
                    "support_decision_retained"
                ],
                "failed_gates": row["failed_gates"],
            }
        )
    supported_edges = [edge for edge in edges if edge["supported"]]
    unsupported_edges = [edge for edge in edges if not edge["supported"]]
    return {
        "schema": "pu_dcgp_v26_a_group_causal_network_evidence_v1",
        "mapping_rule": "NNN.csv = UCE-RNNN = design row NNN = execution order NNN",
        "network_type": "DOE-to-DPV matched total-effect evidence network",
        "supported_edge_count": len(supported_edges),
        "confirmatory_supported_edge_count": sum(
            edge["claim_role"] == "confirmatory" for edge in supported_edges
        ),
        "exploratory_supported_edge_count": sum(
            edge["claim_role"] == "exploratory" for edge in supported_edges
        ),
        "unsupported_edge_count": len(unsupported_edges),
        "edges": edges,
        "non_estimable_nodes": {
            "hydrogen_setting": "fixed at 2.5; no independent A-group variation",
            "powder_carrier_gas_setting": "fixed at 10; no independent A-group variation",
            "hydrogen_to_argon_ratio": "deterministic re-expression of argon because hydrogen is fixed; not a separate causal treatment",
            "latent_physical_mediators": "mechanistic hypotheses only; not identified from A-group DPV data",
        },
        "causal_interpretation": (
            "The nine confirmatory edges are conditionally supported total effects under exact DOE matching, "
            "a linear execution-order sensitivity model, module-direction consistency, hierarchical bootstrap, "
            "and outcome-blind particle-count sensitivity. They are not proof of mediator paths or unconditional "
            "effects under arbitrary unmeasured time-varying drift."
        ),
        "supersedes": (
            "All earlier A-group edge estimates produced with the reordered DPV-to-DOE manifest, including the "
            "one-edge powder-to-diameter result. Synthetic benchmark conclusions are unaffected."
        ),
    }


def render_report(result: dict) -> str:
    lines = [
        "# Corrected A-group causal-network reanalysis",
        "",
        "## Main result",
        "",
        f"Under the corrected one-to-one mapping `{result['mapping_rule']}`, the pure-data network contains "
        f"{result['confirmatory_supported_edge_count']} conditionally supported confirmatory DOE-to-DPV edges "
        f"and {result['exploratory_supported_edge_count']} supported exploratory edge. "
        f"{result['unsupported_edge_count']} exploratory edges remain unsupported.",
        "",
        "| DOE contrast | DPV outcome | Mean effect (95% interval) | Sequence-adjusted | Count sensitivity | Decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    for edge in result["edges"]:
        lower, upper = edge["mean_interval"]
        lines.append(
            f"| {edge['estimand_id']} | {edge['target']} | "
            f"{edge['mean_effect']:+.3f} [{lower:+.3f}, {upper:+.3f}] {edge['unit']} | "
            f"{edge['sequence_adjusted_effect']:+.3f} | "
            f"{edge['count_sensitivity_effect']:+.3f} | "
            f"{edge['status']} |"
        )
    lines.extend(
        [
            "",
            "## What the data support",
            "",
            "- Current 600→800 A: temperature, velocity and detected particle diameter all increase.",
            "- Argon 80→120 scfh: temperature decreases, while velocity and detected particle diameter increase.",
            "- Powder feed 10→30 g/min: temperature and velocity decrease, while detected particle diameter increases.",
            "- Spray distance 80→120 mm: temperature decreases; velocity and particle-diameter edges are not admitted.",
            "",
            "All nine confirmatory directions survive exact matching, leave-one-stratum deletion, linear sequence adjustment, "
            "DOE-module splitting, 2000-replicate hierarchical bootstrap, and the 142-run outcome-blind count sensitivity.",
            "",
            "## Causal boundary",
            "",
            result["causal_interpretation"],
            "",
            "Hydrogen and powder carrier gas are fixed in A and therefore cannot have independent effects estimated. "
            "H2/Ar is only an inverse re-expression of argon under fixed hydrogen and must not be inserted as a second treatment. "
            "The latent plasma and particle-transfer nodes remain physical interpretation, not learned mediator edges.",
            "",
            "## Supersession",
            "",
            result["supersedes"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = build_summary()
    JSON_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
