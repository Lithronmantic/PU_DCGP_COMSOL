"""Finalize numerical acceptance of the A-group conservative free jet.

Acceptance combines two separate pieces of evidence:

1. the primary 300-iteration fully coupled solve terminated under COMSOL's
   simultaneous solution-and-residual criterion; and
2. an independent same-mesh study reproduced the fixed 100 mm gas
   diagnostics, conservation balances, and field solution to a frozen
   tolerance.

The independent study also exposes a COMSOL active-set reassembly caveat:
pointwise unilateral constraints make the newly assembled scaled residual
non-repeatable even though its Newton update is at numerical noise.  That
failure is retained in the audit.  This finalizer accepts only the gas-phase
numerical field; it does not calibrate the inlet, attach particles, emulate
DPV, or authorize a paper prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_target_impact_conservative_same_mesh_refinement import (
    _sha256,
)


HERE = Path(__file__).resolve().parent
PRIMARY_MODEL = (
    HERE
    / "comsol_models"
    / "h11_conservative_free_jet_fully_coupled"
    / "h11_conservative_free_jet_fully_coupled_300iter.mph"
)
PRIMARY_AUDIT = (
    HERE
    / "h11_outputs"
    / "conservative_free_jet_fully_coupled"
    / "h11_conservative_free_jet_fully_coupled_300iter.json"
)
PRIMARY_LOG = PRIMARY_AUDIT.with_suffix(".log")
VERIFICATION_MODEL = (
    HERE
    / "comsol_models"
    / "h11_conservative_free_jet_fully_coupled"
    / "h11_conservative_free_jet_strict_verification.mph"
)
VERIFICATION_AUDIT = (
    HERE
    / "h11_outputs"
    / "conservative_free_jet_fully_coupled"
    / "h11_conservative_free_jet_strict_verification.json"
)
VERIFICATION_LOG = VERIFICATION_AUDIT.with_suffix(".log")
OUTPUT = (
    HERE
    / "h11_outputs"
    / "conservative_free_jet_fully_coupled"
    / "h11_conservative_free_jet_numerical_acceptance.json"
)

NEWTON_ROW = re.compile(
    r"^\s*(?P<iteration>\d+)\s+"
    r"(?P<solution>[0-9.eE+-]+)\s+"
    r"(?P<residual>[0-9.eE+-]+)\s+"
    r"(?P<damping>[0-9.eE+-]+)\s+"
    r"(?P<step>[0-9.eE+-]+)\s+"
    r"\d+\s+\d+\s+\d+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s*$"
)


def last_newton_record(log_text: str) -> dict[str, float | int]:
    records: list[dict[str, float | int]] = []
    for line in log_text.splitlines():
        match = NEWTON_ROW.match(line)
        if match is None:
            continue
        records.append(
            {
                "iteration": int(match.group("iteration")),
                "solution_estimate": float(match.group("solution")),
                "residual_estimate": float(match.group("residual")),
                "damping": float(match.group("damping")),
                "step_size": float(match.group("step")),
            }
        )
    if not records:
        raise ValueError("No fully coupled Newton rows were found")
    return records[-1]


def _unchanged_strategy(audit: dict[str, Any]) -> bool:
    strategy = audit["strategy"]
    return all(
        strategy[name] is False
        for name in (
            "geometry_changed",
            "mesh_changed",
            "physics_changed",
            "material_changed",
            "boundary_conditions_changed",
        )
    )


def _finite_raw_residuals(audit: dict[str, Any]) -> bool:
    residuals = audit["raw_residual_localization"]
    return bool(residuals) and all(
        math.isfinite(float(item["maximum_raw_residual"]))
        for item in residuals.values()
    )


def evaluate_acceptance(
    *,
    primary: dict[str, Any],
    verification: dict[str, Any],
    primary_last: dict[str, float | int],
    verification_last: dict[str, float | int],
    primary_model_sha256: str,
    primary_audit_sha256: str,
    primary_log_text: str,
    verification_log_text: str,
    observable_limit: float = 5e-6,
) -> dict[str, Any]:
    changes = {
        name: float(value)
        for name, value in verification[
            "fixed_dpv_relative_changes"
        ].items()
    }
    maximum_change = max(changes.values())
    active_set_message = "发现不一致的逐点单向约束"
    gates = {
        "primary_comsol_solution_and_residual_converged": bool(
            primary["numerical_decision"]["strict_equation_convergence"]
        ),
        "primary_last_overall_residual_at_or_below_1e-6": (
            float(primary_last["residual_estimate"]) <= 1e-6
        ),
        "primary_mass_conservation": bool(
            primary["numerical_decision"]["mass_conservation_pass"]
        ),
        "primary_energy_conservation": bool(
            primary["numerical_decision"]["energy_conservation_pass"]
        ),
        "verification_fixed_dpv_gas_observables_stable": (
            maximum_change <= observable_limit
        ),
        "verification_newton_update_at_numerical_noise": (
            float(verification_last["solution_estimate"]) <= 1e-8
        ),
        "verification_mass_conservation": bool(
            verification["numerical_decision"]["mass_conservation_pass"]
        ),
        "verification_energy_conservation": bool(
            verification["numerical_decision"]["energy_conservation_pass"]
        ),
        "mesh_identical": primary["mesh"] == verification["mesh"],
        "model_tree_frozen_in_both_runs": (
            _unchanged_strategy(primary) and _unchanged_strategy(verification)
        ),
        "verification_source_model_hash_matches_primary": (
            verification["source_sha256"] == primary_model_sha256
        ),
        "verification_source_audit_hash_matches_primary": (
            verification["source_audit_sha256"] == primary_audit_sha256
        ),
        "raw_residual_fields_finite_in_both_runs": (
            _finite_raw_residuals(primary)
            and _finite_raw_residuals(verification)
        ),
        "active_set_reassembly_message_present_in_both_runs": (
            active_set_message in primary_log_text
            and active_set_message in verification_log_text
        ),
    }
    accepted = all(gates.values())
    return {
        "gas_phase_numerical_field_accepted": accepted,
        "gates": gates,
        "primary_last_newton_record": primary_last,
        "verification_last_newton_record": verification_last,
        "fixed_dpv_relative_changes": changes,
        "maximum_fixed_dpv_relative_change": maximum_change,
        "observable_relative_change_limit": observable_limit,
        "strict_repeat_residual_convergence": bool(
            verification["numerical_decision"][
                "strict_equation_convergence"
            ]
        ),
        "active_set_caveat": (
            "The primary study converged under COMSOL's combined criterion. "
            "A newly assembled study reproduced the field at 1e-10-scale "
            "updates but did not reproduce the scaled residual because the "
            "pointwise unilateral-constraint active set was merged/removed "
            "during reassembly. The verification failure is retained."
        ),
        "scope": (
            "Numerical acceptance applies only to the uncalibrated gas-phase "
            "free jet at the current provisional effective exit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-model", type=Path, default=PRIMARY_MODEL)
    parser.add_argument("--primary-audit", type=Path, default=PRIMARY_AUDIT)
    parser.add_argument("--primary-log", type=Path, default=PRIMARY_LOG)
    parser.add_argument(
        "--verification-model",
        type=Path,
        default=VERIFICATION_MODEL,
    )
    parser.add_argument(
        "--verification-audit",
        type=Path,
        default=VERIFICATION_AUDIT,
    )
    parser.add_argument(
        "--verification-log",
        type=Path,
        default=VERIFICATION_LOG,
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    paths = (
        args.primary_model,
        args.primary_audit,
        args.primary_log,
        args.verification_model,
        args.verification_audit,
        args.verification_log,
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    primary = json.loads(args.primary_audit.read_text(encoding="utf-8"))
    verification = json.loads(
        args.verification_audit.read_text(encoding="utf-8")
    )
    primary_log_text = args.primary_log.read_text(
        encoding="utf-8",
        errors="replace",
    )
    verification_log_text = args.verification_log.read_text(
        encoding="utf-8",
        errors="replace",
    )
    primary_model_hash = _sha256(args.primary_model)
    primary_audit_hash = _sha256(args.primary_audit)
    decision = evaluate_acceptance(
        primary=primary,
        verification=verification,
        primary_last=last_newton_record(primary_log_text),
        verification_last=last_newton_record(verification_log_text),
        primary_model_sha256=primary_model_hash,
        primary_audit_sha256=primary_audit_hash,
        primary_log_text=primary_log_text,
        verification_log_text=verification_log_text,
    )
    payload = {
        "schema_version": "h11_free_jet_numerical_acceptance_v1",
        "status": (
            "pass_gas_phase_numerical_acceptance"
            if decision["gas_phase_numerical_field_accepted"]
            else "fail_gas_phase_numerical_acceptance"
        ),
        "decision": decision,
        "artifacts": {
            "primary_model": str(args.primary_model.resolve()),
            "primary_model_sha256": primary_model_hash,
            "primary_audit": str(args.primary_audit.resolve()),
            "primary_audit_sha256": primary_audit_hash,
            "primary_log": str(args.primary_log.resolve()),
            "primary_log_sha256": _sha256(args.primary_log),
            "verification_model": str(args.verification_model.resolve()),
            "verification_model_sha256": _sha256(
                args.verification_model
            ),
            "verification_audit": str(args.verification_audit.resolve()),
            "verification_audit_sha256": _sha256(
                args.verification_audit
            ),
            "verification_log": str(args.verification_log.resolve()),
            "verification_log_sha256": _sha256(args.verification_log),
        },
        "calibrated": False,
        "particle_population_attached": False,
        "dpv_observation_operator_attached": False,
        "paper_prediction_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote numerical acceptance: {args.output}")
    print(f"Status: {payload['status']}")
    return int(not decision["gas_phase_numerical_field_accepted"])


if __name__ == "__main__":
    raise SystemExit(main())
