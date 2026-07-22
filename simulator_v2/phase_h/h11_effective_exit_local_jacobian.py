"""Run and summarize the frozen local effective-exit COMSOL response design."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from simulator_v2.phase_h.h11_dpv_sampling_volume_contract import (
    DpvSamplingVolumeContract,
)
from simulator_v2.phase_h.h11_effective_exit_local_jacobian_contract import (
    CONTRACT_PATH,
    LocalJacobianContract,
    MODEL_DIR,
    OUTPUT_DIR,
)
from simulator_v2.phase_h.h11_effective_exit_screen_executor import (
    EffectiveExitScreenSpec,
    audit_existing_gas,
    build_skeleton,
    solve_gas,
    solve_particle_count,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256


PILOT_SUMMARY_PATH = (
    OUTPUT_DIR / "h11_effective_exit_local_jacobian_pilot_summary.json"
)
FINAL_SUMMARY_PATH = (
    OUTPUT_DIR / "h11_effective_exit_local_jacobian_final_summary.json"
)
COUNT_CONVERGENCE_PATH = (
    OUTPUT_DIR / "h11_effective_exit_local_jacobian_count_convergence.json"
)


def _summary_path(particles_per_size: int) -> Path:
    contract = LocalJacobianContract()
    if particles_per_size == contract.pilot_particles_per_size:
        return PILOT_SUMMARY_PATH
    if particles_per_size == contract.final_particles_per_size:
        return FINAL_SUMMARY_PATH
    raise ValueError("Unsupported local-Jacobian particle count")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _spec(case_name: str) -> EffectiveExitScreenSpec:
    contract = LocalJacobianContract()
    contract.validate()
    case = contract.cases()[case_name]
    name = (
        f"local_{case_name}_t{int(case['temperature_k'])}_"
        f"u{int(case['speed_m_s'])}"
    )
    return EffectiveExitScreenSpec(
        name=name,
        temperature_k=case["temperature_k"],
        speed_m_s=case["speed_m_s"],
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        particle_output_step_us=10.0,
    )


def _primary_aperture(payload: dict[str, Any]) -> dict[str, Any]:
    radius = DpvSamplingVolumeContract().low_speed_equivalent_radius_mm
    trajectory = payload["particle_solve_audit"]["trajectory_audit"]
    return next(
        item
        for item in trajectory["centerline_aperture_sensitivity"]
        if math.isclose(
            item["aperture_radius_mm"], radius, rel_tol=0.0, abs_tol=1.0e-12
        )
    )


def run_case(
    client: Any,
    case_name: str,
    particles_per_size: int,
    reuse_existing_gas: bool = False,
) -> dict[str, Any]:
    spec = _spec(case_name)
    if reuse_existing_gas:
        if not spec.paths()["gas_model"].is_file():
            raise FileNotFoundError(spec.paths()["gas_model"])
        gas = audit_existing_gas(client, spec)
        _write_json(
            spec.output_dir / f"{spec.name}_gas_reaudit.json",
            gas,
        )
    else:
        gas = solve_gas(client, spec)
        _write_json(spec.paths()["gas_audit"], gas)
    if gas["status"] != "pass_effective_exit_screen_gas":
        return {
            "schema_version": "h11_effective_exit_local_case_v1",
            "status": "fail_local_case_gas",
            "case_name": case_name,
            "particles_per_size": particles_per_size,
            "gas": gas,
        }
    skeleton = build_skeleton(client, spec, gas)
    particle = solve_particle_count(
        client, spec, particles_per_size, skeleton
    )
    _write_json(spec.paths(particles_per_size)["particle_audit"], particle)
    return particle


def _compact_case(case_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    aperture = _primary_aperture(payload)
    median = aperture["empirical_detected_diameter_weighted_median"]
    return {
        "case_name": case_name,
        "temperature_k": _spec(case_name).temperature_k,
        "speed_m_s": _spec(case_name).speed_m_s,
        "status": payload["status"],
        "particles_per_size": int(payload["particles_per_size"]),
        "selected_particle_count": int(aperture["selected_particle_count"]),
        "diameter_nodes_represented": int(aperture["diameter_nodes_represented"]),
        "particle_temperature_c": float(median["temperature_c"]),
        "particle_velocity_m_s": float(median["speed_m_s"]),
        "particle_diameter_um": float(median["diameter_um"]),
        "model_path": payload["particle_solve_audit"]["model_path"],
        "model_sha256": payload["particle_solve_audit"]["model_sha256"],
    }


def _load_payload(case_name: str, particles_per_size: int) -> dict[str, Any] | None:
    path = _spec(case_name).paths(particles_per_size)["particle_audit"]
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(particles_per_size: int) -> dict[str, Any] | None:
    contract = LocalJacobianContract()
    compact = {}
    for case_name in contract.cases():
        payload = _load_payload(case_name, particles_per_size)
        if payload is None or not payload["status"].startswith("pass"):
            return None
        compact[case_name] = _compact_case(case_name, payload)
    tm = compact["temperature_minus"]
    tp = compact["temperature_plus"]
    um = compact["speed_minus"]
    up = compact["speed_plus"]
    jacobian = np.asarray(
        [
            [
                (tp["particle_temperature_c"] - tm["particle_temperature_c"])
                / (2.0 * contract.temperature_step_k),
                (up["particle_temperature_c"] - um["particle_temperature_c"])
                / (2.0 * contract.speed_step_m_s),
            ],
            [
                (tp["particle_velocity_m_s"] - tm["particle_velocity_m_s"])
                / (2.0 * contract.temperature_step_k),
                (up["particle_velocity_m_s"] - um["particle_velocity_m_s"])
                / (2.0 * contract.speed_step_m_s),
            ],
        ],
        dtype=float,
    )
    determinant = float(np.linalg.det(jacobian))
    condition = float(np.linalg.cond(jacobian))
    gates = {
        "all_cases_passed": all(
            case["status"].startswith("pass") for case in compact.values()
        ),
        "all_cases_retain_seven_diameter_nodes": all(
            case["diameter_nodes_represented"] == contract.expected_diameter_nodes
            for case in compact.values()
        ),
        "all_pilot_primary_counts_sufficient": all(
            case["selected_particle_count"]
            >= contract.minimum_pilot_primary_particles
            for case in compact.values()
        ),
        "particle_temperature_increases_with_exit_temperature": bool(
            jacobian[0, 0] > 0.0
        ),
        "particle_velocity_increases_with_exit_speed": bool(
            jacobian[1, 1] > 0.0
        ),
        "jacobian_determinant_sufficient": (
            abs(determinant) >= contract.minimum_absolute_determinant
        ),
        "jacobian_condition_number_below_20": (
            condition <= contract.maximum_jacobian_condition_number
        ),
    }
    return {
        "schema_version": "h11_effective_exit_local_jacobian_summary_v1",
        "status": (
            "pass_local_jacobian_pilot"
            if all(gates.values())
            and particles_per_size == contract.pilot_particles_per_size
            else (
                "pass_local_jacobian_final"
                if all(gates.values())
                else "fail_local_jacobian"
            )
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "particles_per_size": particles_per_size,
        "cases": compact,
        "jacobian_rows_particle_temperature_velocity": jacobian.tolist(),
        "jacobian_columns_exit_temperature_velocity": [
            "exit_temperature_k",
            "exit_velocity_m_s",
        ],
        "determinant": determinant,
        "condition_number": condition,
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def _relative_change(left: float, right: float) -> float:
    return abs(right - left) / max(abs(right), 1.0e-12)


def build_count_convergence() -> dict[str, Any] | None:
    contract = LocalJacobianContract()
    pilot = build_summary(contract.pilot_particles_per_size)
    final = build_summary(contract.final_particles_per_size)
    if pilot is None or final is None:
        return None
    comparisons: dict[str, Any] = {}
    maximum_change = 0.0
    for case_name in contract.cases():
        pilot_payload = _load_payload(case_name, contract.pilot_particles_per_size)
        final_payload = _load_payload(case_name, contract.final_particles_per_size)
        if pilot_payload is None or final_payload is None:
            return None
        pilot_aperture = _primary_aperture(pilot_payload)
        final_aperture = _primary_aperture(final_payload)
        changes: dict[str, Any] = {}
        for outcome in ("temperature_c", "speed_m_s"):
            changes[outcome] = {}
            for quantile in ("q10", "q50", "q90"):
                left = float(
                    pilot_aperture[
                        "empirical_detected_diameter_weighted_quantiles"
                    ][outcome][quantile]
                )
                right = float(
                    final_aperture[
                        "empirical_detected_diameter_weighted_quantiles"
                    ][outcome][quantile]
                )
                change = _relative_change(left, right)
                changes[outcome][quantile] = change
                maximum_change = max(maximum_change, change)
        comparisons[case_name] = {
            "pilot_selected_particle_count": int(
                pilot_aperture["selected_particle_count"]
            ),
            "final_selected_particle_count": int(
                final_aperture["selected_particle_count"]
            ),
            "final_diameter_nodes_represented": int(
                final_aperture["diameter_nodes_represented"]
            ),
            "relative_changes": changes,
        }
    pilot_j = np.asarray(
        pilot["jacobian_rows_particle_temperature_velocity"], dtype=float
    )
    final_j = np.asarray(
        final["jacobian_rows_particle_temperature_velocity"], dtype=float
    )
    gates = {
        "pilot_passed": pilot["status"] == "pass_local_jacobian_pilot",
        "final_passed": final["status"] == "pass_local_jacobian_final",
        "all_final_primary_counts_at_least_70": all(
            case["final_selected_particle_count"]
            >= contract.minimum_final_primary_particles
            for case in comparisons.values()
        ),
        "all_final_cases_retain_seven_diameter_nodes": all(
            case["final_diameter_nodes_represented"]
            == contract.expected_diameter_nodes
            for case in comparisons.values()
        ),
        "all_primary_quantile_changes_below_one_percent": (
            maximum_change <= contract.particle_count_convergence_limit_fraction
        ),
    }
    return {
        "schema_version": "h11_effective_exit_local_jacobian_count_convergence_v1",
        "status": (
            "pass_local_jacobian_particle_count_convergence"
            if all(gates.values())
            else "fail_local_jacobian_particle_count_convergence"
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "pilot_particles_per_size": contract.pilot_particles_per_size,
        "final_particles_per_size": contract.final_particles_per_size,
        "case_comparisons": comparisons,
        "maximum_primary_quantile_relative_change": maximum_change,
        "pilot_jacobian": pilot_j.tolist(),
        "final_jacobian": final_j.tolist(),
        "jacobian_element_absolute_changes": np.abs(final_j - pilot_j).tolist(),
        "pilot_determinant": pilot["determinant"],
        "final_determinant": final["determinant"],
        "pilot_condition_number": pilot["condition_number"],
        "final_condition_number": final["condition_number"],
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(LocalJacobianContract().cases()))
    parser.add_argument("--count", type=int, choices=(127, 1023), default=127)
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument(
        "--reuse-existing-gas",
        action="store_true",
        help="Re-audit and reuse the saved case-specific gas solution.",
    )
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(CONTRACT_PATH)
    if not args.summarize_existing:
        if args.case is None:
            raise ValueError("--case is required unless --summarize-existing is used")
        import mph

        client = mph.start(cores=args.cores, version=args.version)
        try:
            payload = run_case(
                client,
                args.case,
                args.count,
                reuse_existing_gas=args.reuse_existing_gas,
            )
        finally:
            client.clear()
        print(args.case, payload["status"])
    summary = build_summary(args.count)
    if summary is None:
        print("Local Jacobian ladder is incomplete")
        return 0
    summary_path = _summary_path(args.count)
    _write_json(summary_path, summary)
    print(summary_path)
    print(summary["status"])
    print("J=", summary["jacobian_rows_particle_temperature_velocity"])
    print("condition=", summary["condition_number"])
    convergence = build_count_convergence()
    if convergence is not None:
        _write_json(COUNT_CONVERGENCE_PATH, convergence)
        print(COUNT_CONVERGENCE_PATH)
        print(convergence["status"])
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
