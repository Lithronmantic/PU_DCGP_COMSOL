
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.dpv_sampling_volume_contract import (
    DpvSamplingVolumeContract,
)
from pu_dcgp_comsol.comsol.effective_exit_joint_correction_contract import (
    OUTPUT_PATH as CONTRACT_PATH,
    JointCorrectionContract,
)
from pu_dcgp_comsol.comsol.effective_exit_joint_screen import (
    adjacent_count_changes,
    compact_particle_count_case,
)
from pu_dcgp_comsol.comsol.effective_exit_screen_executor import (
    EffectiveExitScreenSpec,
    audit_existing_gas,
    build_skeleton,
    solve_gas,
    solve_particle_count,
)
from pu_dcgp_comsol.comsol.particle_radial_enthalpy_nominal import _sha256


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "effective_exit_joint_correction"
MODEL_DIR = HERE / "comsol_models" / "h11_effective_exit_joint_correction"
SUMMARY_PATH = OUTPUT_DIR / "h11_effective_exit_joint_correction_summary.json"
SPEC = EffectiveExitScreenSpec(
    name="corrected_t11160_u1090",
    temperature_k=11_160.0,
    speed_m_s=1_090.0,
    output_dir=OUTPUT_DIR,
    model_dir=MODEL_DIR,
    particle_output_step_us=10.0,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def pilot_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    contract = JointCorrectionContract()
    solve = payload["particle_solve_audit"]
    trajectory = solve["trajectory_audit"]
    target = trajectory["unweighted_target_comparison"]["dpv_prt_pooled_median"]
    radius = DpvSamplingVolumeContract().low_speed_equivalent_radius_mm
    aperture = next(
        item
        for item in trajectory["centerline_aperture_sensitivity"]
        if math.isclose(
            item["aperture_radius_mm"], radius, rel_tol=0, abs_tol=1e-12
        )
    )
    prediction = aperture["empirical_detected_diameter_weighted_median"]
    temperature_gap = (
        prediction["temperature_c"] - target["temperature_c"]
    ) / target["temperature_c"]
    speed_gap = (
        prediction["speed_m_s"] - target["speed_m_s"]
    ) / target["speed_m_s"]
    gates = {
        "particle_case_passed": (
            payload["status"] == "pass_effective_exit_screen_particle_count"
        ),
        "all_seven_diameter_nodes_represented": (
            aperture["diameter_nodes_represented"] == 7
        ),
        "temperature_gap_within_5_percent": (
            abs(temperature_gap)
            <= contract.pilot_temperature_gap_limit_fraction
        ),
        "speed_gap_within_5_percent": (
            abs(speed_gap) <= contract.pilot_speed_gap_limit_fraction
        ),
    }
    return {
        "status": (
            "pass_corrected_joint_pilot"
            if all(gates.values())
            else "fail_corrected_joint_pilot"
        ),
        "primary_aperture_radius_mm": radius,
        "selected_particle_count": aperture["selected_particle_count"],
        "target": target,
        "prediction": prediction,
        "signed_relative_gaps": {
            "temperature": temperature_gap,
            "speed": speed_gap,
        },
        "gates": gates,
    }


def load_case(count: int) -> dict[str, Any] | None:
    path = SPEC.paths(count)["particle_audit"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def conditional_escalation_is_allowed() -> bool:
    contract = JointCorrectionContract()
    left = load_case(contract.convergence_counts_per_size[-2])
    right = load_case(contract.convergence_counts_per_size[-1])
    if left is None or right is None:
        return False
    cases = [
        compact_particle_count_case(
            count,
            payload["particle_solve_audit"],
        )
        for count, payload in zip(
            contract.convergence_counts_per_size,
            (left, right),
        )
    ]
    comparison = adjacent_count_changes(cases)[0]
    primary_count = cases[-1]["apertures"]["low_speed_primary"][
        "selected_particle_count"
    ]
    return (
        comparison["maximum_relative_change"]
        <= contract.convergence_limit_fraction
        and primary_count < contract.minimum_primary_aperture_particles
    )


def run_correction(
    *,
    selected_counts: set[int],
    cores: int,
    version: str,
    reuse_existing_gas: bool,
) -> None:
    contract = JointCorrectionContract()
    contract.validate()
    allowed = {
        contract.pilot_particles_per_size,
        *contract.convergence_counts_per_size,
        contract.conditional_escalation_particles_per_size,
    }
    if not selected_counts <= allowed:
        raise ValueError("Requested particle count is outside the correction contract")
    if any(count > contract.pilot_particles_per_size for count in selected_counts):
        pilot = load_case(contract.pilot_particles_per_size)
        if pilot is None or pilot_assessment(pilot)["status"] != "pass_corrected_joint_pilot":
            raise RuntimeError("The 127-per-size correction pilot has not passed")
    if (
        contract.conditional_escalation_particles_per_size in selected_counts
        and not conditional_escalation_is_allowed()
    ):
        raise RuntimeError(
            "The 1023 count-only escalation requires a converged 255->511 "
            "comparison and fewer than 70 primary-aperture particles at 511"
        )

    import mph

    client = mph.start(cores=cores, version=version)
    try:
        if reuse_existing_gas and SPEC.paths()["gas_model"].exists():
            gas_audit = audit_existing_gas(client, SPEC)
            print("Re-audited saved corrected gas model", flush=True)
        else:
            print("Solving corrected gas: T=11160 K, U=1090 m/s", flush=True)
            gas_audit = solve_gas(client, SPEC)
        _write_json(SPEC.paths()["gas_audit"], gas_audit)
        if gas_audit["status"] != "pass_effective_exit_screen_gas":
            print(f"Corrected gas failed: {gas_audit['gates']}", flush=True)
            return
        skeleton_audit = build_skeleton(client, SPEC, gas_audit)
        for index, count in enumerate(sorted(selected_counts), start=1):
            print(
                f"[{index}/{len(selected_counts)}] corrected particles_per_size={count}",
                flush=True,
            )
            started = time.time()
            payload = solve_particle_count(
                client,
                SPEC,
                count,
                skeleton_audit,
            )
            _write_json(SPEC.paths(count)["particle_audit"], payload)
            compact = compact_particle_count_case(
                count,
                payload["particle_solve_audit"],
            )
            primary = compact["apertures"]["low_speed_primary"]
            print(
                f"  status={payload['status']}, "
                f"primary_n={primary['selected_particle_count']}, "
                f"wall={time.time()-started:.1f}s",
                flush=True,
            )
            if count == contract.pilot_particles_per_size:
                assessment = pilot_assessment(payload)
                print(
                    "  pilot gaps: "
                    f"T={assessment['signed_relative_gaps']['temperature']:.4f}, "
                    f"U={assessment['signed_relative_gaps']['speed']:.4f}; "
                    f"{assessment['status']}",
                    flush=True,
                )
    finally:
        client.clear()


def build_completed_summary() -> dict[str, Any] | None:
    contract = JointCorrectionContract()
    gas_path = SPEC.paths()["gas_audit"]
    if not gas_path.exists():
        return None
    gas = json.loads(gas_path.read_text(encoding="utf-8"))
    counts = (
        contract.pilot_particles_per_size,
        *contract.convergence_counts_per_size,
    )
    escalation = load_case(contract.conditional_escalation_particles_per_size)
    if escalation is not None:
        counts = (*counts, contract.conditional_escalation_particles_per_size)
    payloads = [load_case(count) for count in counts]
    if any(payload is None for payload in payloads):
        return None
    particle_cases = [
        compact_particle_count_case(count, payload["particle_solve_audit"])
        for count, payload in zip(counts, payloads)
    ]
    comparisons = adjacent_count_changes(particle_cases)
    finest = particle_cases[-1]
    pilot = pilot_assessment(payloads[0])
    gates = {
        "gas_passed": gas["status"] == "pass_effective_exit_screen_gas",
        "correction_pilot_passed": pilot["status"] == "pass_corrected_joint_pilot",
        "finest_primary_count_at_least_70": (
            finest["apertures"]["low_speed_primary"]["selected_particle_count"]
            >= contract.minimum_primary_aperture_particles
        ),
        "finest_adjacent_quantile_change_below_1_percent": (
            comparisons[-1]["maximum_relative_change"]
            <= contract.convergence_limit_fraction
        ),
    }
    escalation_required = (
        len(counts) == 3
        and gates["finest_adjacent_quantile_change_below_1_percent"]
        and not gates["finest_primary_count_at_least_70"]
    )
    return {
        "schema_version": "h11_effective_exit_joint_correction_summary_v1",
        "status": (
            "pass_corrected_joint_screen_and_particle_convergence"
            if all(gates.values())
            else (
                "requires_conditional_particle_count_escalation"
                if escalation_required
                else "fail_corrected_joint_screen_or_particle_convergence"
            )
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "gas_audit": gas,
        "pilot_assessment": pilot,
        "particle_cases": particle_cases,
        "adjacent_count_changes": comparisons,
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--reuse-existing-gas", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument(
        "--count",
        action="append",
        type=int,
        choices=[127, 255, 511, 1023],
    )
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(CONTRACT_PATH)
    if not args.summarize_existing:
        selected = set(args.count) if args.count else {127}
        run_correction(
            selected_counts=selected,
            cores=args.cores,
            version=args.version,
            reuse_existing_gas=args.reuse_existing_gas,
        )
    summary = build_completed_summary()
    if summary is None:
        print("Correction convergence ladder is incomplete; artifacts were retained.")
        return 0
    _write_json(args.output, summary)
    print(f"Summary: {args.output}")
    print(f"Correction status: {summary['status']}")
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
