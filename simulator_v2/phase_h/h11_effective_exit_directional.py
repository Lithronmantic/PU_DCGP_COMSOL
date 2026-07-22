"""Execute and summarize the frozen effective-exit directional pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_effective_exit_directional_contract import (
    OUTPUT_PATH as CONTRACT_PATH,
    EffectiveExitDirectionalCase,
    directional_cases,
)
from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    configure_studies,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_conservative_free_jet_skeleton import (
    MODEL_PATH as GAS_SKELETON_MODEL,
)
from simulator_v2.phase_h.h11_particle_physics_contract import (
    ParticlePhysicsContract,
)
from simulator_v2.phase_h.h11_particle_population_v2_skeleton import (
    audit_model as audit_particle_skeleton,
    build_model as build_particle_skeleton,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import (
    SOLVE_AUDIT_PATH as NOMINAL_PARTICLE_AUDIT,
    _sha256,
    audit_build as audit_radial_particle_build,
    build_model as build_radial_particle_model,
    solve_and_audit as solve_radial_particles,
)


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "effective_exit_directional"
MODEL_DIR = HERE / "comsol_models" / "h11_effective_exit_directional"
SUMMARY_PATH = OUTPUT_DIR / "h11_effective_exit_directional_summary.json"


def case_paths(case: EffectiveExitDirectionalCase) -> dict[str, Path]:
    return {
        "gas_model": MODEL_DIR / f"{case.name}_gas.mph",
        "particle_skeleton": MODEL_DIR / f"{case.name}_particle_skeleton.mph",
        "particle_model": MODEL_DIR / f"{case.name}_particles.mph",
        "case_audit": OUTPUT_DIR / "cases" / f"{case.name}.json",
        "gas_audit": OUTPUT_DIR / "cases" / f"{case.name}_gas.json",
        "gas_log": OUTPUT_DIR / "cases" / f"{case.name}_gas.log",
    }


def _gas_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    contract = FreeJetSolveContract()
    return {
        "temperature_within_property_range": (
            float(metrics["temperature_k"]["minimum"])
            >= contract.property_temperature_floor_k
        ),
        "absolute_pressure_positive": (
            float(metrics["absolute_pressure_pa"]["minimum"])
            >= contract.pressure_floor_pa
        ),
        "mass_imbalance_below_0_5_percent": (
            float(metrics["mass_flux_kg_s"]["imbalance_fraction"])
            < contract.mass_imbalance_limit_fraction
        ),
        "energy_imbalance_below_2_percent": (
            float(
                metrics["energy_balance_w"]["imbalance_fraction_of_inlet"]
            )
            < contract.energy_imbalance_limit_fraction
        ),
        "forward_flow_crosses_dpv_plane": (
            float(
                metrics["fixed_dpv_plane_gas_diagnostics"][
                    "forward_axial_mass_flux_kg_s"
                ]
            )
            > 0
        ),
    }


def solve_gas_case(
    client: Any,
    case: EffectiveExitDirectionalCase,
) -> tuple[dict[str, Any], str]:
    """Solve one bounded gas pilot with the frozen 0-to-1 continuation."""

    paths = case_paths(case)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = client.load(str(GAS_SKELETON_MODEL))
    model.rename(f"{case.name}_gas")
    jm = model.java
    client.java.showProgress(str(paths["gas_log"].resolve()))
    try:
        jm.param().set("T_exit_eff", f"{case.temperature_k:.12g}[K]")
        jm.param().set("u_exit_eff", f"{case.speed_m_s:.12g}[m/s]")
        contract = FreeJetSolveContract()
        solver = configure_studies(jm, contract)
        continuation_started = time.time()
        jm.study("std1").run()
        print("    bounded 0-to-1 gas continuation complete", flush=True)
        refinement_started = time.time()
        jm.study("std_refine").run()
        print("    bounded full-load gas refinement complete", flush=True)
        metrics = evaluate_solution(model, contract)
        gates = _gas_gates(metrics)
        status = (
            "pass_directional_gas_case"
            if all(gates.values())
            else "fail_directional_gas_case_gates"
        )
        model.save(str(paths["gas_model"]))
        payload = {
            "schema_version": "h11_effective_exit_directional_gas_case_v1",
            "status": status,
            "case": asdict(case),
            "source_model": str(GAS_SKELETON_MODEL.resolve()),
            "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
            "final_gas_study": "std_refine",
            "solver": solver,
            "continuation_runtime_sec": (
                refinement_started - continuation_started
            ),
            "refinement_runtime_sec": time.time() - refinement_started,
            "screening_scope": (
                "Bounded segregated lower-limit solve for directional "
                "identifiability. Strict fully coupled residual verification "
                "is deferred to the selected calibration envelope."
            ),
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["gas_model"].resolve()),
            "model_sha256": _sha256(paths["gas_model"]),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        return payload, "std_refine"
    finally:
        client.java.showProgress(False)
        client.remove(model)


def solve_particle_case(
    client: Any,
    case: EffectiveExitDirectionalCase,
    gas_audit: dict[str, Any],
    gas_study: str,
) -> dict[str, Any]:
    """Attach the frozen particle tree and solve it on one directional gas field."""

    paths = case_paths(case)
    particle_contract = ParticlePhysicsContract()
    particle_contract.validate()

    skeleton = None
    particle = None
    try:
        skeleton, skeleton_java = build_particle_skeleton(
            client,
            particle_contract,
            source_model=paths["gas_model"],
        )
        skeleton_audit = audit_particle_skeleton(
            skeleton_java,
            particle_contract,
        )
        skeleton.save(str(paths["particle_skeleton"]))
        skeleton_audit.update(
            {
                "source_model": str(paths["gas_model"].resolve()),
                "source_model_sha256": _sha256(paths["gas_model"]),
                "model_path": str(paths["particle_skeleton"].resolve()),
                "model_sha256": _sha256(paths["particle_skeleton"]),
            }
        )
    finally:
        if skeleton is not None:
            client.remove(skeleton)

    try:
        particle, particle_java = build_radial_particle_model(
            client,
            paths["particle_skeleton"],
            source_study=gas_study,
        )
        build_audit = audit_radial_particle_build(
            particle_java,
            source_study=gas_study,
        )
        solve_audit = solve_radial_particles(particle, particle_java)
        particle.save(str(paths["particle_model"]))
        solve_audit.update(
            {
                "model_path": str(paths["particle_model"].resolve()),
                "model_sha256": _sha256(paths["particle_model"]),
            }
        )
    finally:
        if particle is not None:
            client.remove(particle)

    return {
        "schema_version": "h11_effective_exit_directional_case_v1",
        "status": (
            "pass_directional_gas_and_particle_case"
            if (
                gas_audit["status"] == "pass_directional_gas_case"
                and solve_audit["status"]
                == "pass_nominal_comsol_trajectory_audit"
            )
            else "fail_directional_gas_or_particle_case"
        ),
        "case": asdict(case),
        "gas_audit": gas_audit,
        "particle_skeleton_audit": skeleton_audit,
        "particle_build_audit": build_audit,
        "particle_solve_audit": solve_audit,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def audit_existing_gas_case(
    client: Any,
    case: EffectiveExitDirectionalCase,
) -> tuple[dict[str, Any], str]:
    """Re-audit a saved gas field before reusing it after a particle-only failure."""

    paths = case_paths(case)
    model = client.load(str(paths["gas_model"]))
    try:
        jm = model.java
        temperature = float(str(jm.param().evaluate("T_exit_eff")))
        speed = float(str(jm.param().evaluate("u_exit_eff")))
        if not math.isclose(temperature, case.temperature_k, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Saved gas temperature does not match the case")
        if not math.isclose(speed, case.speed_m_s, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Saved gas speed does not match the case")
        if "std_refine" not in {str(tag) for tag in jm.study().tags()}:
            raise ValueError("Saved gas model lacks the bounded refinement study")
        metrics = evaluate_solution(model, FreeJetSolveContract())
        gates = _gas_gates(metrics)
        payload = {
            "schema_version": "h11_effective_exit_directional_gas_case_v1",
            "status": (
                "pass_directional_gas_case"
                if all(gates.values())
                else "fail_directional_gas_case_gates"
            ),
            "case": asdict(case),
            "source_model": "re_audited_saved_directional_gas_model",
            "final_gas_study": "std_refine",
            "screening_scope": (
                "Re-audited bounded segregated directional gas field."
            ),
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["gas_model"].resolve()),
            "model_sha256": _sha256(paths["gas_model"]),
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        return payload, "std_refine"
    finally:
        client.remove(model)


def compact_particle_case(
    case: EffectiveExitDirectionalCase,
    solve_audit: dict[str, Any],
) -> dict[str, Any]:
    trajectory = solve_audit["trajectory_audit"]
    target = trajectory["unweighted_target_comparison"]["dpv_prt_pooled_median"]
    apertures = []
    for aperture in trajectory["centerline_aperture_sensitivity"]:
        prediction = aperture["empirical_detected_diameter_weighted_median"]
        if prediction["temperature_c"] is None or prediction["speed_m_s"] is None:
            t_gap = None
            u_gap = None
            joint_error = None
        else:
            t_gap = (
                prediction["temperature_c"] - target["temperature_c"]
            ) / target["temperature_c"]
            u_gap = (
                prediction["speed_m_s"] - target["speed_m_s"]
            ) / target["speed_m_s"]
            joint_error = math.hypot(t_gap, u_gap)
        apertures.append(
            {
                "radius_mm": aperture["aperture_radius_mm"],
                "selected_particle_count": aperture["selected_particle_count"],
                "temperature_c": prediction["temperature_c"],
                "speed_m_s": prediction["speed_m_s"],
                "diameter_um": prediction["diameter_um"],
                "signed_relative_temperature_gap": t_gap,
                "signed_relative_speed_gap": u_gap,
                "joint_relative_error": joint_error,
            }
        )
    return {
        "case": case.name,
        "temperature_k": case.temperature_k,
        "speed_m_s": case.speed_m_s,
        "role": case.role,
        "status": solve_audit["status"],
        "crossing_count": trajectory["crossing_count"],
        "target": target,
        "apertures": apertures,
    }


def directional_derivatives(
    compact_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {case["case"]: case for case in compact_cases}
    required = {
        "nominal_t10000_u600",
        "t8000_u600",
        "t12000_u600",
        "t10000_u400",
        "t10000_u800",
    }
    if set(by_name) != required:
        raise ValueError("The complete five-case directional matrix is required")
    aperture_maps = {
        name: {
            round(float(item["radius_mm"]), 12): item
            for item in case["apertures"]
        }
        for name, case in by_name.items()
    }
    common_radii = set.intersection(
        *(set(apertures) for apertures in aperture_maps.values())
    )
    if not common_radii:
        raise ValueError("Directional cases have no common aperture radius")
    rows = []
    for radius_key in sorted(common_radii):
        low_t = aperture_maps["t8000_u600"][radius_key]
        high_t = aperture_maps["t12000_u600"][radius_key]
        low_u = aperture_maps["t10000_u400"][radius_key]
        high_u = aperture_maps["t10000_u800"][radius_key]
        rows.append(
            {
                "radius_mm": low_t["radius_mm"],
                "d_particle_temperature_c_per_exit_k": (
                    high_t["temperature_c"] - low_t["temperature_c"]
                )
                / 4_000.0,
                "d_particle_speed_per_exit_k": (
                    high_t["speed_m_s"] - low_t["speed_m_s"]
                )
                / 4_000.0,
                "d_particle_temperature_c_per_exit_speed_m_s": (
                    high_u["temperature_c"] - low_u["temperature_c"]
                )
                / 400.0,
                "d_particle_speed_per_exit_speed": (
                    high_u["speed_m_s"] - low_u["speed_m_s"]
                )
                / 400.0,
                "temperature_response_monotone": (
                    high_t["temperature_c"] > low_t["temperature_c"]
                ),
                "speed_response_monotone": (
                    high_u["speed_m_s"] > low_u["speed_m_s"]
                ),
            }
        )
    return rows


def build_summary(compact_cases: list[dict[str, Any]]) -> dict[str, Any]:
    derivatives = directional_derivatives(compact_cases)
    all_pass = all(
        case["status"] == "pass_nominal_comsol_trajectory_audit"
        for case in compact_cases
    )
    identifiable = all(
        row["temperature_response_monotone"] and row["speed_response_monotone"]
        for row in derivatives
    )
    return {
        "schema_version": "h11_effective_exit_directional_summary_v1",
        "status": (
            "pass_directional_identifiability"
            if all_pass and identifiable
            else "fail_directional_identifiability"
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "cases": compact_cases,
        "centered_directional_derivatives": derivatives,
        "parameter_release_rule": (
            "Only parameters with stable, physically signed responses may enter "
            "the next training-fold calibration layer."
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_completed_compact_cases() -> list[dict[str, Any]] | None:
    cases = directional_cases()
    nominal = json.loads(NOMINAL_PARTICLE_AUDIT.read_text(encoding="utf-8"))
    compact = [compact_particle_case(cases[0], nominal)]
    for case in cases[1:]:
        path = case_paths(case)["case_audit"]
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        compact.append(
            compact_particle_case(case, payload["particle_solve_audit"])
        )
    return compact


def run_cases(
    selected_names: set[str] | None,
    *,
    cores: int,
    version: str,
    reuse_existing_gas: bool = False,
) -> list[str]:
    cases = [case for case in directional_cases() if not case.reuse_nominal_artifact]
    if selected_names is not None:
        cases = [case for case in cases if case.name in selected_names]
    if not cases:
        return []

    import mph

    completed = []
    client = mph.start(cores=cores, version=version)
    try:
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{len(cases)}] {case.name}: "
                f"T={case.temperature_k:g} K, U={case.speed_m_s:g} m/s",
                flush=True,
            )
            started = time.time()
            paths = case_paths(case)
            if reuse_existing_gas and paths["gas_model"].exists():
                gas_audit, gas_study = audit_existing_gas_case(client, case)
                print("  re-audited saved bounded gas field", flush=True)
            else:
                gas_audit, gas_study = solve_gas_case(client, case)
            _write_json(paths["gas_audit"], gas_audit)
            if gas_audit["status"] != "pass_directional_gas_case":
                failure = {
                    "schema_version": "h11_effective_exit_directional_case_v1",
                    "status": "fail_directional_gas_case",
                    "case": asdict(case),
                    "gas_audit": gas_audit,
                    "calibrated": False,
                    "paper_prediction_allowed": False,
                }
                _write_json(case_paths(case)["case_audit"], failure)
                print(f"  gas gates failed: {gas_audit['gates']}", flush=True)
                continue
            payload = solve_particle_case(client, case, gas_audit, gas_study)
            _write_json(case_paths(case)["case_audit"], payload)
            completed.append(case.name)
            print(
                f"  status={payload['status']}, "
                f"wall={time.time()-started:.1f}s",
                flush=True,
            )
    finally:
        client.clear()
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            case.name
            for case in directional_cases()
            if not case.reuse_nominal_artifact
        ],
    )
    parser.add_argument(
        "--reuse-existing-gas",
        action="store_true",
        help="Re-audit and reuse a matching saved gas model after a particle-only failure.",
    )
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(CONTRACT_PATH)
    if not NOMINAL_PARTICLE_AUDIT.exists():
        raise FileNotFoundError(NOMINAL_PARTICLE_AUDIT)
    selected = set(args.case) if args.case else None
    if not args.summarize_existing:
        run_cases(
            selected,
            cores=args.cores,
            version=args.version,
            reuse_existing_gas=args.reuse_existing_gas,
        )
    compact = _load_completed_compact_cases()
    if compact is None:
        print("Directional matrix is incomplete; case artifacts were retained.")
        return 0
    summary = build_summary(compact)
    _write_json(args.output, summary)
    print(f"Summary: {args.output}")
    print(f"Directional status: {summary['status']}")
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
