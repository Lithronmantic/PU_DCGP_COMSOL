
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.corrected_gas_crosswind_off import paths as gas_paths
from pu_dcgp_comsol.comsol.corrected_gas_crosswind_off_contract import (
    CONTRACT_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    CrosswindOffContract,
)
from pu_dcgp_comsol.comsol.corrected_gas_mesh_convergence import (
    _gas_compact,
    _particle_compact,
    gas_adjacent_change,
    particle_adjacent_change,
)
from pu_dcgp_comsol.comsol.particle_physics_contract import ParticlePhysicsContract
from pu_dcgp_comsol.comsol.particle_population_v2_skeleton import (
    audit_model as audit_particle_skeleton,
    build_model as build_particle_skeleton,
)
from pu_dcgp_comsol.comsol.particle_radial_enthalpy_nominal import (
    _sha256,
    audit_build as audit_particle_build,
    build_model as build_particle_model,
    solve_and_audit as solve_particles,
)


SUMMARY_PATH = OUTPUT_DIR / "h11_corrected_gas_crosswind_off_convergence_summary.json"


def particle_paths(level: int) -> dict[str, Path]:
    stem = f"corrected_t11160_u1090_crosswind_off_mesh_level{level}_particles_n1023"
    return {
        "skeleton": MODEL_DIR / "particle_skeletons" / f"{stem}_skeleton.mph",
        "model": MODEL_DIR / "particles" / f"{stem}.mph",
        "audit": OUTPUT_DIR / "particle_cases" / f"{stem}.json",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def solve_particle_level(client: Any, level: int) -> dict[str, Any]:
    contract = CrosswindOffContract()
    contract.validate()
    gas_artifact = gas_paths(level)
    gas_audit = _load_json(gas_artifact["audit"])
    if gas_audit is None:
        raise FileNotFoundError(gas_artifact["audit"])
    if gas_audit["status"] != "pass_corrected_gas_crosswind_off_case":
        raise RuntimeError(f"Crosswind-off gas mesh level {level} has not passed")
    paths = particle_paths(level)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    particle_contract = ParticlePhysicsContract()
    particle_contract.validate()
    skeleton, skeleton_java = build_particle_skeleton(
        client,
        particle_contract,
        source_model=gas_artifact["model"],
    )
    try:
        skeleton_audit = audit_particle_skeleton(skeleton_java, particle_contract)
        skeleton.save(str(paths["skeleton"]))
        skeleton_audit.update(
            {
                "model_path": str(paths["skeleton"].resolve()),
                "model_sha256": _sha256(paths["skeleton"]),
            }
        )
    finally:
        client.remove(skeleton)

    source_study = "std_refine"
    model, jm = build_particle_model(
        client,
        paths["skeleton"],
        source_study=source_study,
    )
    started = time.time()
    try:
        jm.param().set("particles_per_size", str(contract.particles_per_size))
        jm.param().set(
            "particle_output_step",
            f"{contract.particle_output_step_us:.12g}[us]",
        )
        build_audit = audit_particle_build(jm, source_study=source_study)
        solve_audit = solve_particles(
            model,
            jm,
            maximum_step_s=contract.particle_maximum_step_us * 1e-6,
        )
        model.save(str(paths["model"]))
        solve_audit.update(
            {
                "model_path": str(paths["model"].resolve()),
                "model_sha256": _sha256(paths["model"]),
            }
        )
        payload = {
            "schema_version": "h11_corrected_gas_crosswind_off_particle_case_v1",
            "status": (
                "pass_corrected_gas_crosswind_off_particle_case"
                if solve_audit["status"] == "pass_nominal_comsol_trajectory_audit"
                else "fail_corrected_gas_crosswind_off_particle_case"
            ),
            "mesh_level": level,
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_gas_model": str(gas_artifact["model"].resolve()),
            "source_gas_model_sha256": gas_audit["model_sha256"],
            "source_gas_study": source_study,
            "particle_skeleton_audit": skeleton_audit,
            "particle_build_audit": build_audit,
            "particle_solve_audit": solve_audit,
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(paths["audit"], payload)
        return payload
    finally:
        client.remove(model)


def build_summary() -> dict[str, Any] | None:
    contract = CrosswindOffContract()
    contract.validate()
    gas_payloads = [_load_json(gas_paths(level)["audit"]) for level in (4, 3, 2)]
    particle_payloads = [
        _load_json(particle_paths(level)["audit"]) for level in (4, 3, 2)
    ]
    if any(payload is None for payload in (*gas_payloads, *particle_payloads)):
        return None

    gas_cases = [
        _gas_compact(level, payload, payload["mesh"])
        for level, payload in zip((4, 3, 2), gas_payloads)
    ]
    particle_cases = [
        _particle_compact(level, payload)
        for level, payload in zip((4, 3, 2), particle_payloads)
    ]
    gas_changes = [
        gas_adjacent_change(left, right)
        for left, right in zip(gas_cases[:-1], gas_cases[1:])
    ]
    particle_changes = [
        particle_adjacent_change(left, right)
        for left, right in zip(particle_cases[:-1], particle_cases[1:])
    ]
    finest_gas = gas_changes[-1]
    finest_particle = particle_changes[-1]
    expected_stabilization = {
        "StreamlineDiffusion": 1,
        "RANSStreamlineDiffusion": 1,
        "heatStreamlineDiffusion": 1,
        "CrosswindDiffusion": 0,
        "RANSCrosswindDiffusion": 0,
        "heatCrosswindDiffusion": 0,
    }
    gates = {
        "all_gas_cases_passed": all(
            payload["status"] == "pass_corrected_gas_crosswind_off_case"
            for payload in gas_payloads
        ),
        "mesh_identities_match_contract": all(
            case["mesh"]["elements"] == contract.expected_element_count(level)
            for level, case in zip((4, 3, 2), gas_cases)
        ),
        "mesh_elements_increase_strictly": all(
            right["mesh"]["elements"] > left["mesh"]["elements"]
            for left, right in zip(gas_cases[:-1], gas_cases[1:])
        ),
        "identical_stabilization_on_all_meshes": all(
            payload["stabilization"] == expected_stabilization
            for payload in gas_payloads
        ),
        "all_gas_cases_conservative": all(
            case["mass_imbalance_fraction"]
            <= contract.mass_imbalance_limit_fraction
            and case["energy_imbalance_fraction"]
            <= contract.energy_imbalance_limit_fraction
            for case in gas_cases
        ),
        "finest_gas_temperature_change_below_1_percent": (
            finest_gas["temperature_anomaly_normalized_l2"]
            <= contract.gas_temperature_anomaly_l2_limit_fraction
        ),
        "finest_gas_speed_change_below_1_percent": (
            finest_gas["speed_normalized_l2"]
            <= contract.gas_speed_l2_limit_fraction
        ),
        "finest_gas_pressure_change_below_1e_minus_4": (
            finest_gas["pressure_l2_over_ambient"]
            <= contract.gas_pressure_l2_over_ambient_limit_fraction
        ),
        "all_particle_cases_passed": all(
            case["status"] == "pass_nominal_comsol_trajectory_audit"
            and case["released_particle_count"] == case["crossing_count"]
            and case["diameter_nodes_represented"] == 7
            for case in particle_cases
        ),
        "finest_primary_aperture_count_at_least_70": (
            particle_cases[-1]["selected_particle_count"]
            >= contract.minimum_primary_aperture_particles
        ),
        "finest_particle_quantile_change_below_1_percent": (
            finest_particle["maximum_relative_change"]
            <= contract.particle_quantile_limit_fraction
        ),
    }
    return {
        "schema_version": "h11_corrected_gas_crosswind_off_convergence_summary_v1",
        "status": (
            "pass_crosswind_off_gas_and_particle_mesh_independence"
            if all(gates.values())
            else "fail_crosswind_off_gas_or_particle_mesh_independence"
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "gas_cases": gas_cases,
        "gas_adjacent_changes": gas_changes,
        "particle_cases": particle_cases,
        "particle_adjacent_changes": particle_changes,
        "gates": gates,
        "claim_boundary": (
            "Passing establishes numerical mesh independence for the frozen "
            "crosswind-off equations and DPV operator; it does not establish "
            "experimental predictive validity."
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument(
        "--particle-level",
        type=int,
        action="append",
        choices=(4, 3, 2),
    )
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(CONTRACT_PATH)
    if not args.summarize_existing:
        import mph

        client = mph.start(cores=args.cores, version=args.version)
        try:
            for level in args.particle_level or []:
                print(f"Solving particles on crosswind-off mesh level {level}")
                payload = solve_particle_level(client, level)
                print(level, payload["status"])
        finally:
            client.clear()
    summary = build_summary()
    if summary is None:
        print("Crosswind-off gas/particle mesh ladder is incomplete")
        return 0
    _write_json(args.output, summary)
    print(args.output)
    print(summary["status"])
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
