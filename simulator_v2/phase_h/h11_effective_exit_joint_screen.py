"""Joint effective-exit verification and particle-count convergence."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_dpv_sampling_volume_contract import (
    DpvSamplingVolumeContract,
)
from simulator_v2.phase_h.h11_effective_exit_joint_screen_contract import (
    OUTPUT_PATH as CONTRACT_PATH,
    JointScreenContract,
)
from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    configure_studies,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_conservative_free_jet_skeleton import (
    MODEL_PATH as GAS_SKELETON_MODEL,
)
from simulator_v2.phase_h.h11_effective_exit_directional import _gas_gates
from simulator_v2.phase_h.h11_particle_physics_contract import (
    ParticlePhysicsContract,
)
from simulator_v2.phase_h.h11_particle_population_v2_skeleton import (
    audit_model as audit_particle_skeleton,
    build_model as build_particle_skeleton,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import (
    _sha256,
    audit_build as audit_particle_build,
    build_model as build_particle_model,
    solve_and_audit as solve_particles,
)


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "effective_exit_joint_screen"
MODEL_DIR = HERE / "comsol_models" / "h11_effective_exit_joint_screen"
SUMMARY_PATH = OUTPUT_DIR / "h11_effective_exit_joint_screen_summary.json"


def artifact_paths(particles_per_size: int | None = None) -> dict[str, Path]:
    paths = {
        "gas_model": MODEL_DIR / "joint_t11720_u1075_gas.mph",
        "gas_audit": OUTPUT_DIR / "joint_t11720_u1075_gas.json",
        "gas_log": OUTPUT_DIR / "joint_t11720_u1075_gas.log",
        "particle_skeleton": MODEL_DIR / "joint_t11720_u1075_particle_skeleton.mph",
    }
    if particles_per_size is not None:
        stem = f"joint_t11720_u1075_n{particles_per_size:04d}"
        paths.update(
            {
                "particle_model": MODEL_DIR / f"{stem}.mph",
                "particle_audit": OUTPUT_DIR / "cases" / f"{stem}.json",
            }
        )
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def solve_joint_gas(client: Any) -> dict[str, Any]:
    contract = JointScreenContract()
    contract.validate()
    paths = artifact_paths()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = client.load(str(GAS_SKELETON_MODEL))
    model.rename("joint_t11720_u1075_gas")
    jm = model.java
    client.java.showProgress(str(paths["gas_log"].resolve()))
    try:
        jm.param().set(
            "T_exit_eff",
            f"{contract.rounded_exit_temperature_k:.12g}[K]",
        )
        jm.param().set(
            "u_exit_eff",
            f"{contract.rounded_exit_speed_m_s:.12g}[m/s]",
        )
        gas_contract = FreeJetSolveContract()
        solver = configure_studies(jm, gas_contract)
        continuation_started = time.time()
        jm.study("std1").run()
        print("  joint gas 0-to-1 continuation complete", flush=True)
        refinement_started = time.time()
        jm.study("std_refine").run()
        print("  joint gas full-load refinement complete", flush=True)
        metrics = evaluate_solution(model, gas_contract)
        gates = _gas_gates(metrics)
        model.save(str(paths["gas_model"]))
        return {
            "schema_version": "h11_effective_exit_joint_screen_gas_v1",
            "status": (
                "pass_joint_screen_gas_case"
                if all(gates.values())
                else "fail_joint_screen_gas_case"
            ),
            "parameters": {
                "effective_exit_temperature_k": (
                    contract.rounded_exit_temperature_k
                ),
                "effective_exit_speed_m_s": contract.rounded_exit_speed_m_s,
            },
            "source_model": str(GAS_SKELETON_MODEL.resolve()),
            "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
            "solver": solver,
            "final_gas_study": "std_refine",
            "continuation_runtime_sec": (
                refinement_started - continuation_started
            ),
            "refinement_runtime_sec": time.time() - refinement_started,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["gas_model"].resolve()),
            "model_sha256": _sha256(paths["gas_model"]),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.java.showProgress(False)
        client.remove(model)


def audit_existing_joint_gas(client: Any) -> dict[str, Any]:
    contract = JointScreenContract()
    paths = artifact_paths()
    model = client.load(str(paths["gas_model"]))
    try:
        jm = model.java
        temperature = float(str(jm.param().evaluate("T_exit_eff")))
        speed = float(str(jm.param().evaluate("u_exit_eff")))
        if not math.isclose(
            temperature,
            contract.rounded_exit_temperature_k,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("Saved joint gas temperature does not match")
        if not math.isclose(
            speed,
            contract.rounded_exit_speed_m_s,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("Saved joint gas speed does not match")
        metrics = evaluate_solution(model, FreeJetSolveContract())
        gates = _gas_gates(metrics)
        return {
            "schema_version": "h11_effective_exit_joint_screen_gas_v1",
            "status": (
                "pass_joint_screen_gas_case"
                if all(gates.values())
                else "fail_joint_screen_gas_case"
            ),
            "parameters": {
                "effective_exit_temperature_k": temperature,
                "effective_exit_speed_m_s": speed,
            },
            "source_model": "re_audited_saved_joint_gas_model",
            "final_gas_study": "std_refine",
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["gas_model"].resolve()),
            "model_sha256": _sha256(paths["gas_model"]),
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.remove(model)


def build_joint_particle_skeleton(
    client: Any,
    gas_audit: dict[str, Any],
) -> dict[str, Any]:
    paths = artifact_paths()
    contract = ParticlePhysicsContract()
    contract.validate()
    model, jm = build_particle_skeleton(
        client,
        contract,
        source_model=paths["gas_model"],
    )
    try:
        audit = audit_particle_skeleton(jm, contract)
        model.save(str(paths["particle_skeleton"]))
        audit.update(
            {
                "source_model": str(paths["gas_model"].resolve()),
                "source_model_sha256": gas_audit["model_sha256"],
                "model_path": str(paths["particle_skeleton"].resolve()),
                "model_sha256": _sha256(paths["particle_skeleton"]),
            }
        )
        return audit
    finally:
        client.remove(model)


def solve_particle_count_case(
    client: Any,
    particles_per_size: int,
    skeleton_audit: dict[str, Any],
) -> dict[str, Any]:
    paths = artifact_paths(particles_per_size)
    model, jm = build_particle_model(
        client,
        artifact_paths()["particle_skeleton"],
        source_study="std_refine",
    )
    try:
        jm.param().set("particles_per_size", str(particles_per_size))
        jm.param().set("particle_output_step", "10[us]")
        build_audit = audit_particle_build(jm, source_study="std_refine")
        solve_audit = solve_particles(model, jm)
        model.save(str(paths["particle_model"]))
        solve_audit.update(
            {
                "model_path": str(paths["particle_model"].resolve()),
                "model_sha256": _sha256(paths["particle_model"]),
            }
        )
        return {
            "schema_version": "h11_effective_exit_joint_particle_count_v1",
            "status": (
                "pass_joint_particle_count_case"
                if solve_audit["status"]
                == "pass_nominal_comsol_trajectory_audit"
                else "fail_joint_particle_count_case"
            ),
            "particles_per_size": particles_per_size,
            "particle_output_step_us": 10.0,
            "particle_skeleton_audit": skeleton_audit,
            "particle_build_audit": build_audit,
            "particle_solve_audit": solve_audit,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.remove(model)


def _aperture(
    solve_audit: dict[str, Any],
    radius_mm: float,
) -> dict[str, Any]:
    apertures = solve_audit["trajectory_audit"][
        "centerline_aperture_sensitivity"
    ]
    matches = [
        item
        for item in apertures
        if math.isclose(
            float(item["aperture_radius_mm"]),
            radius_mm,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one aperture at {radius_mm} mm")
    return matches[0]


def compact_particle_count_case(
    particles_per_size: int,
    solve_audit: dict[str, Any],
) -> dict[str, Any]:
    sampling = DpvSamplingVolumeContract()
    radii = {
        "low_speed_primary": sampling.low_speed_equivalent_radius_mm,
        "high_speed_sensitivity": sampling.high_speed_equivalent_radius_mm,
    }
    compact_apertures = {}
    for label, radius in radii.items():
        aperture = _aperture(solve_audit, radius)
        compact_apertures[label] = {
            "radius_mm": radius,
            "selected_particle_count": aperture["selected_particle_count"],
            "diameter_nodes_represented": aperture["diameter_nodes_represented"],
            "median": aperture[
                "empirical_detected_diameter_weighted_median"
            ],
            "quantiles": aperture[
                "empirical_detected_diameter_weighted_quantiles"
            ],
        }
    trajectory = solve_audit["trajectory_audit"]
    return {
        "particles_per_size": particles_per_size,
        "released_particle_count": trajectory["released_particle_count"],
        "crossing_count": trajectory["crossing_count"],
        "status": solve_audit["status"],
        "all_diameter_nodes_reach_observation_plane": trajectory[
            "all_diameter_nodes_reach_observation_plane"
        ],
        "apertures": compact_apertures,
    }


def _relative_change(left: float, right: float) -> float:
    return abs(right - left) / max(abs(right), 1e-12)


def adjacent_count_changes(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(cases, key=lambda item: item["particles_per_size"])
    comparisons = []
    for left, right in zip(ordered, ordered[1:]):
        changes: dict[str, Any] = {}
        for aperture_name in left["apertures"]:
            aperture_changes = {}
            for outcome in ("temperature_c", "speed_m_s"):
                aperture_changes[outcome] = {
                    quantile: _relative_change(
                        left["apertures"][aperture_name]["quantiles"][outcome][
                            quantile
                        ],
                        right["apertures"][aperture_name]["quantiles"][outcome][
                            quantile
                        ],
                    )
                    for quantile in ("q10", "q50", "q90")
                }
            changes[aperture_name] = aperture_changes
        maximum = max(
            value
            for aperture in changes.values()
            for outcome in aperture.values()
            for value in outcome.values()
        )
        comparisons.append(
            {
                "left_particles_per_size": left["particles_per_size"],
                "right_particles_per_size": right["particles_per_size"],
                "changes": changes,
                "maximum_relative_change": maximum,
            }
        )
    return comparisons


def build_summary(
    gas_audit: dict[str, Any],
    particle_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = JointScreenContract()
    contract.validate()
    comparisons = adjacent_count_changes(particle_cases)
    finest = max(particle_cases, key=lambda item: item["particles_per_size"])
    final_change = comparisons[-1]["maximum_relative_change"]
    gates = {
        "gas_conservation_and_bounds": (
            gas_audit["status"] == "pass_joint_screen_gas_case"
        ),
        "all_particle_cases_solved": all(
            case["status"] == "pass_nominal_comsol_trajectory_audit"
            for case in particle_cases
        ),
        "all_diameter_nodes_cross_at_finest_count": finest[
            "all_diameter_nodes_reach_observation_plane"
        ],
        "primary_aperture_count_at_least_70": (
            finest["apertures"]["low_speed_primary"][
                "selected_particle_count"
            ]
            >= contract.minimum_primary_aperture_particles
        ),
        "finest_adjacent_quantile_change_below_1_percent": (
            final_change
            <= contract.adjacent_quantile_change_limit_fraction
        ),
    }
    return {
        "schema_version": "h11_effective_exit_joint_screen_summary_v1",
        "status": (
            "pass_joint_screen_particle_convergence"
            if all(gates.values())
            else "fail_joint_screen_or_particle_convergence"
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "gas_audit": gas_audit,
        "particle_cases": particle_cases,
        "adjacent_count_changes": comparisons,
        "gates": gates,
        "interpretation": (
            "A passing result establishes a numerically resolved pooled-data "
            "screening point at the instrument-informed aperture. It does not "
            "replace grouped held-out calibration."
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def run_joint_screen(
    *,
    selected_counts: set[int] | None,
    cores: int,
    version: str,
    reuse_existing_gas: bool,
) -> None:
    contract = JointScreenContract()
    contract.validate()
    counts = [
        count
        for count in contract.particle_counts_per_size
        if selected_counts is None or count in selected_counts
    ]

    import mph

    client = mph.start(cores=cores, version=version)
    try:
        gas_path = artifact_paths()["gas_model"]
        if reuse_existing_gas and gas_path.exists():
            gas_audit = audit_existing_joint_gas(client)
            print("Re-audited saved joint gas model", flush=True)
        else:
            print(
                "Solving joint gas: T=11720 K, U=1075 m/s",
                flush=True,
            )
            gas_audit = solve_joint_gas(client)
        _write_json(artifact_paths()["gas_audit"], gas_audit)
        if gas_audit["status"] != "pass_joint_screen_gas_case":
            print(f"Joint gas failed gates: {gas_audit['gates']}", flush=True)
            return
        skeleton_audit = build_joint_particle_skeleton(client, gas_audit)
        for index, count in enumerate(counts, start=1):
            print(
                f"[{index}/{len(counts)}] particles_per_size={count} "
                f"(total={7*count})",
                flush=True,
            )
            started = time.time()
            payload = solve_particle_count_case(
                client,
                count,
                skeleton_audit,
            )
            _write_json(artifact_paths(count)["particle_audit"], payload)
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
    finally:
        client.clear()


def load_completed_summary_inputs(
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    gas_path = artifact_paths()["gas_audit"]
    if not gas_path.exists():
        return None
    gas_audit = json.loads(gas_path.read_text(encoding="utf-8"))
    cases = []
    for count in JointScreenContract().particle_counts_per_size:
        path = artifact_paths(count)["particle_audit"]
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(
            compact_particle_count_case(
                count,
                payload["particle_solve_audit"],
            )
        )
    return gas_audit, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--reuse-existing-gas", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument(
        "--count",
        action="append",
        type=int,
        choices=list(JointScreenContract().particle_counts_per_size),
    )
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(CONTRACT_PATH)
    if not args.summarize_existing:
        run_joint_screen(
            selected_counts=set(args.count) if args.count else None,
            cores=args.cores,
            version=args.version,
            reuse_existing_gas=args.reuse_existing_gas,
        )
    completed = load_completed_summary_inputs()
    if completed is None:
        print("Joint-screen count ladder is incomplete; artifacts were retained.")
        return 0
    summary = build_summary(*completed)
    _write_json(args.output, summary)
    print(f"Summary: {args.output}")
    print(f"Joint-screen status: {summary['status']}")
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
