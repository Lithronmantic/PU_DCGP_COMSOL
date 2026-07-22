"""Run one gas mesh with the uniform recovered segregated damping."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    configure_studies,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_conservative_free_jet_skeleton import (
    MODEL_PATH as GAS_SKELETON_MODEL,
)
from simulator_v2.phase_h.h11_corrected_gas_crosswind_off import (
    _set_and_audit_stabilization,
)
from simulator_v2.phase_h.h11_corrected_gas_crosswind_off_contract import (
    CrosswindOffContract,
)
from simulator_v2.phase_h.h11_corrected_gas_crosswind_off_uniform_damped_contract import (
    CONTRACT_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    UniformDampedContract,
)
from simulator_v2.phase_h.h11_corrected_gas_mesh_convergence import (
    _domain_parameters,
    _mesh_audit,
)
from simulator_v2.phase_h.h11_effective_exit_directional import _gas_gates
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256


def paths(mesh_level: int) -> dict[str, Path]:
    stem = (
        "corrected_t11160_u1090_crosswind_off_uniform_damped_"
        f"mesh_level{mesh_level}"
    )
    return {
        "model": MODEL_DIR / f"{stem}.mph",
        "audit": OUTPUT_DIR / "cases" / f"{stem}.json",
        "log": OUTPUT_DIR / "logs" / f"{stem}.log",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _set_and_audit_uniform_damping(
    jm: Any,
    solver: dict[str, Any],
    contract: UniformDampedContract,
) -> dict[str, Any]:
    studies = {
        "std1": (
            solver["continuation_solution"],
            contract.continuation_maximum_iterations,
            contract.continuation_relative_tolerance,
        ),
        "std_refine": (
            solver["refinement_solution"],
            contract.refinement_maximum_iterations,
            contract.refinement_relative_tolerance,
        ),
    }
    audit: dict[str, Any] = {}
    for study_tag, (solution_tag, maximum, tolerance) in studies.items():
        stationary = jm.sol(solution_tag).feature("s1")
        segregated = stationary.feature("se1")
        requested = {
            "ss1": contract.flow_group_damping,
            "ss2": contract.turbulence_group_damping,
        }
        for tag, damping in requested.items():
            substep = segregated.feature(tag)
            substep.set("subdtech", "const")
            substep.set("subdamp", f"{damping:.12g}")
        observed = {
            tag: {
                "method": str(segregated.feature(tag).getString("subdtech")),
                "damping": float(
                    str(segregated.feature(tag).getString("subdamp"))
                ),
            }
            for tag in requested
        }
        expected = {
            tag: {"method": "const", "damping": damping}
            for tag, damping in requested.items()
        }
        observed_maximum = int(str(segregated.getString("maxsegiter")))
        observed_tolerance = float(str(stationary.getString("stol")))
        if (
            observed != expected
            or observed_maximum != maximum
            or observed_tolerance != tolerance
        ):
            raise RuntimeError(
                f"COMSOL did not retain the solver contract for {study_tag}"
            )
        audit[study_tag] = {
            "solution": solution_tag,
            "relative_tolerance": observed_tolerance,
            "maximum_segregated_iterations": observed_maximum,
            "substeps": observed,
        }
    return audit


def solve(client: Any, mesh_level: int) -> dict[str, Any]:
    base = CrosswindOffContract()
    base.validate()
    damped = UniformDampedContract()
    damped.validate()
    if mesh_level not in damped.applies_to_mesh_levels:
        raise ValueError("Mesh level is outside the uniform ladder")
    numerical = FreeJetSolveContract(load_fractions=base.load_fractions)
    numerical.validate()
    artifact = paths(mesh_level)
    for path in artifact.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    model = client.load(str(GAS_SKELETON_MODEL))
    model.rename(artifact["model"].stem)
    jm = model.java
    client.java.showProgress(str(artifact["log"].resolve()))
    started = time.time()
    try:
        jm.param().set(
            "T_exit_eff", f"{base.effective_exit_temperature_k:.12g}[K]"
        )
        jm.param().set(
            "u_exit_eff", f"{base.effective_exit_speed_m_s:.12g}[m/s]"
        )
        stabilization = _set_and_audit_stabilization(jm, base)
        solver = configure_studies(jm, numerical)
        damping = _set_and_audit_uniform_damping(jm, solver, damped)
        mesh = jm.component("comp1").mesh("mesh1")
        mesh.autoMeshSize(mesh_level)
        mesh.feature("size").set("hauto", str(mesh_level))
        mesh.feature("size").set("custom", "off")
        mesh.run()
        mesh_audit = _mesh_audit(jm, mesh_level)
        expected_elements = base.expected_element_count(mesh_level)
        if mesh_audit["elements"] != expected_elements:
            raise RuntimeError(
                f"Mesh level {mesh_level} has {mesh_audit['elements']} "
                f"elements, expected {expected_elements}"
            )
        print(
            f"uniform-damped mesh level {mesh_level}: "
            f"{expected_elements} elements, {len(base.load_fractions)} loads",
            flush=True,
        )
        continuation_started = time.time()
        jm.study("std1").run()
        continuation_runtime = time.time() - continuation_started
        print(f"mesh level {mesh_level} reached full load", flush=True)
        refinement_started = time.time()
        jm.study("std_refine").run()
        refinement_runtime = time.time() - refinement_started
        print(f"mesh level {mesh_level} passed 1e-6 refinement", flush=True)
        metrics = evaluate_solution(model, numerical)
        gates = _gas_gates(metrics)
        domain = _domain_parameters(jm)
        expected_stabilization = {
            "StreamlineDiffusion": 1,
            "RANSStreamlineDiffusion": 1,
            "heatStreamlineDiffusion": 1,
            "CrosswindDiffusion": 0,
            "RANSCrosswindDiffusion": 0,
            "heatCrosswindDiffusion": 0,
        }
        expected_damping = {
            "ss1": {"method": "const", "damping": 0.5},
            "ss2": {"method": "const", "damping": 0.15},
        }
        gates.update(
            {
                "expected_mesh_identity": (
                    mesh_audit["elements"] == expected_elements
                ),
                "positive_mesh_quality": mesh_audit["minimum_quality"] > 0.0,
                "uniform_stabilization_flags": (
                    stabilization == expected_stabilization
                ),
                "uniform_damping_on_both_studies": all(
                    item["substeps"] == expected_damping
                    for item in damping.values()
                ),
                "full_dense_ladder_stored": (
                    metrics["n_continuation_solutions"]
                    == len(base.load_fractions)
                ),
                "fixed_40_by_140_mm_domain": domain
                == {"r_domain": 40.0, "z_domain": 140.0},
            }
        )
        model.save(str(artifact["model"]))
        return {
            "schema_version": (
                "h11_corrected_gas_crosswind_off_uniform_damped_case_v1"
            ),
            "status": (
                "pass_corrected_gas_crosswind_off_uniform_damped_case"
                if all(gates.values())
                else "fail_corrected_gas_crosswind_off_uniform_damped_gates"
            ),
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_model": str(GAS_SKELETON_MODEL.resolve()),
            "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
            "mesh_level": mesh_level,
            "mesh": mesh_audit,
            "domain_mm": domain,
            "stabilization": stabilization,
            "solver": solver,
            "uniform_damping": damping,
            "continuation_runtime_sec": continuation_runtime,
            "refinement_runtime_sec": refinement_runtime,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(artifact["model"].resolve()),
            "model_sha256": _sha256(artifact["model"]),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    except Exception as exc:
        partial = artifact["model"].with_name(
            f"{artifact['model'].stem}_partial.mph"
        )
        model.save(str(partial))
        failure = {
            "schema_version": (
                "h11_corrected_gas_crosswind_off_uniform_damped_failure_v1"
            ),
            "status": "failed_uniform_damped_gas_solve",
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "mesh_level": mesh_level,
            "error": str(exc),
            "partial_model": str(partial.resolve()),
            "partial_model_sha256": _sha256(partial),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(
            artifact["audit"].with_name(
                f"{artifact['audit'].stem}_failure.json"
            ),
            failure,
        )
        raise
    finally:
        client.java.showProgress(False)
        client.remove(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-level", type=int, required=True, choices=(4, 3, 2))
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(CONTRACT_PATH)
    import mph

    client = mph.start(cores=args.cores, version=args.version)
    try:
        payload = solve(client, args.mesh_level)
    finally:
        client.clear()
    _write_json(paths(args.mesh_level)["audit"], payload)
    print(paths(args.mesh_level)["audit"])
    print(payload["status"])
    return 0 if payload["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
