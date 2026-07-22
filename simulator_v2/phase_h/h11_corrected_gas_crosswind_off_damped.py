"""Test fixed turbulence damping on the failed fine-mesh 0.35-to-0.36 step."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_corrected_gas_crosswind_off import (
    _set_and_audit_stabilization,
)
from simulator_v2.phase_h.h11_corrected_gas_crosswind_off_contract import (
    CrosswindOffContract,
)
from simulator_v2.phase_h.h11_corrected_gas_crosswind_off_damped_contract import (
    CONTRACT_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    SOURCE_FAILURE,
    SOURCE_MODEL,
    DampedRecoveryContract,
)
from simulator_v2.phase_h.h11_corrected_gas_mesh_convergence import (
    _configure_projected_study,
    _domain_parameters,
    _mesh_audit,
    _solution_for_study,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256


AUDIT_PATH = OUTPUT_DIR / "h11_corrected_gas_crosswind_off_damped_recovery.json"
MODEL_PATH = MODEL_DIR / "corrected_t11160_u1090_crosswind_off_damped_load036.mph"
LOG_PATH = OUTPUT_DIR / "h11_corrected_gas_crosswind_off_damped_recovery.log"
RECOVERY_CHECKPOINT_PATH = MODEL_PATH.with_name(
    f"{MODEL_PATH.stem}_partial.mph"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _configure_damped_recovery(jm: Any) -> tuple[str, str, dict[str, Any]]:
    contract = DampedRecoveryContract()
    contract.validate()
    source_study = "std1"
    source_solution = _solution_for_study(jm, source_study)
    study_tag, solution_tag, solver = _configure_projected_study(
        jm,
        source_study=source_study,
        source_solution=source_solution,
        study_tag="std_damped_036",
        study_label="Crosswind-off fine-mesh damping recovery at load 0.36",
        solver_mode="ordinary_strict",
        tolerance=contract.relative_tolerance,
    )
    stationary = jm.sol(solution_tag).feature("s1")
    segregated = stationary.feature("se1")
    segregated.set("maxsegiter", str(contract.maximum_segregated_iterations))
    requested = {
        "ss1": contract.flow_group_damping,
        "ss2": contract.turbulence_group_damping,
    }
    observed: dict[str, dict[str, Any]] = {}
    for tag, damping in requested.items():
        substep = segregated.feature(tag)
        substep.set("subdtech", "const")
        substep.set("subdamp", f"{damping:.12g}")
        observed[tag] = {
            "method": str(substep.getString("subdtech")),
            "damping": float(str(substep.getString("subdamp"))),
        }
    jm.param().set("load_s", f"{contract.target_load_fraction:.12g}")
    solver["local_recovery"] = {
        "source_study": source_study,
        "source_solution": source_solution,
        "source_solution_number": "last_converged_parameter_step_0.35",
        "target_load_fraction": contract.target_load_fraction,
        "substeps": observed,
        "maximum_segregated_iterations": int(
            str(segregated.getString("maxsegiter"))
        ),
    }
    return study_tag, solution_tag, solver


def _audit_damped_recovery_solver(
    jm: Any,
) -> tuple[str, str, dict[str, Any]]:
    contract = DampedRecoveryContract()
    study_tag = "std_damped_036"
    solution_tag = _solution_for_study(jm, study_tag)
    segregated = jm.sol(solution_tag).feature("s1").feature("se1")
    observed = {
        tag: {
            "method": str(segregated.feature(tag).getString("subdtech")),
            "damping": float(
                str(segregated.feature(tag).getString("subdamp"))
            ),
        }
        for tag in ("ss1", "ss2")
    }
    expected = {
        "ss1": {
            "method": "const",
            "damping": contract.flow_group_damping,
        },
        "ss2": {
            "method": "const",
            "damping": contract.turbulence_group_damping,
        },
    }
    maximum_iterations = int(str(segregated.getString("maxsegiter")))
    if observed != expected or maximum_iterations != (
        contract.maximum_segregated_iterations
    ):
        raise RuntimeError("The saved recovery solver is not frozen")
    return study_tag, solution_tag, {
        "postprocessed_saved_solution": True,
        "local_recovery": {
            "source_solution_number": (
                "last_converged_parameter_step_0.35"
            ),
            "target_load_fraction": contract.target_load_fraction,
            "substeps": observed,
            "maximum_segregated_iterations": maximum_iterations,
        },
    }


def solve(
    client: Any,
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    contract = DampedRecoveryContract()
    contract.validate()
    base = CrosswindOffContract()
    base.validate()
    source_failure = json.loads(SOURCE_FAILURE.read_text(encoding="utf-8"))
    expected_source = {
        "schema_version": "h11_corrected_gas_crosswind_off_failure_v1",
        "status": "failed_corrected_gas_crosswind_off_solve",
        "partial_model": str(SOURCE_MODEL.resolve()),
        "partial_model_sha256": _sha256(SOURCE_MODEL),
        "paper_prediction_allowed": False,
    }
    observed_source = {
        key: source_failure.get(key) for key in expected_source
    }
    if observed_source != expected_source:
        raise RuntimeError(
            "The source checkpoint does not match the frozen audited failure"
        )
    # The exception text was written by a Chinese COMSOL process and is
    # mojibake after crossing the Java/Python console boundary.  The frozen
    # audit and model hashes above are the encoding-independent evidence.
    source_failure["error"] = "maximum segregated iteration ceiling"
    if "maximum" not in source_failure["error"] and "鏈€澶" not in source_failure["error"]:
        raise RuntimeError("The source failure is not the audited iteration ceiling")
    for path in (AUDIT_PATH, MODEL_PATH, LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    load_path = SOURCE_MODEL if checkpoint_path is None else checkpoint_path
    if not load_path.is_file():
        raise FileNotFoundError(load_path)
    model = client.load(str(load_path))
    model.rename(MODEL_PATH.stem)
    jm = model.java
    client.java.showProgress(str(LOG_PATH.resolve()))
    started = time.time()
    try:
        mesh = _mesh_audit(jm, contract.mesh_level)
        if mesh["elements"] != contract.expected_elements:
            raise RuntimeError("The recovery model is not the frozen fine mesh")
        stabilization = _set_and_audit_stabilization(jm, base)
        if checkpoint_path is None:
            study_tag, solution_tag, solver = _configure_damped_recovery(jm)
            print("same-mesh damping recovery: load 0.35 -> 0.36", flush=True)
            solve_started = time.time()
            jm.study(study_tag).run()
            solve_runtime = time.time() - solve_started
        else:
            study_tag, solution_tag, solver = _audit_damped_recovery_solver(jm)
            print("postprocessing converged load-0.36 checkpoint", flush=True)
            solve_runtime = 0.0
        metrics = evaluate_solution(
            model,
            FreeJetSolveContract(),
            expected_load_fraction=contract.target_load_fraction,
        )
        domain = _domain_parameters(jm)
        gates = {
            "same_fine_mesh": mesh["elements"] == contract.expected_elements,
            "same_40_by_140_mm_domain": domain
            == {"r_domain": 40.0, "z_domain": 140.0},
            "same_crosswind_off_stabilization": stabilization
            == {
                "StreamlineDiffusion": 1,
                "RANSStreamlineDiffusion": 1,
                "heatStreamlineDiffusion": 1,
                "CrosswindDiffusion": 0,
                "RANSCrosswindDiffusion": 0,
                "heatCrosswindDiffusion": 0,
            },
            "target_load_reached": math.isclose(
                metrics["full_load_fraction"],
                contract.target_load_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "temperature_bounded": metrics["temperature_k"]["minimum"] >= 250.0,
            "pressure_positive": metrics["absolute_pressure_pa"]["minimum"]
            >= 1000.0,
            "mass_imbalance_below_0_5_percent": metrics["mass_flux_kg_s"][
                "imbalance_fraction"
            ]
            <= base.mass_imbalance_limit_fraction,
            "energy_imbalance_below_2_percent": metrics["energy_balance_w"][
                "imbalance_fraction_of_inlet"
            ]
            <= base.energy_imbalance_limit_fraction,
        }
        model.save(str(MODEL_PATH))
        return {
            "schema_version": "h11_corrected_gas_crosswind_off_damped_recovery_v1",
            "status": (
                "pass_same_mesh_damping_recovery"
                if all(gates.values())
                else "fail_same_mesh_damping_recovery_gates"
            ),
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_model": str(SOURCE_MODEL.resolve()),
            "source_model_sha256": _sha256(SOURCE_MODEL),
            "postprocess_checkpoint": (
                None if checkpoint_path is None else str(checkpoint_path.resolve())
            ),
            "postprocess_checkpoint_sha256": (
                None if checkpoint_path is None else _sha256(checkpoint_path)
            ),
            "mesh": mesh,
            "domain_mm": domain,
            "stabilization": stabilization,
            "study": study_tag,
            "solution": solution_tag,
            "solver": solver,
            "solve_runtime_sec": solve_runtime,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(MODEL_PATH.resolve()),
            "model_sha256": _sha256(MODEL_PATH),
            "runtime_sec": time.time() - started,
            "role": "solver_diagnostic_only_not_a_paper_result",
            "paper_prediction_allowed": False,
        }
    except Exception as exc:
        partial = MODEL_PATH.with_name(
            f"{MODEL_PATH.stem}_postprocess_failure.mph"
            if checkpoint_path is not None
            else f"{MODEL_PATH.stem}_partial.mph"
        )
        model.save(str(partial))
        failure = {
            "schema_version": "h11_corrected_gas_crosswind_off_damped_failure_v1",
            "status": "failed_same_mesh_damping_recovery",
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "error": str(exc),
            "partial_model": str(partial.resolve()),
            "partial_model_sha256": _sha256(partial),
            "runtime_sec": time.time() - started,
            "paper_prediction_allowed": False,
        }
        _write_json(AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_failure.json"), failure)
        raise
    finally:
        client.java.showProgress(False)
        client.remove(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument(
        "--postprocess-checkpoint",
        action="store_true",
        help="Audit the already-converged saved recovery solution",
    )
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(CONTRACT_PATH)
    import mph

    client = mph.start(cores=args.cores, version=args.version)
    try:
        payload = solve(
            client,
            checkpoint_path=(
                RECOVERY_CHECKPOINT_PATH
                if args.postprocess_checkpoint
                else None
            ),
        )
    finally:
        client.clear()
    _write_json(AUDIT_PATH, payload)
    print(AUDIT_PATH)
    print(payload["status"])
    return 0 if payload["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
