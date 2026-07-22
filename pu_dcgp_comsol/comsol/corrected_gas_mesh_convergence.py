
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from pu_dcgp_comsol.comsol.conservative_free_jet_nominal import (
    FreeJetSolveContract,
    _configure_segregated_solver,
    evaluate_solution,
)
from pu_dcgp_comsol.comsol.corrected_gas_mesh_convergence_contract import (
    OUTPUT_PATH as CONTRACT_PATH,
    CorrectedGasMeshConvergenceContract,
)
from pu_dcgp_comsol.comsol.dpv_sampling_volume_contract import (
    DpvSamplingVolumeContract,
)
from pu_dcgp_comsol.comsol.effective_exit_directional import _gas_gates
from pu_dcgp_comsol.comsol.effective_exit_joint_correction import SPEC
from pu_dcgp_comsol.comsol.particle_physics_contract import (
    ParticlePhysicsContract,
)
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


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "corrected_gas_mesh_convergence"
MODEL_DIR = HERE / "comsol_models" / "h11_corrected_gas_mesh_convergence"
SUMMARY_PATH = OUTPUT_DIR / "h11_corrected_gas_mesh_convergence_summary.json"
BASELINE_GAS_AUDIT = SPEC.paths()["gas_audit"]
BASELINE_GAS_MODEL = SPEC.paths()["gas_model"]
BASELINE_PARTICLE_AUDIT = SPEC.paths(1023)["particle_audit"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def gas_paths(level: int) -> dict[str, Path]:
    stem = f"corrected_t11160_u1090_mesh_level{level}_gas"
    return {
        "model": MODEL_DIR / f"{stem}.mph",
        "audit": OUTPUT_DIR / "cases" / f"{stem}.json",
        "log": OUTPUT_DIR / "logs" / f"{stem}.log",
    }


def bridge_paths() -> dict[str, Path]:
    stem = "corrected_t11160_u1090_mesh_conditional_bridge_gas"
    return {
        "model": MODEL_DIR / f"{stem}.mph",
        "audit": OUTPUT_DIR / "cases" / f"{stem}.json",
        "log": OUTPUT_DIR / "logs" / f"{stem}.log",
    }


def level2_preconditioner_paths() -> dict[str, Path]:
    stem = "corrected_t11160_u1090_mesh_level2_preconditioner"
    return {
        "model": MODEL_DIR / f"{stem}.mph",
        "audit": OUTPUT_DIR / "cases" / f"{stem}.json",
    }


def particle_paths(level: int) -> dict[str, Path]:
    stem = f"corrected_t11160_u1090_mesh_level{level}_n1023"
    return {
        "skeleton": MODEL_DIR / f"{stem}_skeleton.mph",
        "model": MODEL_DIR / f"{stem}.mph",
        "audit": OUTPUT_DIR / "cases" / f"{stem}.json",
    }


def _solution_for_study(jm: Any, study_tag: str) -> str:
    matches = [
        str(tag)
        for tag in jm.sol().tags()
        if str(jm.sol(str(tag)).study()) == study_tag
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one solution for {study_tag}: {matches}"
        )
    return matches[0]


def _mesh_audit(jm: Any, level: int | str) -> dict[str, Any]:
    mesh = jm.component("comp1").mesh("mesh1")
    return {
        "automatic_level": level,
        "elements": int(mesh.getNumElem()),
        "vertices": int(mesh.getNumVertex()),
        "minimum_quality": float(mesh.getMinQuality()),
        "mean_quality": float(mesh.getMeanQuality()),
    }


def _domain_parameters(jm: Any) -> dict[str, float]:
    return {
        name: float(str(jm.param().evaluate(name))) * 1000.0
        for name in ("r_domain", "z_domain")
    }


def _configure_projected_study(
    jm: Any,
    *,
    source_study: str,
    source_solution: str,
    study_tag: str,
    study_label: str,
    solver_mode: str,
    tolerance: float | None = None,
) -> tuple[str, str, dict[str, Any]]:
    if solver_mode not in {
        "ordinary_strict",
        "highly_nonlinear",
        "fixed_damped_level2",
        "interpolated_pid_level2",
    }:
        raise ValueError("Unknown projected-study solver mode")
    hmnf = jm.component("comp1").physics("hmnf")
    initial = hmnf.feature("init1")
    initial.set("u_init", ["0", "0", "0"])
    initial.set("p_init", "p_amb")
    initial.set("Tinit", "T_amb")
    initial.set("k_init", "hmnf.kinit")
    initial.set("om_init", "hmnf.omInit")

    if study_tag in {str(tag) for tag in jm.study().tags()}:
        jm.study().remove(study_tag)
    study = jm.study().create(study_tag)
    study.label(study_label)
    step = study.create("stat", "Stationary")
    physics = [str(tag) for tag in jm.component("comp1").physics().tags()]
    activation: list[str] = []
    for tag in physics:
        activation.extend([tag, "on" if tag == "hmnf" else "off"])
    step.set("activate", activation)
    step.set("useinitsol", "on")
    step.set("initmethod", "sol")
    step.set("initstudy", source_study)
    step.set("initstudystep", "stat")
    step.set("initsol", source_solution)
    step.set("solnum", "last")
    study.createAutoSequences("all")
    solution_tag = _solution_for_study(jm, study_tag)
    gas_contract = FreeJetSolveContract()
    solver_tolerance = (
        gas_contract.refinement_relative_tolerance
        if tolerance is None
        else tolerance
    )
    if solver_tolerance <= 0:
        raise ValueError("Solver tolerance must be positive")
    solver = _configure_segregated_solver(
        jm,
        solution_tag,
        tolerance=solver_tolerance,
        maximum_iterations=(
            gas_contract.refinement_maximum_segregated_iterations
        ),
        contract=gas_contract,
    )
    variable_tags = [
        str(tag)
        for tag in jm.sol(solution_tag).feature().tags()
        if str(jm.sol(solution_tag).feature(str(tag)).getType())
        == "Variables"
    ]
    if len(variable_tags) != 1:
        raise RuntimeError(
            f"Expected one variables feature for wall scaling: {variable_tags}"
        )
    variables = jm.sol(solution_tag).feature(variable_tags[0])
    wall_temperature_scales: dict[str, dict[str, str]] = {}
    for field_tag in ("comp1_hmnf_TWall_d", "comp1_hmnf_TWall_u"):
        field = variables.feature(field_tag)
        field.set("scalemethod", "manual")
        field.set(
            "scaleval",
            f"{CorrectedGasMeshConvergenceContract().level2_wall_temperature_scale_k:.12g}",
        )
        wall_temperature_scales[field_tag] = {
            "scalemethod": str(field.getString("scalemethod")),
            "scaleval": str(field.getString("scaleval")),
        }
    solver["wall_temperature_field_scales"] = wall_temperature_scales
    if solver_mode == "highly_nonlinear":
        stationary = jm.sol(solution_tag).feature("s1")
        segregated = stationary.feature("se1")
        nonlinear_initialization: dict[str, dict[str, str]] = {}
        for substep_tag in ("ss1", "ss2"):
            substep = segregated.feature(substep_tag)
            substep.set("subdtech", "hnlin")
            substep.set("subinitsteph", "1e-4")
            substep.set("subminsteph", "1e-12")
            substep.set("usesubminsteprecovery", "on")
            substep.set("subminsteprecovery", "0.1")
            substep.set("subtermonres", "off")
            substep.set("subtermauto", "itertol")
            substep.set("subiter", "20")
            substep.set("maxsubiter", "100")
            nonlinear_initialization[substep_tag] = {
                key: str(substep.getString(key))
                for key in (
                    "subdtech",
                    "subinitsteph",
                    "subminsteph",
                    "usesubminsteprecovery",
                    "subminsteprecovery",
                    "subtermonres",
                    "subtermauto",
                    "subiter",
                    "maxsubiter",
                )
            }
        solver["nonlinear_initialization"] = nonlinear_initialization
    elif solver_mode in {
        "fixed_damped_level2",
        "interpolated_pid_level2",
    }:
        stationary = jm.sol(solution_tag).feature("s1")
        segregated = stationary.feature("se1")
        contract = CorrectedGasMeshConvergenceContract()
        fixed_damping = {
            "ss1": contract.level2_flow_damping,
            "ss2": contract.level2_turbulence_damping,
        }
        solver["nonlinear_initialization"] = {
            tag: {
                "mode": "fixed_damped_strict_segregated",
                "subdtech": "const",
                "subdamp": f"{damping:.12g}",
            }
            for tag, damping in fixed_damping.items()
        }
        for tag, damping in fixed_damping.items():
            substep = segregated.feature(tag)
            substep.set("subdtech", "const")
            substep.set("subdamp", f"{damping:.12g}")
        if solver_mode == "interpolated_pid_level2":
            proportional, integral, derivative = (
                contract.level2_pseudo_time_pid
            )
            pseudo_time_settings = {
                "segstabacc": "segcflcmp",
                "segcfltech": contract.level2_pseudo_time_controller,
                "subforcecfl": "on",
                "subinitcfl": (
                    f"{contract.level2_pseudo_time_initial_cfl:.12g}"
                ),
                "submincfl": (
                    f"{contract.level2_pseudo_time_target_cfl:.12g}"
                ),
                "subcfltol": (
                    f"{contract.level2_pseudo_time_target_error:.12g}"
                ),
                "subkppid": f"{proportional:.12g}",
                "subkipid": f"{integral:.12g}",
                "subkdpid": f"{derivative:.12g}",
                "segcflaa": (
                    "on"
                    if contract.level2_anderson_acceleration
                    else "off"
                ),
            }
            for key, value in pseudo_time_settings.items():
                segregated.set(key, value)
            solver["pseudo_time_restart"] = {
                key: str(segregated.getString(key))
                for key in pseudo_time_settings
            }
            solver["pseudo_time_restart"]["source"] = (
                "COMSOL 6.3 documented interpolated PID option and default "
                "PID gains; Anderson acceleration disabled after the prior "
                "target-CFL trace reversed from its best error."
            )
    else:
        solver["nonlinear_initialization"] = {
            "mode": "ordinary_strict_segregated_defaults",
            "reason": (
                "The level-3 source already solved with this strict default "
                "strategy; strong level-2 damping is reserved for the final jump."
            ),
        }
    jm.param().set("load_s", "1")
    return study_tag, solution_tag, solver


def _configure_mesh_refinement(
    jm: Any,
    *,
    source_study: str,
    source_solution: str,
    target_level: int,
    solver_mode: str,
    study_tag: str | None = None,
    study_label: str | None = None,
    tolerance: float | None = None,
) -> tuple[str, str, dict[str, Any]]:
    contract = CorrectedGasMeshConvergenceContract()
    contract.validate()
    if target_level not in contract.automatic_mesh_levels[1:]:
        raise ValueError("Only mesh levels 3 and 2 are target refinements")
    mesh = jm.component("comp1").mesh("mesh1")
    mesh.autoMeshSize(target_level)
    mesh.feature("size").set("hauto", str(target_level))
    mesh.feature("size").set("custom", "off")
    mesh.run()
    return _configure_projected_study(
        jm,
        source_study=source_study,
        source_solution=source_solution,
        study_tag=study_tag or f"std_mesh_{target_level}",
        study_label=study_label
        or f"Corrected free-jet automatic mesh level {target_level}",
        solver_mode=solver_mode,
        tolerance=tolerance,
    )


def _configure_fully_coupled_level2(
    jm: Any,
    *,
    source_study: str,
    source_solution: str,
) -> tuple[str, str, dict[str, Any]]:
    contract = CorrectedGasMeshConvergenceContract()
    study_tag, solution_tag, solver = _configure_projected_study(
        jm,
        source_study=source_study,
        source_solution=source_solution,
        study_tag="std_mesh_2",
        study_label=(
            "Corrected level-2 final solution-and-residual verification"
        ),
        solver_mode="ordinary_strict",
        tolerance=contract.level2_final_relative_tolerance,
    )
    stationary = jm.sol(solution_tag).feature("s1")
    child_tags = {
        str(value) for value in stationary.feature().tags()
    }
    if "se1" not in child_tags or "d1" not in child_tags:
        raise RuntimeError(
            "Expected auto-generated segregated and direct solver nodes"
        )
    segregated = stationary.feature("se1")
    fully = (
        stationary.feature("fc1")
        if "fc1" in child_tags
        else stationary.feature().create("fc1", "FullyCoupled")
    )
    fully.active(True)
    segregated.active(False)
    fully.set("linsolver", "d1")
    fully.set("dtech", "hnlin")
    fully.set(
        "initsteph",
        f"{contract.level2_fully_coupled_initial_damping:.12g}",
    )
    fully.set(
        "minsteph",
        f"{contract.level2_fully_coupled_minimum_damping:.12g}",
    )
    fully.set("useminsteprecovery", "on")
    fully.set(
        "minsteprecovery",
        f"{contract.level2_fully_coupled_recovery_damping:.12g}",
    )
    fully.set(
        "maxiter",
        str(contract.level2_fully_coupled_maximum_iterations),
    )
    fully.set("termonres", "both")
    fully.set(
        "reserrfact",
        f"{contract.level2_fully_coupled_residual_factor:.12g}",
    )
    stationary.feature().move("fc1", 2)
    direct = stationary.feature("d1")
    solver["final_nonlinear_solver"] = {
        "method": "fully_coupled_highly_nonlinear_newton",
        "active": bool(fully.isActive()),
        "segregated_active": bool(segregated.isActive()),
        "linear_solver_tag": str(fully.getString("linsolver")),
        "linear_solver_algorithm": str(direct.getString("linsolver")),
        "nonlinear_method": str(fully.getString("dtech")),
        "termination_criterion": str(fully.getString("termonres")),
        "residual_factor": float(str(fully.getString("reserrfact"))),
        "maximum_iterations": int(str(fully.getString("maxiter"))),
        "initial_damping": float(str(fully.getString("initsteph"))),
        "minimum_damping": float(str(fully.getString("minsteph"))),
        "use_minimum_step_recovery": str(
            fully.getString("useminsteprecovery")
        ),
        "recovery_damping": float(
            str(fully.getString("minsteprecovery"))
        ),
    }
    final = solver["final_nonlinear_solver"]
    if (
        not final["active"]
        or final["segregated_active"]
        or final["linear_solver_tag"] != "d1"
        or final["linear_solver_algorithm"] != "pardiso"
        or final["termination_criterion"] != "both"
    ):
        raise RuntimeError(
            "The final fully coupled residual-controlled solver audit failed"
        )
    return study_tag, solution_tag, solver


def _configure_conditional_bridge_mesh(jm: Any) -> dict[str, Any]:
    contract = CorrectedGasMeshConvergenceContract()
    contract.validate()
    mesh = jm.component("comp1").mesh("mesh1")
    mesh.autoMeshSize(3)
    size = mesh.feature("size")
    size.set("custom", "on")
    size.set(
        "hmax",
        f"{contract.conditional_bridge_default_hmax_mm:.12g}[mm]",
    )
    size.set(
        "hmin",
        f"{contract.conditional_bridge_default_hmin_mm:.12g}[mm]",
    )
    size.set(
        "hgrad",
        f"{contract.conditional_bridge_default_growth_rate:.12g}",
    )
    size.set(
        "hcurve",
        f"{contract.conditional_bridge_default_curvature_factor:.12g}",
    )
    mesh.run()
    return {
        **_mesh_audit(jm, "conditional_bridge"),
        "role": "conditional_initialization_bridge_not_evaluation_level",
        "default_size_settings": {
            key: str(size.getString(key))
            for key in (
                "custom",
                "hmax",
                "hmin",
                "hgrad",
                "hcurve",
                "hnarrow",
            )
        },
    }


def solve_gas_bridge(client: Any) -> dict[str, Any]:
    direct_failure = gas_paths(2)["audit"].with_name(
        f"{gas_paths(2)['audit'].stem}_failure.json"
    )
    if not direct_failure.exists():
        raise RuntimeError(
            "The conditional bridge requires a retained direct level-2 failure"
        )
    paths = bridge_paths()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    source_model = gas_paths(3)["model"]
    source_study = "std_mesh_3"
    model = client.load(str(source_model))
    model.rename(paths["model"].stem)
    jm = model.java
    client.java.showProgress(str(paths["log"].resolve()))
    started = time.time()
    try:
        source_solution = _solution_for_study(jm, source_study)
        source_mesh = _mesh_audit(jm, 3)
        source_domain = _domain_parameters(jm)
        target_mesh = _configure_conditional_bridge_mesh(jm)
        study_tag, solution_tag, solver = _configure_projected_study(
            jm,
            source_study=source_study,
            source_solution=source_solution,
            study_tag="std_mesh_bridge",
            study_label=(
                "Corrected free-jet conditional intermediate mesh bridge"
            ),
            solver_mode="ordinary_strict",
        )
        print(
            f"  gas L3->bridge: {source_mesh['elements']}->"
            f"{target_mesh['elements']} elements",
            flush=True,
        )
        try:
            jm.study(study_tag).run()
        except Exception as exc:
            partial_model = paths["model"].with_name(
                f"{paths['model'].stem}_partial.mph"
            )
            model.save(str(partial_model))
            failure = {
                "schema_version": "h11_corrected_gas_mesh_bridge_failure_v1",
                "status": "failed_conditional_mesh_bridge_solve",
                "source_model": str(source_model.resolve()),
                "source_model_sha256": _sha256(source_model),
                "source_study": source_study,
                "source_solution": source_solution,
                "target_study": study_tag,
                "target_solution": solution_tag,
                "source_mesh": source_mesh,
                "target_mesh": target_mesh,
                "solver": solver,
                "error": str(exc),
                "partial_model": str(partial_model.resolve()),
                "partial_model_sha256": _sha256(partial_model),
                "runtime_sec": time.time() - started,
                "calibrated": False,
                "paper_prediction_allowed": False,
            }
            failure_path = paths["audit"].with_name(
                f"{paths['audit'].stem}_failure.json"
            )
            _write_json(failure_path, failure)
            print(f"  wrote bridge failure audit: {failure_path}", flush=True)
            raise
        metrics = evaluate_solution(model, FreeJetSolveContract())
        gates = _gas_gates(metrics)
        domain = _domain_parameters(jm)
        gates["fixed_40_by_140_mm_domain"] = (
            domain == source_domain
            and math.isclose(domain["r_domain"], 40.0)
            and math.isclose(domain["z_domain"], 140.0)
        )
        gates["bridge_elements_between_level3_and_level2"] = (
            source_mesh["elements"]
            < target_mesh["elements"]
            < 55_291
        )
        gates["positive_mesh_quality"] = target_mesh["minimum_quality"] > 0
        model.save(str(paths["model"]))
        payload = {
            "schema_version": "h11_corrected_gas_mesh_bridge_case_v1",
            "status": (
                "pass_conditional_mesh_bridge"
                if all(gates.values())
                else "fail_conditional_mesh_bridge_gates"
            ),
            "role": "initialization_only_not_a_convergence_evaluation_level",
            "triggering_direct_failure": str(direct_failure.resolve()),
            "triggering_direct_failure_sha256": _sha256(direct_failure),
            "source_model": str(source_model.resolve()),
            "source_model_sha256": _sha256(source_model),
            "source_study": source_study,
            "source_solution": source_solution,
            "target_study": study_tag,
            "target_solution": solution_tag,
            "source_mesh": source_mesh,
            "target_mesh": target_mesh,
            "domain_mm": domain,
            "solver": solver,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["model"].resolve()),
            "model_sha256": _sha256(paths["model"]),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(paths["audit"], payload)
        return payload
    finally:
        client.java.showProgress(False)
        client.remove(model)


def solve_gas_level(
    client: Any,
    target_level: int,
    *,
    level2_source: str = "adjacent",
) -> dict[str, Any]:
    contract = CorrectedGasMeshConvergenceContract()
    if level2_source not in {"adjacent", "conditional_bridge"}:
        raise ValueError("Unknown level-2 initialization source")
    if target_level != 2 and level2_source != "adjacent":
        raise ValueError("The conditional bridge is only valid for level 2")
    if target_level == 2 and level2_source == "conditional_bridge":
        bridge = _load_json(bridge_paths()["audit"])
        if bridge is None or bridge["status"] != "pass_conditional_mesh_bridge":
            raise RuntimeError("The conditional mesh bridge has not passed")
        source_level: int | str = "conditional_bridge"
        source_model = bridge_paths()["model"]
        source_study = "std_mesh_bridge"
    else:
        source_level = target_level + 1
        source_model = (
            BASELINE_GAS_MODEL
            if source_level == 4
            else gas_paths(source_level)["model"]
        )
        source_study = (
            "std_refine"
            if source_level == 4
            else f"std_mesh_{source_level}"
        )
    if not source_model.exists():
        raise FileNotFoundError(source_model)
    paths = gas_paths(target_level)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    model = client.load(str(source_model))
    model.rename(paths["model"].stem)
    jm = model.java
    client.java.showProgress(str(paths["log"].resolve()))
    started = time.time()
    try:
        source_solution = _solution_for_study(jm, source_study)
        source_mesh = _mesh_audit(jm, source_level)
        source_domain = _domain_parameters(jm)
        solver_mode = (
            "highly_nonlinear"
            if target_level == 2 and level2_source == "adjacent"
            else (
                "interpolated_pid_level2"
                if target_level == 2
                else "ordinary_strict"
            )
        )
        use_level2_preconditioner = (
            target_level == 2
            and level2_source == "conditional_bridge"
        )
        first_study_tag = (
            "std_mesh_2_preconditioner"
            if use_level2_preconditioner
            else f"std_mesh_{target_level}"
        )
        first_study_label = (
            "Corrected level-2 segregated preconditioner 1e-4"
            if use_level2_preconditioner
            else f"Corrected free-jet automatic mesh level {target_level}"
        )
        study_tag, solution_tag, first_solver = _configure_mesh_refinement(
            jm,
            source_study=source_study,
            source_solution=source_solution,
            target_level=target_level,
            solver_mode=solver_mode,
            study_tag=first_study_tag,
            study_label=first_study_label,
            tolerance=(
                contract.level2_preconditioner_tolerance
                if use_level2_preconditioner
                else None
            ),
        )
        if use_level2_preconditioner:
            solver: dict[str, Any] = {
                "mode": (
                    "segregated_preconditioner_then_fully_coupled_"
                    "residual_verification"
                ),
                "preconditioner_tolerance": (
                    contract.level2_preconditioner_tolerance
                ),
                "final_relative_tolerance": (
                    contract.level2_final_relative_tolerance
                ),
                "acceptance_stage": 2,
                "physics_or_mesh_changes_between_stages": False,
                "stages": [],
            }
        else:
            solver = first_solver
        target_mesh = _mesh_audit(jm, target_level)
        print(
            f"  gas {source_level}->{target_level}: "
            f"{source_mesh['elements']}->{target_mesh['elements']} elements",
            flush=True,
        )
        if target_mesh["elements"] <= source_mesh["elements"]:
            raise RuntimeError(
                "Target automatic mesh did not refine the source mesh; "
                "the equation solve is forbidden"
            )
        current_stage: dict[str, Any] | None = None
        try:
            if use_level2_preconditioner:
                stage_started = time.time()
                current_stage = {
                    "stage": 1,
                    "role": "initialization_only_not_accepted",
                    "study": study_tag,
                    "solution": solution_tag,
                    "relative_tolerance": (
                        contract.level2_preconditioner_tolerance
                    ),
                    "initialization_study": source_study,
                    "initialization_solution": source_solution,
                    "solver": first_solver,
                    "status": "running",
                }
                solver["stages"].append(current_stage)
                print(
                    "  level-2 segregated preconditioner 1/2: "
                    f"tolerance={contract.level2_preconditioner_tolerance:.0e}",
                    flush=True,
                )
                jm.study(study_tag).run()
                current_stage["status"] = "pass"
                current_stage["runtime_sec"] = time.time() - stage_started

                checkpoint_paths = level2_preconditioner_paths()
                for path in checkpoint_paths.values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(checkpoint_paths["model"]))
                checkpoint = {
                    "schema_version": (
                        "h11_corrected_gas_mesh_level2_preconditioner_v1"
                    ),
                    "status": "pass_level2_preconditioner_not_accepted",
                    "source_model": str(source_model.resolve()),
                    "source_model_sha256": _sha256(source_model),
                    "study": study_tag,
                    "solution": solution_tag,
                    "mesh": target_mesh,
                    "domain_mm": _domain_parameters(jm),
                    "solver": first_solver,
                    "relative_tolerance": (
                        contract.level2_preconditioner_tolerance
                    ),
                    "model_path": str(
                        checkpoint_paths["model"].resolve()
                    ),
                    "model_sha256": _sha256(
                        checkpoint_paths["model"]
                    ),
                    "runtime_sec": current_stage["runtime_sec"],
                    "role": "initialization_only_not_paper_prediction",
                    "calibrated": False,
                    "paper_prediction_allowed": False,
                }
                _write_json(checkpoint_paths["audit"], checkpoint)
                current_stage["checkpoint_model"] = checkpoint[
                    "model_path"
                ]
                current_stage["checkpoint_model_sha256"] = checkpoint[
                    "model_sha256"
                ]

                preconditioner_study = study_tag
                preconditioner_solution = solution_tag
                study_tag, solution_tag, final_solver = (
                    _configure_fully_coupled_level2(
                        jm,
                        source_study=preconditioner_study,
                        source_solution=preconditioner_solution,
                    )
                )
                stage_started = time.time()
                current_stage = {
                    "stage": 2,
                    "role": "final_acceptance",
                    "study": study_tag,
                    "solution": solution_tag,
                    "relative_tolerance": (
                        contract.level2_final_relative_tolerance
                    ),
                    "initialization_study": preconditioner_study,
                    "initialization_solution": preconditioner_solution,
                    "solver": final_solver,
                    "status": "running",
                }
                solver["stages"].append(current_stage)
                print(
                    "  level-2 fully coupled residual verification 2/2: "
                    f"tolerance={contract.level2_final_relative_tolerance:.0e}",
                    flush=True,
                )
                jm.study(study_tag).run()
                current_stage["status"] = "pass"
                current_stage["runtime_sec"] = time.time() - stage_started
            else:
                jm.study(study_tag).run()
        except Exception as exc:
            if current_stage is not None:
                current_stage["status"] = "failed"
                current_stage["runtime_sec"] = time.time() - stage_started
            partial_model = paths["model"].with_name(
                f"{paths['model'].stem}_partial.mph"
            )
            model.save(str(partial_model))
            failure = {
                "schema_version": "h11_corrected_gas_mesh_failure_v1",
                "status": "failed_corrected_gas_mesh_equation_solve",
                "source_level": source_level,
                "target_level": target_level,
                "source_model": str(source_model.resolve()),
                "source_model_sha256": _sha256(source_model),
                "source_study": source_study,
                "source_solution": source_solution,
                "target_study": study_tag,
                "target_solution": solution_tag,
                "source_mesh": source_mesh,
                "target_mesh": target_mesh,
                "domain_mm": _domain_parameters(jm),
                "solver": solver,
                "error": str(exc),
                "partial_model": str(partial_model.resolve()),
                "partial_model_sha256": _sha256(partial_model),
                "runtime_sec": time.time() - started,
                "calibrated": False,
                "paper_prediction_allowed": False,
            }
            failure_path = paths["audit"].with_name(
                f"{paths['audit'].stem}_failure.json"
            )
            _write_json(failure_path, failure)
            print(f"  wrote failure audit: {failure_path}", flush=True)
            raise
        metrics = evaluate_solution(model, FreeJetSolveContract())
        gates = _gas_gates(metrics)
        domain = _domain_parameters(jm)
        gates["fixed_40_by_140_mm_domain"] = (
            math.isclose(domain["r_domain"], contract.radial_domain_mm)
            and math.isclose(domain["z_domain"], contract.axial_domain_mm)
            and domain == source_domain
        )
        gates["target_mesh_has_more_elements"] = (
            target_mesh["elements"] > source_mesh["elements"]
        )
        gates["positive_mesh_quality"] = target_mesh["minimum_quality"] > 0
        model.save(str(paths["model"]))
        payload = {
            "schema_version": "h11_corrected_gas_mesh_case_v1",
            "status": (
                "pass_corrected_gas_mesh_case"
                if all(gates.values())
                else "fail_corrected_gas_mesh_case"
            ),
            "source_level": source_level,
            "target_level": target_level,
            "source_model": str(source_model.resolve()),
            "source_model_sha256": _sha256(source_model),
            "source_study": source_study,
            "source_solution": source_solution,
            "target_study": study_tag,
            "target_solution": solution_tag,
            "source_mesh": source_mesh,
            "target_mesh": target_mesh,
            "domain_mm": domain,
            "solver": solver,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["model"].resolve()),
            "model_sha256": _sha256(paths["model"]),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(paths["audit"], payload)
        return payload
    finally:
        client.java.showProgress(False)
        client.remove(model)


def solve_particle_level(
    client: Any,
    level: int,
) -> dict[str, Any]:
    contract = CorrectedGasMeshConvergenceContract()
    contract.validate()
    gas_audit_path = gas_paths(level)["audit"]
    if not gas_audit_path.exists():
        raise FileNotFoundError(gas_audit_path)
    gas_audit = json.loads(gas_audit_path.read_text(encoding="utf-8"))
    if gas_audit["status"] != "pass_corrected_gas_mesh_case":
        raise RuntimeError(f"Gas mesh level {level} has not passed")
    paths = particle_paths(level)
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    particle_contract = ParticlePhysicsContract()
    skeleton, skeleton_java = build_particle_skeleton(
        client,
        particle_contract,
        source_model=gas_paths(level)["model"],
    )
    try:
        skeleton_audit = audit_particle_skeleton(
            skeleton_java,
            particle_contract,
        )
        skeleton.save(str(paths["skeleton"]))
        skeleton_audit.update(
            {
                "model_path": str(paths["skeleton"].resolve()),
                "model_sha256": _sha256(paths["skeleton"]),
            }
        )
    finally:
        client.remove(skeleton)

    model, jm = build_particle_model(
        client,
        paths["skeleton"],
        source_study=gas_audit["target_study"],
    )
    started = time.time()
    try:
        jm.param().set("particles_per_size", str(contract.particles_per_size))
        jm.param().set(
            "particle_output_step",
            f"{contract.particle_output_step_us:.12g}[us]",
        )
        build_audit = audit_particle_build(
            jm,
            source_study=gas_audit["target_study"],
        )
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
            "schema_version": "h11_corrected_gas_mesh_particle_case_v1",
            "status": (
                "pass_corrected_gas_mesh_particle_case"
                if solve_audit["status"]
                == "pass_nominal_comsol_trajectory_audit"
                else "fail_corrected_gas_mesh_particle_case"
            ),
            "mesh_level": level,
            "source_gas_model": str(gas_paths(level)["model"].resolve()),
            "source_gas_model_sha256": gas_audit["model_sha256"],
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


def normalized_l2_change(
    left: Iterable[float],
    right: Iterable[float],
    *,
    offset: float = 0.0,
    absolute_scale: float | None = None,
) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or len(left_values) != len(right_values):
        raise ValueError("Common profiles must have equal nonzero length")
    numerator = math.sqrt(
        sum(
            (target - source) ** 2
            for source, target in zip(left_values, right_values)
        )
        / len(left_values)
    )
    if absolute_scale is not None:
        denominator = absolute_scale
    else:
        left_rms = math.sqrt(
            sum((value - offset) ** 2 for value in left_values)
            / len(left_values)
        )
        right_rms = math.sqrt(
            sum((value - offset) ** 2 for value in right_values)
            / len(right_values)
        )
        denominator = max(left_rms, right_rms, 1e-30)
    return numerator / denominator


def _gas_compact(
    level: int,
    payload: dict[str, Any],
    mesh: dict[str, Any],
) -> dict[str, Any]:
    metrics = payload["metrics"]
    profile = metrics["fixed_dpv_plane_gas_diagnostics"]["gas_profile"]
    contract = CorrectedGasMeshConvergenceContract()
    count = len(contract.common_dpv_profile_radii_mm)
    return {
        "mesh_level": level,
        "status": payload["status"],
        "mesh": mesh,
        "mass_imbalance_fraction": metrics["mass_flux_kg_s"][
            "imbalance_fraction"
        ],
        "energy_imbalance_fraction": metrics["energy_balance_w"][
            "imbalance_fraction_of_inlet"
        ],
        "dpv_profile": {
            "radii_mm": list(contract.common_dpv_profile_radii_mm),
            "temperature_k": profile["temperature_k"][:count],
            "speed_m_s": profile["speed_m_s"][:count],
            "absolute_pressure_pa": profile["absolute_pressure_pa"][:count],
        },
    }


def gas_adjacent_change(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_profile = left["dpv_profile"]
    right_profile = right["dpv_profile"]
    return {
        "left_level": left["mesh_level"],
        "right_level": right["mesh_level"],
        "temperature_anomaly_normalized_l2": normalized_l2_change(
            left_profile["temperature_k"],
            right_profile["temperature_k"],
            offset=300.0,
        ),
        "speed_normalized_l2": normalized_l2_change(
            left_profile["speed_m_s"],
            right_profile["speed_m_s"],
        ),
        "pressure_l2_over_ambient": normalized_l2_change(
            left_profile["absolute_pressure_pa"],
            right_profile["absolute_pressure_pa"],
            absolute_scale=101_325.0,
        ),
    }


def _particle_compact(level: int, payload: dict[str, Any]) -> dict[str, Any]:
    solve = payload["particle_solve_audit"]
    radius = DpvSamplingVolumeContract().low_speed_equivalent_radius_mm
    aperture = next(
        item
        for item in solve["trajectory_audit"][
            "centerline_aperture_sensitivity"
        ]
        if math.isclose(
            item["aperture_radius_mm"],
            radius,
            rel_tol=0,
            abs_tol=1e-12,
        )
    )
    return {
        "mesh_level": level,
        "status": solve["status"],
        "released_particle_count": solve["trajectory_audit"][
            "released_particle_count"
        ],
        "crossing_count": solve["trajectory_audit"]["crossing_count"],
        "selected_particle_count": aperture["selected_particle_count"],
        "diameter_nodes_represented": aperture["diameter_nodes_represented"],
        "quantiles": aperture[
            "empirical_detected_diameter_weighted_quantiles"
        ],
    }


def particle_adjacent_change(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    changes: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for response in ("temperature_c", "speed_m_s"):
        changes[response] = {}
        for quantile in ("q10", "q50", "q90"):
            source = left["quantiles"][response][quantile]
            target = right["quantiles"][response][quantile]
            change = abs(target - source) / max(abs(source), 1e-30)
            changes[response][quantile] = change
            maximum = max(maximum, change)
    return {
        "left_level": left["mesh_level"],
        "right_level": right["mesh_level"],
        "changes": changes,
        "maximum_relative_change": maximum,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary() -> dict[str, Any] | None:
    contract = CorrectedGasMeshConvergenceContract()
    contract.validate()
    gas3 = _load_json(gas_paths(3)["audit"])
    gas2 = _load_json(gas_paths(2)["audit"])
    part3 = _load_json(particle_paths(3)["audit"])
    part2 = _load_json(particle_paths(2)["audit"])
    baseline_gas = _load_json(BASELINE_GAS_AUDIT)
    baseline_particle = _load_json(BASELINE_PARTICLE_AUDIT)
    if any(
        item is None
        for item in (
            gas3,
            gas2,
            part3,
            part2,
            baseline_gas,
            baseline_particle,
        )
    ):
        return None
    gas_cases = [
        _gas_compact(4, baseline_gas, gas3["source_mesh"]),
        _gas_compact(3, gas3, gas3["target_mesh"]),
        _gas_compact(2, gas2, gas2["target_mesh"]),
    ]
    particle_cases = [
        _particle_compact(4, baseline_particle),
        _particle_compact(3, part3),
        _particle_compact(2, part2),
    ]
    gas_changes = [
        gas_adjacent_change(left, right)
        for left, right in zip(gas_cases[:-1], gas_cases[1:])
    ]
    particle_changes = [
        particle_adjacent_change(left, right)
        for left, right in zip(particle_cases[:-1], particle_cases[1:])
    ]
    finest_gas_change = gas_changes[-1]
    finest_particle_change = particle_changes[-1]
    gates = {
        "mesh_elements_increase_strictly": all(
            right["mesh"]["elements"] > left["mesh"]["elements"]
            for left, right in zip(gas_cases[:-1], gas_cases[1:])
        ),
        "all_gas_cases_conservative": all(
            case["mass_imbalance_fraction"]
            <= contract.mass_imbalance_limit_fraction
            and case["energy_imbalance_fraction"]
            <= contract.energy_imbalance_limit_fraction
            for case in gas_cases
        ),
        "finest_gas_temperature_change_below_1_percent": (
            finest_gas_change["temperature_anomaly_normalized_l2"]
            <= contract.gas_temperature_anomaly_l2_limit_fraction
        ),
        "finest_gas_speed_change_below_1_percent": (
            finest_gas_change["speed_normalized_l2"]
            <= contract.gas_speed_l2_limit_fraction
        ),
        "finest_gas_pressure_change_below_1e_minus_4": (
            finest_gas_change["pressure_l2_over_ambient"]
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
            finest_particle_change["maximum_relative_change"]
            <= contract.particle_quantile_limit_fraction
        ),
    }
    return {
        "schema_version": "h11_corrected_gas_mesh_convergence_summary_v1",
        "status": (
            "pass_corrected_gas_and_particle_mesh_convergence"
            if all(gates.values())
            else "fail_corrected_gas_or_particle_mesh_convergence"
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "gas_cases": gas_cases,
        "gas_adjacent_changes": gas_changes,
        "particle_cases": particle_cases,
        "particle_adjacent_changes": particle_changes,
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--gas-level", type=int, action="append", choices=[3, 2])
    parser.add_argument("--gas-bridge", action="store_true")
    parser.add_argument(
        "--level2-source",
        choices=["adjacent", "conditional_bridge"],
        default="adjacent",
    )
    parser.add_argument(
        "--particle-level",
        type=int,
        action="append",
        choices=[3, 2],
    )
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.summarize_existing:
        import mph

        client = mph.start(cores=args.cores, version=args.version)
        try:
            if args.gas_bridge:
                print("Solving conditional gas mesh bridge", flush=True)
                result = solve_gas_bridge(client)
                print(
                    f"  {result['status']}; "
                    f"elements={result['target_mesh']['elements']}; "
                    f"runtime={result['runtime_sec']:.1f}s",
                    flush=True,
                )
            for level in args.gas_level or []:
                print(f"Solving corrected gas mesh level {level}", flush=True)
                result = solve_gas_level(
                    client,
                    level,
                    level2_source=(
                        args.level2_source if level == 2 else "adjacent"
                    ),
                )
                print(
                    f"  {result['status']}; "
                    f"elements={result['target_mesh']['elements']}; "
                    f"runtime={result['runtime_sec']:.1f}s",
                    flush=True,
                )
            for level in args.particle_level or []:
                print(
                    f"Solving particles on corrected gas mesh level {level}",
                    flush=True,
                )
                result = solve_particle_level(client, level)
                compact = _particle_compact(level, result)
                print(
                    f"  {result['status']}; "
                    f"primary_n={compact['selected_particle_count']}; "
                    f"runtime={result['runtime_sec']:.1f}s",
                    flush=True,
                )
        finally:
            client.clear()
    summary = build_summary()
    if summary is None:
        print("Corrected mesh ladder is incomplete; artifacts were retained.")
        return 0
    _write_json(args.output, summary)
    print(f"Summary: {args.output}")
    print(f"Mesh status: {summary['status']}")
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
