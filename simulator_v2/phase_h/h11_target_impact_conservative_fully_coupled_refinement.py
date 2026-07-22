"""H11 layer 11f: residual-controlled fully coupled same-mesh refinement.

The preceding fixed-damping segregated solve can stagnate with small solution
updates while the global continuity residual remains material.  This runner
keeps geometry, mesh, physics, material data, and boundary conditions fixed,
but replaces the active nonlinear attribute with COMSOL's fully coupled
damped Newton method.  Convergence requires both the solution-based and the
residual-based criteria.

The runner is numerical verification only.  It cannot calibrate the inlet or
authorize a paper prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_target_impact_conservative_nominal import (
    ConservativeSolveContract,
    _set_manual_scales,
)
from simulator_v2.phase_h.h11_target_impact_conservative_restart import (
    evaluate_solution,
)
from simulator_v2.phase_h.h11_target_impact_conservative_same_mesh_refinement import (
    _bounded,
    _last_tag,
    _sha256,
    relative_change,
)


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2_f2500_refined.mph"
)
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_bridge"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_bridge"
MODEL_PATH = MODEL_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_fully_coupled.mph"
)
AUDIT_PATH = OUT_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_fully_coupled.json"
)


@dataclass(frozen=True)
class FullyCoupledRefinementContract:
    """Frozen residual-controlled Newton settings."""

    relative_tolerance: float = 1e-6
    maximum_iterations: int = 300
    initial_damping: float = 1e-4
    minimum_damping: float = 1e-10
    recovery_damping: float = 0.1
    residual_factor: float = 1.0
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not 0 < self.relative_tolerance <= 1e-5:
            raise ValueError("Refinement tolerance must lie in (0,1e-5]")
        if self.maximum_iterations < 50:
            raise ValueError("Fully coupled iteration limit is too small")
        damping = (
            self.initial_damping,
            self.minimum_damping,
            self.recovery_damping,
        )
        if any(not math.isfinite(value) or value <= 0 for value in damping):
            raise ValueError("Newton damping values must be finite and positive")
        if self.minimum_damping >= self.initial_damping:
            raise ValueError("Minimum damping must be below initial damping")
        if not self.initial_damping <= self.recovery_damping <= 1:
            raise ValueError(
                "Recovery damping must lie between initial damping and one"
            )
        if not math.isfinite(self.residual_factor):
            raise ValueError("Residual factor must be finite")
        if not 0 < self.residual_factor <= 1:
            raise ValueError("Residual factor must lie in (0,1]")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if (
            not math.isfinite(self.pressure_floor_pa)
            or self.pressure_floor_pa <= 0
        ):
            raise ValueError("Pressure floor must be finite and positive")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError(
                "Numerical refinement cannot claim calibration or prediction"
            )


def _default_artifact_paths(build_only: bool) -> tuple[Path, Path]:
    if not build_only:
        return MODEL_PATH, AUDIT_PATH
    return (
        MODEL_PATH.with_name(f"{MODEL_PATH.stem}_skeleton.mph"),
        AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_skeleton.json"),
    )


def _active(feature: Any) -> bool:
    return bool(feature.isActive())


def _solver_tree(stationary: Any) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for tag_value in stationary.feature().tags():
        tag = str(tag_value)
        feature = stationary.feature(tag)
        nodes.append(
            {
                "tag": tag,
                "type": str(feature.getType()),
                "label": str(feature.label()),
                "active": _active(feature),
            }
        )
    fully = stationary.feature("fc1")
    direct = stationary.feature(str(fully.getString("linsolver")))
    return {
        "stationary_type": str(stationary.getType()),
        "nodes": nodes,
        "fully_coupled": {
            "tag": "fc1",
            "active": _active(fully),
            "linear_solver": str(fully.getString("linsolver")),
            "nonlinear_method": str(fully.getString("dtech")),
            "termination_criterion": str(fully.getString("termonres")),
            "residual_factor": float(str(fully.getString("reserrfact"))),
            "maximum_iterations": int(str(fully.getString("maxiter"))),
            "initial_damping": float(str(fully.getString("initsteph"))),
            "minimum_damping": float(str(fully.getString("minsteph"))),
            "use_recovery": str(fully.getString("useminsteprecovery")),
            "recovery_damping": float(
                str(fully.getString("minsteprecovery"))
            ),
        },
        "direct_solver": {
            "type": str(direct.getType()),
            "algorithm": str(direct.getString("linsolver")),
        },
    }


def configure_fully_coupled_refinement(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: FullyCoupledRefinementContract,
    target_study_tag: str = "std_hmnf_fully_coupled_refine",
) -> tuple[str, str, dict[str, Any]]:
    """Create an inactive-segregated, active-fully-coupled study."""

    contract.validate()
    if not target_study_tag or target_study_tag == source_study_tag:
        raise ValueError(
            "Target study tag must be nonempty and differ from the source"
        )
    hmnf = jm.component("comp1").physics("hmnf")
    if str(hmnf.getType()) != "HighMachNumberFlowTurbulentkomega":
        raise RuntimeError("Source is not the audited all-Mach k-omega model")
    kinetic = hmnf.prop("PhysicalModelProperty").getString(
        "includeKineticEnergy"
    )
    if str(kinetic) != "1":
        raise RuntimeError("Conservative total energy is disabled")

    study_tag = target_study_tag
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study_label = "Residual-controlled fully coupled same-mesh refinement"
    if target_study_tag != "std_hmnf_fully_coupled_refine":
        study_label += f" verification ({target_study_tag})"
    study.label(study_label)
    step = study.create("stat", "Stationary")
    step.set("useinitsol", "on")
    step.set("initmethod", "sol")
    step.set("initstudy", source_study_tag)
    step.set("initstudystep", "stat")
    step.set("initsol", source_solution_tag)
    step.set("solnum", "last")
    study.createAutoSequences("all")
    solution_tag = _last_tag(jm.sol())

    scales = _set_manual_scales(
        jm,
        solution_tag,
        ConservativeSolveContract(automatic_mesh_level=3),
    )
    stationary = jm.sol(solution_tag).feature("s1")
    stationary.set("stol", f"{contract.relative_tolerance:.12g}")
    child_tags = {str(value) for value in stationary.feature().tags()}
    if "se1" not in child_tags:
        raise RuntimeError("Expected the auto-generated segregated attribute")
    if "d1" not in child_tags:
        raise RuntimeError("Expected the auto-generated direct solver")

    segregated = stationary.feature("se1")
    if "fc1" in child_tags:
        fully = stationary.feature("fc1")
    else:
        fully = stationary.feature().create("fc1", "FullyCoupled")
    fully.active(True)
    segregated.active(False)
    fully.set("linsolver", "d1")
    fully.set("dtech", "hnlin")
    fully.set("initsteph", f"{contract.initial_damping:.12g}")
    fully.set("minsteph", f"{contract.minimum_damping:.12g}")
    fully.set("useminsteprecovery", "on")
    fully.set(
        "minsteprecovery",
        f"{contract.recovery_damping:.12g}",
    )
    fully.set("maxiter", str(contract.maximum_iterations))
    fully.set("termonres", "both")
    fully.set("reserrfact", f"{contract.residual_factor:.12g}")
    stationary.feature().move("fc1", 2)

    tree = _solver_tree(stationary)
    expected = tree["fully_coupled"]
    if not expected["active"] or _active(segregated):
        raise RuntimeError("Exactly the fully coupled nonlinear node must be active")
    if expected["linear_solver"] != "d1":
        raise RuntimeError("Fully coupled node is not attached to the direct solver")
    if tree["direct_solver"]["algorithm"] != "pardiso":
        raise RuntimeError("The fully coupled branch must use PARDISO")
    if expected["termination_criterion"] != "both":
        raise RuntimeError("Residual-controlled termination is not active")
    return study_tag, solution_tag, {
        "manual_scales": scales,
        "solver_tree": tree,
        "method": "fully_coupled_highly_nonlinear_newton",
        "termination": "solution_and_residual",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.source_model.exists():
        raise FileNotFoundError(args.source_model)
    default_model, default_audit = _default_artifact_paths(args.build_only)
    model_out = args.model_out or default_model
    audit_path = args.audit or default_audit
    model_out.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    contract = FullyCoupledRefinementContract()
    contract.validate()

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.source_model))
        jm = model.java
        source_study_tag = _last_tag(jm.study())
        source_solution_tag = _last_tag(jm.sol())
        source_metrics = evaluate_solution(model)
        source_mesh = jm.component("comp1").mesh("mesh1")
        mesh_before = {
            "elements": int(source_mesh.getNumElem()),
            "vertices": int(source_mesh.getNumVertex()),
            "minimum_quality": float(source_mesh.getMinQuality()),
            "mean_quality": float(source_mesh.getMeanQuality()),
        }
        study_tag, solution_tag, solver = (
            configure_fully_coupled_refinement(
                jm,
                source_study_tag=source_study_tag,
                source_solution_tag=source_solution_tag,
                contract=contract,
            )
        )
        if args.build_only:
            model.rename(model_out.stem)
            model.save(str(model_out))
            result = {
                "schema_version": "h11_fully_coupled_refinement_skeleton_v1",
                "status": "pass_solver_tree_built_solve_not_run",
                "contract": asdict(contract),
                "strategy": {
                    "geometry_changed": False,
                    "mesh_changed": False,
                    "physics_changed": False,
                    "source_study": source_study_tag,
                    "source_solution": source_solution_tag,
                    "target_study": study_tag,
                    "target_solution": solution_tag,
                },
                "solver": solver,
                "mesh": mesh_before,
                "source_metrics": source_metrics,
                "calibrated": False,
                "paper_prediction_allowed": False,
            }
        else:
            try:
                jm.study(study_tag).run()
            except Exception as exc:
                partial = model_out.with_name(f"{model_out.stem}_partial.mph")
                model.save(str(partial))
                failure = {
                    "schema_version": "h11_fully_coupled_refinement_failure_v1",
                    "status": "failed_residual_controlled_fully_coupled_solve",
                    "contract": asdict(contract),
                    "source_study": source_study_tag,
                    "source_solution": source_solution_tag,
                    "target_study": study_tag,
                    "target_solution": solution_tag,
                    "solver": solver,
                    "mesh": mesh_before,
                    "source_metrics": source_metrics,
                    "error": str(exc),
                    "partial_model": str(partial.resolve()),
                    "partial_model_sha256": _sha256(partial),
                }
                failure_path = audit_path.with_name(
                    f"{audit_path.stem}_failure.json"
                )
                with failure_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        failure,
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"Wrote failure audit: {failure_path}", flush=True)
                raise

            model.rename(model_out.stem)
            model.save(str(model_out))
            refined_metrics = evaluate_solution(model)
            mesh_after_feature = jm.component("comp1").mesh("mesh1")
            mesh_after = {
                "elements": int(mesh_after_feature.getNumElem()),
                "vertices": int(mesh_after_feature.getNumVertex()),
                "minimum_quality": float(
                    mesh_after_feature.getMinQuality()
                ),
                "mean_quality": float(mesh_after_feature.getMeanQuality()),
            }
            if mesh_after != mesh_before:
                raise RuntimeError("Fully coupled refinement changed the mesh")
            source_near = source_metrics["one_mm_upstream_of_target"]
            refined_near = refined_metrics["one_mm_upstream_of_target"]
            changes = {
                "near_target_temperature_fraction": relative_change(
                    source_near["temperature_k"],
                    refined_near["temperature_k"],
                ),
                "near_target_speed_fraction": relative_change(
                    source_near["speed_m_s"],
                    refined_near["speed_m_s"],
                ),
            }
            bounded = _bounded(refined_metrics, contract)
            mass_pass = (
                refined_metrics["mass_flux_kg_s"]["imbalance_fraction"]
                < 0.005
            )
            energy_pass = (
                refined_metrics["energy_balance_w"][
                    "imbalance_fraction_of_inlet"
                ]
                < 0.02
            )
            stable = (
                changes["near_target_temperature_fraction"] < 0.002
                and changes["near_target_speed_fraction"] < 0.01
            )
            result = {
                "schema_version": "h11_fully_coupled_refinement_v1",
                "status": (
                    "pass_fully_coupled_numerical_gates_not_calibrated"
                    if bounded and mass_pass and energy_pass and stable
                    else "pass_solve_fail_one_or_more_numerical_gates"
                ),
                "contract": asdict(contract),
                "strategy": {
                    "geometry_changed": False,
                    "mesh_changed": False,
                    "physics_changed": False,
                    "source_study": source_study_tag,
                    "source_solution": source_solution_tag,
                    "target_study": study_tag,
                    "target_solution": solution_tag,
                },
                "solver": solver,
                "mesh": mesh_after,
                "source_metrics": source_metrics,
                "refined_metrics": refined_metrics,
                "relative_changes": changes,
                "gates": {
                    "same_mesh_identity": True,
                    "physically_bounded": bounded,
                    "mass_imbalance_below_0_5_percent": mass_pass,
                    "energy_imbalance_below_2_percent": energy_pass,
                    "near_target_algebraically_stable": stable,
                },
                "calibrated": False,
                "paper_prediction_allowed": False,
            }
    finally:
        client.clear()

    result.update(
        {
            "runtime_sec": time.time() - started,
            "comsol_version": args.version,
            "cores": args.cores,
            "source_model": str(args.source_model.resolve()),
            "source_sha256": _sha256(args.source_model),
            "model_path": str(model_out.resolve()),
            "model_sha256": _sha256(model_out),
        }
    )
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {model_out}")
    print(f"Wrote audit: {audit_path}")
    print(f"Fully coupled refinement: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
