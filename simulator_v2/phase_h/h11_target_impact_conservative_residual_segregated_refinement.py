"""H11 layer 11g: residual-controlled bounded segregated refinement.

The fully coupled branch cannot use COMSOL's Lower Limit solver attributes and
its first trial left the valid argon property range.  This runner therefore
retains the physically bounded segregated formulation and the stable fixed
substep updates, while requiring the outer Uzawa iteration to satisfy both the
solution-based and residual-based criteria.  The individual substeps are not
over-solved while the complementary variable group is held fixed.

Geometry, mesh, physics, materials, and boundary conditions remain unchanged.
This is numerical verification only, not calibration.
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
from simulator_v2.phase_h.h11_target_impact_conservative_residual_localization import (
    _residual_localization,
    parse_detailed_residual_log,
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
    "f2500_outer_residual.mph"
)
AUDIT_PATH = OUT_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_outer_residual.json"
)


@dataclass(frozen=True)
class ResidualSegregatedContract:
    """Frozen bounded residual-controlled segregated settings."""

    relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 4000
    flow_subiterations: int = 1
    turbulence_subiterations: int = 3
    flow_damping: float = 0.1
    turbulence_damping: float = 0.1
    residual_factor: float = 1.0
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not 0 < self.relative_tolerance <= 1e-5:
            raise ValueError("Refinement tolerance must lie in (0,1e-5]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Segregated iteration limit is too small")
        if self.flow_subiterations < 1:
            raise ValueError("At least one flow subiteration is required")
        if self.turbulence_subiterations < 1:
            raise ValueError("At least one turbulence subiteration is required")
        damping = (self.flow_damping, self.turbulence_damping)
        if any(
            not math.isfinite(value) or not 0 < value <= 0.5
            for value in damping
        ):
            raise ValueError("Substep damping must lie in (0,0.5]")
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


def _solver_tree(segregated: Any) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for tag_value in segregated.feature().tags():
        tag = str(tag_value)
        feature = segregated.feature(tag)
        item: dict[str, Any] = {
            "tag": tag,
            "type": str(feature.getType()),
            "label": str(feature.label()),
            "active": bool(feature.isActive()),
        }
        if item["type"] == "SegregatedStep":
            item.update(
                {
                    "nonlinear_method": str(
                        feature.getString("subdtech")
                    ),
                    "termination_technique": str(
                        feature.getString("subtermconst")
                    ),
                    "fixed_damping": float(
                        str(feature.getString("subdamp"))
                    ),
                    "linear_solver": str(
                        feature.getString("linsolver")
                    ),
                    "fixed_subiterations": int(
                        str(feature.getString("subiter"))
                    ),
                }
            )
        elif item["type"] == "LowerLimit":
            item["limits"] = str(feature.getString("lowerlimit"))
        steps.append(item)
    return {
        "active": bool(segregated.isActive()),
        "maximum_iterations": int(
            str(segregated.getString("maxsegiter"))
        ),
        "termination_technique": str(
            segregated.getString("segterm")
        ),
        "termination_criterion": str(
            segregated.getString("segtermonres")
        ),
        "residual_factor": float(
            str(segregated.getString("segreserrfact"))
        ),
        "stabilization": str(
            segregated.getString("segstabacc")
        ),
        "features": steps,
    }


def configure_residual_segregated_refinement(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: ResidualSegregatedContract,
) -> tuple[str, str, dict[str, Any]]:
    """Create a bounded residual-controlled segregated study."""

    contract.validate()
    hmnf = jm.component("comp1").physics("hmnf")
    if str(hmnf.getType()) != "HighMachNumberFlowTurbulentkomega":
        raise RuntimeError("Source is not the audited all-Mach k-omega model")
    kinetic = hmnf.prop("PhysicalModelProperty").getString(
        "includeKineticEnergy"
    )
    if str(kinetic) != "1":
        raise RuntimeError("Conservative total energy is disabled")

    study_tag = "std_hmnf_residual_segregated_refine"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label("Bounded residual-controlled segregated refinement")
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
    advanced = stationary.feature("aDef")
    advanced.set("storeresidual", "solvingandoutput")
    advanced.set("convinfo", "detailed")
    advanced.set("checkmatherr", "on")
    segregated = stationary.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.maximum_segregated_iterations),
    )
    segregated.set("segterm", "tol")
    segregated.set("segtermonres", "both")
    segregated.set("segreserrfact", f"{contract.residual_factor:.12g}")

    lower = segregated.feature("ll1")
    lower.set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )
    substep_settings = {
        "ss1": (
            contract.flow_damping,
            contract.flow_subiterations,
        ),
        "ss2": (
            contract.turbulence_damping,
            contract.turbulence_subiterations,
        ),
    }
    for step_tag, (damping_value, iterations) in substep_settings.items():
        substep = segregated.feature(step_tag)
        substep.set("subdtech", "const")
        substep.set("subdamp", f"{damping_value:.12g}")
        substep.set("subtermconst", "iter")
        substep.set("subiter", str(iterations))

    tree = _solver_tree(segregated)
    if not tree["active"]:
        raise RuntimeError("Residual-controlled segregated node is inactive")
    if tree["termination_criterion"] != "both":
        raise RuntimeError("Outer residual-controlled termination is inactive")
    substeps = [
        item
        for item in tree["features"]
        if item["type"] == "SegregatedStep"
    ]
    if len(substeps) != 2:
        raise RuntimeError("Expected flow and turbulence segregated steps")
    if any(
        item["nonlinear_method"] != "const"
        or item["termination_technique"] != "iter"
        for item in substeps
    ):
        raise RuntimeError("Fixed bounded substep setup drifted")
    observed_iterations = {
        item["tag"]: item["fixed_subiterations"] for item in substeps
    }
    if observed_iterations != {
        "ss1": contract.flow_subiterations,
        "ss2": contract.turbulence_subiterations,
    }:
        raise RuntimeError("Substep iteration counts drifted")
    limits = [
        item for item in tree["features"] if item["type"] == "LowerLimit"
    ]
    if len(limits) != 1:
        raise RuntimeError("Exactly one physical lower-limit node is required")
    return study_tag, solution_tag, {
        "manual_scales": scales,
        "solver_tree": tree,
        "method": "bounded_fixed_substeps_outer_residual_control",
        "termination": "outer_solution_and_residual",
        "store_last_residual": str(
            advanced.getString("storeresidual")
        ),
        "convergence_log_level": str(
            advanced.getString("convinfo")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--log", type=Path)
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
    log_path = args.log or audit_path.with_suffix(".log")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    contract = ResidualSegregatedContract()
    contract.validate()

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        client.java.showProgress(str(log_path.resolve()))
        model = client.load(str(args.source_model))
        jm = model.java
        source_study_tag = _last_tag(jm.study())
        source_solution_tag = _last_tag(jm.sol())
        source_metrics = evaluate_solution(model)
        source_residuals = _residual_localization(model)
        source_mesh = jm.component("comp1").mesh("mesh1")
        mesh_before = {
            "elements": int(source_mesh.getNumElem()),
            "vertices": int(source_mesh.getNumVertex()),
            "minimum_quality": float(source_mesh.getMinQuality()),
            "mean_quality": float(source_mesh.getMeanQuality()),
        }
        study_tag, solution_tag, solver = (
            configure_residual_segregated_refinement(
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
                "schema_version": (
                    "h11_residual_segregated_refinement_skeleton_v1"
                ),
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
                "source_raw_residuals": source_residuals,
                "calibrated": False,
                "paper_prediction_allowed": False,
            }
        else:
            try:
                jm.study(study_tag).run()
            except Exception as exc:
                partial_metrics: dict[str, Any]
                try:
                    partial_metrics = evaluate_solution(model)
                except Exception as metric_exc:
                    partial_metrics = {"error": str(metric_exc)}
                try:
                    partial_residuals = _residual_localization(model)
                except Exception as residual_exc:
                    partial_residuals = {"error": str(residual_exc)}
                partial = model_out.with_name(f"{model_out.stem}_partial.mph")
                model.save(str(partial))
                failure = {
                    "schema_version": (
                        "h11_outer_residual_segregated_failure_v1"
                    ),
                    "status": (
                        "failed_bounded_outer_residual_"
                        "segregated_solve"
                    ),
                    "contract": asdict(contract),
                    "source_study": source_study_tag,
                    "source_solution": source_solution_tag,
                    "target_study": study_tag,
                    "target_solution": solution_tag,
                    "solver": solver,
                    "mesh": mesh_before,
                    "source_metrics": source_metrics,
                    "partial_metrics": partial_metrics,
                    "source_raw_residuals": source_residuals,
                    "partial_raw_residuals": partial_residuals,
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
            refined_residuals = _residual_localization(model)
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
                raise RuntimeError("Residual refinement changed the mesh")
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
                "schema_version": "h11_outer_residual_segregated_v1",
                "status": (
                    "pass_outer_residual_numerical_gates_"
                    "not_calibrated"
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
                "source_raw_residuals": source_residuals,
                "refined_raw_residuals": refined_residuals,
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
        client.java.showProgress(False)
        client.clear()

    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    result["detailed_convergence"] = parse_detailed_residual_log(log_text)
    result.update(
        {
            "runtime_sec": time.time() - started,
            "comsol_version": args.version,
            "cores": args.cores,
            "source_model": str(args.source_model.resolve()),
            "source_sha256": _sha256(args.source_model),
            "model_path": str(model_out.resolve()),
            "model_sha256": _sha256(model_out),
            "log_path": str(log_path.resolve()),
        }
    )
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {model_out}")
    print(f"Wrote audit: {audit_path}")
    print(f"Residual segregated refinement: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
