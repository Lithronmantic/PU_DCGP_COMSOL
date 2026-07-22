"""H11 layer 11h: replace the artificial upstream opening by a torch face.

The previous target-domain model treated the entire annulus outside the
effective nozzle at z=0 as a pressure-temperature opening.  A read-only audit
showed that this boundary admitted roughly one hundred nozzle mass-flow rates
of cold gas.  In the physical apparatus this plane is occupied by the torch
front face.  This branch therefore leaves only the radial far field open and
lets COMSOL's default no-slip, thermally insulated wall cover the upstream
annulus.

The first solve is an initialization solve at relative tolerance 1e-4.  It is
not a final residual, mesh, calibration, or validation result.
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
    _entities,
    _replace_coupling,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_target_impact_conservative_same_mesh_refinement import (
    _bounded,
    _last_tag,
    _sha256,
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
    "f2500_torch_face_initial.mph"
)
AUDIT_PATH = OUT_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_initial.json"
)


@dataclass(frozen=True)
class TorchFaceInitializationContract:
    """Frozen boundary correction and initial-solve settings."""

    relative_tolerance: float = 1e-4
    maximum_segregated_iterations: int = 4000
    flow_damping: float = 0.1
    turbulence_damping: float = 0.1
    flow_subiterations: int = 1
    turbulence_subiterations: int = 3
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    upstream_annulus_is_wall: bool = True
    upstream_annulus_is_adiabatic: bool = True
    radial_far_field_is_open: bool = True
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not 0 < self.relative_tolerance <= 1e-3:
            raise ValueError("Initialization tolerance must lie in (0,1e-3]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Segregated iteration limit is too small")
        if not 0 < self.flow_damping <= 0.5:
            raise ValueError("Flow damping must lie in (0,0.5]")
        if not 0 < self.turbulence_damping <= 0.5:
            raise ValueError("Turbulence damping must lie in (0,0.5]")
        if self.flow_subiterations < 1:
            raise ValueError("At least one flow subiteration is required")
        if self.turbulence_subiterations < 1:
            raise ValueError("At least one turbulence subiteration is required")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if (
            not math.isfinite(self.pressure_floor_pa)
            or self.pressure_floor_pa <= 0
        ):
            raise ValueError("Pressure floor must be finite and positive")
        if not (
            self.upstream_annulus_is_wall
            and self.upstream_annulus_is_adiabatic
            and self.radial_far_field_is_open
        ):
            raise ValueError("The torch-face boundary contract is incomplete")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError(
                "Torch-face initialization cannot claim prediction"
            )


def _default_artifact_paths(build_only: bool) -> tuple[Path, Path]:
    if not build_only:
        return MODEL_PATH, AUDIT_PATH
    return (
        MODEL_PATH.with_name(f"{MODEL_PATH.stem}_skeleton.mph"),
        AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_skeleton.json"),
    )


def configure_torch_face_boundary(jm: Any) -> dict[str, Any]:
    """Remove the upstream annulus from the open-boundary feature."""

    component = jm.component("comp1")
    hmnf = component.physics("hmnf")
    upstream = sorted(_entities(component, "geom1_sel_ambient_in"))
    radial = sorted(_entities(component, "geom1_sel_far_r"))
    target = sorted(_entities(component, "geom1_sel_target"))
    nozzle = sorted(_entities(component, "geom1_sel_nozzle_in"))
    ambient = hmnf.feature("ambient_open")
    ambient.selection().set(radial)
    ambient.label("Radial far-field pressure-temperature opening")

    _replace_coupling(component, "int_ambient_hmnf", radial)
    expected_walls = sorted({*upstream, *target})
    _replace_coupling(component, "int_wall_hmnf", expected_walls)

    observed = {
        "nozzle": sorted(
            int(value)
            for value in hmnf.feature("nozzle_in").selection().entities()
        ),
        "radial_open": sorted(
            int(value) for value in ambient.selection().entities()
        ),
        "walls": sorted(
            int(value)
            for value in hmnf.feature("wallbc1").selection().entities()
        ),
        "thermal_insulation": sorted(
            int(value)
            for value in hmnf.feature("ins1").selection().entities()
        ),
        "isothermal_target": sorted(
            int(value)
            for value in hmnf.feature(
                "target_temperature"
            ).selection().entities()
        ),
    }
    required = {
        "nozzle": nozzle,
        "radial_open": radial,
        "walls": expected_walls,
        "thermal_insulation": upstream,
        "isothermal_target": target,
    }
    if observed != required:
        raise RuntimeError(
            f"Torch-face boundary selections drifted: {observed} != {required}"
        )
    return {
        "semantics": {
            "upstream_annulus": (
                "stationary no-slip thermally insulated torch face"
            ),
            "radial_far_field": (
                "subsonic pressure-temperature opening with backflow allowed"
            ),
            "target": "stationary no-slip measured-range isothermal wall",
        },
        "selection_entities": observed,
    }


def configure_initialization_study(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: TorchFaceInitializationContract,
) -> tuple[str, str, dict[str, Any]]:
    study_tag = "std_hmnf_torch_face_initial"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label("Torch-face boundary initialization")
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
    segregated = stationary.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.maximum_segregated_iterations),
    )
    segregated.set("segterm", "tol")
    segregated.set("segtermonres", "auto")
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )
    settings = {
        "ss1": (contract.flow_damping, contract.flow_subiterations),
        "ss2": (
            contract.turbulence_damping,
            contract.turbulence_subiterations,
        ),
    }
    for tag, (damping, iterations) in settings.items():
        substep = segregated.feature(tag)
        substep.set("subdtech", "const")
        substep.set("subdamp", f"{damping:.12g}")
        substep.set("subtermconst", "iter")
        substep.set("subiter", str(iterations))
    return study_tag, solution_tag, {
        "manual_scales": scales,
        "nonlinear_method": "bounded_fixed_under_relaxation",
        "termination": "initialization_solution_or_residual",
        "flow_subiterations": contract.flow_subiterations,
        "turbulence_subiterations": contract.turbulence_subiterations,
        "flow_damping": contract.flow_damping,
        "turbulence_damping": contract.turbulence_damping,
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
    contract = TorchFaceInitializationContract()
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
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_before = {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        boundary = configure_torch_face_boundary(jm)
        study_tag, solution_tag, solver = configure_initialization_study(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=contract,
        )
        if args.build_only:
            model.rename(model_out.stem)
            model.save(str(model_out))
            result = {
                "schema_version": "h11_torch_face_skeleton_v1",
                "status": "pass_boundary_and_solver_tree_solve_not_run",
                "contract": asdict(contract),
                "boundary": boundary,
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
                try:
                    partial_metrics: dict[str, Any] = evaluate_solution(model)
                except Exception as metric_exc:
                    partial_metrics = {"error": str(metric_exc)}
                partial = model_out.with_name(f"{model_out.stem}_partial.mph")
                model.save(str(partial))
                failure = {
                    "schema_version": "h11_torch_face_initial_failure_v1",
                    "status": "failed_torch_face_initialization",
                    "contract": asdict(contract),
                    "boundary": boundary,
                    "solver": solver,
                    "mesh": mesh_before,
                    "source_metrics": source_metrics,
                    "partial_metrics": partial_metrics,
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
            metrics = evaluate_solution(model)
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
                raise RuntimeError("Torch-face initialization changed the mesh")
            bounded = _bounded(metrics, contract)
            result = {
                "schema_version": "h11_torch_face_initial_v1",
                "status": (
                    "pass_bounded_initialization_strict_residual_pending"
                    if bounded
                    else "pass_solve_fail_bounded_initialization"
                ),
                "contract": asdict(contract),
                "boundary": boundary,
                "solver": solver,
                "mesh": mesh_after,
                "source_metrics": source_metrics,
                "initial_metrics": metrics,
                "gates": {
                    "same_mesh_identity": True,
                    "physically_bounded": bounded,
                    "strict_residual_verified": False,
                    "domain_independence_verified": False,
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
    print(f"Torch-face initialization: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
