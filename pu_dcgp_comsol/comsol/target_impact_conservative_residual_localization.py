
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.target_impact_conservative_nominal import (
    ConservativeSolveContract,
    _set_manual_scales,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_restart import (
    _scalar,
    evaluate_solution,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_same_mesh_refinement import (
    _last_tag,
    _sha256,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_bridge"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_bridge"
SOURCE_MODEL = MODEL_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_outer_residual_partial.mph"
)
MODEL_PATH = MODEL_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_residual_localization.mph"
)
AUDIT_PATH = OUT_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_residual_localization.json"
)
LOG_PATH = OUT_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_residual_localization.log"
)


RESIDUAL_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("temperature", "T", "domain"),
    ("pressure", "p", "domain"),
    ("radial_velocity", "u", "domain"),
    ("axial_velocity", "w", "domain"),
    ("turbulent_kinetic_energy", "k", "domain"),
    ("specific_dissipation_rate", "om", "domain"),
    ("wall_temperature_down", "hmnf.TWall_d", "boundary"),
)


@dataclass(frozen=True)
class ResidualLocalizationContract:

    relative_tolerance: float = 1e-6
    diagnostic_outer_iterations: int = 5
    flow_subiterations: int = 1
    turbulence_subiterations: int = 3
    flow_damping: float = 0.1
    turbulence_damping: float = 0.1
    residual_factor: float = 1.0
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    store_last_residual: str = "solvingandoutput"
    convergence_log_level: str = "detailed"
    diagnostic_only: bool = True
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not 0 < self.relative_tolerance <= 1e-5:
            raise ValueError("Tolerance must lie in (0, 1e-5]")
        if not 1 <= self.diagnostic_outer_iterations <= 20:
            raise ValueError("Diagnostic iteration count must lie in [1, 20]")
        if self.flow_subiterations != 1:
            raise ValueError("Flow subiterations must match the frozen solver")
        if self.turbulence_subiterations != 3:
            raise ValueError(
                "Turbulence subiterations must match the frozen solver"
            )
        for value in (self.flow_damping, self.turbulence_damping):
            if not math.isfinite(value) or value != 0.1:
                raise ValueError("Diagnostic damping must remain 0.1")
        if self.residual_factor != 1.0:
            raise ValueError("Residual factor must remain 1")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if (
            not math.isfinite(self.pressure_floor_pa)
            or self.pressure_floor_pa <= 0
        ):
            raise ValueError("Pressure floor must be finite and positive")
        if self.store_last_residual != "solvingandoutput":
            raise ValueError("Residual must be stored in the solver output")
        if self.convergence_log_level != "detailed":
            raise ValueError("Field-wise localization requires a detailed log")
        if not self.diagnostic_only:
            raise ValueError("This branch is diagnostic only")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Residual localization cannot claim prediction")


def _default_artifact_paths(
    build_only: bool,
) -> tuple[Path, Path, Path]:
    if not build_only:
        return MODEL_PATH, AUDIT_PATH, LOG_PATH
    return (
        MODEL_PATH.with_name(f"{MODEL_PATH.stem}_skeleton.mph"),
        AUDIT_PATH.with_name(f"{AUDIT_PATH.stem}_skeleton.json"),
        LOG_PATH.with_name(f"{LOG_PATH.stem}_skeleton.log"),
    )


def _replace_maximum_coupling(
    component: Any,
    tag: str,
    dimension: int,
    entities: list[int] | None = None,
) -> None:
    couplings = component.cpl()
    tags = {str(value) for value in couplings.tags()}
    if tag in tags:
        couplings.remove(tag)
    operator = couplings.create(tag, "Maximum")
    operator.selection().geom("geom1", dimension)
    if entities is None:
        operator.selection().all()
    else:
        operator.selection().set(entities)


def configure_residual_localization(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: ResidualLocalizationContract,
) -> tuple[str, str, dict[str, Any]]:

    contract.validate()
    hmnf = jm.component("comp1").physics("hmnf")
    if str(hmnf.getType()) != "HighMachNumberFlowTurbulentkomega":
        raise RuntimeError("Source is not the audited all-Mach k-omega model")
    if str(
        hmnf.prop("PhysicalModelProperty").getString(
            "includeKineticEnergy"
        )
    ) != "1":
        raise RuntimeError("Conservative total energy is disabled")

    study_tag = "std_hmnf_residual_localization"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label("Diagnostic localization of stalled steady residual")
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
    advanced.set("storeresidual", contract.store_last_residual)
    advanced.set("convinfo", contract.convergence_log_level)
    advanced.set("checkmatherr", "on")

    segregated = stationary.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.diagnostic_outer_iterations),
    )
    segregated.set("segterm", "tol")
    segregated.set("segtermonres", "both")
    segregated.set("segreserrfact", f"{contract.residual_factor:.12g}")
    segregated.feature("ss1").set("subdtech", "const")
    segregated.feature("ss1").set(
        "subdamp",
        f"{contract.flow_damping:.12g}",
    )
    segregated.feature("ss1").set("subtermconst", "iter")
    segregated.feature("ss1").set(
        "subiter",
        str(contract.flow_subiterations),
    )
    segregated.feature("ss2").set("subdtech", "const")
    segregated.feature("ss2").set(
        "subdamp",
        f"{contract.turbulence_damping:.12g}",
    )
    segregated.feature("ss2").set("subtermconst", "iter")
    segregated.feature("ss2").set(
        "subiter",
        str(contract.turbulence_subiterations),
    )
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )

    component = jm.component("comp1")
    _replace_maximum_coupling(component, "max_res_domain", 2)
    wall_entities = [
        int(value)
        for value in hmnf.feature("wallbc1").selection().entities()
    ]
    _replace_maximum_coupling(
        component,
        "max_res_boundary",
        1,
        wall_entities,
    )
    return study_tag, solution_tag, {
        "manual_scales": scales,
        "maximum_outer_iterations": int(
            str(segregated.getString("maxsegiter"))
        ),
        "termination_criterion": str(
            segregated.getString("segtermonres")
        ),
        "residual_factor": float(
            str(segregated.getString("segreserrfact"))
        ),
        "store_last_residual": str(
            advanced.getString("storeresidual")
        ),
        "convergence_log_level": str(
            advanced.getString("convinfo")
        ),
        "flow_variables": str(
            segregated.feature("ss1").getString("segcomp")
        ),
        "turbulence_variables": str(
            segregated.feature("ss2").getString("segcomp")
        ),
        "residual_wall_entities": wall_entities,
    }


def _residual_localization(model: Any) -> dict[str, Any]:
    dataset = list(model / "datasets")[-1]
    selector: dict[str, Any] = {}
    inner_indices, _ = model.inner(dataset)
    if len(inner_indices):
        selector["inner"] = "last"

    results: dict[str, Any] = {}
    for name, variable, scope in RESIDUAL_FIELDS:
        operator = (
            "max_res_domain"
            if scope == "domain"
            else "max_res_boundary"
        )
        magnitude = f"abs(residual({variable}))"
        try:
            results[name] = {
                "variable": variable,
                "scope": scope,
                "maximum_raw_residual": _scalar(
                    model.evaluate(
                        f"{operator}({magnitude})",
                        dataset=dataset,
                        **selector,
                    )
                ),
                "r_at_max_m": _scalar(
                    model.evaluate(
                        f"{operator}({magnitude},r)",
                        unit="m",
                        dataset=dataset,
                        **selector,
                    )
                ),
                "z_at_max_m": _scalar(
                    model.evaluate(
                        f"{operator}({magnitude},z)",
                        unit="m",
                        dataset=dataset,
                        **selector,
                    )
                ),
            }
        except Exception as exc:
            results[name] = {
                "variable": variable,
                "scope": scope,
                "error": str(exc),
            }
    return results


def _log_tail(path: Path, lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[
        -lines:
    ]


def parse_detailed_residual_log(text: str) -> dict[str, Any]:

    utf8_group_matches = re.findall(
        r"分离组残差估计\s*[:：]?\s*"
        r"([0-9.eE+\-, \t]+)",
        text,
    )
    utf8_field_matches = re.findall(
        r"以下对象的残差估计\s*[:：]\s*([^\r\n]+)",
        text,
    )
    group_matches = re.findall(
        r"分离组残差估计\s*\r?\n([^\r\n]+)",
        text,
    )
    field_matches = re.findall(
        r"以下对象的残差估计：([^\r\n]+)",
        text,
    )
    if utf8_group_matches:
        group_matches = utf8_group_matches
    if utf8_field_matches:
        field_matches = utf8_field_matches
    groups: list[float] = []
    if group_matches:
        groups = [
            float(value.strip())
            for value in group_matches[-1].split(",")
        ]
    fields: dict[str, dict[str, Any]] = {}
    if field_matches:
        pattern = re.compile(
            r"([^,]+?)\s*\((comp1\.[^)]+)\)\s+"
            r"([0-9.+\-eE]+)"
        )
        for label, variable, value in pattern.findall(field_matches[-1]):
            fields[variable] = {
                "label": label.strip(),
                "scaled_residual_estimate": float(value),
            }
    return {
        "last_group_residual_estimates": groups,
        "last_field_residual_estimates": fields,
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
    default_model, default_audit, default_log = _default_artifact_paths(
        args.build_only
    )
    model_out = args.model_out or default_model
    audit_path = args.audit or default_audit
    log_path = args.log or default_log
    for path in (model_out, audit_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    contract = ResidualLocalizationContract()
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
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_before = {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        study_tag, solution_tag, solver = configure_residual_localization(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=contract,
        )
        if args.build_only:
            status = "pass_residual_localization_skeleton"
            solve_error = None
            residuals: dict[str, Any] = {}
            localized_metrics: dict[str, Any] = {}
        else:
            solve_error = None
            try:
                jm.study(study_tag).run()
                status = "diagnostic_converged_within_iteration_window"
            except Exception as exc:
                solve_error = str(exc)
                if "最大分离式迭代次数" in solve_error or (
                    "maximum number of segregated iterations"
                    in solve_error.lower()
                ):
                    status = "diagnostic_iteration_cap_reached_as_expected"
                else:
                    status = "diagnostic_unexpected_solver_failure"
            localized_metrics = evaluate_solution(model)
            residuals = _residual_localization(model)

        model.rename(model_out.stem)
        model.save(str(model_out))
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
            raise RuntimeError("Residual localization changed the mesh")
        result = {
            "schema_version": "h11_residual_localization_v1",
            "status": status,
            "contract": asdict(contract),
            "strategy": {
                "geometry_changed": False,
                "mesh_changed": False,
                "physics_changed": False,
                "material_changed": False,
                "boundary_conditions_changed": False,
                "source_study": source_study_tag,
                "source_solution": source_solution_tag,
                "diagnostic_study": study_tag,
                "diagnostic_solution": solution_tag,
            },
            "solver": solver,
            "mesh": mesh_after,
            "source_metrics": source_metrics,
            "localized_metrics": localized_metrics,
            "raw_residual_localization": residuals,
            "raw_residual_interpretation": (
                "Use locations within a field only. Raw equation residuals "
                "have different units and cannot rank fields; field-wise "
                "scaled errors must be read from the detailed COMSOL log."
            ),
            "detailed_convergence": parse_detailed_residual_log(
                log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                if log_path.exists()
                else ""
            ),
            "solver_error": solve_error,
            "log_path": str(log_path.resolve()),
            "log_tail": _log_tail(log_path),
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.java.showProgress(False)
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
    print(f"Wrote solver progress log: {log_path}")
    print(f"Residual localization: {result['status']}")
    return int(result["status"] == "diagnostic_unexpected_solver_failure")


if __name__ == "__main__":
    raise SystemExit(main())
