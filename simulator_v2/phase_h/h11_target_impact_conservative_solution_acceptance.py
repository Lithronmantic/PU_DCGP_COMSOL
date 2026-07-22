"""H11 layer 11i: accept a roundoff-limited steady gas-phase baseline.

The torch-face model is bounded and conservative, and an additional 4000
outer iterations reduce every solution increment to about 1e-14.  COMSOL's
relative residual criterion nevertheless remains dominated by pressure and a
wall auxiliary temperature because its residual weights are constructed from
the first two residuals, which are already close to roundoff in this restart.

This branch uses COMSOL's documented solution-based stationary termination,
stores the final assembled residual, and independently requires:

* a successful 1e-6 stationary solve on the unchanged model;
* no material growth of any same-field absolute residual;
* bounded temperature, pressure, and velocity;
* discrete mass imbalance below 0.5 percent;
* total-energy imbalance below 2 percent; and
* stable near-target gas outputs.

The result is a numerically accepted, uncalibrated gas-phase baseline.  It is
not yet a paper prediction and does not bypass mesh or domain independence.
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
from simulator_v2.phase_h.h11_target_impact_conservative_residual_localization import (
    _residual_localization,
    parse_detailed_residual_log,
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
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_bridge"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_bridge"
SOURCE_MODEL = MODEL_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_residual_localization.mph"
)
MODEL_PATH = MODEL_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_numerically_accepted.mph"
)
AUDIT_PATH = OUT_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_numerically_accepted.json"
)
LOG_PATH = OUT_DIR / (
    "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_torch_face_numerically_accepted.log"
)


@dataclass(frozen=True)
class SolutionAcceptanceContract:
    """Frozen numerical gates for the uncalibrated gas-phase baseline."""

    relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 100
    flow_subiterations: int = 1
    turbulence_subiterations: int = 3
    flow_damping: float = 0.1
    turbulence_damping: float = 0.1
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    raw_residual_growth_factor: float = 10.0
    raw_residual_roundoff_floor: float = 1e-12
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02
    near_target_temperature_change_limit_fraction: float = 0.002
    near_target_speed_change_limit_fraction: float = 0.01
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if self.relative_tolerance != 1e-6:
            raise ValueError("Acceptance tolerance must remain 1e-6")
        if not 10 <= self.maximum_segregated_iterations <= 200:
            raise ValueError("Iteration limit must lie in [10, 200]")
        if self.flow_subiterations != 1:
            raise ValueError("Flow subiterations must remain 1")
        if self.turbulence_subiterations != 3:
            raise ValueError("Turbulence subiterations must remain 3")
        if self.flow_damping != 0.1 or self.turbulence_damping != 0.1:
            raise ValueError("Frozen damping is 0.1")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if (
            not math.isfinite(self.pressure_floor_pa)
            or self.pressure_floor_pa <= 0
        ):
            raise ValueError("Pressure floor must be finite and positive")
        if not 1 <= self.raw_residual_growth_factor <= 20:
            raise ValueError("Residual growth factor must lie in [1, 20]")
        if self.raw_residual_roundoff_floor != 1e-12:
            raise ValueError("Residual roundoff floor must remain 1e-12")
        if self.mass_imbalance_limit_fraction != 0.005:
            raise ValueError("Mass gate must remain 0.5 percent")
        if self.energy_imbalance_limit_fraction != 0.02:
            raise ValueError("Energy gate must remain 2 percent")
        if self.near_target_temperature_change_limit_fraction != 0.002:
            raise ValueError("Temperature stability gate must remain 0.2 percent")
        if self.near_target_speed_change_limit_fraction != 0.01:
            raise ValueError("Speed stability gate must remain 1 percent")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Numerical acceptance cannot claim prediction")


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


def configure_solution_acceptance(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: SolutionAcceptanceContract,
) -> tuple[str, str, dict[str, Any]]:
    """Build the solution-terminated acceptance study."""

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

    study_tag = "std_hmnf_solution_acceptance"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label("Roundoff-limited solution and conservation acceptance")
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
    segregated.set("segtermonres", "off")
    for tag, damping, iterations in (
        (
            "ss1",
            contract.flow_damping,
            contract.flow_subiterations,
        ),
        (
            "ss2",
            contract.turbulence_damping,
            contract.turbulence_subiterations,
        ),
    ):
        substep = segregated.feature(tag)
        substep.set("subdtech", "const")
        substep.set("subdamp", f"{damping:.12g}")
        substep.set("subtermconst", "iter")
        substep.set("subiter", str(iterations))
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )
    return study_tag, solution_tag, {
        "manual_scales": scales,
        "maximum_outer_iterations": int(
            str(segregated.getString("maxsegiter"))
        ),
        "termination_criterion": str(
            segregated.getString("segtermonres")
        ),
        "store_last_residual": str(
            advanced.getString("storeresidual")
        ),
        "convergence_log_level": str(
            advanced.getString("convinfo")
        ),
        "flow_subiterations": int(
            str(segregated.feature("ss1").getString("subiter"))
        ),
        "turbulence_subiterations": int(
            str(segregated.feature("ss2").getString("subiter"))
        ),
    }


def residual_growth_audit(
    source: dict[str, Any],
    accepted: dict[str, Any],
    factor: float,
    roundoff_floor: float = 1e-12,
) -> dict[str, Any]:
    """Compare residuals only within the same equation field."""

    fields: dict[str, Any] = {}
    for name, source_item in source.items():
        accepted_item = accepted.get(name, {})
        source_value = source_item.get("maximum_raw_residual")
        accepted_value = accepted_item.get("maximum_raw_residual")
        comparable = (
            source_value is not None
            and accepted_value is not None
            and math.isfinite(float(source_value))
            and math.isfinite(float(accepted_value))
        )
        both_within_roundoff = bool(
            comparable
            and abs(float(source_value)) <= roundoff_floor
            and abs(float(accepted_value)) <= roundoff_floor
        )
        passed = bool(
            comparable
            and (
                both_within_roundoff
                or float(accepted_value)
                <= factor
                * max(
                    float(source_value),
                    float.fromhex("0x1p-1022"),
                )
            )
        )
        fields[name] = {
            "source_maximum_raw_residual": source_value,
            "accepted_maximum_raw_residual": accepted_value,
            "same_field_growth_factor_limit": factor,
            "roundoff_floor": roundoff_floor,
            "both_within_roundoff_floor": both_within_roundoff,
            "comparable": comparable,
            "passed": passed,
        }
    return {
        "fields": fields,
        "all_fields_passed": bool(fields)
        and all(item["passed"] for item in fields.values()),
        "interpretation": (
            "Each field is compared only with itself because residual units "
            "differ across equations."
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
    default_model, default_audit, default_log = _default_artifact_paths(
        args.build_only
    )
    model_out = args.model_out or default_model
    audit_path = args.audit or default_audit
    log_path = args.log or default_log
    for path in (model_out, audit_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    contract = SolutionAcceptanceContract()
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
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_before = {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        study_tag, solution_tag, solver = configure_solution_acceptance(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=contract,
        )

        if args.build_only:
            solve_success = False
            accepted_metrics: dict[str, Any] = {}
            accepted_residuals: dict[str, Any] = {}
            changes: dict[str, float] = {}
            residual_gate: dict[str, Any] = {}
            gates = {
                "solver_returned_converged_solution": False,
                "same_mesh_identity": True,
                "physically_bounded": False,
                "same_field_absolute_residuals_stable": False,
                "mass_imbalance_below_0_5_percent": False,
                "energy_imbalance_below_2_percent": False,
                "near_target_outputs_stable": False,
            }
            status = "pass_solution_acceptance_skeleton_solve_not_run"
        else:
            jm.study(study_tag).run()
            solve_success = True
            accepted_metrics = evaluate_solution(model)
            accepted_residuals = _residual_localization(model)
            residual_gate = residual_growth_audit(
                source_residuals,
                accepted_residuals,
                contract.raw_residual_growth_factor,
                contract.raw_residual_roundoff_floor,
            )
            source_near = source_metrics["one_mm_upstream_of_target"]
            accepted_near = accepted_metrics[
                "one_mm_upstream_of_target"
            ]
            changes = {
                "near_target_temperature_fraction": relative_change(
                    source_near["temperature_k"],
                    accepted_near["temperature_k"],
                ),
                "near_target_speed_fraction": relative_change(
                    source_near["speed_m_s"],
                    accepted_near["speed_m_s"],
                ),
            }
            bounded = _bounded(
                accepted_metrics,
                ConservativeSolveContract(automatic_mesh_level=3),
            )
            mass_pass = (
                accepted_metrics["mass_flux_kg_s"][
                    "imbalance_fraction"
                ]
                < contract.mass_imbalance_limit_fraction
            )
            energy_pass = (
                accepted_metrics["energy_balance_w"][
                    "imbalance_fraction_of_inlet"
                ]
                < contract.energy_imbalance_limit_fraction
            )
            stable = (
                changes["near_target_temperature_fraction"]
                < contract.near_target_temperature_change_limit_fraction
                and changes["near_target_speed_fraction"]
                < contract.near_target_speed_change_limit_fraction
            )
            gates = {
                "solver_returned_converged_solution": solve_success,
                "same_mesh_identity": True,
                "physically_bounded": bounded,
                "same_field_absolute_residuals_stable": residual_gate[
                    "all_fields_passed"
                ],
                "mass_imbalance_below_0_5_percent": mass_pass,
                "energy_imbalance_below_2_percent": energy_pass,
                "near_target_outputs_stable": stable,
            }
            status = (
                "pass_numerically_accepted_uncalibrated_gas_baseline"
                if all(gates.values())
                else "solve_completed_but_acceptance_gate_failed"
            )

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
            raise RuntimeError("Solution acceptance changed the mesh")
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        result = {
            "schema_version": "h11_solution_acceptance_v1",
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
                "acceptance_study": study_tag,
                "acceptance_solution": solution_tag,
                "relative_residual_used_for_acceptance": False,
                "independent_absolute_residual_audit_used": True,
            },
            "solver": solver,
            "mesh": mesh_after,
            "source_metrics": source_metrics,
            "accepted_metrics": accepted_metrics,
            "relative_changes": changes,
            "source_raw_residuals": source_residuals,
            "accepted_raw_residuals": accepted_residuals,
            "same_field_residual_growth_audit": residual_gate,
            "detailed_convergence": parse_detailed_residual_log(log_text),
            "gates": gates,
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
            "log_path": str(log_path.resolve()),
        }
    )
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {model_out}")
    print(f"Wrote audit: {audit_path}")
    print(f"Wrote solver progress log: {log_path}")
    print(f"Solution acceptance: {result['status']}")
    return int(
        not args.build_only
        and result["status"]
        != "pass_numerically_accepted_uncalibrated_gas_baseline"
    )


if __name__ == "__main__":
    raise SystemExit(main())
