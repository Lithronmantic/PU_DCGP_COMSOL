"""H11 layer 11e: tighten the algebraic solve on a fixed all-Mach mesh.

This runner does not alter geometry, mesh, physics, materials, or boundary
conditions.  It starts from a converged all-Mach solution and resolves the
same equations on the same mesh at a stricter relative tolerance.  The result
separates algebraic error from mesh sensitivity before another mesh step.
"""

from __future__ import annotations

import argparse
import hashlib
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


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2_f2500.mph"
)
MODEL_PATH = SOURCE_MODEL.with_name(
    "h11_target_impact_conservative_bridge_3_to_2_f2500_refined.mph"
)
AUDIT_PATH = (
    HERE
    / "h11_outputs"
    / "target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2_f2500_refined.json"
)


@dataclass(frozen=True)
class SameMeshRefinementContract:
    """Frozen algebraic-refinement settings; no calibration is performed."""

    relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 4000
    flow_damping: float = 0.1
    turbulence_damping: float = 0.1
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not 0 < self.relative_tolerance <= 1e-5:
            raise ValueError("Refinement tolerance must lie in (0,1e-5]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Segregated iteration limit is too small")
        if not 0 < self.flow_damping <= 0.5:
            raise ValueError("Flow damping must lie in (0,0.5]")
        if not 0 < self.turbulence_damping <= 0.5:
            raise ValueError("Turbulence damping must lie in (0,0.5]")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if not math.isfinite(self.pressure_floor_pa):
            raise ValueError("Pressure floor must be finite")
        if self.pressure_floor_pa <= 0:
            raise ValueError("Pressure floor must be positive")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Algebraic refinement cannot claim prediction")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _last_tag(collection: Any) -> str:
    tags = [str(value) for value in collection.tags()]
    if not tags:
        raise RuntimeError("Expected nonempty COMSOL collection")
    return tags[-1]


def _bounded(
    metrics: dict[str, Any],
    contract: SameMeshRefinementContract,
) -> bool:
    temperature = metrics["temperature_k"]
    speed = metrics["speed_m_s"]
    pressure = metrics["absolute_pressure_pa"]
    values = (
        temperature["minimum"],
        temperature["maximum"],
        speed["minimum"],
        speed["maximum"],
        pressure["minimum"],
        pressure["maximum"],
    )
    return bool(
        all(math.isfinite(float(value)) for value in values)
        and temperature["minimum"] > contract.property_temperature_floor_k
        and temperature["maximum"] <= 10_000.01
        and speed["minimum"] >= -1e-9
        and speed["maximum"] <= 600.01
        and pressure["minimum"] > contract.pressure_floor_pa
    )


def relative_change(before: float, after: float) -> float:
    """Absolute fractional change with a stable nonzero denominator."""
    return abs(after - before) / max(abs(before), 1e-12)


def configure_same_mesh_refinement(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: SameMeshRefinementContract,
) -> tuple[str, str, dict[str, Any]]:
    contract.validate()
    hmnf = jm.component("comp1").physics("hmnf")
    if str(hmnf.getType()) != "HighMachNumberFlowTurbulentkomega":
        raise RuntimeError("Source is not the final all-Mach k-omega model")
    if (
        str(
            hmnf.prop("PhysicalModelProperty").getString(
                "includeKineticEnergy"
            )
        )
        != "1"
    ):
        raise RuntimeError("Conservative total energy is disabled")

    studies = jm.study()
    study_tag = "std_hmnf_same_mesh_refine"
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label("Conservative all-Mach same-mesh algebraic refinement")
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
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )
    flow_step = segregated.feature("ss1")
    flow_step.set("subdtech", "const")
    flow_step.set("subdamp", f"{contract.flow_damping:.12g}")
    flow_step.set("subiter", "1")
    turbulence_step = segregated.feature("ss2")
    turbulence_step.set("subdtech", "const")
    turbulence_step.set(
        "subdamp",
        f"{contract.turbulence_damping:.12g}",
    )
    turbulence_step.set("subiter", "3")
    return (
        study_tag,
        solution_tag,
        {
            "manual_scales": scales,
            "nonlinear_method": (
                "same_mesh_segregated_fixed_under_relaxation"
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.source_model.exists():
        raise FileNotFoundError(args.source_model)
    contract = SameMeshRefinementContract()
    contract.validate()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

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
        mesh_audit_before = {
            "elements": int(source_mesh.getNumElem()),
            "vertices": int(source_mesh.getNumVertex()),
            "minimum_quality": float(source_mesh.getMinQuality()),
            "mean_quality": float(source_mesh.getMeanQuality()),
        }
        study_tag, solution_tag, solver_strategy = (
            configure_same_mesh_refinement(
                jm,
                source_study_tag=source_study_tag,
                source_solution_tag=source_solution_tag,
                contract=contract,
            )
        )
        try:
            jm.study(study_tag).run()
        except Exception as exc:
            partial = args.model_out.with_name(
                f"{args.model_out.stem}_partial.mph"
            )
            model.save(str(partial))
            failure = {
                "schema_version": "h11_same_mesh_refinement_failure_v1",
                "status": "failed_same_mesh_algebraic_refinement",
                "contract": asdict(contract),
                "source_study": source_study_tag,
                "source_solution": source_solution_tag,
                "target_study": study_tag,
                "target_solution": solution_tag,
                "solver_strategy": solver_strategy,
                "mesh": mesh_audit_before,
                "source_metrics": source_metrics,
                "error": str(exc),
                "partial_model": str(partial.resolve()),
                "partial_model_sha256": _sha256(partial),
            }
            failure_path = args.audit.with_name(
                f"{args.audit.stem}_failure.json"
            )
            with failure_path.open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False)
            print(f"Wrote failure audit: {failure_path}", flush=True)
            raise
        model.rename(args.model_out.stem)
        model.save(str(args.model_out))
        refined_metrics = evaluate_solution(model)
        refined_mesh = jm.component("comp1").mesh("mesh1")
        mesh_audit_after = {
            "elements": int(refined_mesh.getNumElem()),
            "vertices": int(refined_mesh.getNumVertex()),
            "minimum_quality": float(refined_mesh.getMinQuality()),
            "mean_quality": float(refined_mesh.getMeanQuality()),
        }
    finally:
        client.clear()

    if mesh_audit_after != mesh_audit_before:
        raise RuntimeError("Same-mesh refinement changed the mesh")
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
        "mass_imbalance_fraction": relative_change(
            source_metrics["mass_flux_kg_s"]["imbalance_fraction"],
            refined_metrics["mass_flux_kg_s"]["imbalance_fraction"],
        ),
        "energy_imbalance_fraction": relative_change(
            source_metrics["energy_balance_w"][
                "imbalance_fraction_of_inlet"
            ],
            refined_metrics["energy_balance_w"][
                "imbalance_fraction_of_inlet"
            ],
        ),
    }
    bounded_pass = _bounded(refined_metrics, contract)
    mass_pass = (
        refined_metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    )
    energy_pass = (
        refined_metrics["energy_balance_w"][
            "imbalance_fraction_of_inlet"
        ]
        < 0.02
    )
    algebraically_stable = (
        changes["near_target_temperature_fraction"] < 0.002
        and changes["near_target_speed_fraction"] < 0.01
    )
    audit = {
        "schema_version": "h11_same_mesh_refinement_v1",
        "status": (
            "pass_same_mesh_refinement_numerical_gates_not_calibrated"
            if (
                bounded_pass
                and mass_pass
                and energy_pass
                and algebraically_stable
            )
            else "pass_solve_fail_one_or_more_refinement_gates"
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
        "solver_strategy": solver_strategy,
        "mesh": mesh_audit_after,
        "source_metrics": source_metrics,
        "refined_metrics": refined_metrics,
        "relative_changes": changes,
        "gates": {
            "same_mesh_identity": True,
            "physically_bounded": bounded_pass,
            "mass_imbalance_below_0_5_percent": mass_pass,
            "energy_imbalance_below_2_percent": energy_pass,
            "near_target_algebraically_stable": algebraically_stable,
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
        "runtime_sec": time.time() - started,
        "comsol_version": args.version,
        "cores": args.cores,
        "source_model": str(args.source_model.resolve()),
        "source_sha256": _sha256(args.source_model),
        "model_path": str(args.model_out.resolve()),
        "model_sha256": _sha256(args.model_out),
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {args.model_out}")
    print(f"Wrote audit: {args.audit}")
    print(
        f"Same-mesh refinement: {audit['status']}; "
        f"mass={refined_metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}; "
        "energy="
        f"{refined_metrics['energy_balance_w']['imbalance_fraction_of_inlet']:.3%}; "
        f"dT={changes['near_target_temperature_fraction']:.3%}; "
        f"dU={changes['near_target_speed_fraction']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
