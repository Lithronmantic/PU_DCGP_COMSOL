
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

from pu_dcgp_comsol.comsol.target_impact_conservative_nominal import (
    ConservativeSolveContract,
    _set_manual_scales,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_restart import (
    evaluate_solution,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_mesh"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_mesh"


@dataclass(frozen=True)
class ConservativeMeshRefinementContract:

    source_level: int
    target_level: int
    relative_tolerance: float = 1e-4
    maximum_segregated_iterations: int = 4000
    maximum_fully_coupled_iterations: int = 300
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    maximum_admissible_temperature_k: float = 10_000.01
    maximum_admissible_speed_m_s: float = 600.01
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        valid = {2, 3, 4, 5}
        if self.source_level not in valid or self.target_level not in valid:
            raise ValueError("COMSOL automatic mesh levels must lie in {2,3,4,5}")
        if self.source_level - self.target_level != 1:
            raise ValueError("Refinement must advance through one adjacent level")
        if not math.isfinite(self.relative_tolerance) or not (
            0 < self.relative_tolerance <= 1e-3
        ):
            raise ValueError("Relative tolerance must lie in (0,1e-3]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Segregated iteration limit is too small")
        if self.maximum_fully_coupled_iterations < 100:
            raise ValueError("Fully coupled iteration limit is too small")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if self.pressure_floor_pa <= 0:
            raise ValueError("Absolute-pressure floor must be positive")
        if (
            self.maximum_admissible_temperature_k
            <= self.property_temperature_floor_k
        ):
            raise ValueError("Maximum admissible temperature is invalid")
        if self.maximum_admissible_speed_m_s <= 0:
            raise ValueError("Maximum admissible speed must be positive")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Mesh verification cannot claim prediction")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _last_tag(collection: Any) -> str:
    tags = [str(value) for value in collection.tags()]
    if not tags:
        raise RuntimeError("Expected at least one COMSOL tag")
    return tags[-1]


def _bounded(
    metrics: dict[str, Any],
    contract: ConservativeMeshRefinementContract,
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
        and temperature["maximum"]
        <= contract.maximum_admissible_temperature_k
        and speed["minimum"] >= -1e-9
        and speed["maximum"] <= contract.maximum_admissible_speed_m_s
        and pressure["minimum"] > contract.pressure_floor_pa
    )


def configure_refinement(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: ConservativeMeshRefinementContract,
) -> tuple[str, str, dict[str, Any]]:
    contract.validate()
    hmnf = jm.component("comp1").physics("hmnf")
    if str(hmnf.getType()) != "HighMachNumberFlowTurbulentkomega":
        raise RuntimeError("Final physics is no longer all-Mach k-omega")
    if (
        str(
            hmnf.prop("PhysicalModelProperty").getString(
                "includeKineticEnergy"
            )
        )
        != "1"
    ):
        raise RuntimeError("Conservative total energy is disabled")


    initial = hmnf.feature("init1")
    initial.set("u_init", ["0", "0", "0"])
    initial.set("p_init", "p_amb")
    initial.set("Tinit", "T_amb")
    initial.set("k_init", "hmnf.kinit")
    initial.set("om_init", "hmnf.omInit")

    mesh = jm.component("comp1").mesh("mesh1")
    mesh.autoMeshSize(contract.target_level)
    mesh.run()

    study_tag = f"std_hmnf_mesh_{contract.target_level}"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label(
        f"Conservative all-Mach mesh level {contract.target_level} refinement"
    )
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
        ConservativeSolveContract(
            automatic_mesh_level=contract.target_level,
            refinement_maximum_segregated_iterations=(
                contract.maximum_segregated_iterations
            ),
            property_temperature_floor_k=(
                contract.property_temperature_floor_k
            ),
            pressure_floor_pa=contract.pressure_floor_pa,
        ),
    )
    stationary = jm.sol(solution_tag).feature("s1")
    stationary.set("stol", f"{contract.relative_tolerance:.12g}")
    if "se1" not in {
        str(value) for value in stationary.feature().tags()
    }:
        raise RuntimeError("Expected COMSOL segregated all-Mach solver")
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
    for step_tag in ("ss1", "ss2"):
        substep = segregated.feature(step_tag)
        substep.set("subdtech", "hnlin")
        substep.set("subinitsteph", "1e-4")
        substep.set("subminsteph", "1e-12")
        substep.set("usesubminsteprecovery", "on")
        substep.set("subminsteprecovery", "0.1")
        substep.set("subtermonres", "off")
        substep.set("subtermauto", "itertol")
        substep.set("subiter", "20")
        substep.set("maxsubiter", "100")
    nonlinear_method = (
        "segregated_highly_nonlinear_solution_terminated_substeps"
    )
    return study_tag, solution_tag, {
        "manual_scales": scales,
        "nonlinear_method": nonlinear_method,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-level", type=int, required=True)
    parser.add_argument("--target-level", type=int, required=True)
    parser.add_argument("--source-study")
    parser.add_argument("--source-solution")
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.source_model.exists():
        raise FileNotFoundError(args.source_model)
    contract = ConservativeMeshRefinementContract(
        source_level=args.source_level,
        target_level=args.target_level,
    )
    contract.validate()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = args.model_out or (
        MODEL_DIR
        / f"h11_target_impact_conservative_mesh_level_{contract.target_level}.mph"
    )
    audit_path = args.audit or (
        OUT_DIR
        / f"h11_target_impact_conservative_mesh_level_{contract.target_level}.json"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.source_model))
        jm = model.java
        source_study_tag = args.source_study or _last_tag(jm.study())
        source_solution_tag = args.source_solution or _last_tag(jm.sol())
        source_metrics = evaluate_solution(model)
        if not _bounded(source_metrics, contract):
            raise RuntimeError("Source all-Mach solution is not bounded")
        study_tag, solution_tag, solver_strategy = configure_refinement(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=contract,
        )
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_audit = {
            "automatic_level": contract.target_level,
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        try:
            jm.study(study_tag).run()
        except Exception as exc:
            partial = model_path.with_name(f"{model_path.stem}_partial.mph")
            model.save(str(partial))
            failure = {
                "schema_version": "h11_conservative_mesh_failure_v1",
                "status": "failed_target_mesh_equation_solve",
                "contract": asdict(contract),
                "source_study": source_study_tag,
                "source_solution": source_solution_tag,
                "target_study": study_tag,
                "target_solution": solution_tag,
                "solver_strategy": solver_strategy,
                "mesh": mesh_audit,
                "error": str(exc),
                "partial_model": str(partial.resolve()),
                "partial_model_sha256": _sha256(partial),
            }
            failure_path = audit_path.with_name(f"{audit_path.stem}_failure.json")
            with failure_path.open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False)
            print(f"Wrote failure audit: {failure_path}", flush=True)
            raise
        model.rename(
            f"h11_target_impact_conservative_mesh_{contract.target_level}"
        )
        model.save(str(model_path))
        metrics = evaluate_solution(model)
    finally:
        client.clear()

    bounded_pass = _bounded(metrics, contract)
    mass_pass = metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    energy_pass = (
        metrics["energy_balance_w"]["imbalance_fraction_of_inlet"] < 0.02
    )
    audit = {
        "schema_version": "h11_conservative_mesh_refinement_v1",
        "status": (
            "pass_target_mesh_numerical_gates_not_calibrated"
            if bounded_pass and mass_pass and energy_pass
            else "pass_solve_fail_one_or_more_numerical_gates"
        ),
        "contract": asdict(contract),
        "strategy": {
            "type": "adjacent_all_mach_solution_projection",
            "role": "initial_iterate_only",
            "target_equations_resolved": True,
            "source_study": source_study_tag,
            "source_solution": source_solution_tag,
            "target_study": study_tag,
            "target_solution": solution_tag,
        },
        "solver_strategy": solver_strategy,
        "source_health": {
            "temperature_k": source_metrics["temperature_k"],
            "speed_m_s": source_metrics["speed_m_s"],
            "absolute_pressure_pa": source_metrics["absolute_pressure_pa"],
            "bounded": True,
        },
        "mesh": mesh_audit,
        "metrics": metrics,
        "gates": {
            "physically_bounded": bounded_pass,
            "mass_imbalance_below_0_5_percent": mass_pass,
            "energy_imbalance_below_2_percent": energy_pass,
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
        "runtime_sec": time.time() - started,
        "comsol_version": args.version,
        "cores": args.cores,
        "source_model": str(args.source_model.resolve()),
        "source_sha256": _sha256(args.source_model),
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {model_path}")
    print(f"Wrote audit: {audit_path}")
    print(
        f"Conservative mesh {contract.target_level}: {audit['status']}; "
        f"mass={metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}; "
        "energy="
        f"{metrics['energy_balance_w']['imbalance_fraction_of_inlet']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
