
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

from pu_dcgp_comsol.comsol.target_impact_nominal_smoke import (
    NominalSmokeContract,
    evaluate_final_solution,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_mesh_convergence"
OUT_DIR = HERE / "h11_outputs" / "target_impact_mesh_convergence"


@dataclass(frozen=True)
class MeshRefinementContract:

    source_level: int
    target_level: int
    relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 3000
    property_temperature_floor_k: float = 250.0
    maximum_admissible_temperature_k: float = 10_000.01
    maximum_admissible_speed_m_s: float = 600.01
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        valid_levels = {2, 3, 4, 5}
        if self.source_level not in valid_levels or self.target_level not in valid_levels:
            raise ValueError("COMSOL automatic mesh levels must lie in {2, 3, 4, 5}")
        if self.source_level - self.target_level != 1:
            raise ValueError(
                "Mesh refinement must proceed through one adjacent level at a time"
            )
        if not math.isfinite(self.relative_tolerance) or not (
            0 < self.relative_tolerance <= 1e-5
        ):
            raise ValueError("Relative tolerance must lie in (0, 1e-5]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Maximum segregated iterations must be at least 1000")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the argon-table lower bound")
        if self.maximum_admissible_temperature_k <= self.property_temperature_floor_k:
            raise ValueError("Maximum admissible temperature is invalid")
        if self.maximum_admissible_speed_m_s <= 0:
            raise ValueError("Maximum admissible speed must be positive")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Mesh refinement cannot claim calibration or paper use")


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


def _bounded_solution(
    metrics: dict[str, Any], contract: MeshRefinementContract
) -> bool:
    temperature = metrics["temperature_k"]
    speed = metrics["speed_m_s"]
    pressure = metrics["gauge_pressure_pa"]
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
        and temperature["maximum"] <= contract.maximum_admissible_temperature_k
        and speed["minimum"] >= -1e-9
        and speed["maximum"] <= contract.maximum_admissible_speed_m_s
    )


def configure_projected_refinement(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: MeshRefinementContract,
) -> tuple[str, str]:

    contract.validate()
    if str(
        jm.component("comp1")
        .physics("spf")
        .prop("PhysicalModelProperty")
        .getString("Compressibility")
    ) != "WeaklyCompressible":
        raise RuntimeError("Frozen target physics is no longer weakly compressible")
    nonisothermal = jm.component("comp1").multiphysics("nitf1")
    if str(nonisothermal.getString("ThermalTurbType")) != "KaysCrawford":
        raise RuntimeError("Target refinement must use the frozen Kays-Crawford model")

    jm.param().set("load_s", "1")
    mesh = jm.component("comp1").mesh("mesh1")
    mesh.autoMeshSize(contract.target_level)
    mesh.run()

    study_tag = f"std_mesh_{contract.target_level}"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label(f"Full-load mesh level {contract.target_level} refinement")
    stationary_step = study.create("stat", "Stationary")
    stationary_step.set("useinitsol", "on")
    stationary_step.set("initmethod", "sol")
    stationary_step.set("initstudy", source_study_tag)
    stationary_step.set("initstudystep", "stat")
    stationary_step.set("initsol", source_solution_tag)
    stationary_step.set("solnum", "last")
    study.createAutoSequences("all")

    solution_tag = _last_tag(jm.sol())
    stationary_solver = jm.sol(solution_tag).feature("s1")
    stationary_solver.set("stol", f"{contract.relative_tolerance:.12g}")
    segregated = stationary_solver.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.maximum_segregated_iterations),
    )
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.om 0 comp1.k 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] ",
    )
    return study_tag, solution_tag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-level", type=int, required=True)
    parser.add_argument("--target-level", type=int, required=True)
    parser.add_argument("--source-study")
    parser.add_argument("--source-solution")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.source_model.exists():
        raise FileNotFoundError(args.source_model)

    contract = MeshRefinementContract(
        source_level=args.source_level,
        target_level=args.target_level,
    )
    contract.validate()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / (
        f"h11_target_impact_mesh_level_{contract.target_level}.mph"
    )
    audit_path = OUT_DIR / (
        f"h11_target_impact_mesh_level_{contract.target_level}.json"
    )

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.source_model))
        jm = model.java
        source_study_tag = args.source_study or _last_tag(jm.study())
        source_solution_tag = args.source_solution or _last_tag(jm.sol())
        source_metrics = evaluate_final_solution(model)
        if not _bounded_solution(source_metrics, contract):
            raise RuntimeError("Source mesh solution is not physically bounded")

        study_tag, solution_tag = configure_projected_refinement(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=contract,
        )
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_statistics = {
            "automatic_level": contract.target_level,
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        jm.study(study_tag).run()
        model.rename(f"h11_target_impact_mesh_level_{contract.target_level}")
        model.save(str(model_path))
        metrics = evaluate_final_solution(model)
    finally:
        client.clear()

    bounded_pass = _bounded_solution(metrics, contract)
    mass_pass = metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    energy_pass = (
        metrics["energy_balance_w"]["imbalance_fraction_of_inlet"] < 0.02
    )
    audit = {
        "schema_version": "h11_target_impact_mesh_refinement_v1",
        "status": (
            "pass_mesh_level_numerical_gates"
            if bounded_pass and mass_pass and energy_pass
            else "pass_solve_fail_one_or_more_numerical_gates"
        ),
        "contract": asdict(contract),
        "strategy": {
            "type": "adjacent_coarse_to_fine_solution_projection",
            "role": "initial_iterate_only",
            "target_equations_resolved": True,
            "source_study": source_study_tag,
            "source_solution": source_solution_tag,
            "target_study": study_tag,
            "target_solution": solution_tag,
        },
        "source_health": {
            "temperature_k": source_metrics["temperature_k"],
            "speed_m_s": source_metrics["speed_m_s"],
            "gauge_pressure_pa": source_metrics["gauge_pressure_pa"],
            "bounded": True,
        },
        "mesh": mesh_statistics,
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
        f"Mesh {contract.target_level}: {audit['status']}; "
        f"mass={metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}; "
        "energy="
        f"{metrics['energy_balance_w']['imbalance_fraction_of_inlet']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
