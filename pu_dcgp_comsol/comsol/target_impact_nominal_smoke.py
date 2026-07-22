
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

import numpy as np


HERE = Path(__file__).resolve().parent
SKELETON_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_physics_skeleton"
    / "h11_target_impact_physics_skeleton_latest.mph"
)
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_nominal_smoke"
OUT_DIR = HERE / "h11_outputs" / "target_impact_nominal_smoke"
MODEL_PATH = MODEL_DIR / "h11_target_impact_nominal_smoke_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_target_impact_nominal_smoke_audit.json"


@dataclass(frozen=True)
class NominalSmokeContract:

    load_fractions: tuple[float, ...] = (
        0.0,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )
    seed_velocity_m_s: float = 1.0
    property_temperature_floor_k: float = 250.0
    stationary_relative_tolerance: float = 1e-4
    refinement_relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 800
    refinement_maximum_segregated_iterations: int = 2000
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        values = self.load_fractions
        if len(values) < 2 or values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("Continuation must span exactly 0 to 1")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Continuation fractions must be finite and lie in [0, 1]")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Continuation fractions must be strictly increasing")
        if self.seed_velocity_m_s <= 0:
            raise ValueError("Seed velocity must be positive")
        if self.property_temperature_floor_k < 250.0:
            raise ValueError("Temperature floor must respect the argon table lower bound")
        if not 0 < self.stationary_relative_tolerance <= 1e-3:
            raise ValueError("Stationary relative tolerance must lie in (0, 1e-3]")
        if not 0 < self.refinement_relative_tolerance <= self.stationary_relative_tolerance:
            raise ValueError("Refinement tolerance must be no larger than sweep tolerance")
        if self.maximum_segregated_iterations < 300:
            raise ValueError("Segregated iteration limit is too small for the continuation")
        if self.refinement_maximum_segregated_iterations < 300:
            raise ValueError("Refinement iteration limit is too small")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("The nominal smoke contract cannot claim calibration or prediction")

    def continuation_list(self) -> str:
        return " ".join(f"{value:.12g}" for value in self.load_fractions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"Expected one value, received shape {array.shape}")
    return float(array.reshape(-1)[0])


def _field_range(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("COMSOL returned no finite field values")
    return {
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _replace_feature(collection: Any, tag: str, feature_type: str) -> Any:
    if tag in {str(value) for value in collection.tags()}:
        collection.remove(tag)
    return collection.create(tag, feature_type)


def _solution_datasets(model: Any) -> list[Any]:
    datasets = []
    for dataset in model / "datasets":
        properties = set(dataset.properties())
        if "solution" in properties or "data" in properties:
            datasets.append(dataset)
    if not datasets:
        raise RuntimeError("No solution dataset exists after the COMSOL solve")
    return datasets


def configure_continuation(jm: Any, contract: NominalSmokeContract) -> None:

    contract.validate()
    comp = jm.component("comp1")
    spf = comp.physics("spf")
    ht = comp.physics("ht")
    spf.prop("PhysicalModelProperty").set("Compressibility", "WeaklyCompressible")
    spf.prop("PhysicalModelProperty").set("Tref", "T_amb")
    spf.prop("PhysicalModelProperty").set("pref", "p_amb")

    jm.param().set("load_s", "0", "Numerical continuation fraction; not a DOE factor")
    spf.feature("inl1").set(
        "U0in",
        f"({contract.seed_velocity_m_s:.12g}[m/s]"
        f"+load_s*(u_exit_eff-{contract.seed_velocity_m_s:.12g}[m/s]))"
        "*nozzle_shape",
    )
    ht.feature("temp_nozzle").set(
        "T0",
        "T_amb+load_s*(T_exit_eff-T_amb)*nozzle_shape",
    )
    ht.feature("temp_target").set("T0", "T_amb+load_s*(T_target-T_amb)")
    ht.feature("open1").set("Tustr", "T_amb")
    ht.feature("fluid1").set("streamline", "1")
    ht.feature("fluid1").set("crosswind", "1")

    couplings = comp.cpl()
    for tag, label, selection in (
        ("int_nozzle", "Axisymmetric nozzle-flux integral", "geom1_sel_nozzle_in"),
        ("int_open", "Axisymmetric open-boundary flux integral", None),
        ("int_target", "Axisymmetric target-wall flux integral", "geom1_sel_target"),
    ):
        operator = _replace_feature(couplings, tag, "Integration")
        operator.label(label)
        operator.selection().geom("geom1", 1)
        if selection is not None:
            operator.selection().named(selection)
    open_entities = sorted(
        {
            *[int(value) for value in comp.selection("geom1_sel_ambient_in").entities()],
            *[int(value) for value in comp.selection("geom1_sel_far_r").entities()],
        }
    )
    comp.cpl("int_open").selection().set(open_entities)

    study = jm.study("std1")
    parametric = _replace_feature(study.feature(), "param", "Parametric")
    parametric.label("Numerical load continuation")
    parametric.set("pname", ["load_s"])
    parametric.set("plistarr", [contract.continuation_list()])
    parametric.set("punit", [""])
    parametric.set("sweeptype", "filled")
    parametric.set("reusesol", "on")
    parametric.set("keepsol", "all")

    comp.mesh("mesh1").run()
    study.createAutoSequences("all")
    stationary = jm.sol("sol1").feature("s1")
    stationary.set("stol", f"{contract.stationary_relative_tolerance:.12g}")
    segregated = stationary.feature("se1")
    segregated.set("maxsegiter", str(contract.maximum_segregated_iterations))
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.om 0 comp1.k 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] ",
    )


def configure_full_load_refinement(jm: Any, contract: NominalSmokeContract) -> str:

    jm.param().set("load_s", "1")
    studies = jm.study()
    if "std_refine" in {str(tag) for tag in studies.tags()}:
        studies.remove("std_refine")
    study = studies.create("std_refine")
    study.label("Full-load residual refinement")
    study.create("stat", "Stationary")
    study.createAutoSequences("all")

    solution_tag = str(list(jm.sol().tags())[-1])
    stationary = jm.sol(solution_tag).feature("s1")
    stationary.set("stol", f"{contract.refinement_relative_tolerance:.12g}")
    stationary.set("linpmethod", "sol")
    stationary.set("linpsol", "sol1")
    stationary.set("linpsoluse", "sol1")
    stationary.set("linpsolvertype", "solnum")
    stationary.set("linpsolnum", "last")
    segregated = stationary.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.refinement_maximum_segregated_iterations),
    )
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.om 0 comp1.k 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] ",
    )
    return solution_tag


def evaluate_final_solution(model: Any) -> dict[str, Any]:
    datasets = _solution_datasets(model)
    continuation_candidates: list[tuple[int, Any, Any, Any, str]] = []
    for dataset in datasets:
        outer_indices, outer_values = model.outer(dataset)
        if len(outer_indices):
            continuation_candidates.append(
                (len(outer_indices), dataset, outer_indices, outer_values, "outer")
            )
            continue
        inner_indices, inner_values = model.inner(dataset)
        continuation_candidates.append(
            (len(inner_indices), dataset, inner_indices, inner_values, "inner")
        )
    _, _, solution_indices, solution_values, storage_axis = max(
        continuation_candidates,
        key=lambda item: item[0],
    )
    if len(solution_indices) == 0 or not math.isclose(float(solution_values[-1]), 1.0):
        raise RuntimeError("Continuation did not return the full-load solution")

    dataset = datasets[-1]
    refinement_inner, _ = model.inner(dataset)
    evaluation_selector = {"inner": "last"} if len(refinement_inner) else {}
    temperature, speed, pressure = model.evaluate(
        ["T", "spf.U", "p"],
        unit=["K", "m/s", "Pa"],
        dataset=dataset,
        **evaluation_selector,
    )
    nozzle_mass_out = _scalar(
        model.evaluate(
            "int_nozzle(2*pi*r*spf.rho*(u*nr+w*nz))",
            unit="kg/s",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    open_mass_out = _scalar(
        model.evaluate(
            "int_open(2*pi*r*spf.rho*(u*nr+w*nz))",
            unit="kg/s",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    target_mass_out = _scalar(
        model.evaluate(
            "int_target(2*pi*r*spf.rho*(u*nr+w*nz))",
            unit="kg/s",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    input_mass = abs(nozzle_mass_out)
    mass_residual = abs(nozzle_mass_out + open_mass_out + target_mass_out)
    mass_imbalance_fraction = mass_residual / input_mass
    nozzle_energy_out = _scalar(
        model.evaluate(
            "int_nozzle(2*pi*r*ht.nteflux)",
            unit="W",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    open_energy_out = _scalar(
        model.evaluate(
            "int_open(2*pi*r*ht.nteflux)",
            unit="W",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    target_energy_out = _scalar(
        model.evaluate(
            "int_target(2*pi*r*ht.nteflux)",
            unit="W",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    energy_balance = _scalar(
        model.evaluate(
            "ht.energyBalance",
            unit="W",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    total_net_energy_rate = _scalar(
        model.evaluate(
            "ht.ntefluxInt",
            unit="W",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    work_source = _scalar(
        model.evaluate(
            "ht.WInt",
            unit="W",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    heat_source = _scalar(
        model.evaluate(
            "ht.QInt",
            unit="W",
            dataset=dataset,
            **evaluation_selector,
        )
    )
    energy_imbalance_fraction = abs(energy_balance) / abs(nozzle_energy_out)

    return {
        "n_continuation_solutions": int(len(solution_indices)),
        "continuation_storage_axis": storage_axis,
        "full_load_fraction": float(solution_values[-1]),
        "full_load_refinement_solutions": int(len(refinement_inner)),
        "temperature_k": _field_range(temperature),
        "speed_m_s": _field_range(speed),
        "gauge_pressure_pa": _field_range(pressure),
        "one_mm_upstream_of_target": {
            "temperature_k": _scalar(
                model.evaluate(
                    "at2(1e-6[m],d_spray-1[mm],T)",
                    unit="K",
                    dataset=dataset,
                    **evaluation_selector,
                )
            ),
            "speed_m_s": _scalar(
                model.evaluate(
                    "at2(1e-6[m],d_spray-1[mm],spf.U)",
                    unit="m/s",
                    dataset=dataset,
                    **evaluation_selector,
                )
            ),
        },
        "mass_flux_kg_s": {
            "nozzle_outward": nozzle_mass_out,
            "open_boundaries_outward": open_mass_out,
            "target_outward": target_mass_out,
            "absolute_residual": mass_residual,
            "imbalance_fraction": mass_imbalance_fraction,
        },
        "energy_balance_w": {
            "nozzle_outward": nozzle_energy_out,
            "open_boundaries_outward": open_energy_out,
            "target_outward": target_energy_out,
            "total_net_energy_rate": total_net_energy_rate,
            "work_source": work_source,
            "heat_source": heat_source,
            "comsol_energy_balance": energy_balance,
            "imbalance_fraction_of_inlet": energy_imbalance_fraction,
            "definition": "ht.energyBalance=dEi0Int+ntefluxInt-WInt-QInt",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model-in", type=Path, default=SKELETON_MODEL)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument(
        "--resume-solved",
        action="store_true",
        help="Evaluate an already solved --model-out without rerunning COMSOL.",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.model_in.exists():
        raise FileNotFoundError(args.model_in)

    contract = NominalSmokeContract()
    contract.validate()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        if args.resume_solved:
            if not args.model_out.exists():
                raise FileNotFoundError(args.model_out)
            model = client.load(str(args.model_out))
        else:
            model = client.load(str(args.model_in))
            model.rename("h11_target_impact_nominal_smoke")
            configure_continuation(model.java, contract)
            model.java.study("std1").run()
            configure_full_load_refinement(model.java, contract)
            model.java.study("std_refine").run()
            model.save(str(args.model_out))
        metrics = evaluate_final_solution(model)
    finally:
        client.clear()

    mass_gate_passed = metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    temperature_floor_inactive = (
        metrics["temperature_k"]["minimum"]
        > contract.property_temperature_floor_k + 1e-6
    )
    energy_gate_passed = (
        metrics["energy_balance_w"]["imbalance_fraction_of_inlet"] < 0.02
    )
    audit = {
        "schema_version": "h11_target_impact_nominal_smoke_v1",
        "status": (
            "pass_numerical_smoke_not_calibrated"
            if mass_gate_passed and temperature_floor_inactive and energy_gate_passed
            else "pass_solve_fail_conservation_or_boundedness_gate"
        ),
        "contract": asdict(contract),
        "metrics": metrics,
        "numerical_gates": {
            "mass_imbalance_below_0_5_percent": mass_gate_passed,
            "temperature_floor_inactive": temperature_floor_inactive,
            "energy_balance_below_2_percent": energy_gate_passed,
        },
        "interpretation": {
            "physical_formulation_fix": (
                "Weakly compressible flow is required because argon density "
                "varies with the solved temperature."
            ),
            "numerical_fix": (
                "COMSOL-native parametric continuation and a 250 K property-"
                "validity floor prevent nonphysical Newton iterates."
            ),
            "paper_prediction_allowed": False,
            "reason": (
                "The effective inlet state has not yet been calibrated inside "
                "training folds or validated against held-out DPV data."
            ),
        },
        "runtime_sec": time.time() - started,
        "comsol_version": args.version,
        "cores": args.cores,
        "source_model": str(args.model_in.resolve()),
        "source_sha256": _sha256(args.model_in),
        "model_path": str(args.model_out.resolve()),
        "model_sha256": _sha256(args.model_out),
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    print(f"Saved model: {args.model_out}")
    print(f"Wrote audit: {args.audit}")
    print(
        f"Numerical smoke: {audit['status']}; experimental prediction remains LOCKED; "
        f"mass imbalance={metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
