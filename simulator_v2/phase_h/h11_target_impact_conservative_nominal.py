"""H11 layer 10: solve and audit the conservative target-impact baseline.

The runner ramps the uncalibrated effective-exit boundary from an ambient,
low-speed seed to the provisional full-load state using a COMSOL parametric
continuation.  A separate tight full-load solve is initialized from the
continuation endpoint.  The result is a numerical verification artifact only.
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

import numpy as np


HERE = Path(__file__).resolve().parent
SKELETON_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_skeleton"
    / "h11_target_impact_conservative_skeleton_latest.mph"
)
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_nominal"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_nominal"
MODEL_PATH = MODEL_DIR / "h11_target_impact_conservative_nominal_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_target_impact_conservative_nominal_audit.json"


@dataclass(frozen=True)
class ConservativeSolveContract:
    """Numerical settings; none is an experimental calibration result."""

    load_fractions: tuple[float, ...] = (
        0.0,
        0.002,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    )
    seed_velocity_m_s: float = 1.0
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    continuation_relative_tolerance: float = 5e-4
    refinement_relative_tolerance: float = 1e-6
    maximum_segregated_iterations: int = 2000
    refinement_maximum_segregated_iterations: int = 4000
    automatic_mesh_level: int = 4
    pressure_scale_pa: float = 1e5
    temperature_scale_k: float = 1e4
    velocity_scale_m_s: float = 600.0
    turbulent_kinetic_energy_scale_m2_s2: float = 1e3
    specific_dissipation_scale_s_inv: float = 1e6
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        values = self.load_fractions
        if len(values) < 3 or values[0] != 0 or values[-1] != 1:
            raise ValueError("Continuation must span exactly 0 to 1")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Continuation fractions must be finite in [0, 1]")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Continuation fractions must be strictly increasing")
        if self.seed_velocity_m_s <= 0:
            raise ValueError("Seed velocity must be positive")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar property table")
        if self.pressure_floor_pa <= 0:
            raise ValueError("Absolute-pressure floor must be positive")
        if not 0 < self.continuation_relative_tolerance <= 1e-3:
            raise ValueError("Continuation tolerance must lie in (0, 1e-3]")
        if not 0 < self.refinement_relative_tolerance <= 1e-5:
            raise ValueError("Refinement tolerance must lie in (0, 1e-5]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Continuation iteration limit is too small")
        if self.refinement_maximum_segregated_iterations < 1000:
            raise ValueError("Refinement iteration limit is too small")
        if self.automatic_mesh_level not in {2, 3, 4, 5}:
            raise ValueError("Unsupported COMSOL automatic mesh level")
        scales = (
            self.pressure_scale_pa,
            self.temperature_scale_k,
            self.velocity_scale_m_s,
            self.turbulent_kinetic_energy_scale_m2_s2,
            self.specific_dissipation_scale_s_inv,
        )
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("Dependent-variable scales must be finite and positive")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Numerical verification cannot claim paper prediction")

    def continuation_list(self) -> str:
        return " ".join(f"{value:.12g}" for value in self.load_fractions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _replace_feature(collection: Any, tag: str, feature_type: str) -> Any:
    if tag in {str(value) for value in collection.tags()}:
        collection.remove(tag)
    return collection.create(tag, feature_type)


def _last_tag(collection: Any) -> str:
    tags = [str(value) for value in collection.tags()]
    if not tags:
        raise RuntimeError("Expected a nonempty COMSOL collection")
    return tags[-1]


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"Expected scalar, received {array.shape}")
    return float(array.reshape(-1)[0])


def _range(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size != array.size or finite.size == 0:
        raise ValueError("COMSOL field contains undefined values")
    return {"minimum": float(finite.min()), "maximum": float(finite.max())}


def _set_manual_scales(
    jm: Any,
    solution_tag: str,
    contract: ConservativeSolveContract,
) -> dict[str, float]:
    variables = jm.sol(solution_tag).feature("v1")
    mapping = {
        "comp1_p": contract.pressure_scale_pa,
        "comp1_T": contract.temperature_scale_k,
        "comp1_u": contract.velocity_scale_m_s,
        "comp1_k": contract.turbulent_kinetic_energy_scale_m2_s2,
        "comp1_om": contract.specific_dissipation_scale_s_inv,
    }
    present = {str(value) for value in variables.feature().tags()}
    applied = {}
    for tag, value in mapping.items():
        if tag not in present:
            continue
        field = variables.feature(tag)
        field.set("scalemethod", "manual")
        field.set("scaleval", f"{value:.12g}")
        applied[tag] = value
    return applied


def configure_continuation(jm: Any, contract: ConservativeSolveContract) -> None:
    contract.validate()
    comp = jm.component("comp1")
    hmnf = comp.physics("hmnf")
    if str(hmnf.getType()) != "HighMachNumberFlowTurbulentkomega":
        raise RuntimeError("Conservative physics type changed")
    physical = hmnf.prop("PhysicalModelProperty")
    if str(physical.getString("includeKineticEnergy")) != "1":
        raise RuntimeError("Conservative total energy is no longer enabled")

    jm.param().set("load_s", "0", "Numerical continuation fraction")
    nozzle = hmnf.feature("nozzle_in")
    nozzle.set(
        "U0in",
        f"({contract.seed_velocity_m_s:.12g}[m/s]"
        f"+load_s*(u_exit_eff-{contract.seed_velocity_m_s:.12g}[m/s]))"
        "*nozzle_shape",
    )
    nozzle.set("T0", "T_amb+load_s*(T_exit_eff-T_amb)*nozzle_shape")
    hmnf.feature("target_temperature").set(
        "T0",
        "T_amb+load_s*(T_target-T_amb)",
    )

    couplings = comp.cpl()
    for tag, label, selection in (
        ("int_nozzle", "Axisymmetric nozzle integral", "geom1_sel_nozzle_in"),
        ("int_target", "Axisymmetric target integral", "geom1_sel_target"),
    ):
        operator = _replace_feature(couplings, tag, "Integration")
        operator.label(label)
        operator.selection().geom("geom1", 1)
        operator.selection().named(selection)
    operator = _replace_feature(couplings, "int_ambient", "Integration")
    operator.label("Axisymmetric ambient-opening integral")
    operator.selection().geom("geom1", 1)
    ambient_entities = sorted(
        {
            *[
                int(value)
                for value in comp.selection("geom1_sel_ambient_in").entities()
            ],
            *[int(value) for value in comp.selection("geom1_sel_far_r").entities()],
        }
    )
    operator.selection().set(ambient_entities)

    mesh = comp.mesh("mesh1")
    mesh.autoMeshSize(contract.automatic_mesh_level)
    mesh.run()

    study = jm.study("std1")
    parametric = _replace_feature(study.feature(), "param", "Parametric")
    parametric.label("Conservative effective-exit continuation")
    parametric.set("pname", ["load_s"])
    parametric.set("plistarr", [contract.continuation_list()])
    parametric.set("punit", [""])
    parametric.set("sweeptype", "filled")
    parametric.set("reusesol", "on")
    parametric.set("keepsol", "all")
    study.createAutoSequences("all")

    _set_manual_scales(jm, "sol1", contract)
    stationary = jm.sol("sol1").feature("s1")
    stationary.set("stol", f"{contract.continuation_relative_tolerance:.12g}")
    if "se1" in {str(value) for value in stationary.feature().tags()}:
        segregated = stationary.feature("se1")
        segregated.set("maxsegiter", str(contract.maximum_segregated_iterations))
        segregated.feature("ll1").set(
            "lowerlimit",
            f"comp1.k 0 comp1.om 0 comp1.T "
            f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
            f"{contract.pressure_floor_pa:.12g}[Pa] ",
        )


def configure_refinement(
    jm: Any,
    contract: ConservativeSolveContract,
) -> tuple[str, str]:
    jm.param().set("load_s", "1")
    studies = jm.study()
    if "std_refine" in {str(value) for value in studies.tags()}:
        studies.remove("std_refine")
    study = studies.create("std_refine")
    study.label("Conservative full-load residual refinement")
    step = study.create("stat", "Stationary")
    step.set("useinitsol", "on")
    step.set("initmethod", "sol")
    step.set("initstudy", "std1")
    step.set("initstudystep", "stat")
    step.set("initsol", "sol1")
    step.set("solnum", "last")
    study.createAutoSequences("all")
    solution_tag = _last_tag(jm.sol())
    _set_manual_scales(jm, solution_tag, contract)
    stationary = jm.sol(solution_tag).feature("s1")
    stationary.set("stol", f"{contract.refinement_relative_tolerance:.12g}")
    segregated = stationary.feature("se1")
    segregated.set(
        "maxsegiter",
        str(contract.refinement_maximum_segregated_iterations),
    )
    segregated.feature("ll1").set(
        "lowerlimit",
        f"comp1.k 0 comp1.om 0 comp1.T "
        f"{contract.property_temperature_floor_k:.12g}[K] comp1.p "
        f"{contract.pressure_floor_pa:.12g}[Pa] ",
    )
    return "std_refine", solution_tag


def evaluate_solution(model: Any) -> dict[str, Any]:
    datasets = list(model / "datasets")
    if len(datasets) < 2:
        raise RuntimeError("Expected continuation and refinement datasets")
    continuation = datasets[0]
    indices, values = model.inner(continuation)
    if len(indices) == 0 or not math.isclose(float(values[-1]), 1.0):
        raise RuntimeError("Conservative continuation did not reach full load")
    dataset = datasets[-1]
    inner_indices, _ = model.inner(dataset)
    selector = {"inner": "last"} if len(inner_indices) else {}

    temperature, speed, pressure = model.evaluate(
        ["T", "hmnf.U", "hmnf.pA"],
        unit=["K", "m/s", "Pa"],
        dataset=dataset,
        **selector,
    )
    mass_fluxes = {}
    energy_fluxes = {}
    for name, operator in (
        ("nozzle_outward", "int_nozzle"),
        ("ambient_outward", "int_ambient"),
        ("target_outward", "int_target"),
    ):
        mass_fluxes[name] = _scalar(
            model.evaluate(
                f"{operator}(2*pi*r*hmnf.rho*(u*nr+w*nz))",
                unit="kg/s",
                dataset=dataset,
                **selector,
            )
        )
        energy_fluxes[name] = _scalar(
            model.evaluate(
                f"{operator}(2*pi*r*hmnf.nteflux)",
                unit="W",
                dataset=dataset,
                **selector,
            )
        )
    mass_residual = abs(sum(mass_fluxes.values()))
    inlet_mass = abs(mass_fluxes["nozzle_outward"])
    energy_balance = _scalar(
        model.evaluate(
            "hmnf.energyBalance",
            unit="W",
            dataset=dataset,
            **selector,
        )
    )
    total_net_energy = _scalar(
        model.evaluate(
            "hmnf.ntefluxInt",
            unit="W",
            dataset=dataset,
            **selector,
        )
    )
    work = _scalar(
        model.evaluate(
            "hmnf.WInt",
            unit="W",
            dataset=dataset,
            **selector,
        )
    )
    heat = _scalar(
        model.evaluate(
            "hmnf.QInt",
            unit="W",
            dataset=dataset,
            **selector,
        )
    )
    inlet_energy = abs(energy_fluxes["nozzle_outward"])
    return {
        "n_continuation_solutions": int(len(indices)),
        "full_load_fraction": float(values[-1]),
        "temperature_k": _range(temperature),
        "speed_m_s": _range(speed),
        "absolute_pressure_pa": _range(pressure),
        "one_mm_upstream_of_target": {
            "temperature_k": _scalar(
                model.evaluate(
                    "at2(1e-6[m],d_spray-1[mm],T)",
                    unit="K",
                    dataset=dataset,
                    **selector,
                )
            ),
            "speed_m_s": _scalar(
                model.evaluate(
                    "at2(1e-6[m],d_spray-1[mm],hmnf.U)",
                    unit="m/s",
                    dataset=dataset,
                    **selector,
                )
            ),
        },
        "mass_flux_kg_s": {
            **mass_fluxes,
            "absolute_residual": mass_residual,
            "imbalance_fraction": mass_residual / inlet_mass,
        },
        "energy_balance_w": {
            **energy_fluxes,
            "total_net_energy_rate": total_net_energy,
            "work_source": work,
            "heat_source": heat,
            "comsol_energy_balance": energy_balance,
            "imbalance_fraction_of_inlet": abs(energy_balance) / inlet_energy,
            "definition": "hmnf.energyBalance=dEi0Int+ntefluxInt-WInt-QInt",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model-in", type=Path, default=SKELETON_MODEL)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--mesh-level", type=int, default=4)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.model_in.exists():
        raise FileNotFoundError(args.model_in)
    contract = ConservativeSolveContract(automatic_mesh_level=args.mesh_level)
    contract.validate()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.model_in))
        model.rename("h11_target_impact_conservative_nominal")
        jm = model.java
        configure_continuation(jm, contract)
        try:
            jm.study("std1").run()
        except Exception as exc:
            partial_model = args.model_out.with_name(
                f"{args.model_out.stem}_mesh_{contract.automatic_mesh_level}_partial.mph"
            )
            model.save(str(partial_model))
            partial: dict[str, Any] = {
                "schema_version": "h11_conservative_target_failure_v1",
                "status": "failed_continuation",
                "mesh_level": contract.automatic_mesh_level,
                "error": str(exc),
                "partial_model": str(partial_model.resolve()),
                "partial_model_sha256": _sha256(partial_model),
            }
            datasets = list(model / "datasets")
            if datasets:
                try:
                    indices, values = model.inner(datasets[0])
                    partial["stored_parameter_values"] = [
                        float(value) for value in values
                    ]
                    partial["n_stored_solutions"] = int(len(indices))
                    partial["last_stored_load"] = (
                        float(values[-1]) if len(values) else None
                    )
                    if len(indices):
                        fields = model.evaluate(
                            ["T", "hmnf.U", "hmnf.pA"],
                            dataset=datasets[0],
                            inner="last",
                        )
                        partial["last_stored_field_ranges"] = {
                            name: _range(value)
                            for name, value in zip(
                                ("temperature_k", "speed_m_s", "absolute_pressure_pa"),
                                fields,
                            )
                        }
                        ranges = partial["last_stored_field_ranges"]
                        partial["last_stored_solution_bounded"] = bool(
                            ranges["temperature_k"]["minimum"]
                            > contract.property_temperature_floor_k
                            and ranges["temperature_k"]["maximum"] <= 10_000.01
                            and ranges["speed_m_s"]["minimum"] >= -1e-9
                            and ranges["speed_m_s"]["maximum"] <= 600.01
                            and ranges["absolute_pressure_pa"]["minimum"]
                            > contract.pressure_floor_pa
                        )
                except Exception as audit_exc:
                    partial["partial_solution_audit_error"] = str(audit_exc)
            failure_audit = args.audit.with_name(
                f"{args.audit.stem}_mesh_{contract.automatic_mesh_level}_failure.json"
            )
            with failure_audit.open("w", encoding="utf-8") as handle:
                json.dump(partial, handle, indent=2, ensure_ascii=False)
            print(f"Wrote failure audit: {failure_audit}", flush=True)
            raise
        refinement_study, refinement_solution = configure_refinement(jm, contract)
        jm.study(refinement_study).run()
        model.save(str(args.model_out))
        metrics = evaluate_solution(model)
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_statistics = {
            "automatic_level": contract.automatic_mesh_level,
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
    finally:
        client.clear()

    bounded = (
        metrics["temperature_k"]["minimum"]
        > contract.property_temperature_floor_k
        and metrics["absolute_pressure_pa"]["minimum"] > contract.pressure_floor_pa
    )
    mass_pass = metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    energy_pass = (
        metrics["energy_balance_w"]["imbalance_fraction_of_inlet"] < 0.02
    )
    audit = {
        "schema_version": "h11_conservative_target_nominal_v1",
        "status": (
            "pass_conservative_numerical_gates_not_calibrated"
            if bounded and mass_pass and energy_pass
            else "pass_solve_fail_one_or_more_numerical_gates"
        ),
        "contract": asdict(contract),
        "mesh": mesh_statistics,
        "metrics": metrics,
        "gates": {
            "physically_bounded": bounded,
            "mass_imbalance_below_0_5_percent": mass_pass,
            "energy_imbalance_below_2_percent": energy_pass,
        },
        "refinement_solution": refinement_solution,
        "calibrated": False,
        "paper_prediction_allowed": False,
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
        f"Conservative nominal: {audit['status']}; "
        f"mass={metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}; "
        "energy="
        f"{metrics['energy_balance_w']['imbalance_fraction_of_inlet']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
