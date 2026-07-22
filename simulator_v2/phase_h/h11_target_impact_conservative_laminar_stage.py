"""H11 layer 11a: compute a conservative laminar initialization branch.

This stage changes only the turbulence closure used for numerical
initialization.  Geometry, thermodynamic model, inlet profiles, ambient
pressure-temperature opening, and target temperature are identical to the
final all-Mach RANS branch.  Its solution is not a paper result; it exists only
to provide bounded ``u``, ``p``, and ``T`` fields for a subsequent k-omega
restart.
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
    _range,
    _scalar,
)


HERE = Path(__file__).resolve().parent
SKELETON_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_skeleton"
    / "h11_target_impact_conservative_skeleton_latest.mph"
)
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_conservative_laminar"
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_laminar"
MODEL_PATH = MODEL_DIR / "h11_target_impact_conservative_laminar_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_target_impact_conservative_laminar_audit.json"


@dataclass(frozen=True)
class LaminarStageContract:
    """Numerical initialization contract; it cannot authorize prediction."""

    load_fractions: tuple[float, ...] = (
        0.0,
        0.00001,
        0.00002,
        0.00005,
        0.0001,
        0.0002,
        0.00025,
        0.0003,
        0.00035,
        0.0004,
        0.00045,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.001,
        0.0012,
        0.0014,
        0.0016,
        0.0018,
        0.002,
        0.0025,
        0.003,
        0.0035,
        0.004,
        0.0045,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
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
    seed_velocity_m_s: float = 0.0
    automatic_mesh_level: int = 5
    relative_tolerance: float = 5e-4
    maximum_segregated_iterations: int = 2000
    pressure_scale_pa: float = 1e5
    temperature_scale_k: float = 1e4
    velocity_scale_m_s: float = 600.0
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        values = self.load_fractions
        if len(values) < 3 or values[0] != 0 or values[-1] != 1:
            raise ValueError("Laminar continuation must span exactly 0 to 1")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Load fractions must be finite in [0, 1]")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Load fractions must be strictly increasing")
        if self.automatic_mesh_level not in {2, 3, 4, 5}:
            raise ValueError("Unsupported COMSOL automatic mesh level")
        if not math.isfinite(self.seed_velocity_m_s) or self.seed_velocity_m_s < 0:
            raise ValueError("Seed velocity must be finite and nonnegative")
        if not 0 < self.relative_tolerance <= 1e-3:
            raise ValueError("Relative tolerance must lie in (0, 1e-3]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Iteration limit is too small")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.pressure_scale_pa,
                self.temperature_scale_k,
                self.velocity_scale_m_s,
                self.pressure_floor_pa,
            )
        ):
            raise ValueError("Variable scales and pressure floor must be positive")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Initialization stage cannot claim prediction")

    def continuation_list(self) -> str:
        return " ".join(f"{value:.12g}" for value in self.load_fractions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _entities(component: Any, selection_tag: str) -> list[int]:
    return [int(value) for value in component.selection(selection_tag).entities()]


def replace_with_laminar_physics(
    jm: Any,
    contract: LaminarStageContract,
) -> None:
    """Replace k-omega with the same conservative laminar all-Mach interface."""

    comp = jm.component("comp1")
    physics = comp.physics()
    if "hmnf" in {str(value) for value in physics.tags()}:
        physics.remove("hmnf")
    hmnf = physics.create("hmnf", "HighMachNumberFlow", "geom1")
    hmnf.label("Conservative compressible target plume, laminar initializer")

    physical = hmnf.prop("PhysicalModelProperty")
    physical.set("Compressibility", "CompressibleMALT03")
    physical.set("Tref", "T_amb")
    physical.set("includeKineticEnergy", "1")
    hmnf.prop("AdvancedSettingProperty").set("UsePseudoTime", "1")
    hmnf.prop("AdvancedSettingProperty").set("PseudoTimeSetting", "Automatic")

    fluid = hmnf.feature("fluid1")
    fluid.set("fluidType", "idealGas")
    fluid.set("gasConstantType", "specificGC")
    fluid.set("Rs_mat", "userdef")
    fluid.set("Rs", "R_Ar")
    fluid.set("CpOrGammaOption", "Cp")
    fluid.set("Cp_mat", "from_mat")
    fluid.set("k_mat", "from_mat")
    fluid.set("mu_mat", "from_mat")
    fluid.set("PressureWorkFormulationType", "FullFormulation")

    initial = hmnf.feature("init1")
    initial.set("u_init", ["0", "0", "0"])
    initial.set("p_init", "p_amb")
    initial.set("Tinit", "T_amb")

    nozzle = hmnf.create("nozzle_in", "HighMachNumberFlowInlet", 1)
    nozzle.label("Uncalibrated subsonic effective exit, laminar ramp")
    nozzle.selection().named("geom1_sel_nozzle_in")
    nozzle.set("FlowCondition", "Subsonic")
    nozzle.set("BoundaryCondition", "Velocity")
    nozzle.set(
        "U0in",
        f"({contract.seed_velocity_m_s:.12g}[m/s]"
        f"+load_s*(u_exit_eff-{contract.seed_velocity_m_s:.12g}[m/s]))"
        "*nozzle_shape",
    )
    nozzle.set("TemperatureHeatflux", "Temperature")
    nozzle.set("T0", "T_amb+load_s*(T_exit_eff-T_amb)*nozzle_shape")
    nozzle.set("SuppressBackflow", "1")

    ambient_entities = sorted(
        {
            *_entities(comp, "geom1_sel_ambient_in"),
            *_entities(comp, "geom1_sel_far_r"),
        }
    )
    ambient = hmnf.create("ambient_open", "HighMachNumberFlowInlet", 1)
    ambient.label("Ambient pressure-temperature opening")
    ambient.selection().set(ambient_entities)
    ambient.set("FlowCondition", "Subsonic")
    ambient.set("BoundaryCondition", "Pressure")
    ambient.set("p0", "p_amb")
    ambient.set("TemperatureHeatflux", "Temperature")
    ambient.set("T0", "T_amb")
    ambient.set("SuppressBackflow", "0")

    target = hmnf.create("target_temperature", "TemperatureBoundary", 1)
    target.label("Measured-range isothermal workpiece")
    target.selection().named("geom1_sel_target")
    target.set("T0", "T_amb+load_s*(T_target-T_amb)")


def _set_manual_scales(jm: Any, solution_tag: str, contract: LaminarStageContract) -> dict[str, float]:
    variables = jm.sol(solution_tag).feature("v1")
    mapping = {
        "comp1_p": contract.pressure_scale_pa,
        "comp1_T": contract.temperature_scale_k,
        "comp1_u": contract.velocity_scale_m_s,
    }
    present = {str(value) for value in variables.feature().tags()}
    applied: dict[str, float] = {}
    for tag, value in mapping.items():
        if tag not in present:
            continue
        field = variables.feature(tag)
        field.set("scalemethod", "manual")
        field.set("scaleval", f"{value:.12g}")
        applied[tag] = value
    return applied


def configure_laminar_continuation(
    jm: Any,
    contract: LaminarStageContract,
) -> dict[str, float]:
    contract.validate()
    jm.param().set("load_s", "0", "Numerical continuation fraction")
    replace_with_laminar_physics(jm, contract)

    comp = jm.component("comp1")
    mesh = comp.mesh("mesh1")
    mesh.autoMeshSize(contract.automatic_mesh_level)
    mesh.run()

    study = jm.study("std1")
    features = study.feature()
    if "param" in {str(value) for value in features.tags()}:
        features.remove("param")
    parametric = features.create("param", "Parametric")
    parametric.label("Laminar conservative effective-exit continuation")
    parametric.set("pname", ["load_s"])
    parametric.set("plistarr", [contract.continuation_list()])
    parametric.set("punit", [""])
    parametric.set("sweeptype", "filled")
    parametric.set("reusesol", "on")
    parametric.set("keepsol", "all")
    study.createAutoSequences("all")

    applied = _set_manual_scales(jm, "sol1", contract)
    stationary = jm.sol("sol1").feature("s1")
    stationary.set("stol", f"{contract.relative_tolerance:.12g}")
    if "fc1" in {str(value) for value in stationary.feature().tags()}:
        fully_coupled = stationary.feature("fc1")
        fully_coupled.set("dtech", "hnlin")
        fully_coupled.set("initsteph", "1e-4")
        fully_coupled.set("minsteph", "1e-10")
        fully_coupled.set("useminsteprecovery", "on")
        fully_coupled.set("minsteprecovery", "0.1")
        fully_coupled.set("maxiter", "300")
    if "se1" in {str(value) for value in stationary.feature().tags()}:
        segregated = stationary.feature("se1")
        segregated.set("maxsegiter", str(contract.maximum_segregated_iterations))
        segregated.feature("ll1").set(
            "lowerlimit",
            f"comp1.T {contract.property_temperature_floor_k:.12g}[K] "
            f"comp1.p {contract.pressure_floor_pa:.12g}[Pa] ",
        )
    return applied


def evaluate_laminar_solution(model: Any, contract: LaminarStageContract) -> dict[str, Any]:
    datasets = list(model / "datasets")
    if not datasets:
        raise RuntimeError("Laminar continuation dataset is missing")
    dataset = datasets[0]
    indices, values = model.inner(dataset)
    if len(indices) == 0 or not math.isclose(float(values[-1]), 1.0):
        raise RuntimeError("Laminar continuation did not reach full load")
    selector = {"inner": "last"}
    temperature, speed, pressure = model.evaluate(
        ["T", "hmnf.U", "hmnf.pA"],
        unit=["K", "m/s", "Pa"],
        dataset=dataset,
        **selector,
    )
    result = {
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
    }
    result["bounded"] = bool(
        result["temperature_k"]["minimum"] > contract.property_temperature_floor_k
        and result["temperature_k"]["maximum"] <= 10_000.01
        and result["speed_m_s"]["minimum"] >= -1e-9
        and result["speed_m_s"]["maximum"] <= 600.01
        and result["absolute_pressure_pa"]["minimum"] > contract.pressure_floor_pa
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model-in", type=Path, default=SKELETON_MODEL)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--mesh-level", type=int, default=5)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.model_in.exists():
        raise FileNotFoundError(args.model_in)
    contract = LaminarStageContract(automatic_mesh_level=args.mesh_level)
    contract.validate()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.model_in))
        model.rename("h11_target_impact_conservative_laminar")
        jm = model.java
        applied_scales = configure_laminar_continuation(jm, contract)
        try:
            jm.study("std1").run()
        except Exception as exc:
            partial_model = args.model_out.with_name(
                f"{args.model_out.stem}_mesh_{contract.automatic_mesh_level}_partial.mph"
            )
            model.save(str(partial_model))
            failure = {
                "schema_version": "h11_conservative_laminar_failure_v1",
                "status": "failed_laminar_continuation",
                "contract": asdict(contract),
                "manual_scales": applied_scales,
                "error": str(exc),
                "partial_model": str(partial_model.resolve()),
                "partial_model_sha256": _sha256(partial_model),
            }
            datasets = list(model / "datasets")
            if datasets:
                try:
                    indices, values = model.inner(datasets[0])
                    failure["stored_parameter_values"] = [
                        float(value) for value in values
                    ]
                    failure["n_stored_solutions"] = int(len(indices))
                    if len(indices):
                        fields = model.evaluate(
                            ["T", "hmnf.U", "hmnf.pA"],
                            dataset=datasets[0],
                            inner="last",
                        )
                        failure["last_stored_field_ranges"] = {
                            name: _range(value)
                            for name, value in zip(
                                (
                                    "temperature_k",
                                    "speed_m_s",
                                    "absolute_pressure_pa",
                                ),
                                fields,
                            )
                        }
                except Exception as audit_exc:
                    failure["partial_solution_audit_error"] = str(audit_exc)
            failure_audit = args.audit.with_name(
                f"{args.audit.stem}_mesh_{contract.automatic_mesh_level}_failure.json"
            )
            with failure_audit.open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False)
            print(f"Wrote failure audit: {failure_audit}", flush=True)
            raise

        model.save(str(args.model_out))
        metrics = evaluate_laminar_solution(model, contract)
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

    audit = {
        "schema_version": "h11_conservative_laminar_stage_v1",
        "status": (
            "pass_bounded_initialization_only"
            if metrics["bounded"]
            else "fail_unbounded_initialization"
        ),
        "contract": asdict(contract),
        "manual_scales": applied_scales,
        "mesh": mesh_statistics,
        "metrics": metrics,
        "physics_role": (
            "Numerical initialization only; final turbulence closure, "
            "conservation audit, calibration, and held-out validation remain required."
        ),
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
    print(f"Laminar initializer: {audit['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
