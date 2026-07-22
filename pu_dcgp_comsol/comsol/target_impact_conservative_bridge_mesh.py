
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
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_restart"
    / "h11_target_impact_conservative_restart_mesh_3.mph"
)
MODEL_PATH = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2.mph"
)
AUDIT_PATH = (
    HERE
    / "h11_outputs"
    / "target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2.json"
)


@dataclass(frozen=True)
class BridgeMeshContract:

    source_interpolation_fraction: float = 0.0
    interpolation_fraction: float = 0.5
    source_global_hmax_m: float = 1.12e-3
    target_global_hmax_m: float = 5.2e-4
    source_global_hmin_m: float = 1.6e-5
    target_global_hmin_m: float = 6.0e-6
    source_global_hgrad: float = 1.10
    target_global_hgrad: float = 1.08
    source_global_hcurve: float = 0.25
    target_global_hcurve: float = 0.25
    relative_tolerance: float = 1e-4
    maximum_segregated_iterations: int = 4000
    flow_damping: float = 0.1
    turbulence_damping: float = 0.1
    property_temperature_floor_k: float = 250.0
    pressure_floor_pa: float = 1_000.0
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if not 0 <= self.source_interpolation_fraction < 1:
            raise ValueError("Source bridge fraction must lie in [0,1)")
        if not 0 < self.interpolation_fraction <= 1:
            raise ValueError("Target bridge fraction must lie in (0,1]")
        if self.source_interpolation_fraction >= self.interpolation_fraction:
            raise ValueError("Target bridge fraction must exceed the source")
        positive = (
            self.source_global_hmax_m,
            self.target_global_hmax_m,
            self.source_global_hmin_m,
            self.target_global_hmin_m,
            self.source_global_hgrad,
            self.target_global_hgrad,
            self.source_global_hcurve,
            self.target_global_hcurve,
            self.pressure_floor_pa,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Mesh controls and pressure floor must be positive")
        if self.target_global_hmax_m >= self.source_global_hmax_m:
            raise ValueError("Target maximum size must refine the source")
        if self.target_global_hmin_m >= self.source_global_hmin_m:
            raise ValueError("Target minimum size must refine the source")
        if not 0 < self.relative_tolerance <= 1e-3:
            raise ValueError("Relative tolerance must lie in (0,1e-3]")
        if self.maximum_segregated_iterations < 1000:
            raise ValueError("Segregated iteration limit is too small")
        if not 0 < self.flow_damping <= 0.5:
            raise ValueError("Flow damping must lie in (0,0.5]")
        if not 0 < self.turbulence_damping <= 0.5:
            raise ValueError("Turbulence damping must lie in (0,0.5]")
        if self.property_temperature_floor_k < 250:
            raise ValueError("Temperature floor is below the Ar table")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Bridge solve cannot claim prediction")

    @staticmethod
    def log_interpolate(left: float, right: float, fraction: float) -> float:
        return math.exp(
            (1.0 - fraction) * math.log(left)
            + fraction * math.log(right)
        )

    @property
    def bridge_hmax_m(self) -> float:
        return self.log_interpolate(
            self.source_global_hmax_m,
            self.target_global_hmax_m,
            self.interpolation_fraction,
        )

    @property
    def bridge_hmin_m(self) -> float:
        return self.log_interpolate(
            self.source_global_hmin_m,
            self.target_global_hmin_m,
            self.interpolation_fraction,
        )

    @property
    def bridge_hgrad(self) -> float:
        return self.log_interpolate(
            self.source_global_hgrad,
            self.target_global_hgrad,
            self.interpolation_fraction,
        )

    @property
    def bridge_hcurve(self) -> float:
        return self.log_interpolate(
            self.source_global_hcurve,
            self.target_global_hcurve,
            self.interpolation_fraction,
        )

    @property
    def expected_source_hmax_m(self) -> float:
        return self.log_interpolate(
            self.source_global_hmax_m,
            self.target_global_hmax_m,
            self.source_interpolation_fraction,
        )

    @property
    def expected_source_hmin_m(self) -> float:
        return self.log_interpolate(
            self.source_global_hmin_m,
            self.target_global_hmin_m,
            self.source_interpolation_fraction,
        )

    @property
    def expected_source_hgrad(self) -> float:
        return self.log_interpolate(
            self.source_global_hgrad,
            self.target_global_hgrad,
            self.source_interpolation_fraction,
        )

    @property
    def expected_source_hcurve(self) -> float:
        return self.log_interpolate(
            self.source_global_hcurve,
            self.target_global_hcurve,
            self.source_interpolation_fraction,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _fraction_token(fraction: float) -> str:
    scaled = round(fraction * 10_000)
    if not math.isclose(
        scaled / 10_000,
        fraction,
        rel_tol=0.0,
        abs_tol=5e-8,
    ):
        raise ValueError("Bridge fraction supports at most four decimals")
    return f"f{scaled:04d}"


def _default_artifact_paths(
    fraction: float,
    source_fraction: float = 0.0,
) -> tuple[Path, Path]:
    token = _fraction_token(fraction)
    source_token = _fraction_token(source_fraction)
    step_token = (
        token
        if math.isclose(source_fraction, 0.0, abs_tol=1e-15)
        else f"{source_token}_to_{token}"
    )
    return (
        MODEL_PATH.with_name(
            "h11_target_impact_conservative_bridge_3_to_2_"
            f"{step_token}.mph"
        ),
        AUDIT_PATH.with_name(
            "h11_target_impact_conservative_bridge_3_to_2_"
            f"{step_token}.json"
        ),
    )


def _last_tag(collection: Any) -> str:
    tags = [str(value) for value in collection.tags()]
    if not tags:
        raise RuntimeError("Expected nonempty COMSOL collection")
    return tags[-1]


def _bounded(metrics: dict[str, Any], contract: BridgeMeshContract) -> bool:
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


def configure_bridge(
    jm: Any,
    *,
    source_study_tag: str,
    source_solution_tag: str,
    contract: BridgeMeshContract,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    contract.validate()
    comp = jm.component("comp1")
    hmnf = comp.physics("hmnf")
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

    initial = hmnf.feature("init1")
    initial.set("u_init", ["0", "0", "0"])
    initial.set("p_init", "p_amb")
    initial.set("Tinit", "T_amb")
    initial.set("k_init", "hmnf.kinit")
    initial.set("om_init", "hmnf.omInit")

    mesh = comp.mesh("mesh1")
    size = mesh.feature("size")
    observed_source = {
        "hmax_m": float(str(size.getString("hmax"))),
        "hmin_m": float(str(size.getString("hmin"))),
        "hgrad": float(str(size.getString("hgrad"))),
        "hcurve": float(str(size.getString("hcurve"))),
        "hauto": int(str(size.getString("hauto"))),
        "custom": str(size.getString("custom")),
    }
    expected = {
        "hmax_m": contract.expected_source_hmax_m,
        "hmin_m": contract.expected_source_hmin_m,
        "hgrad": contract.expected_source_hgrad,
        "hcurve": contract.expected_source_hcurve,
    }
    for key, value in expected.items():
        if not math.isclose(
            observed_source[key],
            value,
            rel_tol=1e-9,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"Source global mesh control changed: {key}="
                f"{observed_source[key]} expected {value}"
            )
    expected_custom = (
        "off"
        if math.isclose(
            contract.source_interpolation_fraction,
            0.0,
            abs_tol=1e-15,
        )
        else "on"
    )
    if (
        observed_source["hauto"] != 3
        or observed_source["custom"] != expected_custom
    ):
        raise RuntimeError(
            "Source is not the audited mesh fraction "
            f"{contract.source_interpolation_fraction:.4f}"
        )

    size.set("custom", "on")
    size.set("hmax", f"{contract.bridge_hmax_m:.16g}")
    size.set("hmin", f"{contract.bridge_hmin_m:.16g}")
    size.set("hgrad", f"{contract.bridge_hgrad:.16g}")
    size.set("hcurve", f"{contract.bridge_hcurve:.16g}")
    size.set("hnarrow", "1")
    mesh.run()
    bridge_controls = {
        "method": "logarithmic_fraction_of_global_size_controls",
        "source_interpolation_fraction": (
            contract.source_interpolation_fraction
        ),
        "interpolation_fraction": contract.interpolation_fraction,
        "hmax_m": contract.bridge_hmax_m,
        "hmin_m": contract.bridge_hmin_m,
        "hgrad": contract.bridge_hgrad,
        "hcurve": contract.bridge_hcurve,
        "retained_local_size_feature": "size1",
        "retained_corner_refinement": "cr1",
        "retained_boundary_layer": "bl1",
    }

    fraction_token = _fraction_token(contract.interpolation_fraction)
    study_tag = f"std_hmnf_bridge_{fraction_token}"
    studies = jm.study()
    if study_tag in {str(value) for value in studies.tags()}:
        studies.remove(study_tag)
    study = studies.create(study_tag)
    study.label(
        "Conservative all-Mach log-spaced bridge mesh "
        f"({contract.interpolation_fraction:.4f})"
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
                "segregated_pseudotime_with_fixed_under_relaxation"
            ),
        },
        {
            "source": observed_source,
            "bridge": bridge_controls,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--source-fraction", type=float, default=0.0)
    parser.add_argument("--fraction", type=float, default=0.5)
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
    contract = BridgeMeshContract(
        source_interpolation_fraction=args.source_fraction,
        interpolation_fraction=args.fraction,
    )
    contract.validate()
    default_model_out, default_audit = _default_artifact_paths(
        contract.interpolation_fraction,
        contract.source_interpolation_fraction,
    )
    if args.model_out is None:
        args.model_out = default_model_out
    if args.audit is None:
        args.audit = default_audit
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
        if not _bounded(source_metrics, contract):
            raise RuntimeError("Source level-3 solution is not bounded")
        (
            study_tag,
            solution_tag,
            solver_strategy,
            mesh_controls,
        ) = configure_bridge(
            jm,
            source_study_tag=source_study_tag,
            source_solution_tag=source_solution_tag,
            contract=contract,
        )
        mesh = jm.component("comp1").mesh("mesh1")
        mesh_audit = {
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        try:
            jm.study(study_tag).run()
        except Exception as exc:
            partial = args.model_out.with_name(
                f"{args.model_out.stem}_partial.mph"
            )
            model.save(str(partial))
            failure = {
                "schema_version": "h11_conservative_bridge_failure_v1",
                "status": "failed_bridge_mesh_equation_solve",
                "contract": asdict(contract),
                "source_study": source_study_tag,
                "source_solution": source_solution_tag,
                "target_study": study_tag,
                "target_solution": solution_tag,
                "solver_strategy": solver_strategy,
                "mesh_controls": mesh_controls,
                "mesh": mesh_audit,
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
        model.rename(
            "h11_target_impact_conservative_bridge_3_to_2_"
            f"{_fraction_token(contract.interpolation_fraction)}"
        )
        model.save(str(args.model_out))
        metrics = evaluate_solution(model)
    finally:
        client.clear()

    bounded_pass = _bounded(metrics, contract)
    mass_pass = metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    energy_pass = (
        metrics["energy_balance_w"]["imbalance_fraction_of_inlet"] < 0.02
    )
    audit = {
        "schema_version": "h11_conservative_bridge_mesh_v1",
        "status": (
            "pass_bridge_mesh_numerical_gates_not_calibrated"
            if bounded_pass and mass_pass and energy_pass
            else "pass_solve_fail_one_or_more_numerical_gates"
        ),
        "contract": asdict(contract),
        "strategy": {
            "source_role": "initial_iterate_only",
            "target_equations_resolved": True,
            "source_study": source_study_tag,
            "source_solution": source_solution_tag,
            "target_study": study_tag,
            "target_solution": solution_tag,
        },
        "solver_strategy": solver_strategy,
        "mesh_controls": mesh_controls,
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
        "model_path": str(args.model_out.resolve()),
        "model_sha256": _sha256(args.model_out),
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {args.model_out}")
    print(f"Wrote audit: {args.audit}")
    print(
        f"Conservative bridge: {audit['status']}; "
        f"elements={mesh_audit['elements']}; "
        f"mass={metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}; "
        "energy="
        f"{metrics['energy_balance_w']['imbalance_fraction_of_inlet']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
