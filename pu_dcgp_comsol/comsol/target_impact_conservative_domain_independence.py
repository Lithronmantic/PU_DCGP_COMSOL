
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pu_dcgp_comsol.comsol.target_impact_conservative_restart import (
    evaluate_solution,
)
from pu_dcgp_comsol.comsol.target_impact_conservative_same_mesh_refinement import (
    _sha256,
)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_domain"
SKELETON_AUDIT = (
    OUT_DIR / "h11_target_impact_conservative_domain_independence_skeleton.json"
)


@dataclass(frozen=True)
class DomainIndependenceContract:

    radii_mm: tuple[float, float, float] = (40.0, 60.0, 80.0)
    common_radial_points_mm: tuple[float, ...] = (
        0.001,
        2.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
    )
    common_axial_points_mm: tuple[float, ...] = (
        10.0,
        30.0,
        50.0,
        70.0,
        90.0,
        99.0,
    )
    near_target_temperature_change_limit_fraction: float = 0.002
    near_target_speed_change_limit_fraction: float = 0.01
    target_heat_roi_radius_mm: float = 30.0
    target_heat_change_limit_fraction: float = 0.01
    common_temperature_anomaly_l2_limit_fraction: float = 0.01
    common_speed_l2_limit_fraction: float = 0.01
    common_pressure_l2_of_ambient_limit_fraction: float = 1e-4
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02
    require_outer_pair_contraction: bool = True
    require_radial_opening_activity_decrease: bool = True
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if self.radii_mm != (40.0, 60.0, 80.0):
            raise ValueError("Domain audit radii must remain 40/60/80 mm")
        if sorted(self.radii_mm) != list(self.radii_mm):
            raise ValueError("Domain radii must be strictly ordered")
        if (
            not self.common_radial_points_mm
            or not self.common_axial_points_mm
        ):
            raise ValueError("Common field grid cannot be empty")
        if min(self.common_radial_points_mm) <= 0:
            raise ValueError("Axis samples must avoid r=0 singular evaluation")
        if max(self.common_radial_points_mm) >= self.radii_mm[0]:
            raise ValueError("Common field grid must lie inside every domain")
        if self.target_heat_roi_radius_mm != 30.0:
            raise ValueError("Fixed target heat ROI radius must remain 30 mm")
        if self.target_heat_roi_radius_mm != max(
            self.common_radial_points_mm
        ):
            raise ValueError(
                "Target heat ROI must equal the frozen common-grid radius"
            )
        if min(self.common_axial_points_mm) <= 0:
            raise ValueError("Common axial grid must be downstream of the exit")
        if max(self.common_axial_points_mm) >= 100:
            raise ValueError("Common axial grid must remain upstream of target")
        frozen_limits = {
            "near_target_temperature": (
                self.near_target_temperature_change_limit_fraction,
                0.002,
            ),
            "near_target_speed": (
                self.near_target_speed_change_limit_fraction,
                0.01,
            ),
            "target_heat": (
                self.target_heat_change_limit_fraction,
                0.01,
            ),
            "common_temperature": (
                self.common_temperature_anomaly_l2_limit_fraction,
                0.01,
            ),
            "common_speed": (
                self.common_speed_l2_limit_fraction,
                0.01,
            ),
            "common_pressure": (
                self.common_pressure_l2_of_ambient_limit_fraction,
                1e-4,
            ),
            "mass": (self.mass_imbalance_limit_fraction, 0.005),
            "energy": (self.energy_imbalance_limit_fraction, 0.02),
        }
        for name, (observed, expected) in frozen_limits.items():
            if observed != expected:
                raise ValueError(
                    f"Frozen {name} domain-independence limit changed"
                )
        if not self.require_outer_pair_contraction:
            raise ValueError("Adjacent domain changes must contract")
        if not self.require_radial_opening_activity_decrease:
            raise ValueError("Opening activity must decrease outward")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Domain audit cannot claim physical calibration")


def common_sample_points(
    contract: DomainIndependenceContract,
) -> list[dict[str, float]]:
    contract.validate()
    return [
        {"r_mm": radial, "z_mm": axial}
        for axial in contract.common_axial_points_mm
        for radial in contract.common_radial_points_mm
    ]


def relative_scalar_change(source: float, target: float) -> float:
    if not math.isfinite(source) or not math.isfinite(target):
        raise ValueError("Domain comparison values must be finite")
    return abs(target - source) / max(abs(source), abs(target), 1e-30)


def normalized_l2_change(
    source: Iterable[float],
    target: Iterable[float],
    *,
    offset: float = 0.0,
    absolute_scale: float | None = None,
) -> float:
    source_values = [float(value) for value in source]
    target_values = [float(value) for value in target]
    if not source_values or len(source_values) != len(target_values):
        raise ValueError("Common field arrays must have equal nonzero length")
    if not all(
        math.isfinite(value)
        for value in (*source_values, *target_values)
    ):
        raise ValueError("Common field arrays must be finite")
    numerator = math.sqrt(
        sum(
            (target_value - source_value) ** 2
            for source_value, target_value in zip(
                source_values,
                target_values,
            )
        )
        / len(source_values)
    )
    if absolute_scale is not None:
        if not math.isfinite(absolute_scale) or absolute_scale <= 0:
            raise ValueError("Absolute field scale must be finite and positive")
        denominator = absolute_scale
    else:
        source_rms = math.sqrt(
            sum((value - offset) ** 2 for value in source_values)
            / len(source_values)
        )
        target_rms = math.sqrt(
            sum((value - offset) ** 2 for value in target_values)
            / len(target_values)
        )
        denominator = max(source_rms, target_rms, 1e-30)
    return numerator / denominator


def skeleton_payload(
    contract: DomainIndependenceContract,
) -> dict[str, object]:
    points = common_sample_points(contract)
    return {
        "schema_version": "h11_domain_independence_skeleton_v2",
        "status": (
            "pass_comparable_observable_domain_contract_data_pending"
        ),
        "protocol_revision": {
            "supersedes": "h11_domain_independence_skeleton_v1",
            "reason": (
                "The full target boundary grows with radial domain radius, "
                "so full-boundary heat is not the same estimand."
            ),
            "threshold_changed": False,
            "fixed_target_roi_radius_mm": (
                contract.target_heat_roi_radius_mm
            ),
            "roi_selected_before_roi_result_evaluation": True,
        },
        "contract": asdict(contract),
        "common_field_grid": {
            "point_count": len(points),
            "points": points,
            "fields": ["T", "hmnf.U", "p"],
            "normalization": {
                "temperature": "RMS difference / max RMS(T-300 K)",
                "speed": "RMS difference / max RMS(speed)",
                "pressure": "RMS difference / 101325 Pa",
            },
        },
        "required_comparisons": {
            "adjacent_pairs": ["40_to_60_mm", "60_to_80_mm"],
            "outer_pair_controls_acceptance": True,
            "outer_pair_change_must_contract": True,
            "scalars": [
                "one_mm_upstream_temperature",
                "one_mm_upstream_speed",
                "fixed_0_to_30_mm_target_integrated_heat",
                "full_target_integrated_heat_non_gating_diagnostic",
                "mass_imbalance",
                "energy_imbalance",
                "radial_opening_two_way_mass_activity",
            ],
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def _flat_scalar(value: object) -> float:
    return float(value.reshape(-1)[0])


def common_field_expressions(
    contract: DomainIndependenceContract,
    field: str,
) -> list[str]:
    return [
        f"at2({point['r_mm']:.12g}[mm],{point['z_mm']:.12g}[mm],{field})"
        for point in common_sample_points(contract)
    ]


def radial_opening_activity_expressions() -> dict[str, str]:
    normal_mass_flux = "hmnf.rho*(u*nr+w*nz)"
    zero_flux = "0[kg/(m^2*s)]"
    return {
        "outflow": (
            "int_ambient_hmnf(2*pi*r*"
            f"max({normal_mass_flux},{zero_flux}))"
        ),
        "inflow_magnitude": (
            "-int_ambient_hmnf(2*pi*r*"
            f"min({normal_mass_flux},{zero_flux}))"
        ),
        "axisymmetric_area": "int_ambient_hmnf(2*pi*r*1)",
    }


def target_heat_expressions(
    contract: DomainIndependenceContract,
) -> dict[str, str]:

    contract.validate()
    roi_m = contract.target_heat_roi_radius_mm / 1000.0
    return {
        "fixed_roi": (
            "int_target_hmnf(2*pi*r*"
            f"if(r<={roi_m:.12g}[m],hmnf.nteflux,0[W/m^2]))"
        ),
        "full_boundary_diagnostic": (
            "int_target_hmnf(2*pi*r*hmnf.nteflux)"
        ),
    }


def evaluate_domain_model(
    model: object,
    *,
    radius_mm: float,
    contract: DomainIndependenceContract,
) -> dict[str, object]:

    contract.validate()
    if radius_mm not in contract.radii_mm:
        raise ValueError("Model radius is outside the frozen domain audit")
    dataset = list(model / "datasets")[-1]
    selector: dict[str, object] = {}
    inner_indices, _ = model.inner(dataset)
    if len(inner_indices):
        selector["inner"] = "last"

    fields: dict[str, list[float]] = {}
    for name, expression, unit in (
        ("temperature_k", "T", "K"),
        ("speed_m_s", "hmnf.U", "m/s"),
        ("absolute_pressure_pa", "p", "Pa"),
    ):
        expressions = common_field_expressions(contract, expression)
        values = model.evaluate(
            expressions,
            unit=[unit] * len(expressions),
            dataset=dataset,
            **selector,
        )
        fields[name] = [_flat_scalar(value) for value in values]

    activity_expressions = radial_opening_activity_expressions()
    outflow, inflow, opening_area = model.evaluate(
        [
            activity_expressions["outflow"],
            activity_expressions["inflow_magnitude"],
            activity_expressions["axisymmetric_area"],
        ],
        unit=["kg/s", "kg/s", "m^2"],
        dataset=dataset,
        **selector,
    )
    outflow_value = _flat_scalar(outflow)
    inflow_value = _flat_scalar(inflow)
    opening_area_value = _flat_scalar(opening_area)
    if (
        outflow_value < 0
        or inflow_value < 0
        or opening_area_value <= 0
    ):
        raise RuntimeError(
            "Radial opening activity and area must be positive"
        )

    heat_expressions = target_heat_expressions(contract)
    fixed_roi_heat, full_boundary_heat = model.evaluate(
        [
            heat_expressions["fixed_roi"],
            heat_expressions["full_boundary_diagnostic"],
        ],
        unit=["W", "W"],
        dataset=dataset,
        **selector,
    )
    metrics = evaluate_solution(model)
    nozzle_mass = abs(metrics["mass_flux_kg_s"]["nozzle_outward"])
    return {
        "radius_mm": radius_mm,
        "common_fields": fields,
        "one_mm_upstream_of_target": metrics[
            "one_mm_upstream_of_target"
        ],
        "target_heat_w": {
            "fixed_roi_radius_mm": contract.target_heat_roi_radius_mm,
            "fixed_roi": _flat_scalar(fixed_roi_heat),
            "full_boundary_non_gating_diagnostic": _flat_scalar(
                full_boundary_heat
            ),
        },
        "mass_imbalance_fraction": metrics["mass_flux_kg_s"][
            "imbalance_fraction"
        ],
        "energy_imbalance_fraction": metrics["energy_balance_w"][
            "imbalance_fraction_of_inlet"
        ],
        "radial_opening_mass_activity_kg_s": {
            "outflow": outflow_value,
            "inflow_magnitude": inflow_value,
            "two_way_total": outflow_value + inflow_value,
            "net_outflow": outflow_value - inflow_value,
            "two_way_total_over_nozzle": (
                (outflow_value + inflow_value) / nozzle_mass
            ),
            "axisymmetric_area_m2": opening_area_value,
            "area_mean_two_way_flux_density_kg_m2_s": (
                (outflow_value + inflow_value) / opening_area_value
            ),
        },
    }


def adjacent_pair_comparison(
    source: dict[str, object],
    target: dict[str, object],
) -> dict[str, object]:
    source_near = source["one_mm_upstream_of_target"]
    target_near = target["one_mm_upstream_of_target"]
    source_fields = source["common_fields"]
    target_fields = target["common_fields"]
    source_activity = source["radial_opening_mass_activity_kg_s"]
    target_activity = target["radial_opening_mass_activity_kg_s"]
    source_heat = source["target_heat_w"]
    target_heat = target["target_heat_w"]
    return {
        "source_radius_mm": float(source["radius_mm"]),
        "target_radius_mm": float(target["radius_mm"]),
        "near_target_temperature_change_fraction": relative_scalar_change(
            float(source_near["temperature_k"]),
            float(target_near["temperature_k"]),
        ),
        "near_target_speed_change_fraction": relative_scalar_change(
            float(source_near["speed_m_s"]),
            float(target_near["speed_m_s"]),
        ),
        "target_fixed_roi_heat_change_fraction": relative_scalar_change(
            float(source_heat["fixed_roi"]),
            float(target_heat["fixed_roi"]),
        ),
        "target_full_boundary_heat_change_fraction_non_gating": (
            relative_scalar_change(
                float(
                    source_heat["full_boundary_non_gating_diagnostic"]
                ),
                float(
                    target_heat["full_boundary_non_gating_diagnostic"]
                ),
            )
        ),
        "common_field_changes": {
            "temperature_anomaly_normalized_l2": normalized_l2_change(
                source_fields["temperature_k"],
                target_fields["temperature_k"],
                offset=300.0,
            ),
            "speed_normalized_l2": normalized_l2_change(
                source_fields["speed_m_s"],
                target_fields["speed_m_s"],
            ),
            "pressure_l2_over_ambient": normalized_l2_change(
                source_fields["absolute_pressure_pa"],
                target_fields["absolute_pressure_pa"],
                absolute_scale=101_325.0,
            ),
        },
        "radial_opening_two_way_activity_change_fraction": (
            relative_scalar_change(
                float(source_activity["two_way_total"]),
                float(target_activity["two_way_total"]),
            )
        ),
        "radial_opening_activity_decreased": bool(
            float(target_activity["two_way_total"])
            < float(source_activity["two_way_total"])
        ),
    }


def finalize_domain_independence(
    evaluations: list[dict[str, object]],
    contract: DomainIndependenceContract,
) -> dict[str, object]:
    contract.validate()
    observed_radii = tuple(
        float(evaluation["radius_mm"]) for evaluation in evaluations
    )
    if observed_radii != contract.radii_mm:
        raise ValueError("Expected ordered 40/60/80 mm evaluations")
    inner = adjacent_pair_comparison(evaluations[0], evaluations[1])
    outer = adjacent_pair_comparison(evaluations[1], evaluations[2])
    inner_fields = inner["common_field_changes"]
    outer_fields = outer["common_field_changes"]

    outer_limits = {
        "near_target_temperature": (
            outer["near_target_temperature_change_fraction"],
            contract.near_target_temperature_change_limit_fraction,
        ),
        "near_target_speed": (
            outer["near_target_speed_change_fraction"],
            contract.near_target_speed_change_limit_fraction,
        ),
        "target_fixed_roi_heat": (
            outer["target_fixed_roi_heat_change_fraction"],
            contract.target_heat_change_limit_fraction,
        ),
        "common_temperature": (
            outer_fields["temperature_anomaly_normalized_l2"],
            contract.common_temperature_anomaly_l2_limit_fraction,
        ),
        "common_speed": (
            outer_fields["speed_normalized_l2"],
            contract.common_speed_l2_limit_fraction,
        ),
        "common_pressure": (
            outer_fields["pressure_l2_over_ambient"],
            contract.common_pressure_l2_of_ambient_limit_fraction,
        ),
    }
    outer_limit_gates = {
        name: float(value) <= limit
        for name, (value, limit) in outer_limits.items()
    }
    contraction_pairs = {
        "near_target_temperature": (
            inner["near_target_temperature_change_fraction"],
            outer["near_target_temperature_change_fraction"],
        ),
        "near_target_speed": (
            inner["near_target_speed_change_fraction"],
            outer["near_target_speed_change_fraction"],
        ),
        "target_fixed_roi_heat": (
            inner["target_fixed_roi_heat_change_fraction"],
            outer["target_fixed_roi_heat_change_fraction"],
        ),
        "common_temperature": (
            inner_fields["temperature_anomaly_normalized_l2"],
            outer_fields["temperature_anomaly_normalized_l2"],
        ),
        "common_speed": (
            inner_fields["speed_normalized_l2"],
            outer_fields["speed_normalized_l2"],
        ),
        "common_pressure": (
            inner_fields["pressure_l2_over_ambient"],
            outer_fields["pressure_l2_over_ambient"],
        ),
    }
    contraction_gates = {
        name: float(outer_value) <= float(inner_value)
        for name, (inner_value, outer_value) in contraction_pairs.items()
    }
    conservation_gates = {
        f"r{int(evaluation['radius_mm']):03d}": bool(
            float(evaluation["mass_imbalance_fraction"])
            < contract.mass_imbalance_limit_fraction
            and float(evaluation["energy_imbalance_fraction"])
            < contract.energy_imbalance_limit_fraction
        )
        for evaluation in evaluations
    }
    opening_gate = bool(
        inner["radial_opening_activity_decreased"]
        and outer["radial_opening_activity_decreased"]
    )
    gates = {
        "outer_pair_within_preregistered_limits": all(
            outer_limit_gates.values()
        ),
        "outer_pair_changes_contract": all(contraction_gates.values()),
        "all_domains_conservative": all(conservation_gates.values()),
        "radial_opening_activity_decreases_outward": opening_gate,
    }
    return {
        "schema_version": "h11_domain_independence_v2",
        "status": (
            "pass_radial_domain_independence_numerical_gates"
            if all(gates.values())
            else "fail_one_or_more_radial_domain_independence_gates"
        ),
        "protocol_revision": {
            "supersedes": "h11_domain_independence_v1",
            "reason": (
                "Full target heat used a radius-dependent boundary area; "
                "the unchanged 1 percent limit now controls fixed-ROI heat."
            ),
            "threshold_changed": False,
            "fixed_target_roi_radius_mm": (
                contract.target_heat_roi_radius_mm
            ),
            "full_boundary_heat_is_gating": False,
        },
        "contract": asdict(contract),
        "evaluations": evaluations,
        "adjacent_comparisons": {
            "40_to_60_mm": inner,
            "60_to_80_mm": outer,
        },
        "outer_limit_gates": outer_limit_gates,
        "contraction_gates": contraction_gates,
        "conservation_gates": conservation_gates,
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=SKELETON_AUDIT)
    parser.add_argument("--model-40", type=Path)
    parser.add_argument("--model-60", type=Path)
    parser.add_argument("--model-80", type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()
    contract = DomainIndependenceContract()
    models = (args.model_40, args.model_60, args.model_80)
    if any(model is not None for model in models) and not all(
        model is not None for model in models
    ):
        raise ValueError("Provide all three domain models or none")
    if all(model is not None for model in models):
        for path in models:
            if not path.exists():
                raise FileNotFoundError(path)
        import mph

        client = mph.start(cores=args.cores, version=args.version)
        evaluations = []
        try:
            for radius_mm, path in zip(contract.radii_mm, models):
                model = client.load(str(path))
                evaluations.append(
                    {
                        **evaluate_domain_model(
                            model,
                            radius_mm=radius_mm,
                            contract=contract,
                        ),
                        "model_path": str(path.resolve()),
                        "model_sha256": _sha256(path),
                    }
                )
                client.remove(model)
        finally:
            client.clear()
        payload = finalize_domain_independence(evaluations, contract)
    else:
        payload = skeleton_payload(contract)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote domain-independence skeleton: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
