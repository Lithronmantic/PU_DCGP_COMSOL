"""H11 layer 11l: prospective 60/80/100 mm radial-domain continuation.

The preregistered 40/60/80 mm v2 audit is retained as a failure.  Its outer
pair met every absolute field, fixed-ROI heat, and conservation limit, but
some changes did not contract and the total two-way mass activity at the
radial opening increased with the growing cylindrical boundary area.

This continuation is frozen before evaluating a 100 mm solution.  It keeps
the v2 physical-output limits and contraction requirement, while replacing
the dimensionally inappropriate total-opening-activity gate with the
area-mean two-way mass-flux density.  No v2 result or threshold is changed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from simulator_v2.phase_h.h11_target_impact_conservative_domain_independence import (
    adjacent_pair_comparison,
    common_sample_points,
    evaluate_domain_model,
)
from simulator_v2.phase_h.h11_target_impact_conservative_same_mesh_refinement import (
    _sha256,
)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_domain"
SKELETON_AUDIT = (
    OUT_DIR
    / "h11_target_impact_conservative_domain_continuation_skeleton_v1.json"
)


@dataclass(frozen=True)
class DomainContinuationContract:
    """Frozen continuation after the explicitly retained v2 failure."""

    radii_mm: tuple[float, float, float] = (60.0, 80.0, 100.0)
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
    require_area_mean_opening_activity_decrease: bool = True
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if self.radii_mm != (60.0, 80.0, 100.0):
            raise ValueError("Continuation radii must remain 60/80/100 mm")
        if max(self.common_radial_points_mm) != 30.0:
            raise ValueError("Common field and target ROI radius must be 30 mm")
        if self.target_heat_roi_radius_mm != 30.0:
            raise ValueError("Fixed target heat ROI radius must remain 30 mm")
        if min(self.common_radial_points_mm) <= 0:
            raise ValueError("Common points must avoid the axis singularity")
        if max(self.common_axial_points_mm) >= 100.0:
            raise ValueError("Common points must remain upstream of the target")
        frozen_limits = (
            (self.near_target_temperature_change_limit_fraction, 0.002),
            (self.near_target_speed_change_limit_fraction, 0.01),
            (self.target_heat_change_limit_fraction, 0.01),
            (self.common_temperature_anomaly_l2_limit_fraction, 0.01),
            (self.common_speed_l2_limit_fraction, 0.01),
            (self.common_pressure_l2_of_ambient_limit_fraction, 1e-4),
            (self.mass_imbalance_limit_fraction, 0.005),
            (self.energy_imbalance_limit_fraction, 0.02),
        )
        if any(observed != expected for observed, expected in frozen_limits):
            raise ValueError("A frozen v2 numerical limit changed")
        if not self.require_outer_pair_contraction:
            raise ValueError("80-to-100 mm changes must contract")
        if not self.require_area_mean_opening_activity_decrease:
            raise ValueError("Area-mean opening activity must decrease")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Domain continuation cannot claim calibration")


def skeleton_payload(
    contract: DomainContinuationContract,
) -> dict[str, object]:
    contract.validate()
    return {
        "schema_version": "h11_domain_continuation_skeleton_v1",
        "status": "pass_prospective_continuation_contract_data_pending",
        "provenance": {
            "retained_failed_audit": (
                "h11_target_impact_conservative_"
                "domain_independence_v2.json"
            ),
            "v2_result_or_threshold_changed": False,
            "created_before_100_mm_model_evaluation": True,
            "reason": (
                "Total opening exchange scales with cylindrical boundary "
                "area; the continuation gates the area-mean flux density."
            ),
        },
        "contract": asdict(contract),
        "common_field_grid": {
            "point_count": len(common_sample_points(contract)),
            "fixed_target_roi_radius_mm": (
                contract.target_heat_roi_radius_mm
            ),
        },
        "required_comparisons": {
            "adjacent_pairs": ["60_to_80_mm", "80_to_100_mm"],
            "outer_pair_controls_limits": True,
            "outer_pair_must_contract": True,
            "area_mean_opening_activity_must_decrease": True,
            "total_opening_activity_is_non_gating": True,
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def _opening_density(evaluation: dict[str, object]) -> float:
    activity = evaluation["radial_opening_mass_activity_kg_s"]
    value = float(  # type: ignore[index]
        activity["area_mean_two_way_flux_density_kg_m2_s"]
    )
    if not math.isfinite(value) or value < 0:
        raise ValueError("Opening flux density must be finite and nonnegative")
    return value


def finalize_domain_continuation(
    evaluations: list[dict[str, object]],
    contract: DomainContinuationContract,
) -> dict[str, object]:
    contract.validate()
    observed = tuple(float(item["radius_mm"]) for item in evaluations)
    if observed != contract.radii_mm:
        raise ValueError("Expected ordered 60/80/100 mm evaluations")

    inner = adjacent_pair_comparison(evaluations[0], evaluations[1])
    outer = adjacent_pair_comparison(evaluations[1], evaluations[2])
    inner_fields = inner["common_field_changes"]
    outer_fields = outer["common_field_changes"]
    outer_limits = {
        "near_target_temperature": (
            float(outer["near_target_temperature_change_fraction"]),
            contract.near_target_temperature_change_limit_fraction,
        ),
        "near_target_speed": (
            float(outer["near_target_speed_change_fraction"]),
            contract.near_target_speed_change_limit_fraction,
        ),
        "target_fixed_roi_heat": (
            float(outer["target_fixed_roi_heat_change_fraction"]),
            contract.target_heat_change_limit_fraction,
        ),
        "common_temperature": (
            float(  # type: ignore[index]
                outer_fields["temperature_anomaly_normalized_l2"]
            ),
            contract.common_temperature_anomaly_l2_limit_fraction,
        ),
        "common_speed": (
            float(outer_fields["speed_normalized_l2"]),  # type: ignore[index]
            contract.common_speed_l2_limit_fraction,
        ),
        "common_pressure": (
            float(outer_fields["pressure_l2_over_ambient"]),  # type: ignore[index]
            contract.common_pressure_l2_of_ambient_limit_fraction,
        ),
    }
    outer_limit_gates = {
        name: value <= limit
        for name, (value, limit) in outer_limits.items()
    }
    contraction_pairs = {
        "near_target_temperature": (
            float(inner["near_target_temperature_change_fraction"]),
            float(outer["near_target_temperature_change_fraction"]),
        ),
        "near_target_speed": (
            float(inner["near_target_speed_change_fraction"]),
            float(outer["near_target_speed_change_fraction"]),
        ),
        "target_fixed_roi_heat": (
            float(inner["target_fixed_roi_heat_change_fraction"]),
            float(outer["target_fixed_roi_heat_change_fraction"]),
        ),
        "common_temperature": (
            float(  # type: ignore[index]
                inner_fields["temperature_anomaly_normalized_l2"]
            ),
            float(  # type: ignore[index]
                outer_fields["temperature_anomaly_normalized_l2"]
            ),
        ),
        "common_speed": (
            float(inner_fields["speed_normalized_l2"]),  # type: ignore[index]
            float(outer_fields["speed_normalized_l2"]),  # type: ignore[index]
        ),
        "common_pressure": (
            float(inner_fields["pressure_l2_over_ambient"]),  # type: ignore[index]
            float(outer_fields["pressure_l2_over_ambient"]),  # type: ignore[index]
        ),
    }
    contraction_gates = {
        name: outer_value <= inner_value
        for name, (inner_value, outer_value) in contraction_pairs.items()
    }
    conservation_gates = {
        f"r{int(item['radius_mm']):03d}": bool(
            float(item["mass_imbalance_fraction"])
            < contract.mass_imbalance_limit_fraction
            and float(item["energy_imbalance_fraction"])
            < contract.energy_imbalance_limit_fraction
        )
        for item in evaluations
    }
    densities = [_opening_density(item) for item in evaluations]
    density_gates = {
        "60_to_80_mm": densities[1] < densities[0],
        "80_to_100_mm": densities[2] < densities[1],
    }
    gates = {
        "outer_pair_within_frozen_v2_limits": all(
            outer_limit_gates.values()
        ),
        "outer_pair_changes_contract": all(contraction_gates.values()),
        "all_domains_conservative": all(conservation_gates.values()),
        "area_mean_opening_activity_decreases_outward": all(
            density_gates.values()
        ),
    }
    return {
        "schema_version": "h11_domain_continuation_v1",
        "status": (
            "pass_60_80_100_mm_domain_continuation"
            if all(gates.values())
            else "fail_60_80_100_mm_domain_continuation"
        ),
        "provenance": skeleton_payload(contract)["provenance"],
        "contract": asdict(contract),
        "evaluations": evaluations,
        "adjacent_comparisons": {
            "60_to_80_mm": inner,
            "80_to_100_mm": outer,
        },
        "opening_area_mean_flux_density_kg_m2_s": {
            f"r{int(radius):03d}": density
            for radius, density in zip(contract.radii_mm, densities)
        },
        "outer_limit_gates": outer_limit_gates,
        "contraction_gates": contraction_gates,
        "conservation_gates": conservation_gates,
        "opening_density_gates": density_gates,
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=SKELETON_AUDIT)
    parser.add_argument("--model-60", type=Path)
    parser.add_argument("--model-80", type=Path)
    parser.add_argument("--model-100", type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()
    contract = DomainContinuationContract()
    models = (args.model_60, args.model_80, args.model_100)
    if any(path is not None for path in models) and not all(
        path is not None for path in models
    ):
        raise ValueError("Provide all three continuation models or none")
    if all(path is not None for path in models):
        for path in models:
            if not path.exists():  # type: ignore[union-attr]
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
                            contract=contract,  # type: ignore[arg-type]
                        ),
                        "model_path": str(path.resolve()),  # type: ignore[union-attr]
                        "model_sha256": _sha256(path),  # type: ignore[arg-type]
                    }
                )
                client.remove(model)
        finally:
            client.clear()
        payload = finalize_domain_continuation(evaluations, contract)
    else:
        payload = skeleton_payload(contract)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote domain-continuation audit: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
