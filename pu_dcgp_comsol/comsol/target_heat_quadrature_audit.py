
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.target_impact_conservative_same_mesh_refinement import (
    _sha256,
)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "h11_outputs" / "target_impact_conservative_domain"
SKELETON_AUDIT = (
    OUT_DIR / "h11_target_heat_quadrature_audit_skeleton_v1.json"
)


@dataclass(frozen=True)
class TargetHeatQuadratureContract:

    radii_mm: tuple[float, float, float] = (60.0, 80.0, 100.0)
    integration_orders: tuple[int, ...] = (4, 8, 12, 16)
    target_heat_roi_radius_mm: float = 30.0
    successive_high_order_change_limit_fraction: float = 0.001
    target_heat_domain_change_limit_fraction: float = 0.01
    require_outer_pair_contraction: bool = True
    solved_field_changed: bool = False
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if self.radii_mm != (60.0, 80.0, 100.0):
            raise ValueError("Quadrature audit radii must remain 60/80/100 mm")
        if self.integration_orders != (4, 8, 12, 16):
            raise ValueError("Integration orders must remain 4/8/12/16")
        if self.target_heat_roi_radius_mm != 30.0:
            raise ValueError("Fixed target ROI must remain 30 mm")
        if self.successive_high_order_change_limit_fraction != 0.001:
            raise ValueError("Quadrature convergence limit must remain 0.1%")
        if self.target_heat_domain_change_limit_fraction != 0.01:
            raise ValueError("Domain heat-change limit must remain 1%")
        if not self.require_outer_pair_contraction:
            raise ValueError("Outer heat change must contract")
        if self.solved_field_changed:
            raise ValueError("Quadrature audit cannot change solved fields")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("Quadrature audit cannot claim calibration")


def relative_change(source: float, target: float) -> float:
    if not math.isfinite(source) or not math.isfinite(target):
        raise ValueError("Heat values must be finite")
    return abs(target - source) / max(abs(source), abs(target), 1e-30)


def skeleton_payload(
    contract: TargetHeatQuadratureContract,
) -> dict[str, object]:
    contract.validate()
    return {
        "schema_version": "h11_target_heat_quadrature_skeleton_v1",
        "status": "pass_quadrature_contract_ordered_results_pending",
        "provenance": {
            "retained_failed_continuation": (
                "h11_target_impact_conservative_"
                "domain_continuation_v1.json"
            ),
            "continuation_result_or_threshold_changed": False,
            "created_before_ordered_heat_evaluation": True,
            "official_comsol_default_integration_order": 4,
            "reason": (
                "The discontinuous ROI indicator can cut boundary elements; "
                "ordered Gauss quadrature must be shown to converge."
            ),
        },
        "contract": asdict(contract),
        "gating_logic": {
            "all_models_high_order_converged": True,
            "80_to_100_heat_change_below_1_percent": True,
            "80_to_100_heat_change_contracts_from_60_to_80": True,
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def ordered_target_heat(
    model: Any,
    contract: TargetHeatQuadratureContract,
) -> dict[str, object]:

    contract.validate()
    operator = model.java.component("comp1").cpl("int_target_hmnf")
    original_order = str(operator.getString("intorder"))
    dataset = list(model / "datasets")[-1]
    selector: dict[str, object] = {}
    inner_indices, _ = model.inner(dataset)
    if len(inner_indices):
        selector["inner"] = "last"
    roi_m = contract.target_heat_roi_radius_mm / 1000.0
    expression = (
        "int_target_hmnf(2*pi*r*"
        f"if(r<={roi_m:.12g}[m],hmnf.nteflux,0[W/m^2]))"
    )
    values: dict[str, float] = {}
    try:
        for order in contract.integration_orders:
            operator.set("intorder", str(order))
            value = model.evaluate(
                expression,
                unit="W",
                dataset=dataset,
                **selector,
            )
            scalar = float(value.reshape(-1)[0])
            if not math.isfinite(scalar):
                raise RuntimeError("Target heat integral is not finite")
            values[str(order)] = scalar
    finally:
        operator.set("intorder", original_order)
    high_order_change = relative_change(
        values[str(contract.integration_orders[-2])],
        values[str(contract.integration_orders[-1])],
    )
    return {
        "expression": expression,
        "original_integration_order": original_order,
        "integration_order_restored": (
            str(operator.getString("intorder")) == original_order
        ),
        "heat_w_by_integration_order": values,
        "high_order_change_fraction": high_order_change,
        "high_order_converged": (
            high_order_change
            <= contract.successive_high_order_change_limit_fraction
        ),
    }


def finalize_quadrature_audit(
    evaluations: list[dict[str, object]],
    contract: TargetHeatQuadratureContract,
) -> dict[str, object]:
    contract.validate()
    radii = tuple(float(item["radius_mm"]) for item in evaluations)
    if radii != contract.radii_mm:
        raise ValueError("Expected ordered 60/80/100 mm heat evaluations")
    if not all(
        bool(item["integration_order_restored"]) for item in evaluations
    ):
        raise RuntimeError("An integration operator was not restored")
    highest_order = str(contract.integration_orders[-1])
    heat = [
        float(
            item["heat_w_by_integration_order"][highest_order]
        )
        for item in evaluations
    ]
    inner_change = relative_change(heat[0], heat[1])
    outer_change = relative_change(heat[1], heat[2])
    convergence_gates = {
        f"r{int(item['radius_mm']):03d}": bool(
            item["high_order_converged"]
        )
        for item in evaluations
    }
    gates = {
        "all_models_high_order_quadrature_converged": all(
            convergence_gates.values()
        ),
        "outer_heat_change_below_frozen_1_percent": (
            outer_change
            <= contract.target_heat_domain_change_limit_fraction
        ),
        "outer_heat_change_contracts": outer_change <= inner_change,
        "all_integration_operators_restored": True,
    }
    return {
        "schema_version": "h11_target_heat_quadrature_v1",
        "status": (
            "pass_high_order_fixed_roi_heat_domain_gate"
            if all(gates.values())
            else "fail_high_order_fixed_roi_heat_domain_gate"
        ),
        "provenance": skeleton_payload(contract)["provenance"],
        "contract": asdict(contract),
        "evaluations": evaluations,
        "highest_order_heat_w": {
            f"r{int(radius):03d}": value
            for radius, value in zip(contract.radii_mm, heat)
        },
        "highest_order_domain_changes": {
            "60_to_80_mm": inner_change,
            "80_to_100_mm": outer_change,
        },
        "quadrature_convergence_gates": convergence_gates,
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
    contract = TargetHeatQuadratureContract()
    models = (args.model_60, args.model_80, args.model_100)
    if any(path is not None for path in models) and not all(
        path is not None for path in models
    ):
        raise ValueError("Provide all three models or none")
    if all(path is not None for path in models):
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
                        "radius_mm": radius_mm,
                        **ordered_target_heat(model, contract),
                        "model_path": str(path.resolve()),
                        "model_sha256": _sha256(path),
                    }
                )
                client.remove(model)
        finally:
            client.clear()
        payload = finalize_quadrature_audit(evaluations, contract)
    else:
        payload = skeleton_payload(contract)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote target-heat quadrature audit: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
