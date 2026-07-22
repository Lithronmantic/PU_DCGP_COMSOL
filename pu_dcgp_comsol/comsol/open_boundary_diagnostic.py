
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.mass_balance_equation_audit import (
    _integrate_selection,
)


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_outer_residual_partial.mph"
)
AUDIT_PATH = (
    HERE
    / "h11_outputs"
    / "target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2_"
    "f2500_open_boundary_diagnostic.json"
)


def flow_partition(
    outward_kg_s: float,
    inward_positive_kg_s: float,
) -> dict[str, float]:

    values = (outward_kg_s, inward_positive_kg_s)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Directional mass-flow magnitudes must be finite")
    activity = outward_kg_s + inward_positive_kg_s
    if activity <= 0:
        return {
            "signed_outward_kg_s": 0.0,
            "absolute_activity_kg_s": 0.0,
            "outward_fraction": 0.0,
            "inward_fraction": 0.0,
        }
    return {
        "signed_outward_kg_s": outward_kg_s - inward_positive_kg_s,
        "absolute_activity_kg_s": activity,
        "outward_fraction": outward_kg_s / activity,
        "inward_fraction": inward_positive_kg_s / activity,
    }


def _reduce_selection(
    jm: Any,
    *,
    tag: str,
    feature_type: str,
    dataset_tag: str,
    expression: str,
    unit: str,
    entities: list[int],
) -> float:
    numerical = jm.result().numerical()
    if tag in {str(value) for value in numerical.tags()}:
        numerical.remove(tag)
    feature = numerical.create(tag, feature_type)
    try:
        feature.set("data", dataset_tag)
        feature.set("expr", [expression])
        feature.set("unit", [unit])
        feature.selection().set(entities)
        return float(feature.getReal()[0][0])
    finally:
        numerical.remove(tag)


def _boundary_metrics(
    jm: Any,
    *,
    dataset_tag: str,
    name: str,
    entities: list[int],
) -> dict[str, Any]:
    axisymmetric_area = _integrate_selection(
        jm,
        tag=f"int_{name}_area_diag",
        feature_type="IntLine",
        dataset_tag=dataset_tag,
        expression="2*pi*r",
        unit="m^2",
        entities=entities,
    )
    signed = _integrate_selection(
        jm,
        tag=f"int_{name}_signed_mass_diag",
        feature_type="IntLine",
        dataset_tag=dataset_tag,
        expression="2*pi*r*hmnf.rho*(u*nr+w*nz)",
        unit="kg/s",
        entities=entities,
    )
    outward = _integrate_selection(
        jm,
        tag=f"int_{name}_out_mass_diag",
        feature_type="IntLine",
        dataset_tag=dataset_tag,
        expression=(
            "2*pi*r*max(hmnf.rho*(u*nr+w*nz),0[kg/(m^2*s)])"
        ),
        unit="kg/s",
        entities=entities,
    )
    inward = _integrate_selection(
        jm,
        tag=f"int_{name}_in_mass_diag",
        feature_type="IntLine",
        dataset_tag=dataset_tag,
        expression=(
            "-2*pi*r*min(hmnf.rho*(u*nr+w*nz),0[kg/(m^2*s)])"
        ),
        unit="kg/s",
        entities=entities,
    )
    partition = flow_partition(outward, inward)
    closure = signed - partition["signed_outward_kg_s"]
    area_scale = max(axisymmetric_area, 1e-30)

    def area_mean(
        tag_suffix: str,
        expression: str,
        unit: str,
    ) -> float:
        integral = _integrate_selection(
            jm,
            tag=f"int_{name}_{tag_suffix}_diag",
            feature_type="IntLine",
            dataset_tag=dataset_tag,
            expression=f"2*pi*r*({expression})",
            unit=f"({unit})*m^2",
            entities=entities,
        )
        return integral / area_scale

    result: dict[str, Any] = {
        "entities": entities,
        "axisymmetric_area_m2": axisymmetric_area,
        "mass_flow": {
            "direct_signed_outward_kg_s": signed,
            "outward_positive_kg_s": outward,
            "inward_positive_kg_s": inward,
            **partition,
            "partition_identity_error_kg_s": closure,
        },
        "area_mean": {
            "temperature_k": area_mean("temperature", "T", "K"),
            "absolute_pressure_pa": area_mean(
                "pressure",
                "hmnf.pA",
                "Pa",
            ),
            "mach_number": area_mean("mach", "hmnf.Ma", "1"),
            "speed_m_s": area_mean("speed", "hmnf.U", "m/s"),
        },
        "range": {},
    }
    for key, expression, unit in (
        ("temperature_k", "T", "K"),
        ("absolute_pressure_pa", "hmnf.pA", "Pa"),
        ("mach_number", "hmnf.Ma", "1"),
        ("speed_m_s", "hmnf.U", "m/s"),
    ):
        result["range"][key] = {
            "minimum": _reduce_selection(
                jm,
                tag=f"min_{name}_{key}_diag",
                feature_type="MinLine",
                dataset_tag=dataset_tag,
                expression=expression,
                unit=unit,
                entities=entities,
            ),
            "maximum": _reduce_selection(
                jm,
                tag=f"max_{name}_{key}_diag",
                feature_type="MaxLine",
                dataset_tag=dataset_tag,
                expression=expression,
                unit=unit,
                entities=entities,
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--version", default="6.3")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.source_model.exists():
        raise FileNotFoundError(args.source_model)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.source_model))
        jm = model.java
        component = jm.component("comp1")
        dataset = list(model / "datasets")[-1]
        dataset_tag = dataset.tag()
        boundaries = {
            "upstream_ambient": sorted(
                int(value)
                for value in component.selection(
                    "geom1_sel_ambient_in"
                ).entities()
            ),
            "radial_far_field": sorted(
                int(value)
                for value in component.selection(
                    "geom1_sel_far_r"
                ).entities()
            ),
        }
        metrics = {
            name: _boundary_metrics(
                jm,
                dataset_tag=dataset_tag,
                name=name,
                entities=entities,
            )
            for name, entities in boundaries.items()
        }
    finally:
        client.clear()

    inlet_scale = 0.0
    for item in metrics.values():
        inlet_scale += item["mass_flow"]["inward_positive_kg_s"]
    partition_errors = [
        abs(item["mass_flow"]["partition_identity_error_kg_s"])
        for item in metrics.values()
    ]
    audit = {
        "schema_version": "h11_open_boundary_diagnostic_v1",
        "status": "diagnostic_only_domain_independence_not_tested",
        "read_only": True,
        "source_model": str(args.source_model.resolve()),
        "boundary_semantics": {
            "upstream_ambient": (
                "annular ambient opening at z=0 outside the effective nozzle"
            ),
            "radial_far_field": (
                "radial truncation boundary at r=r_domain"
            ),
            "positive_sign": "outward from the computational domain",
        },
        "metrics": metrics,
        "checks": {
            "partition_identity_absolute_error_kg_s": max(
                partition_errors,
                default=0.0,
            ),
            "all_finite": all(
                math.isfinite(value)
                for item in metrics.values()
                for section in (
                    item["area_mean"].values(),
                    (
                        item["mass_flow"][
                            "direct_signed_outward_kg_s"
                        ],
                        item["mass_flow"]["outward_positive_kg_s"],
                        item["mass_flow"]["inward_positive_kg_s"],
                    ),
                )
                for value in section
            ),
            "domain_independence_claim_allowed": False,
            "paper_prediction_allowed": False,
        },
        "interpretation_rule": (
            "This audit identifies boundary interaction but cannot establish "
            "domain independence without repeating the same solved case at "
            "larger radial extents."
        ),
        "comsol_version": args.version,
        "cores": args.cores,
        "total_ambient_inward_activity_kg_s": inlet_scale,
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Wrote open-boundary diagnostic: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
