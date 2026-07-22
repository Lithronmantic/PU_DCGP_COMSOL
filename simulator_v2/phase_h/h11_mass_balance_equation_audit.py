"""Read-only discrete mass-balance audit for the H11 all-Mach solution."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_target_impact_conservative_restart import (
    _scalar,
)


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_target_impact_conservative_bridge"
    / "h11_target_impact_conservative_bridge_3_to_2_f2500_refined.mph"
)
AUDIT_PATH = (
    HERE
    / "h11_outputs"
    / "target_impact_conservative_bridge"
    / "h11_target_impact_conservative_mass_equation_audit.json"
)


def corrected_residual(
    physical_boundary_flux: float,
    wall_upwind_flux: float,
) -> float:
    """Residual implied by -continuityEquation + wall upwind weak flux."""
    return physical_boundary_flux - wall_upwind_flux


def _evaluate_scalar(
    model: Any,
    expression: str,
    *,
    unit: str,
    dataset: Any,
    selector: dict[str, Any],
) -> float:
    return _scalar(
        model.evaluate(
            expression,
            unit=unit,
            dataset=dataset,
            **selector,
        )
    )


def _integrate_selection(
    jm: Any,
    *,
    tag: str,
    feature_type: str,
    dataset_tag: str,
    expression: str,
    unit: str,
    entities: list[int],
) -> float:
    """Integrate an expression on an existing solution dataset."""
    numerical = jm.result().numerical()
    if tag in {str(value) for value in numerical.tags()}:
        numerical.remove(tag)
    feature = numerical.create(tag, feature_type)
    try:
        feature.set("data", dataset_tag)
        feature.set("expr", [expression])
        feature.set("unit", [unit])
        feature.selection().set(entities)
        values = feature.getReal()
        return float(values[0][0])
    finally:
        numerical.remove(tag)


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
        physics = component.physics("hmnf")
        wall_entities = sorted(
            int(value)
            for value in physics.feature("wallbc1").selection().entities()
        )
        target_entities = sorted(
            int(value)
            for value in component.selection(
                "geom1_sel_target"
            ).entities()
        )
        other_wall_entities = sorted(
            set(wall_entities).difference(target_entities)
        )
        domain_entities = sorted(
            int(value)
            for value in physics.feature("fluid1").selection().entities()
        )
        dataset = list(model / "datasets")[-1]
        dataset_tag = dataset.tag()
        selector: dict[str, Any] = {}
        inner_indices, _ = model.inner(dataset)
        if len(inner_indices):
            selector["inner"] = "last"
        evaluate = lambda expression, unit="kg/s": _evaluate_scalar(
            model,
            expression,
            unit=unit,
            dataset=dataset,
            selector=selector,
        )
        nozzle = evaluate("hmnf.nozzle_in.massFlowRate")
        ambient = evaluate("hmnf.ambient_open.massFlowRate")
        target = evaluate(
            "int_target_hmnf(2*pi*r*hmnf.rho*(u*nr+w*nz))"
        )
        all_wall = _integrate_selection(
            jm,
            tag="int_wall_mass_audit",
            feature_type="IntLine",
            dataset_tag=dataset_tag,
            expression="2*pi*r*hmnf.rho*(u*nr+w*nz)",
            unit="kg/s",
            entities=wall_entities,
        )
        other_wall = _integrate_selection(
            jm,
            tag="int_other_wall_mass_audit",
            feature_type="IntLine",
            dataset_tag=dataset_tag,
            expression="2*pi*r*hmnf.rho*(u*nr+w*nz)",
            unit="kg/s",
            entities=other_wall_entities,
        )
        wall_upwind = _integrate_selection(
            jm,
            tag="int_wall_upwind_audit",
            feature_type="IntLine",
            dataset_tag=dataset_tag,
            expression="2*pi*r*hmnf.contCoeffFace*hmnf.unJump",
            unit="kg/s",
            entities=wall_entities,
        )
        domain_continuity = _integrate_selection(
            jm,
            tag="int_domain_continuity_audit",
            feature_type="IntSurface",
            dataset_tag=dataset_tag,
            expression="2*pi*r*hmnf.continuityEquation",
            unit="kg/s",
            entities=domain_entities,
        )
        domain_source = _integrate_selection(
            jm,
            tag="int_domain_source_audit",
            feature_type="IntSurface",
            dataset_tag=dataset_tag,
            expression="2*pi*r*hmnf.Qm",
            unit="kg/s",
            entities=domain_entities,
        )
    finally:
        client.clear()

    physical_boundary_flux = nozzle + ambient + all_wall
    discrete_residual = corrected_residual(
        physical_boundary_flux,
        wall_upwind,
    )
    inlet_scale = max(abs(nozzle), 1e-30)
    audit = {
        "schema_version": "h11_mass_balance_equation_audit_v1",
        "read_only": True,
        "source_model": str(args.source_model.resolve()),
        "equation_view_evidence": {
            "continuity_equation": (
                "d(rho*epsilon_p,t)+div(rho*u)-Qm"
            ),
            "domain_weak_term": (
                "-2*hmnf.continuityEquation*test(p)*pi*r"
            ),
            "wall_weak_continuity_term": (
                "2*hmnf.contCoeffFace*hmnf.unJump*test(p)*pi*r"
            ),
            "discrete_identity": (
                "physical_boundary_flux-wall_upwind_flux=0"
            ),
        },
        "entities": {
            "domains": domain_entities,
            "all_walls": wall_entities,
            "target": target_entities,
            "other_walls": other_wall_entities,
        },
        "mass_flux_kg_s": {
            "nozzle_outward": nozzle,
            "ambient_outward": ambient,
            "target_raw_outward": target,
            "other_walls_raw_outward": other_wall,
            "all_walls_raw_outward": all_wall,
            "physical_boundary_sum": physical_boundary_flux,
            "wall_upwind_weak_flux": wall_upwind,
            "domain_continuity_integral": domain_continuity,
            "domain_mass_source": domain_source,
            "discrete_corrected_residual": discrete_residual,
        },
        "normalized": {
            "legacy_three_boundary_residual_fraction": (
                abs(nozzle + ambient + target) / inlet_scale
            ),
            "complete_physical_boundary_residual_fraction": (
                abs(physical_boundary_flux - domain_source) / inlet_scale
            ),
            "discrete_corrected_residual_fraction": (
                abs(discrete_residual - domain_source) / inlet_scale
            ),
            "domain_vs_boundary_fraction": (
                abs(domain_continuity - physical_boundary_flux)
                / inlet_scale
            ),
        },
        "finite": all(
            math.isfinite(value)
            for value in (
                nozzle,
                ambient,
                target,
                all_wall,
                other_wall,
                wall_upwind,
                domain_continuity,
                domain_source,
                discrete_residual,
            )
        ),
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Wrote read-only mass-equation audit: {args.audit}")
    print(
        "Legacy/complete/discrete residuals: "
        f"{audit['normalized']['legacy_three_boundary_residual_fraction']:.3%}/"
        f"{audit['normalized']['complete_physical_boundary_residual_fraction']:.3%}/"
        f"{audit['normalized']['discrete_corrected_residual_fraction']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
