"""H11 layer 1: gun-attached COMSOL geometry with a fixed DPV plane.

This is a geometry contract, not a plasma solution.  It establishes the
coordinate semantics required by the A experiment before any flow, heat, or
particle physics is added:

* ``r=0, z=0`` is attached to the spray-gun exit reference;
* ``z_dpv`` is fixed relative to the gun;
* ``d_spray`` records the independently prescribed robot/workpiece standoff;
* changing ``d_spray`` must not move the DPV plane or truncate the free-jet
  domain.

The unresolved A-workpiece state is deliberately not guessed.  The layer-1
model contains the free-jet envelope and a named DPV cross-section only.
Adding a target boundary is a later geometry branch that requires the
operator-confirmed target and optical geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_gun_fixed_dpv_geometry"
OUT_DIR = HERE / "h11_outputs" / "gun_fixed_dpv_geometry"
MODEL_PATH = MODEL_DIR / "h11_gun_fixed_dpv_geometry_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_geometry_audit.json"
SPRAY_DISTANCE_AUDIT_MM = (80.0, 100.0, 120.0)


@dataclass(frozen=True)
class GeometryContract:
    """Confirmed coordinates plus provisional far-field extents."""

    z_dpv_mm: float = 100.0
    spray_distance_mm: float = 100.0
    radial_domain_mm: float = 40.0
    axial_domain_mm: float = 140.0
    selection_tolerance_mm: float = 0.01
    coordinate_frame: str = "gun_attached_axisymmetric_rz"
    workpiece_state: str = "unresolved_not_modelled"

    def validate(self) -> None:
        if self.z_dpv_mm <= 0:
            raise ValueError("z_dpv_mm must be positive")
        if self.radial_domain_mm <= 0:
            raise ValueError("radial_domain_mm must be positive")
        if self.axial_domain_mm <= self.z_dpv_mm:
            raise ValueError("axial_domain_mm must extend beyond the DPV plane")
        if not 80.0 <= self.spray_distance_mm <= 120.0:
            raise ValueError("spray_distance_mm is outside the executed A range")
        if self.selection_tolerance_mm <= 0:
            raise ValueError("selection_tolerance_mm must be positive")


def set_param(params: Any, name: str, value: str, description: str) -> None:
    params.set(name, value, description)


def _create_boundary_box(
    geom: Any,
    tag: str,
    label: str,
    *,
    xmin: str,
    xmax: str,
    ymin: str,
    ymax: str,
) -> None:
    selection = geom.feature().create(tag, "BoxSelection")
    selection.label(label)
    selection.set("entitydim", "1")
    selection.set("condition", "inside")
    selection.set("xmin", xmin)
    selection.set("xmax", xmax)
    selection.set("ymin", ymin)
    selection.set("ymax", ymax)


def build_geometry_model(client: Any, contract: GeometryContract) -> tuple[Any, Any]:
    """Build the geometry-only COMSOL model."""

    contract.validate()
    model = client.create("h11_gun_fixed_dpv_geometry")
    jm = model.java
    params = jm.param()

    set_param(params, "z_dpv", f"{contract.z_dpv_mm:.9g}[mm]", "DPV plane fixed relative to gun")
    set_param(
        params,
        "d_spray",
        f"{contract.spray_distance_mm:.9g}[mm]",
        "Independent robot/workpiece spray-distance setting",
    )
    set_param(params, "r_domain", f"{contract.radial_domain_mm:.9g}[mm]", "Provisional radial far field")
    set_param(params, "z_domain", f"{contract.axial_domain_mm:.9g}[mm]", "Provisional axial far field")
    set_param(
        params,
        "sel_tol",
        f"{contract.selection_tolerance_mm:.9g}[mm]",
        "Tolerance for coordinate-based boundary selections",
    )

    comp = jm.component().create("comp1", True)
    comp.label("Gun-attached free-jet component")
    geom = comp.geom().create("geom1", 2)
    geom.label("Gun-attached axisymmetric r-z geometry")
    geom.axisymmetric(True)

    upstream = geom.feature().create("r_up", "Rectangle")
    upstream.label("Free jet from gun to fixed DPV plane")
    upstream.set("size", ["r_domain", "z_dpv"])
    upstream.set("pos", ["0", "0"])

    downstream = geom.feature().create("r_down", "Rectangle")
    downstream.label("Downstream buffer independent of spray distance")
    downstream.set("size", ["r_domain", "z_domain-z_dpv"])
    downstream.set("pos", ["0", "z_dpv"])

    # Keep the interface at z=z_dpv as an interior boundary.  Later physics
    # will evaluate population statistics on this boundary without treating
    # it as a wall.
    union = geom.feature().create("uni1", "Union")
    union.label("Free-jet envelope with retained DPV cross-section")
    union.selection("input").set(["r_up", "r_down"])
    union.set("intbnd", True)

    _create_boundary_box(
        geom,
        "sel_dpv",
        "Fixed DPV observation cross-section",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="z_dpv-sel_tol",
        ymax="z_dpv+sel_tol",
    )
    _create_boundary_box(
        geom,
        "sel_axis",
        "Axis of symmetry",
        xmin="-sel_tol",
        xmax="sel_tol",
        ymin="-sel_tol",
        ymax="z_domain+sel_tol",
    )
    _create_boundary_box(
        geom,
        "sel_inlet",
        "Gun-exit reference boundary",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="sel_tol",
    )
    _create_boundary_box(
        geom,
        "sel_far_r",
        "Provisional radial far-field boundary",
        xmin="r_domain-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="z_domain+sel_tol",
    )
    _create_boundary_box(
        geom,
        "sel_far_z",
        "Provisional axial far-field boundary",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="z_domain-sel_tol",
        ymax="z_domain+sel_tol",
    )

    geom.run()
    return model, jm


def _call_int(node: Any, method_name: str) -> int | None:
    try:
        return int(getattr(node, method_name)())
    except Exception:
        return None


def _geometry_snapshot(jm: Any, spray_distance_mm: float) -> dict[str, Any]:
    """Rebuild at one spray distance and report topology/extent invariants."""

    jm.param().set(
        "d_spray",
        f"{spray_distance_mm:.9g}[mm]",
        "Independent robot/workpiece spray-distance setting",
    )
    geom = jm.component("comp1").geom("geom1")
    geom.run()

    try:
        bbox = [float(v) for v in geom.getBoundingBox()]
    except Exception:
        bbox = [
            0.0,
            float(jm.param().evaluate("r_domain")) * 1e3,
            0.0,
            float(jm.param().evaluate("z_domain")) * 1e3,
        ]

    return {
        "spray_distance_mm": spray_distance_mm,
        "z_dpv_mm": float(jm.param().evaluate("z_dpv")) * 1e3,
        "bounding_box_native": bbox,
        "domains": _call_int(geom, "getNDomains"),
        "boundaries": _call_int(geom, "getNBoundaries"),
        "edges": _call_int(geom, "getNEdges"),
        "vertices": _call_int(geom, "getNVertices"),
    }


def _signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(round(float(v), 12) for v in snapshot["bounding_box_native"]),
        snapshot["domains"],
        snapshot["boundaries"],
        snapshot["edges"],
        snapshot["vertices"],
        round(float(snapshot["z_dpv_mm"]), 12),
    )


def _selection_audit(jm: Any) -> dict[str, list[int]]:
    comp = jm.component("comp1")
    expected_counts = {
        "geom1_sel_dpv": 1,
        "geom1_sel_axis": 2,
        "geom1_sel_inlet": 1,
        "geom1_sel_far_r": 2,
        "geom1_sel_far_z": 1,
    }
    entities: dict[str, list[int]] = {}
    for tag, expected_count in expected_counts.items():
        selected = [int(value) for value in comp.selection(tag).entities()]
        if len(selected) != expected_count:
            raise RuntimeError(
                f"Selection {tag} contains {len(selected)} boundaries; "
                f"expected {expected_count}: {selected}"
            )
        entities[tag] = selected

    dpv = set(entities["geom1_sel_dpv"])
    physical_boundaries = set().union(
        entities["geom1_sel_axis"],
        entities["geom1_sel_inlet"],
        entities["geom1_sel_far_r"],
        entities["geom1_sel_far_z"],
    )
    if dpv & physical_boundaries:
        raise RuntimeError("The DPV interior plane overlaps a physical boundary selection")
    return entities


def audit_geometry(jm: Any, nominal_spray_distance_mm: float) -> dict[str, Any]:
    snapshots = [_geometry_snapshot(jm, value) for value in SPRAY_DISTANCE_AUDIT_MM]
    signatures = [_signature(item) for item in snapshots]
    invariant = all(signature == signatures[0] for signature in signatures[1:])
    if not invariant:
        raise RuntimeError("Geometry or DPV plane changed when only d_spray changed")

    # The model artifact itself is saved at the contract's nominal setting,
    # not at the final value used by the invariance loop.
    jm.param().set(
        "d_spray",
        f"{nominal_spray_distance_mm:.9g}[mm]",
        "Independent robot/workpiece spray-distance setting",
    )
    jm.component("comp1").geom("geom1").run()
    selection_entities = _selection_audit(jm)

    return {
        "status": "pass",
        "tested_spray_distances_mm": list(SPRAY_DISTANCE_AUDIT_MM),
        "geometry_invariant_to_spray_distance": True,
        "dpv_plane_fixed_in_gun_frame": True,
        "spray_distance_used_by_geometry": False,
        "workpiece_boundary_included": False,
        "boundary_selection_entities": selection_entities,
        "snapshots": snapshots,
        "gate_to_next_layer": (
            "Passes coordinate/topology gate only. Flow physics remains closed "
            "until apparatus inputs and the A-workpiece geometry are resolved."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    contract = GeometryContract()
    contract.validate()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("H11 layer 1: gun-attached fixed-DPV geometry")
    import mph

    client = mph.start(cores=args.cores, version=args.version)
    print(f"COMSOL {client.version}, cores={client.cores}, standalone={client.standalone}")

    try:
        model, jm = build_geometry_model(client, contract)
        audit = audit_geometry(jm, contract.spray_distance_mm)
        model.save(str(args.model))
    finally:
        client.clear()

    payload = {
        "schema_version": "h11_geometry_audit_v1",
        "contract": asdict(contract),
        "comsol_version": args.version,
        "cores": args.cores,
        "runtime_sec": time.time() - started,
        "model_path": str(args.model.resolve()),
        **audit,
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Saved model: {args.model}")
    print(f"Wrote audit: {args.audit}")
    print("Geometry gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
