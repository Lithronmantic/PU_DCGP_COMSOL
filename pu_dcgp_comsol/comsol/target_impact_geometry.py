
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_geometry"
OUT_DIR = HERE / "h11_outputs" / "target_impact_geometry"
MODEL_PATH = MODEL_DIR / "h11_target_impact_geometry_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_target_impact_geometry_audit.json"
PHOTO_PATH = HERE / "evidence" / "dpv_no_workpiece_setup.jpg"
SPRAY_DISTANCE_AUDIT_MM = (80.0, 100.0, 120.0)


@dataclass(frozen=True)
class DualBranchGeometryContract:

    measurement_workpiece_present: bool = False
    measurement_dpv_positions_mm: tuple[float, ...] = (90.0, 100.0, 110.0)
    a_measurement_dpv_position_mm: float = 100.0
    impact_workpiece_present: bool = True
    spray_distance_range_mm: tuple[float, float] = (80.0, 120.0)
    nominal_spray_distance_mm: float = 100.0
    radial_domain_mm: float = 40.0
    provisional_effective_exit_radius_mm: float = 4.0
    target_temperature_range_c: tuple[float, float] = (97.0, 119.0)
    nominal_target_temperature_c: float = 108.0
    selection_tolerance_mm: float = 0.01

    def validate(self) -> None:
        if self.measurement_workpiece_present:
            raise ValueError("The confirmed DPV measurement setup has no workpiece")
        if not self.impact_workpiece_present:
            raise ValueError("The impact simulation branch requires a workpiece")
        if not self.measurement_dpv_positions_mm:
            raise ValueError("At least one DPV measurement coordinate is required")
        if any(value <= 0 for value in self.measurement_dpv_positions_mm):
            raise ValueError("DPV coordinates must be positive")
        if self.a_measurement_dpv_position_mm not in self.measurement_dpv_positions_mm:
            raise ValueError("A-group DPV coordinate must be in the measurement positions")
        low, high = self.spray_distance_range_mm
        if not 0 < low <= self.nominal_spray_distance_mm <= high:
            raise ValueError("Nominal spray distance is outside the executed range")
        if not 0 < self.provisional_effective_exit_radius_mm < self.radial_domain_mm:
            raise ValueError("Effective exit radius must lie inside the radial domain")
        target_low, target_high = self.target_temperature_range_c
        if not target_low <= self.nominal_target_temperature_c <= target_high:
            raise ValueError("Nominal target temperature is outside the measured range")
        if self.selection_tolerance_mm <= 0:
            raise ValueError("Selection tolerance must be positive")

    def branch_semantics(self) -> dict[str, Any]:
        return {
            "measurement": {
                "workpiece_present": self.measurement_workpiece_present,
                "axial_coordinate": "z_dpv",
                "allowed_values_mm": list(self.measurement_dpv_positions_mm),
                "spray_distance_in_equations": False,
                "role": "validate latent particle state against DPV observations",
            },
            "impact": {
                "workpiece_present": self.impact_workpiece_present,
                "axial_coordinate": "d_spray",
                "allowed_range_mm": list(self.spray_distance_range_mm),
                "dpv_coordinate_in_equations": False,
                "role": "predict particle/plume state at the workpiece",
            },
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_param(params: Any, name: str, value: str, description: str) -> None:
    params.set(name, value, description)


def _create_box_selection(
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


def _selection_entities(comp: Any, tag: str) -> list[int]:
    return [int(value) for value in comp.selection(tag).entities()]


def build_model(client: Any, contract: DualBranchGeometryContract) -> tuple[Any, Any]:

    contract.validate()
    model = client.create("h11_target_impact_geometry")
    jm = model.java
    params = jm.param()

    _set_param(
        params,
        "d_spray",
        f"{contract.nominal_spray_distance_mm:.9g}[mm]",
        "Gun-to-workpiece distance; controls target-wall coordinate",
    )
    _set_param(params, "r_domain", f"{contract.radial_domain_mm:.9g}[mm]", "Radial far field")
    _set_param(
        params,
        "r_exit_eff",
        f"{contract.provisional_effective_exit_radius_mm:.9g}[mm]",
        "Provisional effective nozzle radius",
    )
    _set_param(
        params,
        "T_target",
        f"{contract.nominal_target_temperature_c:.9g}[degC]",
        "Nominal measured workpiece temperature for isothermal-wall sensitivity",
    )
    _set_param(
        params,
        "T_target_low",
        f"{contract.target_temperature_range_c[0]:.9g}[degC]",
        "Lower observed workpiece-temperature bound",
    )
    _set_param(
        params,
        "T_target_high",
        f"{contract.target_temperature_range_c[1]:.9g}[degC]",
        "Upper observed workpiece-temperature bound",
    )
    _set_param(
        params,
        "sel_tol",
        f"{contract.selection_tolerance_mm:.9g}[mm]",
        "Coordinate-selection tolerance",
    )

    comp = jm.component().create("comp1", True)
    comp.label("Gun-attached workpiece-impact component")
    geom = comp.geom().create("geom1", 2)
    geom.label("Axisymmetric plume truncated by target at d_spray")
    geom.axisymmetric(True)

    core = geom.feature().create("core", "Rectangle")
    core.label("Plume core to target")
    core.set("size", ["r_exit_eff", "d_spray"])
    core.set("pos", ["0", "0"])

    ambient = geom.feature().create("ambient", "Rectangle")
    ambient.label("Ambient annulus to target")
    ambient.set("size", ["r_domain-r_exit_eff", "d_spray"])
    ambient.set("pos", ["r_exit_eff", "0"])

    union = geom.feature().create("uni1", "Union")
    union.label("Impact domain with retained radial core interface")
    union.selection("input").set(["core", "ambient"])
    union.set("intbnd", True)

    _create_box_selection(
        geom,
        "sel_nozzle_in",
        "Finite effective nozzle inlet",
        xmin="-sel_tol",
        xmax="r_exit_eff+sel_tol",
        ymin="-sel_tol",
        ymax="sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_ambient_in",
        "Upstream ambient opening outside nozzle",
        xmin="r_exit_eff-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_axis",
        "Axis of symmetry",
        xmin="-sel_tol",
        xmax="sel_tol",
        ymin="-sel_tol",
        ymax="d_spray+sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_far_r",
        "Radial far-field opening",
        xmin="r_domain-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="-sel_tol",
        ymax="d_spray+sel_tol",
    )
    _create_box_selection(
        geom,
        "sel_target",
        "Workpiece target wall",
        xmin="-sel_tol",
        xmax="r_domain+sel_tol",
        ymin="d_spray-sel_tol",
        ymax="d_spray+sel_tol",
    )
    geom.run()
    return model, jm


def _snapshot(jm: Any, spray_distance_mm: float) -> dict[str, Any]:
    jm.param().set(
        "d_spray",
        f"{spray_distance_mm:.9g}[mm]",
        "Gun-to-workpiece distance; controls target-wall coordinate",
    )
    comp = jm.component("comp1")
    geom = comp.geom("geom1")
    geom.run()
    bbox = [float(value) for value in geom.getBoundingBox()]
    target = _selection_entities(comp, "geom1_sel_target")
    return {
        "spray_distance_mm": spray_distance_mm,
        "bounding_box_native": bbox,
        "target_boundary_entities": target,
        "domains": int(geom.getNDomains()),
        "boundaries": int(geom.getNBoundaries()),
        "edges": int(geom.getNEdges()),
        "vertices": int(geom.getNVertices()),
    }


def audit_geometry(jm: Any, contract: DualBranchGeometryContract) -> dict[str, Any]:
    snapshots = [_snapshot(jm, value) for value in SPRAY_DISTANCE_AUDIT_MM]
    topology = {
        (item["domains"], item["boundaries"], item["edges"], item["vertices"])
        for item in snapshots
    }
    if len(topology) != 1:
        raise RuntimeError(f"Impact topology changed over spray-distance sweep: {snapshots}")
    if any(len(item["target_boundary_entities"]) != 2 for item in snapshots):
        raise RuntimeError(f"Target selection is incomplete: {snapshots}")

    for item in snapshots:
        axial_max_m = item["bounding_box_native"][3]
        if abs(axial_max_m * 1000.0 - item["spray_distance_mm"]) > 1e-9:
            raise RuntimeError(f"Target coordinate does not equal spray distance: {item}")

    jm.param().set(
        "d_spray",
        f"{contract.nominal_spray_distance_mm:.9g}[mm]",
        "Gun-to-workpiece distance; controls target-wall coordinate",
    )
    jm.component("comp1").geom("geom1").run()

    comp = jm.component("comp1")
    selection_entities = {
        tag: _selection_entities(comp, f"geom1_{tag}")
        for tag in ("sel_nozzle_in", "sel_ambient_in", "sel_axis", "sel_far_r", "sel_target")
    }
    if set(selection_entities["sel_nozzle_in"]) & set(selection_entities["sel_ambient_in"]):
        raise RuntimeError("Nozzle and ambient inlet selections overlap")

    return {
        "schema_version": "h11_dual_branch_target_geometry_v1",
        "status": "pass",
        "dual_branch_contract": contract.branch_semantics(),
        "impact_geometry": {
            "coordinate_frame": "gun_attached_axisymmetric_rz",
            "spray_distance_controls_target_coordinate": True,
            "dpv_coordinate_present": False,
            "workpiece_present": True,
            "target_temperature_range_c": list(contract.target_temperature_range_c),
            "selection_entities": selection_entities,
            "snapshots": snapshots,
        },
        "measurement_evidence": {
            "workpiece_present": False,
            "photo_path": str(PHOTO_PATH.resolve()) if PHOTO_PATH.exists() else None,
            "photo_sha256": _sha256(PHOTO_PATH) if PHOTO_PATH.exists() else None,
            "visual_scope": (
                "Photograph supports a no-workpiece DPV setup and opposing optical "
                "hardware; it does not establish metric dimensions."
            ),
        },
        "scientific_resolution": (
            "DPV position and gun-to-workpiece spray distance are coordinates in "
            "different physical branches and are never compared inside one geometry."
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

    contract = DualBranchGeometryContract()
    contract.validate()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    print("H11 layer 4: workpiece-impact geometry separated from DPV measurement")
    client = mph.start(cores=args.cores, version=args.version)
    print(f"COMSOL {client.version}, cores={client.cores}, standalone={client.standalone}")
    try:
        model, jm = build_model(client, contract)
        audit = audit_geometry(jm, contract)
        model.save(str(args.model))
    finally:
        client.clear()

    audit.update(
        {
            "contract": asdict(contract),
            "comsol_version": args.version,
            "cores": args.cores,
            "runtime_sec": time.time() - started,
            "model_path": str(args.model.resolve()),
            "model_sha256": _sha256(args.model),
        }
    )
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    print(f"Saved model: {args.model}")
    print(f"Wrote audit: {args.audit}")
    print("Dual-branch geometry gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
