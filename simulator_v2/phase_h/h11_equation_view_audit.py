"""Read-only COMSOL Equation View audit for the H11 all-Mach model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
    / "h11_target_impact_conservative_equation_view.json"
)


def _rows(java_table: Any) -> list[list[str]]:
    return [[str(cell) for cell in row] for row in java_table]


def collect_feature_info(parent: Any) -> dict[str, Any]:
    """Collect every exposed Equation View table without changing locks."""
    info_list = parent.featureInfo()
    output: dict[str, Any] = {}
    for tag_value in info_list.tags():
        tag = str(tag_value)
        info = parent.featureInfo(tag)
        tables: dict[str, Any] = {}
        for table_name in ("Expression", "Weak", "Constraint", "Shape"):
            try:
                tables[table_name] = _rows(
                    info.getInfoTable(table_name, "recursive", "all")
                )
            except Exception as exc:
                tables[table_name] = {"error": str(exc)}
        output[tag] = {
            "label": str(info.label()),
            "tables": tables,
        }
    return output


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
        physics = jm.component("comp1").physics("hmnf")
        audit = {
            "schema_version": "h11_equation_view_audit_v1",
            "read_only": True,
            "source_model": str(args.source_model.resolve()),
            "physics_tag": "hmnf",
            "physics_type": str(physics.getType()),
            "physics_info": collect_feature_info(physics),
            "feature_info": {},
        }
        for tag_value in physics.feature().tags():
            tag = str(tag_value)
            feature = physics.feature(tag)
            audit["feature_info"][tag] = {
                "label": str(feature.label()),
                "type": str(feature.getType()),
                "equation_view": collect_feature_info(feature),
            }
    finally:
        client.clear()

    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Wrote read-only Equation View audit: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
