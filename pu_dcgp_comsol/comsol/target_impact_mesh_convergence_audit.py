
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE / "h11_outputs" / "target_impact_mesh_convergence"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "h11_target_impact_three_mesh_audit.json"


def _nested(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def _predicted_difference_ratio(
    order: float,
    *,
    h_coarse: float,
    h_medium: float,
    h_fine: float,
) -> float:
    numerator = h_medium**order - h_fine**order
    denominator = h_coarse**order - h_medium**order
    return numerator / denominator


def _bisect_order(
    target_ratio: float,
    predictor: Callable[[float], float],
    *,
    lower: float = 1e-4,
    upper: float = 12.0,
    iterations: int = 100,
) -> float | None:
    low_value = predictor(lower) - target_ratio
    high_value = predictor(upper) - target_ratio
    if not math.isfinite(low_value) or not math.isfinite(high_value):
        return None
    if low_value == 0:
        return lower
    if high_value == 0:
        return upper
    if low_value * high_value > 0:
        return None
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        mid_value = predictor(midpoint) - target_ratio
        if abs(mid_value) < 1e-12:
            return midpoint
        if low_value * mid_value <= 0:
            upper = midpoint
            high_value = mid_value
        else:
            lower = midpoint
            low_value = mid_value
    return (lower + upper) / 2


def audit_quantity(
    values: tuple[float, float, float],
    elements: tuple[int, int, int],
    *,
    acceptable_fine_medium_difference: float = 0.02,
    safety_factor: float = 1.25,
) -> dict[str, Any]:

    coarse, medium, fine = values
    n_coarse, n_medium, n_fine = elements
    if not (0 < n_coarse < n_medium < n_fine):
        raise ValueError("Element counts must increase from coarse to fine")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Mesh outputs must be finite")
    h_coarse, h_medium, h_fine = (
        1 / math.sqrt(n_coarse),
        1 / math.sqrt(n_medium),
        1 / math.sqrt(n_fine),
    )
    coarse_difference = medium - coarse
    fine_difference = fine - medium
    monotonic = coarse_difference * fine_difference > 0
    relative_fine_medium = abs(fine_difference) / max(abs(fine), 1e-15)

    observed_order = None
    extrapolated = None
    fine_gci_fraction = None
    asymptotic = False
    observed_difference_ratio = None
    if monotonic and coarse_difference != 0:
        observed_difference_ratio = fine_difference / coarse_difference
        if observed_difference_ratio > 0:
            predictor = lambda order: _predicted_difference_ratio(
                order,
                h_coarse=h_coarse,
                h_medium=h_medium,
                h_fine=h_fine,
            )
            observed_order = _bisect_order(observed_difference_ratio, predictor)
    if observed_order is not None and observed_order > 0:
        denominator = (h_medium / h_fine) ** observed_order - 1
        extrapolated = fine + (fine - medium) / denominator
        fine_gci_fraction = (
            safety_factor
            * abs(fine - medium)
            / max(abs(fine), 1e-15)
            / denominator
        )
        asymptotic = True

    return {
        "values_coarse_medium_fine": list(values),
        "element_counts_coarse_medium_fine": list(elements),
        "characteristic_h_relative": [h_coarse, h_medium, h_fine],
        "monotonic": monotonic,
        "observed_difference_ratio": observed_difference_ratio,
        "observed_order": observed_order,
        "richardson_extrapolated": extrapolated,
        "fine_gci_fraction": fine_gci_fraction,
        "asymptotic_range_supported": asymptotic,
        "fine_medium_relative_difference": relative_fine_medium,
        "fine_medium_below_2_percent": (
            relative_fine_medium < acceptable_fine_medium_difference
        ),
        "gate_passed": bool(
            asymptotic
            and fine_gci_fraction is not None
            and fine_gci_fraction < acceptable_fine_medium_difference
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = {}
    for level in (4, 3, 2):
        path = args.input_dir / f"h11_target_impact_mesh_level_{level}.json"
        with path.open(encoding="utf-8") as handle:
            records[level] = json.load(handle)
    elements = tuple(int(records[level]["mesh"]["elements"]) for level in (4, 3, 2))
    quantities = {
        "temperature_1mm_upstream_k": (
            "metrics",
            "one_mm_upstream_of_target",
            "temperature_k",
        ),
        "speed_1mm_upstream_m_s": (
            "metrics",
            "one_mm_upstream_of_target",
            "speed_m_s",
        ),
    }
    convergence = {
        name: audit_quantity(
            tuple(_nested(records[level], path) for level in (4, 3, 2)),
            elements,
        )
        for name, path in quantities.items()
    }
    mass = {
        str(level): _nested(
            records[level], ("metrics", "mass_flux_kg_s", "imbalance_fraction")
        )
        for level in (4, 3, 2)
    }
    energy = {
        str(level): _nested(
            records[level],
            ("metrics", "energy_balance_w", "imbalance_fraction_of_inlet"),
        )
        for level in (4, 3, 2)
    }
    audit = {
        "schema_version": "h11_target_impact_three_mesh_audit_v1",
        "status": (
            "pass_three_mesh_gate"
            if all(value["gate_passed"] for value in convergence.values())
            else "fail_three_mesh_gate"
        ),
        "mesh_levels_coarse_medium_fine": [4, 3, 2],
        "quantities": convergence,
        "conservation_by_level": {
            "mass_imbalance_fraction": mass,
            "energy_imbalance_fraction_of_inlet": energy,
        },
        "interpretation": (
            "No GCI is reported unless a positive observed order exists on the "
            "nonuniform three-grid sequence. A failed gate requires a redesigned "
            "systematic mesh and cannot be repaired by selecting one grid."
        ),
        "paper_prediction_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Wrote audit: {args.output}")
    print(f"Three-mesh gate: {audit['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
