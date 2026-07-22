"""Manifest-backed loader for the executed APS-YSZ DPV runs."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import RunBatch
from .interfaces import RunDataSource


DEFAULT_MANIFEST_PATH = Path(__file__).with_name("data") / "run_manifest.json"


class ManifestDataSource(RunDataSource):
    """Load run settings and jointly valid DPV particle rows from one manifest."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        groups: tuple[str, ...] = ("A",),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.groups = groups

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def load(self) -> RunBatch:
        manifest = self.read_manifest()
        data_root = (self.manifest_path.parent / manifest["data_root"]).resolve()
        runs = manifest["runs"]
        runs = [run for run in runs if run["group"] in self.groups]

        treatment_names = (
            "current_a",
            "argon_flow_scfh",
            "powder_feed_g_min",
            "spray_distance_mm",
        )
        controlled_process_names = (
            "hydrogen_setting",
            "powder_carrier_gas_setting",
        )
        experiment = manifest["experiment"]
        context_names = ("execution_order", "measurement_position_mm")
        csv_columns = manifest["particle_schema"]["csv_columns"]
        outcome_names = tuple(csv_columns)
        particle_samples: dict[str, list[np.ndarray]] = {
            outcome: [] for outcome in outcome_names
        }

        for run in runs:
            values_by_outcome = self._load_particle_file(
                data_root / run["dpv_csv"],
                csv_columns,
            )
            for outcome in outcome_names:
                particle_samples[outcome].append(values_by_outcome[outcome])

        return RunBatch(
            run_ids=tuple(run["run_id"] for run in runs),
            groups=tuple(run["group"] for run in runs),
            doe_modules=tuple(run["doe_module"] for run in runs),
            treatment_names=treatment_names,
            treatment_values=np.asarray(
                [[run[name] for name in treatment_names] for run in runs],
                dtype=float,
            ),
            controlled_process_names=controlled_process_names,
            controlled_process_values=np.tile(
                np.asarray(
                    [experiment[name] for name in controlled_process_names],
                    dtype=float,
                ),
                (len(runs), 1),
            ),
            context_names=context_names,
            context_values=np.asarray(
                [[run[name] for name in context_names] for run in runs],
                dtype=float,
            ),
            particle_samples={
                outcome: tuple(samples)
                for outcome, samples in particle_samples.items()
            },
        )

    @staticmethod
    def _load_particle_file(
        csv_path: Path,
        csv_columns: dict[str, str],
    ) -> dict[str, np.ndarray]:
        valid_rows: list[tuple[float, ...]] = []
        column_names = tuple(csv_columns.values())

        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                values = tuple(float(row[column]) for column in column_names)
                array = np.asarray(values)
                if np.isfinite(array).all() and (array > 0).all():
                    valid_rows.append(values)

        if not valid_rows:
            raise ValueError(f"No jointly valid DPV particles in {csv_path}")

        matrix = np.asarray(valid_rows, dtype=float)
        return {
            outcome: matrix[:, index]
            for index, outcome in enumerate(csv_columns)
        }


def subset_run_batch(runs: RunBatch, indices: np.ndarray) -> RunBatch:
    """Select runs while preserving aligned settings and particle arrays."""

    selected = np.asarray(indices, dtype=int)
    return RunBatch(
        run_ids=tuple(runs.run_ids[index] for index in selected),
        groups=tuple(runs.groups[index] for index in selected),
        doe_modules=tuple(runs.doe_modules[index] for index in selected),
        treatment_names=runs.treatment_names,
        treatment_values=runs.treatment_values[selected],
        controlled_process_names=runs.controlled_process_names,
        controlled_process_values=runs.controlled_process_values[selected],
        context_names=runs.context_names,
        context_values=runs.context_values[selected],
        particle_samples={
            outcome: tuple(samples[index] for index in selected)
            for outcome, samples in runs.particle_samples.items()
        },
    )
