import tempfile
import unittest
from pathlib import Path

from experiments.pu_dcgp.paper_figures import (
    build_figure_5,
    build_registry,
    figure_specs,
    load_formal_summary,
    load_powder_diameter_effect,
)


class PaperFigureTest(unittest.TestCase):
    def test_five_planned_figures_have_builders(self):
        planned = {spec.figure_id for spec in figure_specs()}
        self.assertEqual(planned, set(build_registry()))

    def test_formal_summary_is_complete(self):
        summary = load_formal_summary()
        self.assertTrue(summary["formal_complete"])
        self.assertEqual(summary["completed_dataset_count"], 3000)
        self.assertEqual(summary["record_count"], 12000)

    def test_powder_diameter_parser_recovers_frozen_band(self):
        rows = load_powder_diameter_effect()
        self.assertEqual(len(rows), 19)
        self.assertAlmostEqual(min(row["sim_low"] for row in rows), -7.3288)
        self.assertAlmostEqual(max(row["sim_high"] for row in rows), -0.2753)

    def test_powder_diameter_figure_writes_png(self):
        with tempfile.TemporaryDirectory() as directory:
            output = build_figure_5(Path(directory))
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
