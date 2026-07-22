"""Regression tests for the frozen H11 effective-exit local Jacobian pilot."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from simulator_v2.phase_h.h11_effective_exit_local_jacobian import (
    COUNT_CONVERGENCE_PATH,
    FINAL_SUMMARY_PATH,
    PILOT_SUMMARY_PATH,
    build_summary,
)
from simulator_v2.phase_h.h11_effective_exit_local_jacobian_contract import (
    LocalJacobianContract,
    OUTPUT_DIR,
)


class EffectiveExitLocalJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = LocalJacobianContract()
        cls.contract.validate()
        cls.summary = json.loads(PILOT_SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.final = json.loads(FINAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.convergence = json.loads(
            COUNT_CONVERGENCE_PATH.read_text(encoding="utf-8")
        )

    def test_frozen_four_case_design_is_symmetric(self):
        cases = self.contract.cases()

        self.assertEqual(cases["temperature_minus"]["temperature_k"], 10_760.0)
        self.assertEqual(cases["temperature_plus"]["temperature_k"], 11_560.0)
        self.assertEqual(cases["speed_minus"]["speed_m_s"], 990.0)
        self.assertEqual(cases["speed_plus"]["speed_m_s"], 1_190.0)

    def test_all_four_gas_solutions_pass_frozen_physics_gates(self):
        for case_name in self.contract.cases():
            spec_name = {
                "temperature_minus": "local_temperature_minus_t10760_u1090",
                "temperature_plus": "local_temperature_plus_t11560_u1090",
                "speed_minus": "local_speed_minus_t11160_u990",
                "speed_plus": "local_speed_plus_t11160_u1190",
            }[case_name]
            payload = json.loads(
                (OUTPUT_DIR / f"{spec_name}_gas.json").read_text(encoding="utf-8")
            )

            self.assertEqual(payload["status"], "pass_effective_exit_screen_gas")
            self.assertTrue(all(payload["gates"].values()))

    def test_pilot_sampling_retains_all_diameter_nodes(self):
        for case in self.summary["cases"].values():
            self.assertEqual(case["particles_per_size"], 127)
            self.assertEqual(case["selected_particle_count"], 14)
            self.assertEqual(case["diameter_nodes_represented"], 7)

    def test_centered_local_jacobian_is_identifiable(self):
        jacobian = np.asarray(
            self.summary["jacobian_rows_particle_temperature_velocity"],
            dtype=float,
        )

        self.assertGreater(jacobian[0, 0], 0.0)
        self.assertGreater(jacobian[1, 1], 0.0)
        self.assertGreater(
            abs(float(np.linalg.det(jacobian))),
            self.contract.minimum_absolute_determinant,
        )
        self.assertLessEqual(
            float(np.linalg.cond(jacobian)),
            self.contract.maximum_jacobian_condition_number,
        )

    def test_saved_summary_is_reproducible_from_case_audits(self):
        rebuilt = build_summary(127)

        self.assertIsNotNone(rebuilt)
        self.assertEqual(rebuilt["status"], "pass_local_jacobian_pilot")
        np.testing.assert_allclose(
            rebuilt["jacobian_rows_particle_temperature_velocity"],
            self.summary["jacobian_rows_particle_temperature_velocity"],
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_final_ladder_and_particle_count_convergence_pass(self):
        self.assertEqual(self.final["status"], "pass_local_jacobian_final")
        self.assertEqual(
            self.convergence["status"],
            "pass_local_jacobian_particle_count_convergence",
        )
        self.assertTrue(all(self.convergence["gates"].values()))
        self.assertLessEqual(
            self.convergence["maximum_primary_quantile_relative_change"],
            self.contract.particle_count_convergence_limit_fraction,
        )


if __name__ == "__main__":
    unittest.main()
