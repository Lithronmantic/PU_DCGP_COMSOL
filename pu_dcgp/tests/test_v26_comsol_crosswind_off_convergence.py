
from __future__ import annotations

import unittest

from pu_dcgp_comsol.comsol.corrected_gas_crosswind_off_contract import (
    CrosswindOffContract,
)
from pu_dcgp_comsol.comsol.corrected_gas_crosswind_off_uniform_damped_contract import (
    UniformDampedContract,
    build_contract as build_uniform_damped_contract,
)
from pu_dcgp_comsol.comsol.corrected_gas_mesh_convergence import (
    gas_adjacent_change,
    normalized_l2_change,
    particle_adjacent_change,
)


class CrosswindOffConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.contract = CrosswindOffContract()
        self.contract.validate()

    def test_three_mesh_and_stabilization_choices_are_frozen(self):
        self.assertEqual(self.contract.automatic_mesh_levels, (4, 3, 2))
        self.assertEqual(self.contract.expected_elements, (8714, 15772, 55291))
        self.assertEqual(
            (
                self.contract.streamline_diffusion,
                self.contract.rans_streamline_diffusion,
                self.contract.heat_streamline_diffusion,
                self.contract.crosswind_diffusion,
                self.contract.rans_crosswind_diffusion,
                self.contract.heat_crosswind_diffusion,
            ),
            (1, 1, 1, 0, 0, 0),
        )

    def test_numerical_and_particle_gates_are_frozen_before_results(self):
        self.assertEqual(
            (
                self.contract.gas_temperature_anomaly_l2_limit_fraction,
                self.contract.gas_speed_l2_limit_fraction,
                self.contract.gas_pressure_l2_over_ambient_limit_fraction,
                self.contract.particle_quantile_limit_fraction,
            ),
            (0.01, 0.01, 1e-4, 0.01),
        )
        self.assertEqual(self.contract.particles_per_size, 1023)
        self.assertEqual(self.contract.minimum_primary_aperture_particles, 70)

    def test_normalized_l2_change_is_zero_for_identical_profiles(self):
        self.assertEqual(normalized_l2_change([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_gas_change_uses_temperature_anomaly_and_ambient_pressure_scale(self):
        left = {
            "mesh_level": 3,
            "dpv_profile": {
                "temperature_k": [400.0, 500.0],
                "speed_m_s": [100.0, 200.0],
                "absolute_pressure_pa": [101325.0, 101425.0],
            },
        }
        right = {
            "mesh_level": 2,
            "dpv_profile": {
                "temperature_k": [401.0, 502.0],
                "speed_m_s": [101.0, 202.0],
                "absolute_pressure_pa": [101326.0, 101427.0],
            },
        }
        change = gas_adjacent_change(left, right)
        self.assertEqual((change["left_level"], change["right_level"]), (3, 2))
        self.assertGreater(change["temperature_anomaly_normalized_l2"], 0.0)
        self.assertGreater(change["speed_normalized_l2"], 0.0)
        self.assertGreater(change["pressure_l2_over_ambient"], 0.0)

    def test_particle_change_covers_temperature_and_speed_quantiles(self):
        quantiles = {
            "temperature_c": {"q10": 100.0, "q50": 200.0, "q90": 300.0},
            "speed_m_s": {"q10": 50.0, "q50": 100.0, "q90": 150.0},
        }
        left = {"mesh_level": 3, "quantiles": quantiles}
        right = {
            "mesh_level": 2,
            "quantiles": {
                response: {key: value * 1.005 for key, value in values.items()}
                for response, values in quantiles.items()
            },
        }
        change = particle_adjacent_change(left, right)
        self.assertAlmostEqual(change["maximum_relative_change"], 0.005)

    def test_recovered_damping_is_uniform_across_studies_and_meshes(self):
        contract = UniformDampedContract()
        contract.validate()
        self.assertEqual(
            (contract.flow_group_damping, contract.turbulence_group_damping),
            (0.5, 0.15),
        )
        self.assertEqual(contract.applies_to_studies, ("std1", "std_refine"))
        self.assertEqual(contract.applies_to_mesh_levels, (4, 3, 2))

    def test_uniform_contract_requires_the_passed_local_recovery(self):
        artifact = build_uniform_damped_contract()
        self.assertEqual(
            artifact["status"],
            "pass_frozen_uniform_full_ladder_damping_contract",
        )
        self.assertFalse(artifact["paper_prediction_allowed"])


if __name__ == "__main__":
    unittest.main()
