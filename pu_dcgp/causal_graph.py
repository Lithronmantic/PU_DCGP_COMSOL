
from dataclasses import dataclass

import numpy as np

from .contracts import FloatArray, RunBatch


@dataclass(frozen=True, slots=True)
class CausalNode:

    name: str
    label: str
    role: str
    observed: bool
    varies_in_a: bool
    separately_estimable: bool
    fixed_value: float | None = None


@dataclass(frozen=True, slots=True)
class CausalEdge:

    source: str
    target: str
    mechanism: str


@dataclass(frozen=True, slots=True)
class CausalGraphSpec:

    nodes: tuple[CausalNode, ...]
    edges: tuple[CausalEdge, ...]

    def is_acyclic(self) -> bool:
        children = {node.name: [] for node in self.nodes}
        indegree = {node.name: 0 for node in self.nodes}
        for edge in self.edges:
            children[edge.source].append(edge.target)
            indegree[edge.target] += 1
        frontier = [name for name, degree in indegree.items() if degree == 0]
        visited = 0
        while frontier:
            source = frontier.pop()
            visited += 1
            for target in children[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    frontier.append(target)
        return visited == len(self.nodes)


def aps_ysz_a_causal_graph() -> CausalGraphSpec:

    nodes = (
        CausalNode("current_a", "Arc current", "varied_treatment", True, True, True),
        CausalNode(
            "argon_flow_scfh",
            "Argon flow",
            "varied_treatment",
            True,
            True,
            True,
        ),
        CausalNode(
            "hydrogen_setting",
            "Hydrogen setting",
            "controlled_constant",
            True,
            False,
            False,
            2.5,
        ),
        CausalNode(
            "hydrogen_to_argon_ratio",
            "H2/Ar setting ratio",
            "deterministic_derived",
            True,
            True,
            False,
        ),
        CausalNode(
            "powder_feed_g_min",
            "Powder feed",
            "varied_treatment",
            True,
            True,
            True,
        ),
        CausalNode(
            "powder_carrier_gas_setting",
            "Powder carrier-gas setting",
            "controlled_constant",
            True,
            False,
            False,
            10.0,
        ),
        CausalNode(
            "spray_distance_mm",
            "Spray distance",
            "varied_exploratory_treatment",
            True,
            True,
            False,
        ),
        CausalNode(
            "execution_order",
            "Execution order / system history",
            "observed_context",
            True,
            True,
            False,
        ),
        CausalNode(
            "plasma_energy_momentum_state",
            "Plasma energy and momentum state",
            "latent_mediator",
            False,
            True,
            False,
        ),
        CausalNode(
            "powder_injection_loading_state",
            "Powder injection and loading state",
            "latent_mediator",
            False,
            True,
            False,
        ),
        CausalNode(
            "in_flight_particle_state",
            "In-flight heat and momentum transfer",
            "latent_mediator",
            False,
            True,
            False,
        ),
        CausalNode(
            "temperature_c",
            "Particle-temperature distribution",
            "distributional_outcome",
            True,
            True,
            False,
        ),
        CausalNode(
            "velocity_m_s",
            "Particle-velocity distribution",
            "distributional_outcome",
            True,
            True,
            False,
        ),
        CausalNode(
            "particle_diameter_um",
            "Particle-diameter distribution",
            "distributional_outcome",
            True,
            True,
            False,
        ),
    )
    edges = (
        CausalEdge(
            "argon_flow_scfh",
            "hydrogen_to_argon_ratio",
            "ratio denominator",
        ),
        CausalEdge(
            "hydrogen_setting",
            "hydrogen_to_argon_ratio",
            "ratio numerator",
        ),
        CausalEdge("current_a", "plasma_energy_momentum_state", "arc power"),
        CausalEdge(
            "argon_flow_scfh",
            "plasma_energy_momentum_state",
            "primary-gas flow and momentum",
        ),
        CausalEdge(
            "hydrogen_setting",
            "plasma_energy_momentum_state",
            "secondary-gas energy transport",
        ),
        CausalEdge(
            "hydrogen_to_argon_ratio",
            "plasma_energy_momentum_state",
            "gas-composition channel",
        ),
        CausalEdge(
            "execution_order",
            "plasma_energy_momentum_state",
            "torch and system history",
        ),
        CausalEdge(
            "powder_feed_g_min",
            "powder_injection_loading_state",
            "particle mass loading",
        ),
        CausalEdge(
            "powder_carrier_gas_setting",
            "powder_injection_loading_state",
            "injection momentum and dispersion",
        ),
        CausalEdge(
            "powder_injection_loading_state",
            "plasma_energy_momentum_state",
            "plasma loading and cooling",
        ),
        CausalEdge(
            "plasma_energy_momentum_state",
            "in_flight_particle_state",
            "available heat and momentum",
        ),
        CausalEdge(
            "powder_injection_loading_state",
            "in_flight_particle_state",
            "particle entry and residence conditions",
        ),
        CausalEdge(
            "spray_distance_mm",
            "in_flight_particle_state",
            "flight time and heat-loss path",
        ),
        CausalEdge(
            "in_flight_particle_state",
            "temperature_c",
            "thermal response",
        ),
        CausalEdge(
            "in_flight_particle_state",
            "velocity_m_s",
            "kinematic response",
        ),
        CausalEdge(
            "in_flight_particle_state",
            "particle_diameter_um",
            "size-selective transport and detection",
        ),
    )
    return CausalGraphSpec(nodes=nodes, edges=edges)


def hydrogen_to_argon_setting_ratio(runs: RunBatch) -> FloatArray:

    hydrogen_index = runs.controlled_process_names.index("hydrogen_setting")
    argon_index = runs.treatment_names.index("argon_flow_scfh")
    return np.asarray(
        runs.controlled_process_values[:, hydrogen_index]
        / runs.treatment_values[:, argon_index],
        dtype=float,
    )
