"""Reusable COMSOL executor for frozen effective-exit screening cases."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
    configure_studies,
    evaluate_solution,
)
from simulator_v2.phase_h.h11_conservative_free_jet_skeleton import (
    MODEL_PATH as GAS_SKELETON_MODEL,
)
from simulator_v2.phase_h.h11_effective_exit_directional import _gas_gates
from simulator_v2.phase_h.h11_particle_physics_contract import (
    ParticlePhysicsContract,
)
from simulator_v2.phase_h.h11_particle_population_v2_skeleton import (
    audit_model as audit_particle_skeleton,
    build_model as build_particle_skeleton,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import (
    _sha256,
    audit_build as audit_particle_build,
    build_model as build_particle_model,
    solve_and_audit as solve_particles,
)


@dataclass(frozen=True)
class EffectiveExitScreenSpec:
    name: str
    temperature_k: float
    speed_m_s: float
    output_dir: Path
    model_dir: Path
    particle_output_step_us: float = 10.0

    def validate(self) -> None:
        if not self.name or any(char in self.name for char in "\\/:*?\"<>|"):
            raise ValueError("A filesystem-safe screen name is required")
        if not 8_000 <= self.temperature_k <= 12_000:
            raise ValueError("Screen temperature lies outside the tested envelope")
        if not 400 <= self.speed_m_s <= 1_200:
            raise ValueError("Screen speed lies outside the tested envelope")
        if not 2 <= self.particle_output_step_us <= 20:
            raise ValueError("Particle output step is outside the audited range")

    def paths(self, particles_per_size: int | None = None) -> dict[str, Path]:
        self.validate()
        paths = {
            "gas_model": self.model_dir / f"{self.name}_gas.mph",
            "gas_audit": self.output_dir / f"{self.name}_gas.json",
            "gas_log": self.output_dir / f"{self.name}_gas.log",
            "particle_skeleton": self.model_dir / f"{self.name}_particle_skeleton.mph",
        }
        if particles_per_size is not None:
            if particles_per_size <= 0:
                raise ValueError("Particle count must be positive")
            stem = f"{self.name}_n{particles_per_size:04d}"
            paths.update(
                {
                    "particle_model": self.model_dir / f"{stem}.mph",
                    "particle_audit": self.output_dir / "cases" / f"{stem}.json",
                }
            )
        return paths


def solve_gas(client: Any, spec: EffectiveExitScreenSpec) -> dict[str, Any]:
    spec.validate()
    paths = spec.paths()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = client.load(str(GAS_SKELETON_MODEL))
    model.rename(f"{spec.name}_gas")
    jm = model.java
    client.java.showProgress(str(paths["gas_log"].resolve()))
    try:
        jm.param().set("T_exit_eff", f"{spec.temperature_k:.12g}[K]")
        jm.param().set("u_exit_eff", f"{spec.speed_m_s:.12g}[m/s]")
        contract = FreeJetSolveContract()
        solver = configure_studies(jm, contract)
        continuation_started = time.time()
        jm.study("std1").run()
        print(f"  {spec.name}: gas continuation complete", flush=True)
        refinement_started = time.time()
        jm.study("std_refine").run()
        print(f"  {spec.name}: gas refinement complete", flush=True)
        metrics = evaluate_solution(model, contract)
        gates = _gas_gates(metrics)
        model.save(str(paths["gas_model"]))
        return {
            "schema_version": "h11_effective_exit_screen_gas_v1",
            "status": (
                "pass_effective_exit_screen_gas"
                if all(gates.values())
                else "fail_effective_exit_screen_gas"
            ),
            "spec": {
                **asdict(spec),
                "output_dir": str(spec.output_dir.resolve()),
                "model_dir": str(spec.model_dir.resolve()),
            },
            "source_model": str(GAS_SKELETON_MODEL.resolve()),
            "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
            "solver": solver,
            "final_gas_study": "std_refine",
            "continuation_runtime_sec": (
                refinement_started - continuation_started
            ),
            "refinement_runtime_sec": time.time() - refinement_started,
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["gas_model"].resolve()),
            "model_sha256": _sha256(paths["gas_model"]),
            "runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.java.showProgress(False)
        client.remove(model)


def audit_existing_gas(
    client: Any,
    spec: EffectiveExitScreenSpec,
) -> dict[str, Any]:
    spec.validate()
    paths = spec.paths()
    model = client.load(str(paths["gas_model"]))
    try:
        jm = model.java
        temperature = float(str(jm.param().evaluate("T_exit_eff")))
        speed = float(str(jm.param().evaluate("u_exit_eff")))
        if not math.isclose(
            temperature, spec.temperature_k, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("Saved gas temperature does not match the spec")
        if not math.isclose(speed, spec.speed_m_s, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Saved gas speed does not match the spec")
        metrics = evaluate_solution(model, FreeJetSolveContract())
        gates = _gas_gates(metrics)
        return {
            "schema_version": "h11_effective_exit_screen_gas_v1",
            "status": (
                "pass_effective_exit_screen_gas"
                if all(gates.values())
                else "fail_effective_exit_screen_gas"
            ),
            "spec": {
                **asdict(spec),
                "output_dir": str(spec.output_dir.resolve()),
                "model_dir": str(spec.model_dir.resolve()),
            },
            "source_model": "re_audited_saved_effective_exit_screen_gas",
            "final_gas_study": "std_refine",
            "metrics": metrics,
            "gates": gates,
            "model_path": str(paths["gas_model"].resolve()),
            "model_sha256": _sha256(paths["gas_model"]),
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.remove(model)


def build_skeleton(
    client: Any,
    spec: EffectiveExitScreenSpec,
    gas_audit: dict[str, Any],
) -> dict[str, Any]:
    paths = spec.paths()
    contract = ParticlePhysicsContract()
    contract.validate()
    model, jm = build_particle_skeleton(
        client,
        contract,
        source_model=paths["gas_model"],
    )
    try:
        audit = audit_particle_skeleton(jm, contract)
        model.save(str(paths["particle_skeleton"]))
        audit.update(
            {
                "source_model": str(paths["gas_model"].resolve()),
                "source_model_sha256": gas_audit["model_sha256"],
                "model_path": str(paths["particle_skeleton"].resolve()),
                "model_sha256": _sha256(paths["particle_skeleton"]),
            }
        )
        return audit
    finally:
        client.remove(model)


def solve_particle_count(
    client: Any,
    spec: EffectiveExitScreenSpec,
    particles_per_size: int,
    skeleton_audit: dict[str, Any],
) -> dict[str, Any]:
    paths = spec.paths(particles_per_size)
    model, jm = build_particle_model(
        client,
        spec.paths()["particle_skeleton"],
        source_study="std_refine",
    )
    try:
        jm.param().set("particles_per_size", str(particles_per_size))
        jm.param().set(
            "particle_output_step",
            f"{spec.particle_output_step_us:.12g}[us]",
        )
        build_audit = audit_particle_build(jm, source_study="std_refine")
        solve_audit = solve_particles(model, jm)
        model.save(str(paths["particle_model"]))
        solve_audit.update(
            {
                "model_path": str(paths["particle_model"].resolve()),
                "model_sha256": _sha256(paths["particle_model"]),
            }
        )
        return {
            "schema_version": "h11_effective_exit_screen_particle_count_v1",
            "status": (
                "pass_effective_exit_screen_particle_count"
                if solve_audit["status"]
                == "pass_nominal_comsol_trajectory_audit"
                else "fail_effective_exit_screen_particle_count"
            ),
            "spec": {
                **asdict(spec),
                "output_dir": str(spec.output_dir.resolve()),
                "model_dir": str(spec.model_dir.resolve()),
            },
            "particles_per_size": particles_per_size,
            "particle_skeleton_audit": skeleton_audit,
            "particle_build_audit": build_audit,
            "particle_solve_audit": solve_audit,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.remove(model)
