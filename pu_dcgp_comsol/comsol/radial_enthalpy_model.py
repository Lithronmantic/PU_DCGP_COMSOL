
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RadialEnthalpyConfig:
    shells: int = 8
    reference_temperature_k: float = 300.0
    density_kg_m3: float = 5890.0
    heat_capacity_j_kg_k: float = 713.0
    thermal_conductivity_w_m_k: float = 2.32
    solidus_k: float = 2923.13
    liquidus_k: float = 3023.13
    latent_heat_j_kg: float = 7.07e5
    stefan_boltzmann_w_m2_k4: float = 5.670374419e-8

    def validate(self) -> None:
        if self.shells < 6:
            raise ValueError("Use at least six radial shells")
        if not self.reference_temperature_k < self.solidus_k < self.liquidus_k:
            raise ValueError("Reference, solidus, and liquidus temperatures are invalid")
        if min(
            self.density_kg_m3,
            self.heat_capacity_j_kg_k,
            self.thermal_conductivity_w_m_k,
            self.latent_heat_j_kg,
        ) <= 0:
            raise ValueError("Thermophysical values must be positive")

    @property
    def solidus_enthalpy_j_kg(self) -> float:
        return self.heat_capacity_j_kg_k * (
            self.solidus_k - self.reference_temperature_k
        )

    @property
    def apparent_heat_capacity_j_kg_k(self) -> float:
        return self.heat_capacity_j_kg_k + self.latent_heat_j_kg / (
            self.liquidus_k - self.solidus_k
        )

    @property
    def liquidus_enthalpy_j_kg(self) -> float:
        return self.solidus_enthalpy_j_kg + self.apparent_heat_capacity_j_kg_k * (
            self.liquidus_k - self.solidus_k
        )


def temperature_from_enthalpy(
    enthalpy_j_kg: float | np.ndarray,
    config: RadialEnthalpyConfig | None = None,
) -> float | np.ndarray:
    cfg = config or RadialEnthalpyConfig()
    cfg.validate()
    h = np.asarray(enthalpy_j_kg, dtype=float)
    solid = cfg.reference_temperature_k + h / cfg.heat_capacity_j_kg_k
    mushy = cfg.solidus_k + (
        h - cfg.solidus_enthalpy_j_kg
    ) / cfg.apparent_heat_capacity_j_kg_k
    liquid = cfg.liquidus_k + (
        h - cfg.liquidus_enthalpy_j_kg
    ) / cfg.heat_capacity_j_kg_k
    value = np.where(
        h <= cfg.solidus_enthalpy_j_kg,
        solid,
        np.where(h < cfg.liquidus_enthalpy_j_kg, mushy, liquid),
    )
    if np.ndim(enthalpy_j_kg) == 0:
        return float(value)
    return value


def melt_fraction_from_enthalpy(
    enthalpy_j_kg: float | np.ndarray,
    config: RadialEnthalpyConfig | None = None,
) -> float | np.ndarray:
    cfg = config or RadialEnthalpyConfig()
    cfg.validate()
    value = np.clip(
        (
            np.asarray(enthalpy_j_kg, dtype=float)
            - cfg.solidus_enthalpy_j_kg
        )
        / (cfg.liquidus_enthalpy_j_kg - cfg.solidus_enthalpy_j_kg),
        0.0,
        1.0,
    )
    if np.ndim(enthalpy_j_kg) == 0:
        return float(value)
    return value


def shell_geometry(
    diameter_m: float,
    config: RadialEnthalpyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or RadialEnthalpyConfig()
    cfg.validate()
    if diameter_m <= 0:
        raise ValueError("Particle diameter must be positive")
    radius = diameter_m / 2.0
    boundaries = np.linspace(0.0, radius, cfg.shells + 1)
    volumes = 4.0 * math.pi / 3.0 * (
        boundaries[1:] ** 3 - boundaries[:-1] ** 3
    )
    interface_areas = 4.0 * math.pi * boundaries[1:] ** 2
    return volumes, interface_areas


def radial_enthalpy_rhs(
    enthalpy_j_kg: Sequence[float],
    *,
    diameter_m: float,
    gas_temperature_k: float,
    heat_transfer_coefficient_w_m2_k: float,
    emissivity: float = 0.6,
    ambient_radiation_temperature_k: float = 300.0,
    config: RadialEnthalpyConfig | None = None,
) -> np.ndarray:

    cfg = config or RadialEnthalpyConfig()
    cfg.validate()
    h = np.asarray(enthalpy_j_kg, dtype=float)
    if h.shape != (cfg.shells,):
        raise ValueError(f"Expected {cfg.shells} shell enthalpies")
    if heat_transfer_coefficient_w_m2_k < 0:
        raise ValueError("Heat-transfer coefficient cannot be negative")
    if not 0 <= emissivity <= 1:
        raise ValueError("Emissivity must be in [0, 1]")

    temperatures = np.asarray(temperature_from_enthalpy(h, cfg), dtype=float)
    volumes, areas = shell_geometry(diameter_m, cfg)
    dr = diameter_m / (2.0 * cfg.shells)
    heat_rate = np.zeros(cfg.shells)
    for boundary in range(1, cfg.shells):
        flux = (
            cfg.thermal_conductivity_w_m_k
            * areas[boundary - 1]
            / dr
            * (temperatures[boundary] - temperatures[boundary - 1])
        )
        heat_rate[boundary - 1] += flux
        heat_rate[boundary] -= flux

    surface_area = areas[-1]
    heat_rate[-1] += (
        heat_transfer_coefficient_w_m2_k
        * surface_area
        * (gas_temperature_k - temperatures[-1])
    )
    heat_rate[-1] += (
        emissivity
        * cfg.stefan_boltzmann_w_m2_k4
        * surface_area
        * (
            ambient_radiation_temperature_k**4
            - temperatures[-1] ** 4
        )
    )
    return heat_rate / (cfg.density_kg_m3 * volumes)


def comsol_temperature_expression(
    enthalpy_expression: str,
) -> str:
    return (
        f"if({enthalpy_expression}<=H_ysz_sol,"
        f"T_ysz_ref+{enthalpy_expression}/cp_ysz_ref,"
        f"if({enthalpy_expression}<H_ysz_liq,"
        f"T_ysz_sol+({enthalpy_expression}-H_ysz_sol)/cp_ysz_app,"
        f"T_ysz_liq+({enthalpy_expression}-H_ysz_liq)/cp_ysz_ref))"
    )


def comsol_shell_rhs_expressions(
    config: RadialEnthalpyConfig | None = None,
) -> list[str]:

    cfg = config or RadialEnthalpyConfig()
    cfg.validate()
    result: list[str] = []
    for index in range(1, cfg.shells + 1):
        volume = (
            "4*pi/3*((%d*dr_p)^3-(%d*dr_p)^3)" % (index, index - 1)
        )
        terms: list[str] = []
        if index > 1:
            area_inner = f"4*pi*(({index - 1})*dr_p)^2"
            terms.append(
                f"k_ysz_ref*({area_inner})/dr_p*(Tsh{index - 1}-Tsh{index})"
            )
        if index < cfg.shells:
            area_outer = f"4*pi*(({index})*dr_p)^2"
            terms.append(
                f"k_ysz_ref*({area_outer})/dr_p*(Tsh{index + 1}-Tsh{index})"
            )
        else:
            area_surface = f"4*pi*(({index})*dr_p)^2"
            terms.extend(
                [
                    f"hconv_p*({area_surface})*(T-Tsh{index})",
                    (
                        f"eps_ysz*sigma_sb*({area_surface})*"
                        f"(T_amb^4-Tsh{index}^4)"
                    ),
                ]
            )
        result.append(f"({' + '.join(terms)})/(rho_ysz*({volume}))")
    return result
