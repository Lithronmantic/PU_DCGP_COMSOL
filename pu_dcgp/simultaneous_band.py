"""Gaussian posterior simultaneous bands for quantile-effect curves."""

from dataclasses import dataclass

import numpy as np

from .contracts import FloatArray


@dataclass(frozen=True, slots=True)
class GaussianSimultaneousBand:
    """A max-standardized-deviation band for one effect curve."""

    level: float
    draw_count: int
    random_seed: int
    critical_value: float
    lower_bound: FloatArray
    upper_bound: FloatArray


def gaussian_simultaneous_band(
    point_effect: FloatArray,
    covariance: FloatArray,
    level: float,
    draw_count: int,
    random_seed: int,
) -> GaussianSimultaneousBand:
    """Estimate the joint-Gaussian max-t critical value reproducibly."""

    point_effect = np.asarray(point_effect, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    active = standard_errors > 0.0
    if np.any(active):
        active_covariance = covariance[np.ix_(active, active)]
        active_scale = standard_errors[active]
        correlation = active_covariance / np.outer(active_scale, active_scale)
        correlation = (correlation + correlation.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        standard_draws = np.random.default_rng(random_seed).standard_normal(
            (draw_count, len(active_scale))
        )
        correlated_draws = standard_draws @ factor.T
        maximum_deviations = np.max(np.abs(correlated_draws), axis=1)
        critical_value = float(
            np.quantile(maximum_deviations, level, method="higher")
        )
    else:
        critical_value = 0.0
    half_width = critical_value * standard_errors
    return GaussianSimultaneousBand(
        level=level,
        draw_count=draw_count,
        random_seed=random_seed,
        critical_value=critical_value,
        lower_bound=point_effect - half_width,
        upper_bound=point_effect + half_width,
    )
