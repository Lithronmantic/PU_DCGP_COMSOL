"""Linear covariance propagation for matched distributional contrasts."""

from dataclasses import dataclass

import numpy as np

from .contracts import FloatArray, JointDistributionPrediction


@dataclass(frozen=True, slots=True)
class QuantileContrastMoments:
    """Mean and covariance of an average intervention-minus-reference curve."""

    point_effect: FloatArray
    covariance: FloatArray
    marginal_variance: FloatArray


def average_paired_quantile_contrast(
    prediction: JointDistributionPrediction,
    outcome: str,
    stratum_count: int,
) -> QuantileContrastMoments:
    """Contrast paired points ordered as all references then interventions."""

    means = np.asarray(prediction.means[outcome])
    quantile_count = means.shape[1]
    point_count = 2 * stratum_count
    point_effect = np.mean(
        means[stratum_count:point_count] - means[:stratum_count],
        axis=0,
    )
    identity = np.eye(quantile_count) / stratum_count
    contrast_operator = np.hstack(
        [-identity] * stratum_count + [identity] * stratum_count
    )
    flat_covariance = np.asarray(
        prediction.covariances[outcome]
    ).reshape(
        point_count * quantile_count,
        point_count * quantile_count,
    )
    covariance = contrast_operator @ flat_covariance @ contrast_operator.T
    covariance = (covariance + covariance.T) / 2
    return QuantileContrastMoments(
        point_effect=point_effect,
        covariance=covariance,
        marginal_variance=np.diag(covariance).copy(),
    )
