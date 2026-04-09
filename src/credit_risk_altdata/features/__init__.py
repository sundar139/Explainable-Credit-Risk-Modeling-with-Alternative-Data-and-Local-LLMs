"""Feature engineering package interfaces."""

from credit_risk_altdata.features.pipeline import (
    FeaturePipelineError,
    FeaturePipelineResult,
    build_feature_matrices,
)

__all__ = [
    "FeaturePipelineError",
    "FeaturePipelineResult",
    "build_feature_matrices",
]
