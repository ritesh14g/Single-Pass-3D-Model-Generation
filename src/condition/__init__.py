"""Stage 2 — conditioning: the noise-robustness layer (spec §5).

Design principle (§5.1): **exclude what is destroyed, correct what is merely
degraded.** No algorithm recovers detail that severe motion blur physically
removed, and "deblur and use anyway" injects false geometry. Every frame leaves
this stage with one of two verdicts — CORRECT or REJECT — plus a fusion weight.
"""

from src.condition.blur import BlurAssessment, BlurProfile, BlurVerdict, assess_blur, profile_video_blur
from src.condition.artifacts import ArtifactAssessment, assess_artifacts, suppress_block_artifacts
from src.condition.illumination import ExposureChain, IlluminationResult, condition_illumination
from src.condition.dynamic_mask import DynamicMasker, DynamicMaskResult
from src.condition.gps_filter import GpsFilterReport, filter_telemetry, huber_weights

__all__ = [
    "GpsFilterReport",
    "filter_telemetry",
    "huber_weights",
    "BlurAssessment",
    "BlurProfile",
    "BlurVerdict",
    "assess_blur",
    "profile_video_blur",
    "ArtifactAssessment",
    "assess_artifacts",
    "suppress_block_artifacts",
    "ExposureChain",
    "IlluminationResult",
    "condition_illumination",
    "DynamicMasker",
    "DynamicMaskResult",
]
