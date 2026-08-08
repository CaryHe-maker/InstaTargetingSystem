"""Tracker backend implementations and model adapters."""

from instatarget.tracker.backend import TrackerBackend, TrackerBackendImpl
from instatarget.tracker.depth_encoder import DepthEncoder, DepthFeatures, DepthPrediction
from instatarget.tracker.depth_preprocessor import DepthPreprocessor, DepthPreprocessResult
from instatarget.tracker.fusion_head import FusionHead, FusionInput
from instatarget.tracker.hit_backend import HiTBackend, HiTPrediction, HiTSession
from instatarget.tracker.observation import buildRgbObservation, clipLocalBox
from instatarget.tracker.template import TemplateCache, TemplateSample, TemplateSnapshot

__all__ = [
    "HiTBackend",
    "HiTPrediction",
    "HiTSession",
    "DepthEncoder",
    "DepthFeatures",
    "DepthPrediction",
    "DepthPreprocessResult",
    "DepthPreprocessor",
    "FusionHead",
    "FusionInput",
    "TemplateCache",
    "TemplateSample",
    "TemplateSnapshot",
    "TrackerBackend",
    "TrackerBackendImpl",
    "buildRgbObservation",
    "clipLocalBox",
]
