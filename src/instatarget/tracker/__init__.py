"""Tracker backend implementations and model adapters."""

from instatarget.tracker.backend import TrackerBackend, TrackerBackendImpl
from instatarget.tracker.depth_preprocessor import DepthPreprocessor, DepthPreprocessResult
from instatarget.tracker.hit_backend import HiTBackend, HiTPrediction, HiTSession
from instatarget.tracker.observation import buildRgbObservation, clipLocalBox
from instatarget.tracker.pytorch_hit import PytorchHiTSession, createHiTSession
from instatarget.tracker.template import TemplateCache, TemplateSample, TemplateSnapshot

__all__ = [
    "HiTBackend",
    "HiTPrediction",
    "HiTSession",
    "PytorchHiTSession",
    "createHiTSession",
    "DepthPreprocessResult",
    "DepthPreprocessor",
    "TemplateCache",
    "TemplateSample",
    "TemplateSnapshot",
    "TrackerBackend",
    "TrackerBackendImpl",
    "buildRgbObservation",
    "clipLocalBox",
]
