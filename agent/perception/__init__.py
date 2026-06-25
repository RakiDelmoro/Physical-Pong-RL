"""Perception: designed scaffold (general, never trains) + learned model
(domain-specific, online, with discovered behavior classes)."""

from .proposal import propose_regions, saliency_zscore
from .tracker import ObjectTracker, Track, DIM_COMPACT, DIM_ELONGATED, DIM_PLANAR
from .model import (
    ClassModel, BehaviorClass, ObjectState, Reservoir, OnlineRLS,
)
from .perception import Perception

__all__ = [
    "propose_regions", "saliency_zscore",
    "ObjectTracker", "Track",
    "DIM_COMPACT", "DIM_ELONGATED", "DIM_PLANAR",
    "ClassModel", "BehaviorClass", "ObjectState", "Reservoir", "OnlineRLS",
    "Perception",
]
