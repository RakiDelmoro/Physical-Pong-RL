"""Perception: the SCAFFOLD (general, never trains) -- finds and tracks
objects, lighting-invariant. The labeler moved to the world model (option C):
the dynamics model discovers which object the action moves. The OnlineRLS
primitive is kept for the Horde's GVF readouts."""

from .proposal import propose_regions, saliency_zscore, propose_regions_conditioned
from .tracker import ObjectTracker, Track, DIM_COMPACT, DIM_ELONGATED, DIM_PLANAR
from .model import OnlineRLS
from .perception import Perception

__all__ = [
    "propose_regions", "saliency_zscore", "propose_regions_conditioned",
    "ObjectTracker", "Track",
    "DIM_COMPACT", "DIM_ELONGATED", "DIM_PLANAR",
    "OnlineRLS",
    "Perception",
]
