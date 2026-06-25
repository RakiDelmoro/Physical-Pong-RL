"""
The SCAFFOLD (part 2) — a general object tracker.

Designed, domain-general. Knows NOTHING about balls, paddles, screens, or
Pong. Three evolved-style rules turn proposal candidates into persistent
objects:

  PERSISTENCE  : a region must appear MIN_AGE frames to be "confirmed"; it
                 dies after MAX_MISS frames unseen. A one-frame glare flash
                 never becomes an object.
  COHESION     : an object moves as a whole; its size stays roughly stable
                 (gated by log-area ratio).
  CONTINUITY   : an object travels on a smooth constant-velocity path; a
                 candidate too far from any track's prediction starts a NEW
                 track rather than hijacking an existing one.

DIMENSIONALITY is a scaffold CUE, not identity: each confirmed track is
tagged 0d (compact) / 1d (elongated) / 2d (planar). No names, no roles.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

DIM_COMPACT = "0d"       # roughly round/square blob  (e.g. a ball)
DIM_ELONGATED = "1d"     # tall-thin or wide-short    (e.g. a paddle)
DIM_PLANAR = "2d"        # large filled rectangle      (e.g. a screen)


def classify_dim(aspect, extent, min_dim):
    """Tag a region by dimensionality. A CUE, not an identity.
    min_dim = min(w, h) in px; a planar region must be large in BOTH dims
    (a screen), not just a small filled square (a ball)."""
    if aspect > 2.5 or aspect < 0.4:
        return DIM_ELONGATED
    if extent > 0.8 and min_dim >= 30:
        return DIM_PLANAR
    return DIM_COMPACT


class Track:
    """A single persistent object hypothesis."""

    def __init__(self, track_id, cand, frame):
        self.id = track_id
        self.cx, self.cy = cand["cx"], cand["cy"]
        self.w, self.h = cand["w"], cand["h"]
        self.area, self.aspect, self.extent = cand["area"], cand["aspect"], cand["extent"]
        self.dim = classify_dim(self.aspect, self.extent, min(self.w, self.h))
        self.vx = self.vy = 0.0
        self.age = 1
        self.missed = 0
        self.confirmed = False
        self.path = [(self.cx, self.cy)]

    def _predicted(self):
        return self.cx + self.vx, self.cy + self.vy

    def update(self, cand, frame, pos_ema=0.85, shape_ema=0.5):
        ncx, ncy = cand["cx"], cand["cy"]
        # position: responsive (so the per-frame displacement the model learns
        # from actually reflects the action; a heavy position EMA smears the
        # action signal across frames and the model can't learn causality).
        self.vx = pos_ema * (ncx - self.cx) + (1 - pos_ema) * self.vx
        self.vy = pos_ema * (ncy - self.cy) + (1 - pos_ema) * self.vy
        self.cx = pos_ema * ncx + (1 - pos_ema) * self.cx
        self.cy = pos_ema * ncy + (1 - pos_ema) * self.cy
        # shape: stable (used for cohesion gating + dimensionality cue; jitter
        # here would split/merge tracks spuriously).
        for a in ("w", "h", "area", "aspect", "extent"):
            setattr(self, a, shape_ema * cand[a] + (1 - shape_ema) * getattr(self, a))
        self.dim = classify_dim(self.aspect, self.extent, min(self.w, self.h))
        self.age += 1
        self.missed = 0
        self.confirmed = self.age >= ObjectTracker.MIN_AGE
        self.path.append((self.cx, self.cy))
        if len(self.path) > 64:
            self.path.pop(0)

    def coast(self):
        self.cx += self.vx
        self.cy += self.vy
        self.missed += 1

    def as_dict(self):
        return {"id": self.id, "cx": self.cx, "cy": self.cy,
                "vx": self.vx, "vy": self.vy, "w": self.w, "h": self.h,
                "area": self.area, "aspect": self.aspect, "extent": self.extent,
                "dim": self.dim, "age": self.age, "missed": self.missed,
                "confirmed": self.confirmed, "path": list(self.path)}


class ObjectTracker:
    """General object tracker = the perception SCAFFOLD. See module docstring."""

    MIN_AGE = 3
    MAX_MISS = 10
    GATE_POS = 35.0
    GATE_LOGAREA = 1.0
    COST_DIM = 40.0

    def __init__(self):
        self._tracks = []
        self._next_id = 1

    def reset(self):
        self._tracks = []
        self._next_id = 1

    def update(self, candidates, frame):
        preds = [t._predicted() for t in self._tracks]
        T, C = len(self._tracks), len(candidates)
        BIG = 1e6
        cost = np.full((T, C), BIG)
        for ti, t in enumerate(self._tracks):
            px, py = preds[ti]
            gate_pos = self.GATE_POS + 2.0 * (t.vx ** 2 + t.vy ** 2) ** 0.5
            log_a = np.log(max(t.area, 1.0))
            for ci, c in enumerate(candidates):
                dpos = ((c["cx"] - px) ** 2 + (c["cy"] - py) ** 2) ** 0.5
                if dpos > gate_pos:
                    continue  # continuity
                dla = abs(np.log(max(c["area"], 1.0)) - log_a)
                if dla > self.GATE_LOGAREA:
                    continue  # cohesion
                dim_pen = (0.0 if classify_dim(c["aspect"], c["extent"],
                                                min(c["w"], c["h"])) == t.dim
                           else self.COST_DIM)
                cost[ti, ci] = dpos + 20.0 * dla + dim_pen

        matched_t, matched_c = set(), set()
        if T and C:
            rows, cols = linear_sum_assignment(cost)
            for ti, ci in zip(rows, cols):
                if cost[ti, ci] >= BIG:   # gated out (no real match)
                    continue
                self._tracks[ti].update(candidates[ci], frame)
                matched_t.add(ti)
                matched_c.add(ci)

        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.coast()
        for ci, c in enumerate(candidates):
            if ci not in matched_c:
                self._tracks.append(Track(self._next_id, c, frame))
                self._next_id += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.MAX_MISS]
        return [t.as_dict() for t in self._tracks if t.confirmed]
