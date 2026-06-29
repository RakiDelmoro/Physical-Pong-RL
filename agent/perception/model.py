"""
Online learning primitives used by the Horde (Alberta Plan Step 3).

Only `OnlineRLS` remains. The class-model labeler that used to live here
(BehaviorClass, ObjectState, ClassModel, the ghost/revival/binding/merging
tower) was deleted under option C: the world model IS the labeler now --
"which object am I controlling?" is a dynamics question, answered by
querying the world model's own learned dynamics (see world_model.py
`controlled_track`). No separate labeler, no coupling to association.
"""

import numpy as np


class OnlineRLS:
    """Recursive Least Squares with forgetting. Closed-form, per-frame,
    stores no data -- the Alberta Plan's "learn on every step" learner.

        model:  y = W @ h
        per sample (h, y):
            gain = P @ h / (forget + h^T P h)
            W   += (y - W@h) outer gain
            P   -= (P@h) outer gain / forget
    forgetting < 1 -> discount old data; adapt to drift."""

    def __init__(self, in_dim, out_dim, forgetting=0.999, ridge_init=1.0):
        self.d, self.m = in_dim, out_dim
        self.lam = forgetting
        self.P = np.eye(in_dim) / ridge_init
        self.W = np.zeros((out_dim, in_dim), dtype=np.float64)
        self.n_updates = 0

    def update(self, h, y):
        h = np.asarray(h, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Ph = self.P @ h
        denom = self.lam + h @ Ph
        if abs(denom) < 1e-12:
            return
        gain = Ph / denom
        err = y - self.W @ h
        self.W += np.outer(err, gain)
        self.P -= np.outer(Ph, gain) / self.lam
        self.n_updates += 1

    def predict(self, h):
        return self.W @ np.asarray(h, dtype=np.float64)
