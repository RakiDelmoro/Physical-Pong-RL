"""
The MODEL half of perception — DISCOVERED behavior classes (the learner).

This is the learned, domain-specific part. It learns each object's dynamics
from experience, AND it discovers behavior classes on its own — no human
names them.

  Objective : squared error of each object's next DISPLACEMENT prediction.
  Target    : Δ_i(t) = s_i(t+1) - s_i(t)  (DISPLACEMENT, normalized 0..1).
             Predicting displacement (not absolute position) makes the model
             POSITION-INVARIANT: two objects with the same dynamics but
             different locations predict the same Δ, so they bind to the same
             behavior class. Dynamics is about how things MOVE, not where
             they ARE.
  Input     : the object's last `delay` positions  +  the action a(t).
  Training  : one RLS update per frame, from (features -> true Δ).
              No labels, no batch, no reward.

ARCHITECTURE
  Per object: a fixed Kolmogorov-flavored reservoir (Fourier + Chebyshev
  bases over a delay embedding of recent positions) + a learned linear
  readout (W) updated online by Recursive Least Squares. The reservoir is
  DESIGNED and never trains (Alberta Plan Step 1); only the readout trains.

DISCOVERED CLASSES (the research move)
  Each new tracked object first trains its OWN tentative W (a private
  reservoir+readout) from its own experience — the "watch and see" phase.
  Once that W is confident (enough observations), its PREDICTIONS on a fixed
  set of probe situations are compared to every existing behavior class's
  predictions. "Same behavior" is defined behaviorally: two models are the
  same iff they predict the same next-positions for the same situations.
  (This is robust to feature scaling / representation — it is the
  behavior-level definition of "same", not a weight-space one.)

    close to an existing class  -> the object BINDS to it; it shares that
                                   class's W; its new observations train the
                                   SHARED class W.  (knowledge reused)
    far from all classes        -> the tentative W is PROMOTED to a new class.

  Periodically, class-models whose predictions have grown close MERGE into
  one. No human names the classes. "controlled"/"bounces" emerge as whatever
  clusters the data forms.

IDENTITY BY BEHAVIOR (the payoff)
  The action is an input block in every model. The class whose predicted
  position changes most when the action changes is the "controlled" class
  (the agent moves it). Any object bound to that class IS a controlled
  object. Lighting can't break this — behavior is lighting-invariant by
  definition. The model IS the namer, and classes make the naming reusable.
"""

import numpy as np


# =====================================================================
# Fixed Kolmogorov-flavored reservoir (never trained)
# =====================================================================

def fourier_features(x, n_freq):
    """x in [0,1] -> [1, sin(2pi f x), cos(2pi f x)] for f=1..n_freq.
    Born for oscillatory / bouncing motion (the ball)."""
    x = np.asarray(x, dtype=np.float64)
    out = [np.ones_like(x)]
    for f in range(1, n_freq + 1):
        out.append(np.sin(2.0 * np.pi * f * x))
        out.append(np.cos(2.0 * np.pi * f * x))
    return np.stack(out, axis=-1)


def chebyshev_features(x, degree):
    """x in [0,1] (mapped to [-1,1]) -> [T_0..T_degree].
    Born for bounded, smooth, saturating motion (the paddle)."""
    x = np.clip(2.0 * np.asarray(x, dtype=np.float64) - 1.0, -1.0, 1.0)
    feats = [np.ones_like(x), x]
    for n in range(2, degree + 1):
        feats.append(2.0 * x * feats[-1] - feats[-2])
    return np.stack(feats, axis=-1)


class Reservoir:
    """Builds the feature vector from a delay-embedded history + action.
    Stateless apart from config. Never trains."""

    def __init__(self, delay, num_actions, n_freq, cheb_degree):
        self.delay = delay
        self.num_actions = num_actions
        self.n_freq = n_freq
        self.cheb_degree = cheb_degree
        self.per_coord = (1 + 2 * n_freq) + (cheb_degree + 1)
        self.embed_dim = self.delay * 2 * self.per_coord   # 2 coords
        self.total_in = self.embed_dim + num_actions

    def features(self, hist, action_id):
        """hist: list of `delay` normalized [cx, cy] positions."""
        parts = []
        for s in hist:
            cx, cy = float(s[0]), float(s[1])
            parts.append(fourier_features(np.array([cx]), self.n_freq).ravel())
            parts.append(chebyshev_features(np.array([cx]), self.cheb_degree).ravel())
            parts.append(fourier_features(np.array([cy]), self.n_freq).ravel())
            parts.append(chebyshev_features(np.array([cy]), self.cheb_degree).ravel())
        h = np.concatenate(parts)
        oh = np.zeros(self.num_actions, dtype=np.float64)
        oh[int(action_id)] = 1.0
        return np.concatenate([h, oh])


# =====================================================================
# Online RLS — the ONLY thing that trains (closed-form, per-frame)
# =====================================================================

class OnlineRLS:
    """Recursive Least Squares with forgetting. Solves ridge regression
    incrementally — one sample at a time, O(d^2), mathematically identical to
    the batch ridge solution but stores no data.
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


# (The class-model labeler -- BehaviorClass, ObjectState, ClassModel, and the
# ghost/revival/binding/promotion/merging tower -- used to live here. It was
# DELETED under option C: the world model IS the labeler now. "Which object am
# I controlling?" and "which objects behave the same way?" are DYNAMICS
# questions, answered by querying the world model's own learned dynamics, not
# by a separate labeler coupled to the association. See world_model.py
# `controlled_track` and WORLD_MODEL_PLAN.md "option C". The primitives above
# (Reservoir, OnlineRLS, the feature helpers) are KEPT -- the Horde's GVFs are
# OnlineRLS readouts on shared features.)
