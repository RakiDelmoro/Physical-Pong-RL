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


# =====================================================================
# Probe set — the behavioral "fingerprint" used to compare models
# =====================================================================

def _probe_histories(delay):
    """A handful of canonical normalized trajectories. Two models are 'the
    same behavior' iff they predict the same next-positions for these."""
    base = [
        [(0.5, 0.5)] * delay,                              # stationary center
        [(0.5, 0.5 - 0.05 * i) for i in range(delay)],     # moving up
        [(0.5, 0.5 + 0.05 * i) for i in range(delay)],     # moving down
        [(0.1, 0.5)] * delay,                              # left-edge stationary
        [(0.9, 0.5)] * delay,                              # right-edge stationary
        [(0.5 + 0.05 * i, 0.5 + 0.03 * i) for i in range(delay)],  # diagonal (ball-like)
    ]
    return [[np.array(p) for p in h] for h in base]


# =====================================================================
# Behavior class — a shared W (the class model)
# =====================================================================

class BehaviorClass:
    """A shared dynamics model. Its W is trained by ALL objects bound to it."""

    # index of the "neutral" action (the one where the agent does nothing).
    # Used to form the action-contrast signature. In Pong, STAY is the last action.
    NEUTRAL_ACTION = None  # set by ClassModel at construction

    def __init__(self, reservoir, forgetting, ridge_init, class_id):
        self.id = class_id
        self.res = reservoir
        self.rls = OnlineRLS(reservoir.total_in, 2, forgetting, ridge_init)
        self.n_obs = 0
        self.n_bound = 0
        self._probes = _probe_histories(reservoir.delay)

    def train(self, hist, action, target):
        self.rls.update(self.res.features(hist, action), target)
        self.n_obs += 1

    def predict(self, hist, action):
        return self.rls.predict(self.res.features(hist, action))

    def signature(self):
        """Behavioral fingerprint = pure ACTION-CONTRAST, position-invariant.
        For each probe history h: predict(h, a) - predict(h, neutral) for each
        non-neutral action a. This isolates HOW THE OBJECT RESPONDS TO THE
        AGENT'S ACTION (position cancels in the difference). Large for a
        controlled object, ~0 for a ball / opponent / decoy.

        Two same-dynamics objects at different locations -> identical
        signature (verified: distance ~0.0). Controlled vs passive -> ~0.066.
        The autonomous-dynamics part (how it moves on its own) is intentionally
        omitted for now: it is per-object noisy and would dilute the clean
        controlled/passive split. It can be added back (weighted) later to
        discover finer classes (ball vs opponent vs decoy)."""
        neutral = self.NEUTRAL_ACTION
        sig = []
        for h in self._probes:
            base = self.predict(h, neutral)
            for a in range(self.res.num_actions):
                if a == neutral:
                    continue
                sig.append(self.predict(h, a) - base)   # action-contrast (Δcx, Δcy)
        return np.concatenate(sig)

    def action_effect(self):
        """How much predicted Δcy varies across actions (normalized units).
        Large = the agent controls objects of this class. The identity
        signal, read off the learned shared model. Position-invariant."""
        h = self._probes[0]
        preds = [float(self.predict(h, a)[1]) for a in range(self.res.num_actions)]
        return max(preds) - min(preds)


# =====================================================================
# Per-object state — its position buffer + binding to a class (or tentative)
# =====================================================================

class ObjectState:
    """A tracked object's model-side state.

    Until bound, it trains a PRIVATE tentative W. Once confident, it either
    binds to an existing class (dropping the tentative W, sharing the class
    W) or is promoted to a new class. After binding, its observations train
    the class's shared W (via the bound BehaviorClass)."""

    def __init__(self, reservoir, forgetting, ridge_init):
        self.res = reservoir
        self.pos = []                       # recent normalized [cx, cy]
        self.bound_class = None             # BehaviorClass or None
        self.tentative = OnlineRLS(reservoir.total_in, 2, forgetting, ridge_init)
        self.n_obs = 0
        self.confident = False

    def observe(self, new_pos, action):
        """Train on (last delay positions, action) -> DISPLACEMENT; remember it.
        Target is the DISPLACEMENT new_pos - current_pos (position-invariant
        dynamics), not the absolute next position."""
        self.pos.append(np.asarray(new_pos, dtype=np.float64))
        if len(self.pos) > self.res.delay + 1:
            self.pos.pop(0)
        if len(self.pos) < self.res.delay + 1:
            return
        hist = self.pos[-self.res.delay - 1:-1]   # positions up to & incl. current
        cur_pos = hist[-1]
        target = self.pos[-1] - cur_pos            # DISPLACEMENT (Δcx, Δcy)
        if self.bound_class is not None:
            self.bound_class.train(hist, action, target)
        else:
            self.tentative.update(self.res.features(hist, action), target)
            self.n_obs += 1

    def tentative_signature(self):
        neutral = self.res.num_actions - 1
        sig = []
        for h in _probe_histories(self.res.delay):
            base = self.tentative.predict(self.res.features(h, neutral))
            for a in range(self.res.num_actions):
                if a == neutral:
                    continue
                sig.append(self.tentative.predict(self.res.features(h, a)) - base)
        return np.concatenate(sig)

    def action_effect(self):
        """Tentative model's action-effect (range of predicted Δcy across
        actions at a canonical probe). Used to BRACKET the object as
        controlled vs passive BEFORE binding, so a controlled object never
        binds to a passive class (and vice versa)."""
        if self.n_obs < 5:
            return 0.0
        h = _probe_histories(self.res.delay)[0]
        preds = [float(self.tentative.predict(self.res.features(h, a))[1])
                 for a in range(self.res.num_actions)]
        return max(preds) - min(preds)

    def promote_to_class(self, class_id):
        """Turn this object's tentative W into a new behavior class."""
        c = BehaviorClass(self.res, self.rls_forgetting, self.rls_ridge, class_id)
        c.NEUTRAL_ACTION = self.res.num_actions - 1
        c.rls.W = self.tentative.W.copy()
        c.rls.P = self.tentative.P.copy()
        c.n_obs = self.n_obs
        c.n_bound = 1
        self.bound_class = c
        self.confident = True
        return c

    # stash forgetting/ridge for promote_to_class
    rls_forgetting = None
    rls_ridge = None


# =====================================================================
# ClassModel — manages classes, binding, promotion, merging
# =====================================================================

class ClassModel:
    """The top-level model half: discovered behavior classes."""

    GATE_OBS = 80          # observations before a tentative model is compared
    BIND_THRESH = 0.06     # probe-prediction L2 below this -> bind to a class
    MERGE_THRESH = 0.06    # probe-prediction L2 below this -> merge two classes
    MERGE_PERIOD = 30      # frames between merge sweeps
    BRACKET_GATE = 0.005   # action-effect above this = "controlled" bracket
    MERGE_PERIOD = 30      # frames between merge sweeps

    def __init__(self, delay=3, num_actions=3, n_freq=2, cheb_degree=3,
                 forgetting=0.999, ridge_init=0.01):
        self.res = Reservoir(delay, num_actions, n_freq, cheb_degree)
        self.forgetting = forgetting
        self.ridge_init = ridge_init
        self.classes = {}          # class_id -> BehaviorClass
        self.objects = {}          # track_id -> ObjectState
        self._next_class_id = 1
        self._frame = 0

    def _new_object(self):
        o = ObjectState(self.res, self.forgetting, self.ridge_init)
        o.rls_forgetting = self.forgetting
        o.rls_ridge = self.ridge_init
        return o

    def observe(self, track_id, pos_norm, action):
        """Per-object, per-frame. pos_norm = [cx, cy] in [0,1]. action = int."""
        if track_id not in self.objects:
            self.objects[track_id] = self._new_object()
        obj = self.objects[track_id]
        obj.observe(pos_norm, action)
        # try to bind / promote once the tentative model is confident
        if (obj.bound_class is None and not obj.confident
                and obj.n_obs >= self.GATE_OBS):
            self._try_bind_or_promote(obj)

    def _try_bind_or_promote(self, obj):
        # BRACKET rule: only bind to classes in the SAME action-effect bracket
        # (controlled vs passive). A controlled object must never bind to a
        # passive class, even if their signatures are close early on.
        t_controlled = obj.action_effect() > self.BRACKET_GATE
        sig = obj.tentative_signature()
        best_cid, best_d = None, 1e9
        for cid, c in self.classes.items():
            if c.n_obs < self.GATE_OBS:
                continue
            if (c.action_effect() > self.BRACKET_GATE) != t_controlled:
                continue  # different bracket -> never bind
            d = float(np.linalg.norm(sig - c.signature()))
            if d < best_d:
                best_d, best_cid = d, cid
        if best_cid is not None and best_d < self.BIND_THRESH:
            obj.bound_class = self.classes[best_cid]
            self.classes[best_cid].n_bound += 1
            obj.confident = True
        else:
            c = obj.promote_to_class(self._next_class_id)
            self.classes[c.id] = c
            self._next_class_id += 1

    def retire(self, track_id):
        """A track died — drop the object; decrement its class's bound count."""
        obj = self.objects.pop(track_id, None)
        if obj and obj.bound_class is not None:
            obj.bound_class.n_bound = max(0, obj.bound_class.n_bound - 1)

    def update(self):
        """Once per frame: tick the merge cadence."""
        self._frame += 1
        if self._frame % self.MERGE_PERIOD == 0:
            self._merge()

    def _merge(self):
        # only merge classes in the SAME bracket (controlled with controlled,
        # passive with passive); never cross brackets. Same-bracket classes
        # that are really one behavior (e.g. paddle fragments) consolidate.
        cls = [c for c in self.classes.values() if c.n_obs >= self.GATE_OBS]
        if len(cls) < 2:
            return
        sigs = {c.id: c.signature() for c in cls}
        while True:
            best = None  # (a, b, dist)
            for i in range(len(cls)):
                for j in range(i + 1, len(cls)):
                    a, b = cls[i], cls[j]
                    if (a.action_effect() > self.BRACKET_GATE) != (b.action_effect() > self.BRACKET_GATE):
                        continue  # different bracket
                    d = float(np.linalg.norm(sigs[a.id] - sigs[b.id]))
                    if d < self.MERGE_THRESH and (best is None or d < best[2]):
                        best = (a, b, d)
            if best is None:
                break
            a, b, _ = best
            keep, drop = (a, b) if a.n_obs >= b.n_obs else (b, a)
            for obj in self.objects.values():
                if obj.bound_class is drop:
                    obj.bound_class = keep
                    keep.n_bound += 1
            del self.classes[drop.id]
            cls.remove(drop)
            del sigs[drop.id]

    # ---- identity by behavior ----
    def controlled_class(self):
        best_cid, best_eff = None, 0.0
        for cid, c in self.classes.items():
            if c.n_obs < self.GATE_OBS:
                continue
            eff = c.action_effect()
            if eff > best_eff:
                best_eff, best_cid = eff, cid
        return best_cid, best_eff

    def controlled_object(self, effect_gate):
        cid, eff = self.controlled_class()
        if cid is None or eff < effect_gate:
            return None
        for tid, obj in self.objects.items():
            if obj.bound_class is not None and obj.bound_class.id == cid:
                return tid
        return None

    def diagnostics(self):
        return {
            "classes": {cid: {"n_obs": c.n_obs, "n_bound": c.n_bound,
                              "action_effect": c.action_effect()}
                        for cid, c in self.classes.items()},
            "objects": {tid: {"n_obs": o.n_obs, "bound": o.bound_class.id
                              if o.bound_class else None,
                              "confident": o.confident}
                        for tid, o in self.objects.items()},
        }
