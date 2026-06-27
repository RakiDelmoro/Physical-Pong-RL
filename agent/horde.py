"""
The HORDE of General Value Functions (Alberta Plan Step 3).

A GVF (general value function) is a prediction of the discounted future sum
of a CUMULANT signal, under a (fixed or learned) policy, with a termination
and a discount gamma:

    GVF(s) = E[ C_{t+1} + gamma*C_{t+2} + gamma^2*C_{t+3} + ... ]

where C is the cumulant (what this GVF cares about), gamma is how far ahead it
looks. Many GVFs share ONE feature representation -- that is the Horde. Each
GVF is just another linear readout on the shared features; the math is the
same as one value function, the conceptual shift is "a POPULATION of
predictors" instead of "one model".

Why this is the Alberta-Plan next step (and why it's unblocked NOW):
  - W1 (reward rate) and W2 (point foresight) are already GVFs-in-spirit:
    they predict future reward (cumulant = reward). The event-vs-motion
    detector predicts future episode-boundaries (cumulant = "was this an
    event"). The Horde makes that explicit and general.
  - Each GVF is a *useful signal to someone* (the plan's IA orientation, sec
    1d): "the ball will beat you on the left in 0.3s" is a GVF whose cumulant
    is "ball crossed my line." A human partner (or the agent's own controller)
    can read any GVF as a forecast.
  - The GVF vector itself can BE the state (proto-value-functions / successor
    features) -- the Horde dissolves the perception/value chicken-and-egg.
  - It does NOT need W3 (multi-step rollout): each GVF is learned ONLINE from
    the real stream by temporal-difference (TD) learning, one step at a time.
    Deep imagination (Dyna) comes later, ON TOP of a trained Horde.

DESIGN (scaffold/surface, same as the whole agent):
  SCAFFOLD (universal, the GVF definition): a GVF is (cumulant, gamma) + a
    linear readout on shared features, learned by TD(0):
      delta = cumulant + gamma * V(s') - V(s)
      w += alpha * delta * features(s)
    This is the standard TD rule -- universal, no Pong knowledge.
  SURFACE (learned, Pong-specific): the WEIGHTS of each GVF's readout, and the
    cumulant definitions (which are given by what the agent cares about --
    "did the ball cross my line" -- stated in the agent's own discovered
    state, not in hardcoded pixel coords).

The shared features come from the world model's frozen-for-GVFs representation
(the RandomFourierFeatures basis over (state, action) -- the same input front
the world model uses, kept fixed so the GVFs don't fight the world model's
training). Each GVF is a tiny OnlineRLS readout (closed-form, online, no
training phase -- the Alberta Plan's "learn on every step").

Interface:
  horde = Horde(feature_fn, num_features, gvfs=[...])
  horde.observe(state, action, next_state, controlled_id)  # one TD update per GVF
  horde.values(state, action) -> {name: predicted_value}

The GVFs are stated in DISCOVERED terms: "ball-reaches-my-side" uses the
world model's slot assignment + the controlled slot (both learned), not a
hardcoded x-coordinate. So a GVF's cumulant is "did the ball (the fast
free-moving slot) cross the controlled slot's x-line" -- general, learned
identity, no Pong geometry.
"""

import numpy as np

from .perception.model import OnlineRLS


# =====================================================================
# A single General Value Function
# =====================================================================

class GVF:
    """One general value function: a discounted future-cumulant predictor.

      cumulant_fn(state, next_state, ctx) -> float : what this GVF cares about
        (the per-step signal whose discounted future sum we predict).
      gamma : discount / horizon (how far ahead this GVF looks).
      name  : human-readable (for diagnostics / IA signals).

    Learned by TD(0) on shared features:
      delta = cumulant + gamma * V(s') - V(s)
      V(s)  = w . features(s)
    The readout w is an OnlineRLS (closed-form, online) on the shared features.
    """

    def __init__(self, name, cumulant_fn, gamma, n_features,
                 forgetting=0.999, ridge_init=1.0):
        self.name = name
        self.cumulant_fn = cumulant_fn
        self.gamma = float(gamma)
        self.readout = OnlineRLS(n_features, 1, forgetting, ridge_init)
        self.last_feat = None
        self.n_updates = 0
        self.last_delta = 0.0   # the TD error (for prioritized sweeping later)

    def update(self, feat, next_feat, cumulant):
        """One TD(0) update. `feat` = features(s), `next_feat` = features(s').
        RLS does the closed-form update; we feed it the TD TARGET
        cumulant + gamma * V(s') and it learns w so V(s) approximates that."""
        v_next = float(self.readout.predict(next_feat)[0]) if next_feat is not None else 0.0
        target = np.array([cumulant + self.gamma * v_next])
        self.readout.update(feat, target)
        v = float(self.readout.predict(feat)[0])
        self.last_delta = float(target[0] - v)
        self.last_feat = feat
        self.n_updates += 1

    def value(self, feat):
        """The GVF's prediction (discounted future cumulant) at features `feat`."""
        return float(self.readout.predict(feat)[0])


# =====================================================================
# The Horde -- many GVFs sharing one feature representation
# =====================================================================

class Horde:
    """A collection of GVFs sharing one feature representation. The Alberta
    Plan's "many value functions, one representation, all online."

    feature_fn(state, action) -> np.array : the shared features (provided by
      the world model -- its frozen Fourier basis over (state, action)).
    gvf_specs : list of (name, cumulant_fn, gamma) -- the Horde's membership.
    """

    def __init__(self, feature_fn, gvf_specs):
        self.feature_fn = feature_fn
        # probe the feature size
        self._n_feat = None
        self.gvfs = []
        for name, cum_fn, gamma in gvf_specs:
            self.gvfs.append({"name": name, "cum": cum_fn, "gamma": gamma,
                              "gvf": None})  # GVF built lazily once we know n_feat

    def _ensure_gvfs(self, n_feat):
        if self._n_feat == n_feat:
            return
        self._n_feat = n_feat
        for g in self.gvfs:
            g["gvf"] = GVF(g["name"], g["cum"], g["gamma"], n_feat)

    def observe(self, state, action, next_state, ctx=None):
        """One online step: compute the cumulant for each GVF from the real
        transition, and do a TD update on each. `ctx` carries anything the
        cumulant functions need (e.g. the controlled slot, the slot map)."""
        feat = self.feature_fn(state, action)
        if self._n_feat is None:
            self._ensure_gvfs(len(feat))
        # the next-state features under the SAME action (the GVF's policy is
        # "the agent's own action" here -- on-policy; a learned policy per GVF
        # is the Step-4 control upgrade, not needed yet)
        next_feat = self.feature_fn(next_state, action) if next_state is not None else None
        for g in self.gvfs:
            cum = float(g["cum"](state, next_state, ctx))
            g["gvf"].update(feat, next_feat, cum)

    def values(self, state, action):
        """All GVF predictions at (state, action) -- the Horde's forecast
        vector. A useful signal to a partner (the IA orientation) and a
        candidate state representation (proto-value-functions)."""
        feat = self.feature_fn(state, action)
        if self._n_feat is None:
            self._ensure_gvfs(len(feat))
        return {g["name"]: g["gvf"].value(feat) for g in self.gvfs}

    def diagnostics(self):
        return {"gvfs": [{g["name"]: {"n_updates": g["gvf"].n_updates,
                                      "last_delta": g["gvf"].last_delta}}
                         for g in self.gvfs]}
