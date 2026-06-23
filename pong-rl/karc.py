"""
KARC: Kolmogorov-Arnold Reservoir Computing -- the real-time online version.

Two models, both pure closed-form ridge regression updated INCREMENTALLY by
Recursive Least Squares (RLS) every frame. No neural network, no gradient
descent, no replay buffer, no target network, no GPU.

=== What KARC actually is (the whole thing) ===

1. A FIXED feature map (basis functions). We pick them once and never train
   them. State is expanded into a high-dim feature vector:
       Phi(s) = [Fourier/Chebyshev bases of each coordinate]
   This is the "reservoir" replacement -- a fixed nonlinear lens.

2. A LINEAR readout W (the ONLY thing trained). Next-state prediction:
       s_next = W @ Phi(s_delay_embedded)
   W is found by ridge regression. Online, W is updated by RLS: one small
   matrix update per new sample, mathematically identical to re-solving the
   full ridge problem with all data so far, but O(d^2) per step (microseconds).

=== This file: two models ===

Model A -- AutonomousBallKARC
    Predicts [ball_x, ball_y, opp_y]_{t+1} from the delay-embedded state
    [ball_x, ball_y, my_y, opp_y]_{t..t-k+1}. Pure autonomous forecasting,
    exactly as KARC is published (no action input). Trains by just watching
    the ball bounce -- servo can be parked.

Model B -- ServoKARC  (the research extension: control-affine)
    Predicts my_y_{t+1} from delay-embedded my_y history + the action taken.
    Control-affine form:
        my_y_{t+1} = A @ Phi(my_delay) + B @ action_onehot
    (A,B) trained jointly by ridge/RLS with action one-hot as extra input
    columns. This is "data-driven Koopman with control" -- the KARC control
    extension. Lets the planner ask "if I press UP, where does my paddle go?"
"""

import numpy as np

# ---------------------------------------------------------------------------
# Basis functions (fixed feature maps)
# ---------------------------------------------------------------------------

def chebyshev_features(x, degree):
    """
    x: scalar or 1d array. Returns Chebyshev T_0..T_degree evaluated at x
    (clipped to [-1,1]). T_0=1, T_1=x, T_n=2x*T_{n-1}-T_{n-2}.
    Returns shape (..., degree+1).
    """
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    feats = [np.ones_like(x), x]
    for n in range(2, degree + 1):
        feats.append(2.0 * x * feats[-1] - feats[-2])
    return np.stack(feats, axis=-1)


def fourier_features(x, n_freq):
    """
    x: scalar or 1d array in [0,1]. Returns [1, sin(2*pi*f*x), cos(2*pi*f*x)]
    for f=1..n_freq. Good for oscillatory / bouncing motion.
    Returns shape (..., 1 + 2*n_freq).
    """
    x = np.asarray(x, dtype=np.float64)
    out = [np.ones_like(x)]
    for f in range(1, n_freq + 1):
        out.append(np.sin(2.0 * np.pi * f * x))
        out.append(np.cos(2.0 * np.pi * f * x))
    return np.stack(out, axis=-1)


# ---------------------------------------------------------------------------
# A unified feature builder for the 4-dim state
# ---------------------------------------------------------------------------

class StateFeatures:
    """
    Builds the KARC feature vector from a 4-dim state [bx, by, my, opp].

    For each coordinate: apply Chebyshev (bounded, smooth paddle/servo) or
    Fourier (oscillatory ball). The full feature = concatenation across
    coordinates. No cross-coordinate products (keeps dim modest; can add
    interactions later if needed).
    """

    def __init__(self, basis="chebyshev", degree=4, n_freq=3):
        self.basis = basis
        self.degree = degree
        self.n_freq = n_freq
        # compute dim once
        if basis == "chebyshev":
            self.per_coord = degree + 1
        elif basis == "fourier":
            self.per_coord = 1 + 2 * n_freq
        else:
            raise ValueError(f"unknown basis {basis}")
        self.dim = 4 * self.per_coord  # 4 coordinates

    def __call__(self, s):
        """s: (4,) array -> (dim,) feature vector."""
        feats = []
        for i in range(4):
            x = s[i]
            if self.basis == "chebyshev":
                feats.append(chebyshev_features(x, self.degree).ravel())
            else:
                feats.append(fourier_features(x, self.n_freq).ravel())
        return np.concatenate(feats)


# ---------------------------------------------------------------------------
# Online Recursive Least Squares (the "training" -- one update per frame)
# ---------------------------------------------------------------------------

class OnlineRLS:
    """
    Recursive Least Squares with forgetting factor. Solves ridge regression
    incrementally -- one sample at a time, O(d^2) per update, mathematically
    identical to the batch ridge solution W = Y H^T (H H^T + lam I)^-1 but
    without ever storing all the data.

    Model:  y = W @ h     (h = feature vector dim d, y = output dim m)
    Update per sample (h, y):
        gain = P @ h / (forget + h^T @ P @ h)
        W += (y - W@h) outer gain
        P -= gain outer (P @ h) / forget   (P = inverse covariance, d x d)

    forgetting=1.0  -> never forget (use ALL history equally)
    forgetting<1.0   -> discount old data; adapts to drift (servo wear etc.)
    """

    def __init__(self, in_dim, out_dim, forgetting=0.999, ridge_init=1.0):
        self.d = in_dim
        self.m = out_dim
        self.lam = forgetting
        # P starts as ridge prior; W starts at zero.
        self.P = np.eye(in_dim) / ridge_init
        self.W = np.zeros((out_dim, in_dim), dtype=np.float64)
        self.n_updates = 0

    def update(self, h, y):
        """h: (d,) feature. y: (m,) target. Updates W in place."""
        h = np.asarray(h, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Ph = self.P @ h
        denom = self.lam + h @ Ph
        if abs(denom) < 1e-12:
            return
        gain = Ph / denom
        err = y - self.W @ h
        # rank-1 W update
        self.W += np.outer(err, gain)
        # rank-1 P update
        self.P -= np.outer(Ph, gain) / self.lam
        self.n_updates += 1

    def predict(self, h):
        return self.W @ np.asarray(h, dtype=np.float64)

    def reset(self):
        self.P = np.eye(self.d) / 1.0
        self.W = np.zeros((self.m, self.d), dtype=np.float64)
        self.n_updates = 0


# ---------------------------------------------------------------------------
# Model A: Autonomous ball/opponent forecaster (vanilla KARC)
# ---------------------------------------------------------------------------

class AutonomousBallKARC:
    """
    Predicts [ball_x, ball_y, opp_y]_{t+1} from the delay-embedded state
    [bx, by, my, opp]_{t..t-k+1}.

    Autonomous: NO action input (the ball moves on its own). This is KARC
    exactly as published. Trains by watching the game run -- servo can be
    parked at STAY.
    """

    # indices into the 4-state for what we PREDICT (ball + opponent)
    PRED_IDX = [0, 1, 3]

    def __init__(self, delay=4, basis="chebyshev", degree=4,
                 forgetting=0.9995):
        self.delay = delay
        self.feat = StateFeatures(basis=basis, degree=degree)
        # delay-embedded feature dim = delay * per-coord feature, times 4 coords
        self.embed_dim = delay * self.feat.dim
        self.rls = OnlineRLS(self.embed_dim, len(self.PRED_IDX),
                             forgetting=forgetting)
        self._buffer = []  # rolling history of 4-dim states

    def _embed(self, history):
        """history: list of last `delay` 4-states -> feature vector."""
        # concatenate features across delay steps
        parts = []
        for s in history:
            parts.append(self.feat(s))
        return np.concatenate(parts)

    def observe_and_train(self, state):
        """
        Call EVERY frame with the true state. Automatically builds the delay
        embedding and trains RLS on (state_{t-delay+1..t}) -> state_{t+1}.

        Actually we train on the PREVIOUS completed tuple: we just got
        state_{t}, and we already had states t-delay..t-1, so we can now form
        the target state_{t} (the newest one) from the embedding of
        [t-delay .. t-1]. So we train one step BEHIND -- completely online.
        """
        self._buffer.append(state.copy())
        if len(self._buffer) > self.delay + 1:
            self._buffer.pop(0)
        if len(self._buffer) < self.delay + 1:
            return  # not enough history yet
        # embedding from states [0..delay-1], target = state[delay]
        history = self._buffer[:-1]   # the delay oldest
        target = self._buffer[-1]     # the newest
        h = self._embed(history)
        y = target[self.PRED_IDX]
        self.rls.update(h, y)

    def predict_next(self, current_history):
        """current_history: list of last `delay` states -> predicted next
        [ball_x, ball_y, opp_y]."""
        if len(current_history) < self.delay:
            # not enough history -> return last known (carry forward)
            last = current_history[-1] if current_history else np.array(
                [0.5, 0.5, 0.5, 0.5])
            return last[self.PRED_IDX].copy()
        h = self._embed(current_history[-self.delay:])
        return self.rls.predict(h)

    def rollout(self, current_history, horizon):
        """
        Roll the ball prediction forward `horizon` steps. Returns
        predicted [ball_x, ball_y, opp_y] for each future step
        (shape (horizon, 3)). Feeds predictions back in as if they were
        observations (autonomous rollout -- my_y is held constant at its
        last value since we don't know future actions yet).
        """
        hist = [s.copy() for s in current_history[-self.delay:]]
        preds = []
        last_my = hist[-1][2] if hist else 0.5
        for _ in range(horizon):
            if len(hist) < self.delay:
                # pad with copies
                while len(hist) < self.delay:
                    hist.insert(0, hist[0].copy() if hist else np.array(
                        [0.5, 0.5, 0.5, 0.5]))
            p = self.predict_next(hist)
            # build a synthetic next-state: predicted ball/opp + held my_y
            next_state = np.array([p[0], p[1], last_my, p[2]],
                                 dtype=np.float64)
            preds.append(p.copy())
            hist.append(next_state)
            if len(hist) > self.delay:
                hist.pop(0)
        return np.array(preds)  # (horizon, 3)


# ---------------------------------------------------------------------------
# Model B: Servo model (control-affine KARC extension)
# ---------------------------------------------------------------------------

class ServoKARC:
    """
    Predicts my_y_{t+1} from delay-embedded my_y history + action taken.

    Control-affine form, trained as a single ridge problem with action
    one-hot appended to the feature:
        feature = [my_delay_features , action_onehot]
        my_y_{t+1} = W @ feature
    This lets us query "if I take action a from this history, where's my
    paddle next?" -- needed by the planner.
    """

    def __init__(self, delay=4, num_actions=3, basis="chebyshev", degree=3,
                 forgetting=0.999):
        self.delay = delay
        self.num_actions = num_actions
        # feature on my_y only (1 coordinate) across delay steps
        self.per_coord = (degree + 1) if basis == "chebyshev" else (1 + 2 * 3)
        self.embed_dim = delay * self.per_coord
        self.action_dim = num_actions
        self.total_in = self.embed_dim + self.action_dim
        self.rls = OnlineRLS(self.total_in, 1, forgetting=forgetting)
        self._my_buffer = []
        self._act_buffer = []

    def _my_features(self, my_history):
        """my_history: list of my_y scalars (length delay) -> feature."""
        parts = []
        for y in my_history:
            s = np.array([y, 0.0, 0.0, 0.0])  # only coord 0 matters
            parts.append(self.feat_single(y))
        return np.concatenate(parts)

    def feat_single(self, y):
        if hasattr(self, "_basis") and self._basis == "fourier":
            return fourier_features(np.array([y]), 3).ravel()
        return chebyshev_features(np.array([y]), 3).ravel()

    def __init_basis__(self, basis):
        self._basis = basis

    def observe_and_train(self, my_y, action_id, next_my_y):
        """
        Train on one transition: from my_y, taking action_id, we observed
        next_my_y. my_y and next_my_y are scalars (normalized 0..1).
        """
        self._my_buffer.append(float(my_y))
        self._act_buffer.append(int(action_id))
        if len(self._my_buffer) > self.delay:
            self._my_buffer.pop(0)
            self._act_buffer.pop(0)
        if len(self._my_buffer) < self.delay:
            return
        # embedding of my history + the LAST action taken -> target next_my_y
        feat = self._my_features(self._my_buffer)
        oh = np.zeros(self.action_dim, dtype=np.float64)
        oh[int(action_id)] = 1.0
        h = np.concatenate([feat, oh])
        y = np.array([float(next_my_y)], dtype=np.float64)
        self.rls.update(h, y)

    def predict_next(self, my_history, action_id):
        """my_history: list of delay scalars. Returns predicted next my_y."""
        if len(my_history) < self.delay:
            last = my_history[-1] if my_history else 0.5
            return float(last)
        feat = self._my_features(my_history)
        oh = np.zeros(self.action_dim, dtype=np.float64)
        oh[int(action_id)] = 1.0
        h = np.concatenate([feat, oh])
        return float(self.rls.predict(h)[0])

    def rollout_action(self, my_history, action_id, horizon):
        """Predict my_y for `horizon` steps if we HOLD action_id."""
        hist = [float(y) for y in my_history[-self.delay:]]
        preds = []
        for _ in range(horizon):
            if len(hist) < self.delay:
                while len(hist) < self.delay:
                    hist.insert(0, hist[0] if hist else 0.5)
            p = self.predict_next(hist, action_id)
            preds.append(p)
            hist.append(p)
            if len(hist) > self.delay:
                hist.pop(0)
        return np.array(preds)
