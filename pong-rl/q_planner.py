"""
Q: the bounded, trust-gated residual strategic learner.

This is the "learn to beat the opponent" layer. It sits ON TOP of a safe
guess (closed-form physics + "opponent chases" assumption) and learns only
the safe guess's MISTAKES. Three guards keep it from the bias trap:

  1. RESIDUAL ONLY  -- Q adds a small correction to the safe guess's score;
                       it can only re-rank reachable intercepts, never invent
                       crazy actions. Worst case = the safe guess.
  2. TRUST GATE     -- Q is not ACTED ON until it has empirically beaten the
                       bare safe guess over a window of points. While
                       untrusted, the safe guess drives, and the safe guess's
                       data is what trains Q. No self-corruption.
  3. LOW CAPACITY   -- Q is a linear RLS (reuses karc.OnlineRLS). It literally
                       cannot memorize a degenerate self-induced slice.

The safe guess (SafeGuess) is deliberately a CLEAN, SEPARATE, TESTABLE class
so that if Q ever struggles, we can test the foundation directly (the
safe-guess keystone) without untangling it from Q.

=== What each piece does (junior version) ===

SafeGuess:   "if I hit at height Y, ball lands at P, takes T frames, and the
             opponent reaches it iff T*speed >= distance." Pure math + one
             assumption (opponent chases). No learning.

QModel:      a tiny learned scoreboard. Looks at the situation + a candidate
             intercept, outputs a small number = "how much is the safe guess
             off here?" Trains on realized point outcomes. Trust-gated.

QPlanner:    lists reachable intercept heights, scores each with
             safe_guess + gate*Q, picks the best. Also handles exploration
             (try intercepts Q is unsure about) and remembers the last
             decision so Q can train when the point resolves.
"""

import numpy as np

from karc import OnlineRLS
from safety import UP, DOWN, STAY

# ---------------------------------------------------------------------------
# Firmware constants (mirror sense-pong/main/pong.c). The agent sees normalized
# 0..1 coords; we convert to firmware px for the physics, then back.
# ---------------------------------------------------------------------------
FW = 480.0                 # firmware screen size (px)
BALL_SPEED = 4.0           # |ball_vx| after a paddle hit (px/tick)
PADDLE_H = 80.0            # paddle height (px)
# Opponent chase speed. We LEARN this online (don't hard-trust the firmware
# value), but use the firmware value as the initial estimate.
CPU_SPEED_INIT = 3.0       # px/tick, initial estimate of opponent speed


def reflect01(p):
    """Reflect a position into [0,1] with wall bounces (triangle wave).
    A ball moving past y=1 bounces back, etc. Used for landing-point math."""
    p = p % 2.0
    if p < 0:
        p += 2.0
    if p > 1.0:
        p = 2.0 - p
    return p


# ---------------------------------------------------------------------------
# SafeGuess: the closed-form, testable foundation
# ---------------------------------------------------------------------------

class SafeGuess:
    """
    The safe guess r̂: given the state and a candidate intercept height,
    predict the win-probability using closed-form ball physics + the
    "opponent chases the ball" assumption.

    This is deliberately pure and side-effect-free so it can be unit-tested
    in isolation (the safe-guess keystone) if Q ever struggles.

    Convention: normalized coords 0..1, where 0 = top, 1 = bottom.
      my paddle (agent)  on the LEFT  at small x
      opponent           on the RIGHT at large x
    """

    def __init__(self, opp_speed_init=CPU_SPEED_INIT / FW):
        # learned opponent speed (normalized per frame). Start from firmware.
        self.opp_speed = float(opp_speed_init)

    def estimate(self, state, intercept_y, ball_vy_norm):
        """
        state:        (4,) [ball_x, ball_y, my_y, opp_y] normalized 0..1
        intercept_y:  candidate height where we'd hit (normalized 0..1)
        ball_vy_norm: ball vertical velocity at contact, normalized per frame
                      (firmware vy / FW). Sign: + = moving down.

        returns dict with:
          landing_y     : where the ball lands at the opponent's x-line (0..1)
          flight_time   : frames until the ball reaches the opponent's line
          reach_margin  : T*opp_speed - |landing_y - opp_now|   (the core
                          reachability number; >0 -> opponent CAN reach,
                          <0 -> opponent CANNOT reach)
          win_prob      : safe-guess estimate of win probability for this
                          intercept (high when opponent CANNOT reach)
          features      : np.array of physics-grounded features for Q
        """
        # --- contact happens at my paddle (left). After the hit, vx flips to
        #     +BALL_SPEED (toward opponent). ball_vy is unchanged by the hit.
        # Where my paddle is in x (left margin). In normalized x, ~0.05..0.08.
        # We use the ball's x at contact as the launch x; for a hit it's
        # near the left paddle. Use a fixed launch x = left paddle position.
        launch_x = 0.06            # ~PADDLE_MARGIN/DISP_W (24/480)
        target_x = 1.0 - 0.06      # opponent paddle line (right margin)

        # flight time in frames: distance / speed. vx is constant BALL_SPEED.
        dx_frames = (target_x - launch_x) * FW          # px to travel
        T = dx_frames / BALL_SPEED                      # frames

        # landing y: start at intercept_y, move ball_vy_norm*T, bounce walls.
        landing_y = reflect01(intercept_y + ball_vy_norm * T)

        opp_now = state[3]
        distance = abs(landing_y - opp_now)
        reach_margin = T * self.opp_speed - distance

        # win probability: opponent CANNOT reach -> we likely win.
        # Map reach_margin (negative = safe for us) to a 0..1 win prob.
        # Use a smooth sigmoid-ish ramp around 0.
        win_prob = 1.0 / (1.0 + np.exp(reach_margin * 12.0))
        # (reach_margin large positive -> opp reaches easily -> win_prob ~0)
        # (reach_margin large negative -> opp can't reach   -> win_prob ~1)

        # features for Q (physics-grounded, minimal)
        features = np.array([
            reach_margin,              # the core reachability number
            landing_y,                 # where it lands
            T / 60.0,                  # flight time (normalized ~0..1)
            opp_now,                   # opponent current y
            (landing_y - opp_now),     # signed distance (captures asymmetry)
            ball_vy_norm * 10.0,       # ball vertical speed (scaled)
        ], dtype=np.float64)

        return {
            'landing_y': float(landing_y),
            'flight_time': float(T),
            'reach_margin': float(reach_margin),
            'win_prob': float(win_prob),
            'features': features,
        }


# ---------------------------------------------------------------------------
# QModel: the small residual learner + trust gate + exploration
# ---------------------------------------------------------------------------

class QModel:
    """
    A small linear RLS model that learns the residual:
        delta(s, a) = realized_outcome - safe_guess_win_prob(s, a)
    i.e. "how much is the safe guess off for this kind of situation?"

    Trust gate: Q is only ACTED ON once it has beaten the bare safe guess
    over a window of points. Until then it trains in the background.

    Exploration: an uncertainty bonus (from the RLS posterior) makes the
    planner try intercepts Q is unsure about, so Q gets varied data.
    """

    def __init__(self, n_features=6, forgetting=0.995,
                 gate_window=40, gate_margin=0.5, explore_kappa=0.3):
        self.rls = OnlineRLS(n_features, 1, forgetting=forgetting)
        # trust gate bookkeeping
        self.gate_window = gate_window
        self.gate_margin = gate_margin
        self._baseline_rewards = []   # rewards when gate was CLOSED
        self._q_rewards = []          # rewards when gate was OPEN
        self.gate_open = False
        # exploration
        self.explore_kappa = explore_kappa

    def predict(self, features):
        """Q's residual correction for a candidate. Returns scalar."""
        return float(self.rls.predict(features)[0])

    def uncertainty(self, features):
        """Posterior std of Q's prediction for this feature vector
        (from the RLS inverse-covariance P). High = Q is unsure here."""
        h = np.asarray(features, dtype=np.float64)
        var = float(h @ self.rls.P @ h)
        return np.sqrt(max(var, 0.0))

    def score(self, features, safe_win_prob):
        """Total score for a candidate = safe guess + (gate * Q) + exploration.
        Used by the planner to rank candidates."""
        q = self.predict(features)
        gated_q = q if self.gate_open else 0.0
        explore = self.explore_kappa * self.uncertainty(features)
        return safe_win_prob + gated_q + explore

    def update(self, features, realized_reward, safe_win_prob):
        """Train Q on one realized outcome.
        realized_reward: +1 (won point), -1 (lost), 0 (rally still going).
        target delta = realized - safe_guess."""
        target = np.array([realized_reward - safe_win_prob], dtype=np.float64)
        self.rls.update(features, target)

    # --- trust gate ---
    def record_point(self, reward, q_was_acting):
        """Call when a point resolves. reward = +1/-1. q_was_acting = was the
        gate open for the decision that led to this point?"""
        if q_was_acting:
            self._q_rewards.append(float(reward))
            self._q_rewards = self._q_rewards[-self.gate_window:]
        else:
            self._baseline_rewards.append(float(reward))
            self._baseline_rewards = self._baseline_rewards[-self._window_keep():]
        self._reeval_gate()

    def _window_keep(self):
        return self.gate_window

    def _reeval_gate(self):
        """Open the gate only if Q-augmented choices beat the baseline clearly
        over the recent window, with enough samples. Close it if they stop."""
        n_q = len(self._q_rewards)
        n_b = len(self._baseline_rewards)
        # need some baseline samples to compare against
        if n_b < 10:
            self.gate_open = False
            return
        base_mean = (np.mean(self._baseline_rewards)
                     if self._baseline_rewards else 0.0)
        if n_q < 10:
            self.gate_open = False
            return
        q_mean = np.mean(self._q_rewards)
        self.gate_open = bool(q_mean > base_mean + self.gate_margin)


# ---------------------------------------------------------------------------
# QPlanner: enumerate reachable intercepts, score, pick the best
# ---------------------------------------------------------------------------

class QPlanner:
    """
    The strategic planner. Replaces the "move toward the one intercept"
    step with "pick WHICH intercept to aim for using safe_guess + Q."

    The actual movement toward the chosen intercept (and the timing logic)
    stays in the existing agent -- QPlanner only decides the TARGET height.

    Usage in the agent loop:
        planner.begin_decision(state, my_hist, ...)   # when ball incoming
        target_y = planner.choose_intercept(...)       # pick target height
        ... agent moves paddle toward target_y ...
        planner.notify_point(reward)                   # when a point resolves
    """

    # candidate intercept heights to consider (normalized 0..1)
    # spread across the paddle-reachable range; finer near center where it
    # matters, but a uniform grid is fine for v1.
    N_CANDIDATES = 9

    def __init__(self, n_features=6, explore_kappa=0.3):
        self.safe = SafeGuess()
        self.q = QModel(n_features=n_features, explore_kappa=explore_kappa)
        # remember the last decision so we can train Q when the point resolves
        self._pending = None   # dict: {features, safe_win_prob, q_was_acting}

    def candidate_heights(self, my_y, reachable_band=0.4):
        """Return a grid of intercept heights to consider. Bounded by what
        the paddle can reach (a band around current my_y)."""
        lo = max(0.05, my_y - reachable_band / 2)
        hi = min(0.95, my_y + reachable_band / 2)
        return np.linspace(lo, hi, self.N_CANDIDATES)

    def choose_intercept(self, state, ball_vy_norm, my_y, reachable_band=0.4):
        """Pick the best intercept height. Returns (target_y, info dict)."""
        candidates = self.candidate_heights(my_y, reachable_band)
        best_y = None
        best_score = -1e9
        best_info = None
        for y in candidates:
            est = self.safe.estimate(state, y, ball_vy_norm)
            s = self.q.score(est['features'], est['win_prob'])
            if s > best_score:
                best_score = s
                best_y = float(y)
                best_info = est
        # remember this decision for training when the point resolves
        self._pending = {
            'features': best_info['features'].copy(),
            'safe_win_prob': best_info['win_prob'],
            'q_was_acting': self.q.gate_open,
            'intercept_y': best_y,
        }
        return best_y, best_info

    def notify_point(self, reward):
        """A point resolved. Train Q on the last decision and update the
        trust gate. reward = +1 (won) / -1 (lost)."""
        if self._pending is None:
            return
        self.q.update(self._pending['features'],
                      float(reward),
                      self._pending['safe_win_prob'])
        self.q.record_point(float(reward), self._pending['q_was_acting'])
        self._pending = None

    # diagnostic access (for logging)
    def diagnostics(self):
        return {
            'gate_open': self.q.gate_open,
            'q_updates': self.q.rls.n_updates,
            'opp_speed': self.safe.opp_speed,
            'pending': self._pending is not None,
        }
