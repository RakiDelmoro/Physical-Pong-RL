"""
KARC Agent -- the real-time brain for physical Pong.

Pure online from frame 1. No neural network, no GPU, no replay buffer, no
target network. Two closed-form models (AutonomousBallKARC + ServoKARC)
updated every frame by RLS.

=== The planner (Option 3 redesign): physics direction + Model B timing ===

The original planner asked Model B "which action should I take?" -- and fell
into a closed-loop bias trap: the planner kept picking UP, the paddle pinned
at the top wall, Model B only ever saw "UP does nothing from the top", so it
learned a true-but-useless lesson and the trap reinforced itself.

The redesign splits the decision into two parts that use DIFFERENT sources:

  DIRECTION (physics rule -- UNCORRUPTIBLE, never depends on Model B):
      "Is the ball going to be above or below me at the intercept?"
          above -> UP, below -> DOWN, aligned -> STAY
      This cannot be trapped because it has no learned weights. It is the
      90% of Pong strategy that's just "move toward the ball."

  TIMING (Model B -- the genuine KARC value: latency compensation):
      "If I start moving now, will my paddle ARRIVE at the intercept at the
       right time?" Model B predicts the paddle's trajectory under the chosen
       direction. We estimate the arrival frame and compare it to the ball's
       intercept frame:
          arrive too early (overshoot risk) -> STAY and wait
          arrive around the right time      -> MOVE now
          can't make it in time             -> MOVE now (best effort)
      This is exactly the forward-prediction / latency-compensation role
      KARC was built for. And timing is far less bias-prone than direction,
      because it's about *when*, not *where*.

  If Model B is untrained/unreliable (can't distinguish actions), timing
  collapses to "just move toward the ball now" -- the safe physics default.

=== Periodic re-probing (Option 1): keep Model B's data diverse ===

Every REPROBE_PERIOD frames in PLAN phase we run a short forced
STAY->UP->STAY->DOWN cycle. This injects balanced paddle data (UP and DOWN
from a neutral position) so Model B keeps relearning the true action effects
even while the planner runs. It directly fights the bias trap: even if the
planner spends most frames driving the paddle one way, the re-probes remind
Model B what each button does from the middle.

=== The 3 phases ===

Phase 1 -- WATCH (1..PHASE1_END): servo PARKED, train Model A only.
Phase 2 -- PROBE (PHASE1_END..PHASE2_END): gentle cycling, train both models.
Phase 3 -- PLAN (PHASE2_END+): physics+timing planner + periodic re-probes,
           both models keep training every frame.

=== Robustness guards (unchanged) ===

  GATE 1: screen_conf < SCREEN_CONF_MIN -> FREEZE (no training, park).
  GATE 2: ball lost >= BALL_LOSS_LIMIT frames -> park, wait for ball.
  GATE 3: train each model only on frames its element was actually seen.

Interface matches the old Agent (observe/act/store/maybe_learn/save/load).
"""

import os
import numpy as np
import cv2

from state_extract import StateExtractor
from karc import AutonomousBallKARC, ServoKARC
from safety import SafetyFilter, UP, DOWN, STAY
from q_planner import QPlanner

NUM_ACTIONS = 3

# Phase boundaries (in agent steps = frames)
PHASE1_END = 100      # until here: WATCH
PHASE2_END = 300      # until here: PROBE
PLAN_HORIZON = 6      # frames ahead the ball/paddle predictor looks

# Robustness thresholds
SCREEN_CONF_MIN = 0.9   # below this the screen detector is unreliable -> FREEZE
BALL_LOSS_LIMIT = 8     # ball lost this many frames in a row -> STOP planning

# Planner thresholds (normalized 0..1 coordinates)
ALIGN_TOL      = 0.04   # |my_y - intercept_y| below this -> "aligned" -> STAY
ARRIVE_TOL     = 0.04   # paddle within this of intercept_y counts as "arrived"
EARLY_MARGIN   = 2      # if paddle would arrive >EARLY_MARGIN frames before
                        # the ball, wait (STAY) to avoid overshoot

# Periodic re-probing (Option 1) -- fights Model B bias during PLAN phase
REPROBE_PERIOD = 400    # every N PLAN-phase frames, inject a re-probe cycle
REPROBE_HOLD   = 20     # frames to hold each step of the re-probe (STAY/UP/STAY/DOWN)


class KARCAgent:
    def __init__(self, delay=4, basis="chebyshev",
                 plan_horizon=PLAN_HORIZON, dwell=5, conf_thresh=0.4,
                 device="cpu"):
        self.num_actions = NUM_ACTIONS
        self.extractor = StateExtractor()
        self.ball = AutonomousBallKARC(delay=delay, basis=basis)
        self.servo = ServoKARC(delay=delay, num_actions=NUM_ACTIONS,
                               basis=basis)
        self.safety = SafetyFilter(dwell=dwell, conf_thresh=conf_thresh)
        self.qplanner = QPlanner()
        self.plan_horizon = plan_horizon
        self.steps = 0
        # rolling history of states and my_y for the planner
        self._state_hist = []      # list of 4-states
        self._my_hist = []         # list of my_y scalars
        self._last_action = STAY
        self._last_state = None
        self._probe_counter = 0    # phase-2 probe scheduler
        # robustness bookkeeping
        self._ball_loss_streak = 0
        self._frozen = False
        self._freeze_reason = ""
        self.last_found = {'ball': False, 'my': False, 'opp': False}
        self.screen_conf = 1.0
        # planner diagnostics (for the log)
        self._last_intercept_y = 0.5
        self._last_my_y = 0.5
        self._last_intercept_t = -1     # frames until ball reaches my side
        self._last_arrive_t = -1        # frames until paddle reaches intercept
        self._last_decision = "init"    # "physics" | "timing-wait" | "timing-move" | "reprobe" | ...
        self._last_q_winprob = 0.0     # safe-guess win prob of the chosen intercept
        self._last_q_gate = False      # was Q's gate open for the last decision?
        # re-probe scheduler
        self._plan_frames = 0           # frames spent in PLAN phase
        self._reprobe_active = False
        self._reprobe_step = 0

    # ------------------------------------------------------------------ #
    # observe(rgb, screen_conf, reward) -> state                         #
    # ------------------------------------------------------------------ #
    def observe(self, rgb, screen_conf=1.0, reward=0):
        self.screen_conf = screen_conf

        # --- GATE 1: screen detector unreliable -> FREEZE everything ---
        if screen_conf < SCREEN_CONF_MIN:
            self._frozen = True
            self._freeze_reason = (f"screen_conf={screen_conf:.2f}"
                                   f"<{SCREEN_CONF_MIN}")
            if self._last_state is None:
                self._last_state = np.array([0.5, 0.5, 0.5, 0.5])
            return self._last_state.copy()

        was_frozen = self._frozen
        self._frozen = False
        self._freeze_reason = ""

        # --- POINT SCORED -> ball reset to center by firmware ---
        if reward != 0:
            self.extractor.reset_ball(center=0.5)
            # tell Q a point resolved so it can train on the last decision
            self.qplanner.notify_point(reward)
            self._ball_loss_streak = 0

        state, found = self.extractor.extract(rgb)
        self.last_found = found

        if found['ball']:
            self._ball_loss_streak = 0
        else:
            self._ball_loss_streak += 1

        if not was_frozen and found['ball']:
            self._state_hist.append(state)
            if len(self._state_hist) > self.ball.delay + 2:
                self._state_hist.pop(0)
            self._my_hist.append(float(state[2]))
            if len(self._my_hist) > self.servo.delay + 2:
                self._my_hist.pop(0)
        elif found['my'] and not was_frozen:
            self._my_hist.append(float(state[2]))
            if len(self._my_hist) > self.servo.delay + 2:
                self._my_hist.pop(0)

        # --- Train Model A: only when the ball was actually seen ---
        if found['ball']:
            self.ball.observe_and_train(state)

        # --- Train Model B: only when my paddle was actually seen ---
        if (found['my'] and self._last_state is not None
                and len(self._my_hist) >= 2):
            prev_my = self._my_hist[-2]
            cur_my = self._my_hist[-1]
            self.servo.observe_and_train(prev_my, self._last_action, cur_my)

        self._last_state = state.copy()
        return state

    def state_for_inference(self):
        return self._last_state

    # ------------------------------------------------------------------ #
    # act(state, train) -> action int                                    #
    # ------------------------------------------------------------------ #
    def act(self, state, train=True):
        self.steps += 1

        # --- GATE 1: frozen -> park safely ---
        if self._frozen:
            action = self.safety.filter(STAY, confidence=0.0)
            self._last_action = action
            self._last_decision = "frozen"
            return action

        # --- GATE 2: ball lost too long -> park safely ---
        if self._ball_loss_streak >= BALL_LOSS_LIMIT:
            action = self.safety.filter(STAY, confidence=0.0)
            self._last_action = action
            self._last_decision = "ball-lost"
            return action

        phase = self._phase()
        if phase == 1:
            desired = STAY
            self._last_decision = "watch"
        elif phase == 2:
            desired = self._probe_action()
            self._last_decision = "probe"
        else:
            self._plan_frames += 1
            # --- Option 1: periodic re-probe to keep Model B diverse ---
            if self._reprobe_due():
                desired = self._reprobe_action()
                self._last_decision = "reprobe"
            else:
                desired = self._physics_timing_plan()

        action = self.safety.filter(desired, confidence=1.0)
        self._last_action = action
        return action

    # The remaining methods are interface-compat no-ops / save-load.
    def store(self, *a, **k):
        pass

    def maybe_learn(self):
        return None

    def save(self, path):
        # np.savez auto-appends .npz; save/load must agree on the SAME path.
        # We canonicalize to a .npz path so load() finds it exactly.
        if not path.endswith(".npz"):
            path = path + ".npz"
        np.savez(path,
                 ball_W=self.ball.rls.W,
                 servo_W=self.servo.rls.W,
                 q_W=self.qplanner.q.rls.W,
                 q_P=self.qplanner.q.rls.P,
                 q_baseline_rewards=self.qplanner.q._baseline_rewards,
                 q_q_rewards=self.qplanner.q._q_rewards,
                 steps=self.steps)

    def load(self, path):
        if not path.endswith(".npz"):
            path = path + ".npz"
        if not os.path.isfile(path):
            print(f"[karc] no checkpoint at {path}, starting fresh")
            return
        z = np.load(path)
        self.ball.rls.W = z["ball_W"]
        self.servo.rls.W = z["servo_W"]
        if "q_W" in z.files:
            self.qplanner.q.rls.W = z["q_W"]
            self.qplanner.q.rls.P = z["q_P"]
            self.qplanner.q._baseline_rewards = list(z["q_baseline_rewards"])
            self.qplanner.q._q_rewards = list(z["q_q_rewards"])
            # recompute gate from the loaded reward history
            self.qplanner.q._reeval_gate()
            print(f"[karc] loaded Q: {int(self.qplanner.q.rls.n_updates)} updates, "
                  f"gate {'OPEN' if self.qplanner.q.gate_open else 'closed'}")
        self.steps = int(z["steps"])
        print(f"[karc] loaded checkpoint at step {self.steps}")

    # ------------------------------------------------------------------ #
    # Internals: phases & probing                                        #
    # ------------------------------------------------------------------ #
    def _phase(self):
        if self.steps < PHASE1_END:
            return 1
        if self.steps < PHASE2_END:
            return 2
        return 3

    def _probe_action(self):
        """Phase 2: slow gentle cycling STAY -> UP -> STAY -> DOWN."""
        hold = self.safety.dwell * 4
        seq = [STAY, UP, STAY, DOWN]
        desired = seq[(self._probe_counter // hold) % len(seq)]
        self._probe_counter += 1
        return desired

    # ------------------------------------------------------------------ #
    # Option 1: periodic re-probing during PLAN phase                    #
    # ------------------------------------------------------------------ #
    def _reprobe_due(self):
        """True if we should be running a re-probe cycle this frame."""
        # Start a re-probe every REPROBE_PERIOD plan-frames, unless one is
        # already running.
        if self._reprobe_active:
            return True
        if self._plan_frames % REPROBE_PERIOD == 0 and self._plan_frames > 0:
            self._reprobe_active = True
            self._reprobe_step = 0
            return True
        return False

    def _reprobe_action(self):
        """A short STAY -> UP -> STAY -> DOWN cycle held REPROBE_HOLD each.
        Injects balanced paddle data from a roughly-neutral position so Model
        B keeps relearning the true action effects (fights the bias trap)."""
        seq = [STAY, UP, STAY, DOWN]
        idx = self._reprobe_step // REPROBE_HOLD
        desired = seq[idx % len(seq)]
        self._reprobe_step += 1
        # end the re-probe after one full cycle
        if self._reprobe_step >= REPROBE_HOLD * len(seq):
            self._reprobe_active = False
            self._reprobe_step = 0
        return desired

    # ------------------------------------------------------------------ #
    # Q planner: pick WHICH intercept to aim for (Q), then move toward it #
    # ------------------------------------------------------------------ #
    def _physics_timing_plan(self):
        """The Q-augmented planner.

        1. Predict the ball forward with Model A; find WHEN it reaches my
           side (intercept_t) and the rough intercept_y.
        2. Q picks WHICH height to aim for: among reachable intercept heights,
           score each with safe_guess + gate*Q + exploration, pick the best.
           (Q is trust-gated; while untrained it just uses the safe guess.)
        3. TIMING (unchanged): use Model B to decide MOVE-now vs WAIT so we
           arrive at the chosen height at the right time.
        """
        # need enough history + a visible ball to plan
        if len(self._state_hist) < self.ball.delay or not self.last_found['ball']:
            self._last_decision = "no-history"
            return STAY

        hist = self._state_hist[-self.ball.delay:]
        my_hist = self._my_hist[-self.servo.delay:]
        my_now = my_hist[-1] if my_hist else 0.5

        # --- 1. predict the ball forward, find intercept timing ---
        ball_pred = self.ball.rollout(hist, self.plan_horizon)
        # ball_pred: (H, 3) = [bx, by, opp_y] per future step
        intercept_y = None
        intercept_t = None
        for i, p in enumerate(ball_pred):
            bx = p[0]
            if bx <= 0.08:
                intercept_y = p[1]
                intercept_t = i + 1
                break
        if intercept_y is None:
            intercept_y = ball_pred[-1][1]
            intercept_t = self.plan_horizon

        # ball vertical velocity at contact, reconstructed from recent history
        # (Model A predicts positions; we need vy for the landing-point math).
        # Use the last two known ball_y values; normalized per frame.
        if len(self._state_hist) >= 2:
            ball_vy_norm = (self._state_hist[-1][1] - self._state_hist[-2][1])
        else:
            ball_vy_norm = 0.0

        # --- 2. Q picks which height to aim for ---
        # QPlanner scores reachable candidate heights and returns the best.
        state_now = self._last_state.copy()
        target_y, est = self.qplanner.choose_intercept(
            state_now, ball_vy_norm, my_now)

        # diagnostics
        self._last_intercept_y = float(target_y)
        self._last_my_y = float(my_now)
        self._last_intercept_t = int(intercept_t)
        self._last_q_winprob = float(est['win_prob'])
        self._last_q_gate = self.qplanner.q.gate_open

        # --- 3. DIRECTION toward the Q-chosen target (physics, uncorruptible) ---
        dy = target_y - my_now
        if dy < -ALIGN_TOL:
            direction = UP
        elif dy > ALIGN_TOL:
            direction = DOWN
        else:
            # already aligned with the target -> hold position
            self._last_decision = "aligned"
            self._last_arrive_t = 0
            return STAY

        # --- 4. TIMING (Model B): will I arrive at target_y in time? ---
        pred_paddle = self.servo.rollout_action(my_hist, direction,
                                                self.plan_horizon)
        arrive_t = None
        for i, py in enumerate(pred_paddle):
            if abs(py - target_y) <= ARRIVE_TOL:
                arrive_t = i + 1
                break
        self._last_arrive_t = int(arrive_t) if arrive_t is not None else -1

        stay_pred = self.servo.predict_next(my_hist, STAY)
        dir_pred = pred_paddle[0] if len(pred_paddle) else my_now
        model_b_alive = abs(dir_pred - stay_pred) > 0.005

        if not model_b_alive or arrive_t is None:
            self._last_decision = "physics"
            return direction

        if arrive_t <= intercept_t:
            if intercept_t - arrive_t > EARLY_MARGIN:
                self._last_decision = "timing-wait"
                return STAY
            else:
                self._last_decision = "timing-move"
                return direction
        else:
            self._last_decision = "timing-late"
            return direction
