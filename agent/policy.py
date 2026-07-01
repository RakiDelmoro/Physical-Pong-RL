"""
The POLICY (Alberta Plan Step 4): the decision layer that turns perception +
the world model into an ACTION emitted to the servo.

DESIGN -- model-based planning over the imagination we just built (the direct
payoff of W3 / Way C), with a closed-form safe prior as the fallback:

  SCAFFOLD (the SAFE PRIOR, never learns, closed-form physics):
    "move my paddle toward where the ball will be when it reaches my x-line."
    A sensible reactive controller that needs no learning -- the worst-case
    fallback. GENERAL: 'move the thing I control toward where the fast free
    object will arrive at my plane' -- 'controlled' and 'fast free' are both
    DISCOVERED by the world model (controlled_track + kind-speed), not
    hardcoded Pong vocabulary.

  SURFACE (the PLANNER, uses the imagination directly):
    For each candidate action, ROLL OUT the imagined future (the world
    model's rollout = exact free-flight physics + the computed Way C bounce)
    and sum the predicted reward over the horizon. Pick the action with the
    best imagined total. This is "try each option in my head, pick the one
    whose imagined result is best" -- the direct payoff of the multi-frame
    imagination. No learned policy -> no bias-trap (the agent cannot
    self-induce a degenerate data slice, because it doesn't learn a policy
    from its own actions; it imagines every frame from a model trained on the
    diverse real stream).

  TRUST (measure, then trust):
    The planner only ACTS once it has EMPIRICALLY BEATEN the bare safe prior
    over a recent window of points (fewer opponent points let through). Until
    then, act on the safe prior alone -- worst case = the reflex controller,
    and the experience stays diverse. Same safety philosophy as the plan's
    trust-gate, simpler because there is no learned policy to corrupt.

Interface:
  policy = Policy(world_model, num_actions=3, neutral=2)
  action = policy.act(tracks, controlled_id)            # one decision / frame
  policy.observe(reward_delta)                          # feed the trust gate

The policy emits an action INDEX (0/1/2); the host bridge maps that to a
servo angle. It never touches hardware.
"""

import numpy as np


class Policy:
    """Model-based planner over the imagination, with a safe-prior fallback
    and a measure-then-trust gate."""

    # ---- planner config ----
    HORIZON = 25            # imagined frames ahead per candidate action
    # ACTION COMMIT WINDOW (MPC semantics): the candidate action is held
    # fixed for the WHOLE imagined horizon -- 'what happens if I KEEP doing
    # this?' The planner picks the sustained action whose imagined future is
    # safest, executes it for ONE real step, then re-plans next frame. This
    # is what lets the planner DISCRIMINATE: if the reflex takes over mid-
    # rollout, every candidate converges to the SAME (reflex) trajectory and
    # the action's effect washes out -- measured directly, commit=4 gave 0
    # discriminating frames; commit=HORIZON gave 10/90. A short commit only
    # sees the action's one-frame twitch before the reflex coasts; the full
    # commit sees a sustained strategy. (The reflex is still the fallback
    # DECIDER when the planner is untrusted; this is only the imagined
    # rollout's inner assumption.)
    COMMIT_WINDOW = HORIZON
    # The imagined RETURN discounts later frames UP (gamma > 1): the action
    # barely affects the early imagined frames (the ball is far from my
    # paddle) and only matters when the ball reaches my plane near the end of
    # the horizon. A flat sum is dominated by the action-independent base rate
    # (the 'points happen at rate r/frame' prior); up-weighting later frames
    # amplifies the action-dependent part so the planner chooses on signal,
    # not noise. GENERAL: 'care most about the part of the future my action
    # can actually change.'
    RETURN_GAMMA = 1.08     # per-frame up-weight

    # ---- trust-gate config ----
    TRUST_WARMUP = 150      # frames before the gate is considered
    TRUST_WINDOW = 80       # the recent-points window we compare over
    TRUST_MARGIN = 0.3      # residual must beat prior by this opponent-points/
                            # frame in the window to take control
    # EXPLORE FRACTION (breaks the trust-gate chicken-and-egg). The old gate
    # never let the planner act until it was trusted, but couldn't trust it
    # until it had acted -> n_plan stayed 0 -> never promoted. So while
    # UNTRUSTED, let the planner act on a FRACTION of frames (exploration) to
    # gather evidence, and use the safe prior the rest (safety). Those
    # explored frames' outcomes populate _plan_bad; once the planner's
    # opponent-points-per-frame beats the prior's by TRUST_MARGIN, the gate
    # promotes and the planner takes over fully. Classic explore/exploit in a
    # trust gate. GENERAL: 'let an unproven strategy try sometimes so you can
    # measure it, but mostly play safe.'
    TRUST_EXPLORE_FRAC = 0.5

    def __init__(self, world_model, num_actions=3, neutral=None):
        self.wm = world_model
        self.num_actions = int(num_actions)
        self.neutral = int(neutral) if neutral is not None else num_actions - 1
        # trust-gate bookkeeping: opponent-points-per-frame under each
        # provenance (prior vs planner), over the recent window.
        self._prior_bad = []     # list of per-frame opponent-point counts
        self._plan_bad = []
        self._trusted = False
        self._frame = 0
        # RNG for the explore fraction (which frames the planner tries on
        # while untrusted). Unseeded -> robust to *which* frames; the gate
        # compares rates over a window, not specific frames.
        self._rng = np.random.default_rng()
        # the last decision's provenance, for observe() to credit correctly
        self._last_was_planner = False

    # ==================================================================
    # THE SAFE PRIOR -- closed-form, never learns
    # ==================================================================
    def _safe_prior(self, state, controlled_slot, ball_slot):
        """Move my paddle toward where the ball will be when it reaches my
        x-plane. Returns an action index. GENERAL: 'move the controlled
        object toward the fast free object's arrival point at my plane.'"""
        if controlled_slot is None or ball_slot is None:
            return self.neutral
        cj = controlled_slot * self.wm.STATE_PER_OBJ
        bj = ball_slot * self.wm.STATE_PER_OBJ
        my_x, my_y = state[cj], state[cj + 1]
        bx, by, bvx, bvy = state[bj], state[bj + 1], state[bj + 2], state[bj + 3]
        if abs(bvx) < 1e-6:
            return self.neutral
        toward_me = (bvx < 0 and bx > my_x) or (bvx > 0 and bx < my_x)
        if not toward_me:
            return self.neutral
        t = (my_x - bx) / bvx
        if t <= 0 or t > 200:
            return self.neutral
        arrive_y = float(np.clip(by + bvy * t, 0.0, 1.0))
        dy = arrive_y - my_y
        if abs(dy) < 0.02:
            return self.neutral
        # action 0 = up (decrease y), 1 = down (increase y) under the sim
        # convention; the SIGN is discovered from the world model's
        # controlled-slot action/y correlation (not hardcoded).
        sign = self._action_sign()
        return 0 if dy * sign < 0 else 1

    def _action_sign(self):
        """Sign of the action's effect on the controlled slot's y: +1 if
        action 'up' (scalar +1) decreases y, -1 if it increases y. DISCOVERED
        from the world model's controlled correlation; defaults to the sim
        convention (+1) until discovered."""
        corr = getattr(self.wm, "_last_controlled_corr", None)
        if corr is None:
            return 1.0
        return 1.0 if corr >= 0 else -1.0

    # ==================================================================
    # THE PLANNER -- imagine each action, pick the best imagined future
    # ==================================================================
    # The planner discriminates actions by a DIRECT state signal, not the
    # predicted reward scalar. The reward predictor was trained on one-step
    # REAL transitions; over a long IMAGINED rollout it is off-distribution
    # and noisy, so summing it gave a wobbly score and the planner picked
    # wrong. Instead: read the imagined movie directly -- 'how close did the
    # ball get to my wall?' Higher = safer (if my paddle intercepted, the Way
    # C reflection bounced it back, so min-x stays at my paddle plane; if it
    # got past me, min-x -> 0). GENERAL: 'the action is good if the imagined
    # future keeps the threat furthest from my goal line' -- no reward
    # predictor, no Pong vocab.
    SAFETY_MARGIN = 0.01   # only diverge from the prior if meaningfully safer

    def _imagined_return(self, state, first_action, controlled_id):
        """Roll out the imagined future under `first_action` (then the safe
        prior for subsequent imagined steps) and return a SAFETY score: the
        MINIMUM ball-x reached over the horizon (higher = the ball stayed
        further from my wall = safer). The ball = the fast free slot at the
        rollout's start (slot identity is stable through the rollout -- pure
        kinematics, no reassignment). If the ball isn't in the state, return
        a neutral 0.0 so the prior wins the tie (no opinion)."""
        controlled = self.wm._controlled_slot
        ball_slot = self._ball_slot(state, controlled)
        if ball_slot is None:
            return 0.0   # no ball in the state -> no opinion, prior wins
        # COMMIT the candidate action for the first COMMIT_WINDOW imagined
        # steps (a sustained strategy), THEN let the safe prior take over.
        # action_fn(s, t) chooses the action for step t+1 (step 0 already used
        # `first_action`), so we hold the candidate while t+1 < COMMIT_WINDOW.
        candidate = int(first_action)
        commit = int(self.COMMIT_WINDOW)
        def action_fn(s, t):
            if t + 1 < commit:
                return candidate
            return self._safe_prior(s, self.wm._controlled_slot,
                                    self._ball_slot(s, self.wm._controlled_slot))
        states, _, _, _ = self.wm.rollout_from_state(
            state, first_action=first_action, action_fn=action_fn,
            horizon=self.HORIZON, controlled_id=controlled_id)
        bj = ball_slot * self.wm.STATE_PER_OBJ
        # the closest the ball got to my wall (x=0) over the imagined future;
        # higher = safer. Track the ball slot's x (slot identity is stable
        # through the rollout).
        return min(float(s[bj]) for s in states)

    def _planner_action(self, state, controlled_id):
        """Pick the action with the best (highest) imagined SAFETY score. The
        SAFE PRIOR's own imagined score is the BAR TO BEAT -- a candidate
        action only takes over if it is MEANINGFULLY safer (by SAFETY_MARGIN).
        Ties and near-ties keep the prior (the planner is conservative; this
        stops it thrashing to an arbitrary action when the imagination can't
        discriminate -- e.g. the ball is not threatening this horizon)."""
        prior_a = self._safe_prior(
            state, self.wm._controlled_slot,
            self._ball_slot(state, self.wm._controlled_slot))
        best_a = prior_a
        best_r = self._imagined_return(state, prior_a, controlled_id)
        for a in range(self.num_actions):
            if a == prior_a:
                continue
            r = self._imagined_return(state, a, controlled_id)
            if r > best_r + self.SAFETY_MARGIN:
                best_r, best_a = r, a
        return best_a

    def _ball_slot(self, state, controlled_slot):
        """The fast free-moving slot (the ball), excluding the controlled
        slot. DISCOVERED: the slot with the largest |velocity| that isn't
        controlled. Falls back to None if nothing moves."""
        best, best_spd = None, -1.0
        for s in range(self.wm.MAX_OBJECTS):
            if s == controlled_slot:
                continue
            j = s * self.wm.STATE_PER_OBJ
            spd = float(np.hypot(state[j + 2], state[j + 3]))
            if spd > best_spd:
                best_spd, best = spd, s
        return best if best_spd > 1e-3 else None

    # ==================================================================
    # THE PUBLIC DECISION
    # ==================================================================
    def act(self, tracks, controlled_id):
        """One decision per frame. Returns an action index (0/1/2)."""
        state = self.wm.state_from_tracks(tracks, controlled_id)
        prior_a = self._safe_prior(
            state, self.wm._controlled_slot,
            self._ball_slot(state, self.wm._controlled_slot))
        if self._trusted:
            chosen = self._planner_action(state, controlled_id)
            self._last_was_planner = (chosen != prior_a)
        else:
            # UNTRUSTED: explore on a fraction of frames (let the planner act
            # so the gate can measure it), else play safe with the prior.
            explore = float(self._rng.random()) < self.TRUST_EXPLORE_FRAC
            if explore:
                chosen = self._planner_action(state, controlled_id)
                self._last_was_planner = (chosen != prior_a)
            else:
                chosen = prior_a
                self._last_was_planner = False
        self._frame += 1
        return int(chosen)

    # ==================================================================
    # LEARNING FROM THE OUTCOME -- feeds the trust gate
    # ==================================================================
    def observe(self, reward_delta):
        """Credit this frame's outcome to whichever provenance acted, and
        update the trust flag. The planner itself learns NOTHING (it imagines
        every frame) -- this is only the trust-gate accounting."""
        badness = max(0.0, -float(reward_delta))   # opponent points = bad for us
        if self._last_was_planner:
            self._plan_bad.append(badness)
        else:
            self._prior_bad.append(badness)
        # keep only the recent window
        if len(self._prior_bad) > self.TRUST_WINDOW:
            self._prior_bad = self._prior_bad[-self.TRUST_WINDOW:]
        if len(self._plan_bad) > self.TRUST_WINDOW:
            self._plan_bad = self._plan_bad[-self.TRUST_WINDOW:]
        self._update_trust()

    def _update_trust(self):
        """The planner is trusted iff enough frames have passed AND its
        opponent-points-per-frame is lower than the prior's by TRUST_MARGIN.
        Once trusted, it stays trusted unless it starts doing much worse."""
        if self._frame < self.TRUST_WARMUP:
            return
        if len(self._plan_bad) < 20 or len(self._prior_bad) < 20:
            return
        prior_rate = float(np.mean(self._prior_bad))
        plan_rate = float(np.mean(self._plan_bad))
        if not self._trusted:
            if plan_rate < prior_rate - self.TRUST_MARGIN:
                self._trusted = True
        else:
            if plan_rate > prior_rate + self.TRUST_MARGIN:
                self._trusted = False

    def diagnostics(self):
        def _rate(xs):
            return float(np.mean(xs)) if xs else None
        return {
            "trusted": self._trusted,
            "frame": self._frame,
            "n_prior": len(self._prior_bad),
            "n_plan": len(self._plan_bad),
            "prior_opp_pts_per_frame": _rate(self._prior_bad),
            "plan_opp_pts_per_frame": _rate(self._plan_bad),
        }
