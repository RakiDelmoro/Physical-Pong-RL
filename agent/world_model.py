"""
The WORLD MODEL -- `M(state, action) -> (next_state, reward)`, as a
DYNAMIC PREDICTIVE CODING (DPC) model.

Alberta Plan Step 8 (Prototype-AI I). The keystone that unblocks planning,
the Horde, and control. Perception answers "what's happening now?"; the
world model answers "what will happen next, and will I score?"

THE DESIGN PHILOSOPHY (the same one the whole agent follows):

  DESIGN THE SCAFFOLD (universal truth, never trains); LEARN THE SURFACE
  (the specifics that adapt). Perception does this ("objects persist + move
  smoothly" is the scaffold; "which object is the ball" is learned). The
  world model does the SAME: the scaffold is the universal kinematics
  "motion = position + velocity; velocity persists"; the learned surface is
  the CORRECTION to that scaffold (bounces, paddle hits).

ARCHITECTURE -- adaptive residual + a DPC hierarchy:

  predicted next state = PHYSICS DEFAULT  +  LEARNED CORRECTION
                         (scaffold, exact)   (surface, learned)

  PHYSICS DEFAULT (hardcoded structure, ADAPTIVE parameters):
      next_pos = cur_pos + cur_vel        # kinematics: position advances by
      next_vel = cur_vel                  # velocity; velocity persists.
    Universal math -- true in any world with moving objects, not a Pong
    assumption. The PARAMETERS (the vel values) are the OBSERVED velocities
    in the state, which update every frame from perception. Applied EXACTLY,
    so the boring 'ball flies straight' part of a rollout has ZERO compounding
    error; the model's entire capacity goes to the bounces (the 10% that is
    actually hard).

  LEARNED CORRECTION -- a hierarchy of neural nets trained ONLINE by
  backpropagation on the prediction error (Dynamic Predictive Coding, Jiang &
  Rao 2024 -- the minimal slice needed for our problem):

    LOWER LEVEL: a small MLP takes (state, action) -> a state_dim CORRECTION
      to the physics default. This is the one-step predictor (handles the
      boring 90% + the smooth part of surprises).

    HIGHER LEVEL: a slow recurrent net takes the last several states (longer
      timescale) -> a small REGIME vector that MODULATES the lower level's
      hidden representation. This is the DPC trick: a slower brain reads the
      APPROACH pattern ("the ball has been heading toward the paddle for
      several frames -> a bounce is coming") and tells the lower level to
      expect it -- where a one-step model smears the rare bounce.

  REWARD HEAD: a third little output on the lower level -> predicted reward.

WHY A NEURAL NET + DPC (the thing that was broken before):

  The prior model used a frozen Fourier basis + a linear RLS readout. It
  learned the bounce DIRECTION but not its sharp MAGNITUDE -- a smooth basis
  smears a sudden velocity flip into a weak deceleration (~55% of the needed
  flip). Four attempts to fix it (a discrete-event detector, a least-squares
  velocity, a DPC-higher-level on top of RLS, per-weight meta-learned step
  sizes) all hit the SAME wall: the bounce is a TEMPORALLY-SPARSE rare event
  (1-2 frames in a sea of ~zeros), and any gate/learner trained FRAME-BY-FRAME
  against that ~zero target collapses and misses the rare event.

  The DPC higher level on top of RLS got CLOSEST (its correction hit -0.013
  vs the needed -0.0138) but its GATE -- trained frame-by-frame -- couldn't
  learn when to fire. Real DPC fixes exactly that: the modulation/gate is
  learned END-TO-END by backprop over the WHOLE sequence, optimizing "did the
  imagined future come true," not a per-frame mostly-zero target. That is the
  specific tool for the specific wall we kept hitting.

ON-GOAL (Alberta Plan):

  Online, continual, experience-only -- trained by prediction error every
  frame, no separate training phase. No Pong knowledge: the hierarchy +
  prediction-error learning + the physics-default scaffold are general; the
  correction patterns, the regime, and the modulation are all learned from
  data. "Scale with computation -- search and learning over human insight"
  -- neural nets are exactly that, which the plan explicitly favors.

Servo split: the reward + ball/opponent dynamics + bounces are servo-free
(reward comes from the SenseCAP). The my-paddle action channel fills in when
the servo arrives -- same model, same training, no rewrite.
"""

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Frozen nonlinear basis -- random Fourier features (never trains).
# Kept as the INPUT front for the lower level: a fixed nonlinear map of the
# (state, action) input that the net sits on top of. This gives the net a
# rich, fixed nonlinear expansion to read from (a good inductive bias for a
# small online learner) while the net's weights carry the learned part.
# =====================================================================

class RandomFourierFeatures:
    def __init__(self, in_dim, n_features=1000, gamma=0.5, seed=0):
        self.in_dim = int(in_dim)
        self.M = int(n_features)
        self.gamma = float(gamma)
        rng = np.random.default_rng(seed)
        self.k = rng.normal(0.0, self.gamma, size=(self.M, self.in_dim))
        self._scale = np.sqrt(2.0 / self.M)
        self.out_dim = 1 + 2 * self.M

    def features(self, x):
        x = np.asarray(x, dtype=np.float64)
        proj = self.k @ x
        return np.concatenate([[1.0],
                               self._scale * np.cos(2.0 * np.pi * proj),
                               self._scale * np.sin(2.0 * np.pi * proj)])


# =====================================================================
# The lower level -- an MLP that predicts the RESIDUAL correction + reward.
# =====================================================================

class _LowerNet(nn.Module):
    """Lower level: (RFF(state,action) features) -> (state_dim correction,
    1 reward). A small MLP. Sits on the frozen Fourier basis (fixed nonlinear
    input expansion) so the net can stay small and train fast online."""

    def __init__(self, in_dim, state_dim, hidden=128, seed=0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.state_head = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)
        # The correction is BOUNDED: tanh * CORR_SCALE caps the net's
        # contribution so it can never overpower the physics default and
        # diverge over a multi-step rollout. The net learns the SURPRISE only
        # (bounces are ~0.01-0.03 in normalized units); a cap of 0.05 lets a
        # full bounce through while preventing the unbounded growth that an
        # unconstrained net shows when fed its own predictions (out-of-
        # distribution during rollout). Scaffold = the bound (a correction is
        # small by definition); surface = the learned direction/magnitude.
        self.CORR_SCALE = 0.05
        # small init so the net starts near zero correction (the physics
        # default is right by construction at start; the net only learns the
        # surprise). Seed the init for reproducibility.
        torch.manual_seed(seed)
        with torch.no_grad():
            for m in (self.state_head, self.reward_head):
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x):
        h = self.body(x)
        return self.state_head(h), self.reward_head(h).squeeze(-1)

    def correction(self, h):
        """The BOUNDED state correction from a hidden vector."""
        return torch.tanh(self.state_head(h)) * self.CORR_SCALE


# =====================================================================
# The world model (DPC: lower level + higher level)
# =====================================================================

class WorldModel:
    """Dynamic-Predictive-Coding world model.

    predicted next state = physics_default(state) + correction(state, a)
      where correction comes from a LOWER-level MLP, MODULATED by a
      HIGHER-level recurrent net over the recent state history (the regime).
    predicted reward = the lower level's reward head.

    Same interface as the prior RLS model (step / predict_reward /
    predict_next_state / rollout, same __init__ signature) so the existing
    W1/W2/W3 tests run unchanged -- `n_features` is repurposed as the lower
    net's hidden size, `gamma` is accepted for signature compatibility.
    """

    MAX_OBJECTS = 8
    STATE_PER_OBJ = 4      # cx, cy, vx, vy  (per slot)
    POS_PER_OBJ = 2        # cx, cy
    VEL_PER_OBJ = 2        # vx, vy
    PERSIST_GATE = 0.15
    KIND_THRESH = 0.005
    # ---- DPC higher level ----
    HIST_LEN = 8           # state-history window the higher level reads
    N_REGIME = 8           # regime vector dimension (modulates the lower net)

    def __init__(self, frame_h, frame_w, num_actions=3,
                 n_features=1000, gamma=0.5, seed=0,
                 forgetting=0.999, ridge_init=1.0):
        self.frame_h = float(frame_h)
        self.frame_w = float(frame_w)
        self.num_actions = int(num_actions)
        self.state_dim = self.MAX_OBJECTS * self.STATE_PER_OBJ
        self.pos_dim = self.MAX_OBJECTS * self.POS_PER_OBJ
        self.vel_dim = self.MAX_OBJECTS * self.VEL_PER_OBJ
        self.action_dim = self.num_actions
        self.in_dim = self.state_dim + self.action_dim
        # `n_features` (from the RLS-era signature) is the frozen RFF basis
        # size (a rich fixed nonlinear input expansion). The net's HIDDEN size
        # is a separate small constant -- a small net trains stably ONLINE
        # (a 1000-wide net is too big to converge reliably in ~1200 frames;
        # 128 does). `gamma` is the RFF bandwidth (a gentle knob).
        self._seed = int(seed)
        self._hidden = 128
        self.basis = RandomFourierFeatures(self.in_dim, n_features=int(n_features),
                                           gamma=gamma, seed=seed)
        # ---- lower level (MLP) ----
        self._device = torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
        self._lower = _LowerNet(self.basis.out_dim, self.state_dim,
                                hidden=self._hidden, seed=seed
                                ).to(self._device)
        self._opt = torch.optim.Adam(self._lower.parameters(), lr=3e-4)
        # ---- higher level (a small RNN over the state history -> regime) ----
        # The regime MODULATES the lower net's hidden representation: it
        # scales the post-body hidden vector element-wise before the heads.
        # Learned end-to-end with the lower net (one optimizer), so the gate
        # is learned by backprop over the sequence, not frame-by-frame -- the
        # fix for the rare-event wall we kept hitting.
        self._higher = nn.GRUCell(self.state_dim, self.N_REGIME).to(self._device)
        self._regime_scale = nn.Linear(self.N_REGIME,
                                       self._hidden).to(self._device)
        self._opt.add_param_group({"params": self._higher.parameters()})
        self._opt.add_param_group({"params": self._regime_scale.parameters()})
        self._hx = torch.zeros(self.N_REGIME, device=self._device)
        # a small fixed RFF over the state history to seed the GRU's input
        # (gives the recurrent net a rich per-step input without training a
        # big input embed).
        self._hist_basis = RandomFourierFeatures(self.state_dim,
                                                 n_features=64, gamma=gamma,
                                                 seed=seed + 1)
        self._state_hist = []   # list of recent states (np arrays) for rollout
        self._last_state = None
        self._last_action = None
        self.n_obs = 0
        self._frame = 0
        # surprise-distribution histogram for event-vs-motion discovery (see
        # _update_surprise_stats / _is_event). 64 log-spaced buckets over
        # residual norms from 1e-4 to ~1e0.
        self._surv_hist = np.zeros(64, dtype=np.float64)
        self._surv_n = 0
        # ---- slot assignment (unchanged from the prior model) ----
        self._slot_map = {}
        self._next_slot = 0
        self._slot_last = {}
        self._slot_frame = {}
        self._slot_speeds = {}
        self._controlled_slot = None
        self._track_speeds = {}

    # ---- state construction from perception's tracks (UNCHANGED) ----
    def _new_slot(self):
        i = self._next_slot
        self._next_slot += 1
        return i

    def _nearest_free_slot(self, pos, claimed):
        best_slot, best_d = None, self.PERSIST_GATE
        for slot, (lpos, lvel) in self._slot_last.items():
            if slot in claimed:
                continue
            dead = self._frame - self._slot_frame.get(slot, self._frame)
            pred = lpos + lvel * dead
            d = float(np.linalg.norm(pred - pos))
            if d < best_d:
                best_d, best_slot = d, slot
        return best_slot

    def _kind_speed(self, t):
        vx = np.clip(t["vx"] / self.frame_w, -1.0, 1.0)
        vy = np.clip(t["vy"] / self.frame_h, -1.0, 1.0)
        inst = float(np.hypot(vx, vy))
        hist = self._track_speeds.setdefault(t["id"], [])
        hist.append(inst)
        if len(hist) > 20:
            hist.pop(0)
        return float(np.median(hist)) if len(hist) >= 3 else inst

    def _slot_kind_speed(self, slot):
        ss = self._slot_speeds.get(slot)
        if not ss:
            return None
        return float(np.median(ss))

    def _merge_by_kind(self, speed, claimed, exclude=None):
        best_slot, best_d = None, self.KIND_THRESH
        for slot in self._slot_speeds:
            if slot in claimed or slot == exclude:
                continue
            ks = self._slot_kind_speed(slot)
            if ks is None:
                continue
            d = abs(speed - ks)
            if d < best_d:
                best_d, best_slot = d, slot
        return best_slot

    def _consolidate_slots(self):
        if self._controlled_slot is None:
            return
        slots = sorted(self._slot_speeds.keys())
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots[i], slots[j]
                if a == self._controlled_slot or b == self._controlled_slot:
                    continue
                ka, kb = self._slot_kind_speed(a), self._slot_kind_speed(b)
                if ka is None or kb is None:
                    continue
                if abs(ka - kb) < self.KIND_THRESH:
                    keep, drop = a, b
                    for tid, sl in list(self._slot_map.items()):
                        if sl == drop:
                            self._slot_map[tid] = keep
                    self._slot_speeds[keep].extend(self._slot_speeds.get(drop, []))
                    self._slot_speeds.pop(drop, None)
                    self._slot_last.pop(drop, None)
                    self._slot_frame.pop(drop, None)
                    return

    def state_from_tracks(self, tracks, controlled_id=None):
        """Joint state vector from perception's tracks (unchanged slot
        assignment -- keeps the ball in one stable slot across resets)."""
        s = np.zeros(self.state_dim, dtype=np.float64)
        claimed = {}
        for t in tracks:
            pos = np.array([t["cx"] / self.frame_w, t["cy"] / self.frame_h])
            vel = np.array([np.clip(t["vx"] / self.frame_w, -1.0, 1.0),
                            np.clip(t["vy"] / self.frame_h, -1.0, 1.0)])
            tid = t["id"]
            slot = self._slot_map.get(tid)
            if slot is None or slot in claimed:
                speed = self._kind_speed(t)
                slot = None
                if speed > 0.003:
                    slot = self._merge_by_kind(speed, claimed,
                                               exclude=self._controlled_slot)
                if slot is None:
                    slot = self._nearest_free_slot(pos, claimed)
                if slot is None:
                    if self._next_slot >= self.MAX_OBJECTS:
                        continue
                    slot = self._new_slot()
                self._slot_map[tid] = slot
            claimed[slot] = (pos, vel)
        if controlled_id is not None:
            cslot = self._slot_map.get(controlled_id)
            if cslot is not None:
                self._controlled_slot = cslot
        for slot, (pos, vel) in claimed.items():
            j = slot * self.STATE_PER_OBJ
            s[j + 0], s[j + 1] = pos[0], pos[1]
            s[j + 2], s[j + 3] = vel[0], vel[1]
            self._slot_last[slot] = (pos.copy(), vel.copy())
            self._slot_frame[slot] = self._frame
            speed = float(np.hypot(vel[0], vel[1]))
            self._slot_speeds.setdefault(slot, []).append(speed)
            if len(self._slot_speeds[slot]) > 20:
                self._slot_speeds[slot].pop(0)
        self._consolidate_slots()
        return s

    # ---- the physics default (scaffold, exact, adaptive parameters) ----
    def _physics_default(self, state):
        cur_pos = state[:self.pos_dim]
        cur_vel = state[self.pos_dim:self.pos_dim + self.vel_dim]
        next_pos = cur_pos + cur_vel
        next_vel = cur_vel
        return np.concatenate([next_pos, next_vel])

    # ---- tensors ----
    def _features_t(self, state, action):
        oh = np.zeros(self.action_dim, dtype=np.float64)
        oh[int(action)] = 1.0
        x = np.concatenate([state, oh])
        feat = self.basis.features(x)
        return torch.as_tensor(feat, dtype=torch.float32, device=self._device)

    def _state_t(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32),
                               dtype=torch.float32, device=self._device)

    # ---- the per-frame step: observe (tracks, action, reward) ----
    def step(self, tracks, action, reward_delta, controlled_id=None):
        """One frame. Trains the net on the PREVIOUS transition to predict the
        RESIDUAL (real_next - physics_default(prev)) + reward, end-to-end with
        the higher-level regime modulation. Online, one update per frame."""
        reward_delta = float(reward_delta)
        cur_state = self.state_from_tracks(tracks, controlled_id)
        # maintain the state history for the higher level
        self._state_hist.append(cur_state.copy())
        if len(self._state_hist) > self.HIST_LEN:
            self._state_hist.pop(0)
        if self._last_state is not None and self._last_action is not None:
            default_next = self._physics_default(self._last_state)
            residual = cur_state - default_next
            # ---- discover EVENT vs MOTION from the surprise distribution ----
            # Maintain a running median + MAD (median absolute deviation) of
            # the residual norm. A frame whose residual norm far exceeds the
            # typical (median + k*MAD) is a DISCONTINUITY -- a teleport / episode
            # reset / non-physical event, NOT motion. The agent DISCOVERS this
            # cluster from its own surprise statistics (no hand threshold, no
            # "this is a reset" flag): the residual distribution is bimodal
            # (most tiny = motion, a few huge = events), and median+MAD finds
            # the split. We do NOT train the MOTION correction head on event
            # frames (that would teach it fake physics -- "objects teleport to
            # center"); the REWARD head still trains (a point event co-occurs
            # with the teleport, so that signal is real and wanted). This is
            # the Alberta-Plan "learn the structure of your world from the
            # stream" -- events emerge as a learned concept, not a hand-flag.
            rnorm = float(np.linalg.norm(residual))
            self._update_surprise_stats(rnorm)
            is_event = self._is_event(rnorm)
            self._train_step(self._last_state, self._last_action,
                             residual, reward_delta, skip_motion=is_event)
            if not is_event:
                self.n_obs += 1
        self._last_state = cur_state
        self._last_action = int(action)
        self._frame += 1

    # ---- surprise-distribution tracking (event vs motion, learned) ----
    def _update_surprise_stats(self, rnorm):
        """Online accumulate the residual-norm distribution (median + MAD via
        a running histogram). Cheap, no storage of samples."""
        # log-spaced buckets from 1e-4 to ~1.0 cover the range (residuals are
        # normalized units; teleports are ~0.4, motion ~0.005).
        if rnorm < 1e-4:
            return
        b = int(np.clip(np.log10(rnorm / 1e-4) / 4.0 * 64, 0, 63))
        self._surv_hist[b] += 1
        self._surv_n += 1

    def _surprise_median_mad(self):
        """Return (median, mad) of the residual-norm distribution from the
        histogram. None until enough samples."""
        if self._surv_n < 50:
            return None, None
        h = self._surv_hist
        total = h.sum()
        if total == 0:
            return None, None
        # cumulative -> median bucket
        cum = np.cumsum(h)
        med_b = int(np.searchsorted(cum, total / 2.0))
        med = 10 ** ((med_b + 0.5) / 64.0 * 4.0) * 1e-4
        # MAD = median of |x - med|; approximate via the bucket of |x-med|
        # using the same histogram (good enough for a threshold).
        # simpler robust proxy: the 25th-percentile residual norm (the "typical
        # motion" scale); MAD ~= median - p25 for a right-skewed dist.
        p25_b = int(np.searchsorted(cum, total / 4.0))
        p25 = 10 ** ((p25_b + 0.5) / 64.0 * 4.0) * 1e-4
        mad = max(med - p25, med * 0.5)  # floor so we don't over-flag when narrow
        return med, mad

    def _is_event(self, rnorm):
        """A frame is an EVENT (non-physical discontinuity) if its residual
        norm is far above the typical motion scale: rnorm > median + k*MAD,
        with k large (8) so only true outliers (teleports, ~100x motion) flag.
        Returns False until enough stats exist (train on everything early)."""
        med, mad = self._surprise_median_mad()
        if med is None:
            return False
        return rnorm > med + 12.0 * mad

    def _train_step(self, prev_state, action, residual, reward, skip_motion=False):
        """One online gradient update: predict the residual + reward for the
        previous (state, action) given the current regime, backprop the error,
        then advance the regime GRU with the previous state. The regime
        modulation and the lower net are trained TOGETHER (end-to-end) -- the
        gate is learned by backprop, not frame-by-frame.

        If skip_motion is set, this frame was flagged as an EVENT (a
        discontinuity / teleport / non-physical reset), discovered from the
        surprise distribution. Don't train the MOTION correction head on it
        (that would teach fake physics); only train the REWARD head (a point
        event co-occurs with the teleport -- that signal is real and wanted)."""
        self._lower.train()
        feat = self._features_t(prev_state, action)
        prev_t = self._state_t(prev_state)
        h = self._lower.body(feat)
        # regime modulation: scale the hidden vector element-wise (the gate).
        # _hx is detached (carried forward as a constant), so the regime
        # weights train via the modulation's gradient, not via BPTT.
        mod = torch.sigmoid(self._regime_scale(self._hx))
        h_mod = h * mod
        pred_corr = self._lower.correction(h_mod)
        pred_r = self._lower.reward_head(h_mod).squeeze()
        tgt_corr = torch.as_tensor(residual, dtype=torch.float32,
                                   device=self._device)
        tgt_r = torch.tensor(reward, dtype=torch.float32, device=self._device)
        # RARE-EVENT LOSS WEIGHTING: the bounce is a TEMPORALLY-SPARSE event --
        # the residual target is ~0 for ~90% of frames, so a plain MSE would
        # learn "predict ~0" and ignore the rare bounce (the wall we kept
        # hitting). Weight each frame's state-correction loss by (1 + |residual|/scale)
        # so a frame with a large residual (a real event) counts far more than
        # a quiet frame. General (class-imbalance weighting, not Pong knowledge)
        # and the direct attack on the temporal-sparsity wall. The reward head
        # is a single scalar; weight it by (1 + |reward|) the same way.
        w = 1.0 + torch.mean(torch.abs(tgt_corr)) / 0.02   # scalar weight
        wr = 1.0 + abs(reward) / 0.02
        motion_loss = w * torch.mean((pred_corr - tgt_corr) ** 2) if not skip_motion \
            else torch.zeros((), device=self._device)
        loss = motion_loss + wr * (pred_r - tgt_r) ** 2
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._lower.parameters())
            + list(self._higher.parameters())
            + list(self._regime_scale.parameters()), 1.0)
        self._opt.step()
        # advance the regime GRU with the previous state. DETACH the new
        # hidden state so the next frame's backward doesn't try to flow through
        # this frame's graph (truncated BPTT -- standard for online RNNs; one
        # frame of gradient per update, regime state carried forward as a
        # constant). The regime STILL trains: its weights get gradients through
        # the modulation each frame.
        with torch.no_grad():
            self._hx = self._higher(prev_t, self._hx).detach()

    # ---- prediction ----
    def _predict(self, state, action, hx=None):
        """Return (predicted ABSOLUTE next state, predicted reward) for (s, a).

        Lower level: physics_default + correction(s, a), where the correction
        is MODULATED by the higher-level regime (the GRU hidden state). The
        regime is read from the recent state history -- a longer-timescale
        signal that can foresee the rare bounce from the approach pattern.

        If `hx` is given (rollout), use it as the regime state and return the
        NEXT regime state too, so a rollout can carry the regime forward."""
        self._lower.eval()
        feat = self._features_t(state, action)
        h = self._lower.body(feat)
        regime = self._hx if hx is None else hx
        mod = torch.sigmoid(self._regime_scale(regime))
        h_mod = h * mod
        pred_corr = self._lower.correction(h_mod).detach().cpu().numpy()
        pred_r = float(self._lower.reward_head(h_mod).detach().reshape(()).cpu().numpy())
        default_next = self._physics_default(state)
        next_state = default_next + pred_corr
        # advance the regime for rollout carry-forward
        with torch.no_grad():
            next_hx = self._higher(self._state_t(state), regime)
        if hx is None:
            return next_state, pred_r
        # in rollout mode, also return the imagined SURPRISE = the norm of the
        # learned correction (the deviation from the physics default) -- the
        # same signal used to detect real-stream events, applied to an imagined
        # frame so the dynamic rollout can stop when its own imagination is
        # no longer believable.
        corr_norm = float(np.linalg.norm(pred_corr))
        return next_state, pred_r, next_hx, corr_norm

    # ---- model-assisted coast (close the perception<->model loop) ----
    def velocity_hint(self, track):
        """A bounce-aware velocity guess for a COASTING track (one whose blob
        merged with another at a contact/collision, so perception can't see
        it this frame). `track` is a perception track dict (pixel coords + id).
        Returns (vx, vy) in PIXEL units, or None if the model is cold / the
        slot is not found / the guess is not believable.

        This closes the perception<->model loop (a listed ALBERTA_PLAN
        departure: 'perception -> model is one-directional so far; the
        feedback loop is a later step'). Instead of `coast()` FREEEZING the
        pre-bounce velocity through the contact (wrong sign -- the ball
        already bounced), we ask the world model 'where will this object be
        next, and how fast?' -- its prediction = physics_default + LEARNED
        correction, and the correction is where the bounce lives. So the gap
        velocity becomes bounce-aware instead of frozen-wrong.

        HONEST CIRCULARITY (the caveat): the world model was trained on the
        corrupted (frozen-wrong) velocity, so its bounce correction is only
        ~40% of a full flip (measured). So this moves the gap velocity from
        'fully wrong' to 'less wrong', not to 'right' -- and it feeds the
        less-wrong velocity back in as input, which may bootstrap stronger
        over time (a virtuous cycle) or stall. We measure (W3b).

        Believability gate: if the predicted velocity magnitude is implausible
        (> 3x the track's last known speed, or > 0.3 normalized), return None
        and fall back to freeze -- a wild guess is worse than a frozen one.
        """
        if self._last_state is None or self.n_obs < 200:
            return None   # cold model -- freeze is safer than a random guess
        tid = track.get("id")
        slot = self._slot_map.get(tid)
        if slot is None:
            return None   # this track has no world-model slot yet
        # predict one step from the world model's last state under the last
        # action; read the predicted velocity for this slot.
        action = (self._last_action if self._last_action is not None else 0)
        try:
            nxt, _r = self._predict(self._last_state, action)
        except Exception:
            return None
        j = slot * self.STATE_PER_OBJ
        pvx = float(nxt[j + 2]) * self.frame_w   # normalized -> pixel units
        pvy = float(nxt[j + 3]) * self.frame_h
        # believability gate: reject implausible guesses (fall back to freeze)
        cur_spd = float(np.hypot(track.get("vx", 0.0), track.get("vy", 0.0)))
        pspdl = float(np.hypot(pvx, pvy))
        # 0.3 normalized ~= a large fraction of the frame in one step
        if pspdl > 0.3 * max(self.frame_w, self.frame_h):
            return None
        if cur_spd > 1e-3 and pspdl > 3.0 * cur_spd:
            return None
        return pvx, pvy

    def predict_reward(self, tracks, action, controlled_id=None):
        """Predict the reward_delta following this (state, action). The
        foresight signal."""
        s = self.state_from_tracks(tracks, controlled_id)
        _, r = self._predict(s, action)
        return r

    def predict_next_state(self, tracks, action, controlled_id=None):
        """Predict the next joint state (object positions/velocities)."""
        s = self.state_from_tracks(tracks, controlled_id)
        nxt, _ = self._predict(s, action)
        return nxt

    # ---- the dynamic-rollout stop rule (no Pong geometry) ----
    def _is_unrealistic(self, state, corr_norm):
        """Is this IMAGINED state believable? Returns (bool, reason).

        General signals -- nothing here knows it is Pong:
          (a) OFF-SCREEN: an object's position leaves the normalized frame
              (with a small margin). "An object left the world."
          (b) VELOCITY EXPLOSION: some velocity magnitude is implausibly
              large vs normalized units. "Something moves impossibly fast."
          (c) SURPRISE SPIKE: the imagined correction -- the deviation from
              the physics default, the SAME surprise statistic used to detect
              real-stream events -- far exceeds the model's own typical motion
              scale (median + 12*MAD). "The model does not believe its own
              imagination anymore."
        This is the honest stopping rule: imagination continues only as long
        as the model's own signals say it is still realistic."""
        pos = state[:self.pos_dim]
        if np.any(pos < -0.05) or np.any(pos > 1.05):
            return True, "off_screen"
        vel = state[self.pos_dim:self.pos_dim + self.vel_dim]
        if np.max(np.abs(vel)) > 0.3:
            return True, "velocity_explosion"
        med, mad = self._surprise_median_mad()
        if med is not None and corr_norm > med + 12.0 * mad:
            return True, "surprise_spike"
        return False, None

    def rollout(self, tracks, first_action, action_fn, horizon=20,
                controlled_id=None, dynamic=True):
        """Imagine forward from the current state. Each imagined step =
        physics_default + regime-modulated correction; the regime is carried
        forward in the imagined future (the higher level fires on the imagined
        approach pattern -- this is what makes an imagined bounce sharp).
        All in the agent's head; the servo never moves.

        DYNAMIC rollout (the honest version of imagination): STOP as soon as
        the imagination becomes unrealistic, detected by the model's own
        signals (see _is_unrealistic) -- no Pong geometry. The length is
        EARNED by realism, not fixed: a short trustworthy trajectory beats a
        long dishonest one. This is how the Alberta Plan's Dyna (Step 7)
        actually imagines -- until the model's confidence runs out -- and it
        sidesteps the compounding-error wall the fixed-length W3 kept hitting:
        we never reach the compounded-garbage regime, we get a SHORT HONEST
        rollout instead. A human imagining the future does this: "the ball
        flies straight... straight... reaches the paddle... and now I'm not
        sure, so I stop."

        first_action : action at the FIRST imagined step.
        action_fn(s, t) -> action : chooses each subsequent imagined action.
        dynamic : if True (default), stop on unrealistic imagination; if False,
          run the full `horizon` (the old fixed-length behavior, kept for
          comparison / diagnostics).

        Returns (states, rewards, cumulative_reward, meta) where meta =
        {"stop_reason": str|None, "stopped_at": int} -- stopped_at is the
        number of imagined steps taken (== horizon if it ran the full way).
        states always includes the starting state, so len(states) ==
        stopped_at + 1."""
        s = self.state_from_tracks(tracks, controlled_id)
        states = [s.copy()]
        rewards = []
        a = int(first_action)
        hx = self._hx
        stop_reason = None
        t = 0
        for t in range(horizon):
            s, r, hx, corr_norm = self._predict(s, a, hx=hx)
            states.append(s.copy())
            rewards.append(float(r))
            if dynamic:
                bad, why = self._is_unrealistic(s, corr_norm)
                if bad:
                    stop_reason = why
                    break
            a = int(action_fn(s, t))
        meta = {"stop_reason": stop_reason, "stopped_at": t + 1}
        return states, rewards, float(np.sum(rewards)), meta

    def diagnostics(self):
        return {
            "n_obs": self.n_obs,
            "n_features": int(self._lower.body[0].out_features),
            "gamma": self.basis.gamma,
            "slot_map": dict(self._slot_map),
            "controlled_slot": self._controlled_slot,
        }
