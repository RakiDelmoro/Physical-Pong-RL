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
# Magnitude-Direction Decoupled optimizer (Hägele et al. 2026,
# arXiv:2606.25971). See WORLD_MODEL_PLAN.md 'MD Decoupling'.
# =====================================================================
# The root-cause fix for long-run drift in the online/continual world model.
# Plain Adam updates a weight matrix's DIRECTION and MAGNITUDE tangled
# together: the magnitude creeps up on its own (updates are ~perpendicular
# to the weight, so the norm inflates even with no radial gradient), which
# silently SHRINKS the directional update (angular change ~ ||dW||/||W||),
# so online learning slows down and predictions drift -- exactly the W2
# degradation we measured (foresight 0.857 -> 0.725 over 5000 frames).
# Weight decay only patches this indirectly (shrinks the norm to fight the
# creep) and distorts the LR schedule.
#
# MD Decoupling factorizes each 2D weight W into a fixed-norm DIRECTION on a
# Frobenius sphere + learnable per-row and per-column GAINS (softplus-
# reparameterized so they stay positive), updated at SEPARATE learning rates:
#     W = diag(g_row) * What * diag(g_col),   ||What||_F = R (fixed)
# The direction cannot inflate (pinned to the sphere), so the LR directly
# sets the angular update at every step and the rate does NOT decay over
# training. The gains recover the fine-grained scale control that pinning the
# norm gives up. The model sees the single fused weight W (Algorithm 2 of the
# paper). No weight decay, no warmup. GENERAL (an optimizer fix, no Pong
# knowledge); also helps the corrupted-history bounce relearn at a steady,
# non-decaying rate (the virtuous cycle that stalled under plain Adam).
#
# Non-matrix params (biases) use plain Adam. Gradient clipping is applied to
# the full parameter set first (unchanged from the prior plain-Adam path).

class _MDOptimizer:
    """Magnitude-Direction Decoupled optimizer for the world model's learned
    net. Matrix-shaped (2D) parameters get the MD factorization; biases and
    any non-2D params get plain Adam. The model always sees the fused weight.

    Spheres use the INIT Frobenius norm as the radius (so the initial weight
    is unchanged -- gains start at softplus-inv(1) ~= 0.541, i.e. gain = 1,
    and What = W_init; no scale reset, so an already-trained net is not
    disturbed). Only the UPDATE DYNAMICS change, which is the point."""

    def __init__(self, matrix_params, other_params,
                 lr_dir=3e-4, lr_gain=3e-3, eps=1e-8, betas=(0.9, 0.999)):
        self._lr_dir = float(lr_dir)
        self._lr_gain = float(lr_gain)
        self._eps = float(eps)
        self._b1, self._b2 = float(betas[0]), float(betas[1])
        self._t = 0
        self._md = {}          # id(p) -> state dict (lazy init on first step)
        self._matrix = []      # the 2D params, in a stable order
        for p in matrix_params:
            if p.dim() == 2:
                self._md[id(p)] = None
                self._matrix.append(p)
            else:
                other_params = list(other_params) + [p]
        self._other = list(other_params)
        self._adam_other = (torch.optim.Adam(self._other, lr=lr_dir, eps=eps,
                                             betas=betas)
                            if self._other else None)

    # ---- softplus gain map + its derivative + the inverse for init ----
    @staticmethod
    def _sp(x):
        return torch.nn.functional.softplus(x)
    @staticmethod
    def _sp_inv(y):
        # y > 0; softplus(x) = log(1+e^x) = y  ->  x = log(e^y - 1)
        return torch.log(torch.expm1(y) + 1e-12)
    @staticmethod
    def _sp_grad(x):
        return torch.sigmoid(x)

    def zero_grad(self):
        for p in self._matrix + self._other:
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()

    def _adam_step(self, t, grad, m, v, lr):
        """Manual Adam update on a tensor; returns the updated tensor. Bias-
        correction uses the optimizer's global step count."""
        m.mul_(self._b1).add_(grad, alpha=1.0 - self._b1)
        v.mul_(self._b2).addcmul_(grad, grad, value=1.0 - self._b2)
        bc1 = 1.0 - self._b1 ** self._t
        bc2 = 1.0 - self._b2 ** self._t
        mhat = m / bc1
        vhat = v / bc2
        return t - lr * mhat / (vhat.sqrt() + self._eps)

    def step(self):
        self._t += 1
        for p in self._matrix:
            if p.grad is None:
                continue
            st = self._md[id(p)]
            G = p.grad
            if st is None:
                # lazy init: sphere radius = init Frobenius norm; What = W_init;
                # gains -> 1 (raw = softplus-inv(1)).
                R = float(p.data.norm(p='fro').item())
                R = R if R > 1e-8 else 1.0
                out_, in_ = p.shape
                dev, dt = p.device, p.dtype
                one = torch.tensor(1.0, device=dev, dtype=dt)
                ghat_row = self._sp_inv(one).expand(out_).clone().to(dev)
                ghat_col = self._sp_inv(one).expand(in_).clone().to(dev)
                What = p.data.clone()
                st = {
                    "R": R,
                    "What": What,
                    "ghat_row": ghat_row,
                    "ghat_col": ghat_col,
                    "m_w": torch.zeros_like(What), "v_w": torch.zeros_like(What),
                    "m_gr": torch.zeros_like(ghat_row), "v_gr": torch.zeros_like(ghat_row),
                    "m_gc": torch.zeros_like(ghat_col), "v_gc": torch.zeros_like(ghat_col),
                }
                self._md[id(p)] = st
            R = st["R"]
            What = st["What"]
            ghat_row, ghat_col = st["ghat_row"], st["ghat_col"]
            g_row = self._sp(ghat_row)          # (out,)
            g_col = self._sp(ghat_col)          # (in,)
            # 1. recover the on-sphere direction from the fused weight
            What = (p.data / g_row.unsqueeze(1)) / g_col.unsqueeze(0)
            # 2. gain gradients (Algorithm 2):  Wg = What * G (elementwise)
            Wg = What * G
            g_grow = (Wg * g_col.unsqueeze(0)).sum(dim=1)        # rowsum(Wg diag(g_col))
            g_gcol = (g_row.unsqueeze(1) * Wg).sum(dim=0)       # colsum(diag(g_row) Wg)
            # backprop through softplus
            g_ghat_row = g_grow * self._sp_grad(ghat_row)
            g_ghat_col = g_gcol * self._sp_grad(ghat_col)
            # 3. direction gradient: G_What = diag(g_row) G diag(g_col)
            G_what = g_row.unsqueeze(1) * G * g_col.unsqueeze(0)
            # 4. Adam step on the direction, then project back onto the sphere
            What_new = self._adam_step(What, G_what, st["m_w"], st["v_w"],
                                       self._lr_dir)
            What_new = What_new / What_new.norm(p='fro') * R
            st["What"] = What_new
            # 5. Adam step on the raw gains (own LR)
            st["ghat_row"] = self._adam_step(ghat_row, g_ghat_row,
                                             st["m_gr"], st["v_gr"], self._lr_gain)
            st["ghat_col"] = self._adam_step(ghat_col, g_ghat_col,
                                             st["m_gc"], st["v_gc"], self._lr_gain)
            # 6. reassemble the fused weight for the next forward
            g_row_new = self._sp(st["ghat_row"])
            g_col_new = self._sp(st["ghat_col"])
            with torch.no_grad():
                p.data.copy_(g_row_new.unsqueeze(1) * What_new * g_col_new.unsqueeze(0))
        if self._adam_other is not None:
            self._adam_other.step()


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
        # ---- higher level (a small RNN over the state history -> regime) ----
        # The regime MODULATES the lower net's hidden representation: it
        # scales the post-body hidden vector element-wise before the heads.
        # Learned end-to-end with the lower net (one optimizer), so the gate
        # is learned by backprop over the sequence, not frame-by-frame -- the
        # fix for the rare-event wall we kept hitting.
        self._higher = nn.GRUCell(self.state_dim, self.N_REGIME).to(self._device)
        self._regime_scale = nn.Linear(self.N_REGIME,
                                       self._hidden).to(self._device)
        # ---- optimizer: Magnitude-Direction Decoupling (arXiv:2606.25971) ----
        # The root-cause fix for the measured long-run drift (W2 foresight
        # 0.857 -> 0.725 over 5000 frames under plain Adam, no weight decay):
        # the weight magnitude creeps up, silently shrinking the directional
        # update so online learning stalls. MD pins each matrix's DIRECTION to
        # a fixed Frobenius sphere + learns per-row/per-column GAINS at their
        # own LR, so the angular update is set by the LR and does NOT decay.
        matrix_params, other_params = [], []
        for sub in self._lower.modules():
            if isinstance(sub, nn.Linear):
                matrix_params.append(sub.weight)
                other_params.append(sub.bias)
        # GRUCell: weight_ih/weight_hh are matrices; biases are not.
        matrix_params += [self._higher.weight_ih, self._higher.weight_hh]
        other_params += [self._higher.bias_ih, self._higher.bias_hh]
        # regime_scale Linear
        matrix_params.append(self._regime_scale.weight)
        other_params.append(self._regime_scale.bias)
        self._opt = _MDOptimizer(matrix_params, other_params,
                                 lr_dir=3e-4, lr_gain=3e-3)
        self._hx = torch.zeros(self.N_REGIME, device=self._device)
        # a small fixed RFF over the state history to seed the GRU's input
        # (gives the recurrent net a rich per-step input without training a
        # big input embed).
        self._hist_basis = RandomFourierFeatures(self.state_dim,
                                                 n_features=64, gamma=gamma,
                                                 seed=seed + 1)
        self._state_hist = []   # list of recent states (np arrays) for rollout
        self._action_hist = []  # aligned action history (for the controlled-
        # FULL (untrimmed) state + action history, for the controlled-object
        # action->motion model (a short window can be all-one-action -> no
        # slope; the full history guarantees enough action variation).
        self._state_hist_full = []
        self._acts_full = []
        # the controlled track's history, for the action->motion model
        # (clean: track-id identity is stable). List of (action, cx_px,
        # cy_px, vx_px, vy_px) for the controlled track each frame. POSITION
        # is what we regress on (its per-frame DELTA = the true displacement
        # under linear motion, even though the reported velocity is an EMA
        # that LAGS and underestimates -- that underestimate made the imagined
        # paddle ~5x too slow, so the planner's intercept predictions were
        # wrong. Position-deltas fix it.).
        self._ctrl_track_vel = []
        # PER-TRACK-ID position history for the controlled-object discovery
        # query. Keyed by TRACK ID (stable identity within a track's life),
        # NOT by slot (which churns -- the paddle's slot reassigns as its
        # track dies at ball-contacts and restarts). Each value is a list of
        # (action, cx_px, cy_px) capped at CORR_HIST. The controlled object
        # is the track whose POSITION DELTA correlates with the action --
        # measured on the track's own stable identity, so it survives slot
        # churn and even track-id churn (each active paddle track accumulates
        # its own correlation; we return whichever is best RIGHT NOW).
        self._track_pos = {}
        self._controlled_track_id = None   # hysteresis on the TRACK ID
        self._controlled_last_pos = None  # (cx, cy) px of the controlled
                                          # track's last LIVE position (for
                                          # object-permanence bridging)
        self._controlled_shape = None     # (w, h) of the controlled track
                                          # (heir must match its shape class)
                                # object correlation query; see controlled_track)
        # a SEPARATE, longer history for the controlled-object correlation
        # query (the DPC higher level's _state_hist is only HIST_LEN=8, too
        # short to measure an action-velocity correlation; the class model
        # used 80). Does not touch the DPC higher level.
        self._corr_hist = []     # recent states for the correlation query
        self._corr_acts = []     # aligned actions
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
        self._slot_dim = {}        # per-slot dimensionality cue ('0d'/'1d'/'2d')
        self._slot_shape = {}      # per-slot (w, h) in normalized units (for surface extent)
        self._controlled_slot = None
        self._track_speeds = {}

    # ---- state construction from perception's tracks (UNCHANGED) ----
    def _new_slot(self):
        # Fresh slot index, MONOTONIC but bounded by MAX_OBJECTS. Recycling
        # of dead slots (occupant churned) is handled by _merge_by_kind /
        # _nearest_free_slot (which reuse dead slots of matching kind/pos) and
        # the _recycle_dead_slot fallback in state_from_tracks (for when no
        # dead slot matches and the monotonic counter is exhausted -- the
        # ball-resets-often case: every reset is a new track id needing a
        # slot, so a non-recycling counter permanently starves new objects).
        i = self._next_slot
        self._next_slot += 1
        return i

    def _recycle_dead_slot(self, claimed, live_slots):
        # When the monotonic _next_slot has passed MAX_OBJECTS, reuse a slot
        # index that is NOT occupied this frame (not in `claimed`, not in
        # `live_slots`). This covers BOTH dead slots still in _slot_speeds
        # (occupant churned) AND indices that were consolidated out of
        # _slot_speeds (which _new_slot can no longer hand out and a
        # _slot_speeds-only scan would miss -- the bug that starved new ball
        # track ids of slots). Picks the smallest free index. Returns None
        # only if every slot index is occupied this frame (genuinely full).
        used = set(claimed) | set(live_slots)
        for i in range(self.MAX_OBJECTS):
            if i not in used:
                return i
        return None

    def _reset_slot(self, slot):
        # Clear a recycled slot's kind/pos history so the new occupant starts
        # fresh (its dynamics are learned online; stale history would mislead
        # kind-merging for FUTURE new tracks).
        self._slot_speeds.pop(slot, None)
        self._slot_last.pop(slot, None)
        self._slot_frame.pop(slot, None)
        self._slot_dim.pop(slot, None)

    def _nearest_free_slot(self, pos, claimed, exclude=None):
        best_slot, best_d = None, self.PERSIST_GATE
        for slot, (lpos, lvel) in self._slot_last.items():
            if slot in claimed or (exclude is not None and slot in exclude):
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

    def _merge_by_kind(self, speed, dim, claimed, exclude=None):
        # DECOUPLED from controlled: `exclude` is the set of LIVE slots (still
        # occupied by a current track), not the controlled slot. A new track
        # merges only into a DEAD same-speed slot (its occupant churned/gone),
        # so the ball never folds into the paddle's live slot even when their
        # speeds overlap. Controlled-independent. ALSO requires a matching
        # DIMENSIONALITY cue: a fast 0d ball and a fast-moving 1d paddle can
        # have overlapping speeds in this scene, but they are different
        # object KINDS -- a 0d track must not fold into a 1d slot (or the ball
        # drops out of the state when the paddle churns). Shape-class is the
        # kind signal speed alone lacked.
        best_slot, best_d = None, self.KIND_THRESH
        for slot in self._slot_speeds:
            if slot in claimed or (exclude is not None and slot in exclude):
                continue
            ks = self._slot_kind_speed(slot)
            if ks is None:
                continue
            if self._slot_dim.get(slot) != dim:
                continue   # different object kind -> never share a slot
            d = abs(speed - ks)
            if d < best_d:
                best_d, best_slot = d, slot
        return best_slot

    def _consolidate_slots(self, live_slots):
        # DECOUPLED from controlled. Consolidation merges duplicate same-kind
        # (same-speed) slots -- the cleanup when _merge_by_kind fails to fold a
        # new ball track into the existing ball slot in one step. It no longer
        # reads _controlled_slot; a frame warmup guards the cold start, and a
        # LIVE-SLOTS guard (controlled-independent) prevents merging two
        # DIFFERENT live objects (e.g. the two paddles, both ~stationary ->
        # same speed 0 -> would wrongly merge). Only merge when at least one
        # slot is DEAD (no current track) -- a dead slot's occupant is gone, so
        # folding it into a same-speed slot is the 'same object re-detected'
        # case consolidation exists for.
        if self._frame < 30:
            return   # cold-start warmup (frame-based, not controlled-based)
        slots = sorted(self._slot_speeds.keys())
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots[i], slots[j]
                if a in live_slots and b in live_slots:
                    continue   # both live -> different objects, do not merge
                ka, kb = self._slot_kind_speed(a), self._slot_kind_speed(b)
                if ka is None or kb is None:
                    continue
                if self._slot_dim.get(a) != self._slot_dim.get(b):
                    continue   # different object kinds -> never consolidate
                if abs(ka - kb) < self.KIND_THRESH:
                    keep, drop = a, b
                    for tid, sl in list(self._slot_map.items()):
                        if sl == drop:
                            self._slot_map[tid] = keep
                    self._slot_speeds[keep].extend(self._slot_speeds.get(drop, []))
                    self._slot_speeds.pop(drop, None)
                    self._slot_last.pop(drop, None)
                    self._slot_frame.pop(drop, None)
                    self._slot_dim.pop(drop, None)
                    return

    def state_from_tracks(self, tracks, controlled_id=None):
        """Joint state vector from perception's tracks (DECOUPLED slot
        assignment -- keeps the ball in one stable slot across resets without
        reading the controlled property)."""
        s = np.zeros(self.state_dim, dtype=np.float64)
        claimed = {}
        current_tids = {t["id"] for t in tracks}
        # LIVE slots: slots whose previous occupant (track id) is STILL in the
        # current tracks. A new track must NOT merge into a live slot (that
        # would fold two different objects into one slot -- e.g. the ball into
        # the paddle's slot, which happens because a moving paddle's speed
        # overlaps the ball's). A new track merges only into a DEAD slot (its
        # occupant churned/gone -- the canonical case is the ball resetting:
        # old ball tid gone -> old ball slot dead -> new ball merges back in).
        # This is CONTROLLED-INDEPENDENT: it distinguishes 'dead ball slot' from
        # 'live paddle slot' by track-id liveness, not by which object is
        # controlled -- breaking the circular dependency.
        live_slots = {slot for tid, slot in self._slot_map.items()
                      if tid in current_tids}
        for t in tracks:
            pos = np.array([t["cx"] / self.frame_w, t["cy"] / self.frame_h])
            vel = np.array([np.clip(t["vx"] / self.frame_w, -1.0, 1.0),
                            np.clip(t["vy"] / self.frame_h, -1.0, 1.0)])
            tid = t["id"]
            dim = t.get("dim")            # dimensionality cue (the kind signal)
            shape = (float(t.get("w", 0.0)) / self.frame_w,
                     float(t.get("h", 0.0)) / self.frame_h)
            slot = self._slot_map.get(tid)
            if slot is None or slot in claimed:
                speed = self._kind_speed(t)
                slot = None
                if speed > 0.003 and dim is not None:
                    slot = self._merge_by_kind(speed, dim, claimed,
                                               exclude=live_slots)
                if slot is None:
                    slot = self._nearest_free_slot(pos, claimed,
                                                   exclude=live_slots)
                if slot is None:
                    if self._next_slot < self.MAX_OBJECTS:
                        slot = self._new_slot()
                    else:
                        # monotonic slot space exhausted -- RECYCLE a dead
                        # slot (occupant churned) so a new object (e.g. a
                        # reset ball with a brand-new track id) still gets a
                        # slot instead of being dropped from the state.
                        slot = self._recycle_dead_slot(claimed, live_slots)
                        if slot is None:
                            continue   # all slots live -- genuinely full
                        self._reset_slot(slot)
                self._slot_map[tid] = slot
            claimed[slot] = (pos, vel, dim, shape)
        # NOTE: under option C, _controlled_slot is set ONLY by controlled_track
        # (the dynamics query). state_from_tracks no longer writes it from
        # controlled_id -- that was the circular write (controlled_track reads
        # the slots to discover controlled, then state_from_tracks rewrote
        # _controlled_slot from the controlled_id controlled_track returned).
        # controlled_id is still accepted (signature compat) but ignored here.
        for slot, (pos, vel, dim, shape) in claimed.items():
            j = slot * self.STATE_PER_OBJ
            s[j + 0], s[j + 1] = pos[0], pos[1]
            s[j + 2], s[j + 3] = vel[0], vel[1]
            self._slot_last[slot] = (pos.copy(), vel.copy())
            self._slot_frame[slot] = self._frame
            if dim is not None:
                self._slot_dim[slot] = dim
            self._slot_shape[slot] = shape
            speed = float(np.hypot(vel[0], vel[1]))
            self._slot_speeds.setdefault(slot, []).append(speed)
            if len(self._slot_speeds[slot]) > 20:
                self._slot_speeds[slot].pop(0)
        self._consolidate_slots(live_slots)
        return s

    # ---- the physics default (scaffold, exact, adaptive parameters) ----
    def _physics_default(self, state):
        # INTERLEAVED layout: slot s occupies state[s*STATE_PER_OBJ : +4] =
        # (cx, cy, vx, vy). Apply the universal kinematics 'position advances
        # by velocity; velocity persists' PER SLOT. (The prior block-layout
        # reading was a BUG -- it garbled the state, so the imagined ball never
        # advanced correctly and imagination drifted. The residual + reward
        # heads still trained because they learn whatever mapping the state
        # carries, but the rollout's physics default was wrong.)
        out = np.array(state, dtype=np.float64, copy=True)
        for s in range(self.MAX_OBJECTS):
            j = s * self.STATE_PER_OBJ
            out[j + 0] += out[j + 2]   # next_cx = cx + vx
            out[j + 1] += out[j + 3]   # next_cy = cy + vy
            # next_vx, next_vy unchanged (velocity persists)
        return out

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
        # record the controlled track's perception velocity for the action
        # model (clean, track-id-stable; the slot-state signal is churn-noisy)
        if controlled_id is not None:
            for t in tracks:
                if t["id"] == controlled_id:
                    self._ctrl_track_vel.append(
                        (int(action), float(t.get("cx", 0.0)),
                         float(t.get("cy", 0.0)),
                         float(t.get("vx", 0.0)),
                         float(t.get("vy", 0.0))))
                    if len(self._ctrl_track_vel) > 300:
                        self._ctrl_track_vel.pop(0)
                    break
        # maintain the state + action histories for the higher level and for
        # the controlled-object correlation query (controlled_track).
        self._state_hist.append(cur_state.copy())
        self._action_hist.append(int(action))
        self._state_hist_full.append(cur_state.copy())
        self._acts_full.append(int(action))
        # cap the full history (keep enough for a stable action-model fit;
        # not the whole session -- a few hundred frames is plenty and bounds
        # memory).
        if len(self._state_hist_full) > 600:
            self._state_hist_full.pop(0); self._acts_full.pop(0)
        if len(self._state_hist) > self.HIST_LEN:
            self._state_hist.pop(0)
        if len(self._action_hist) > self.HIST_LEN:
            self._action_hist.pop(0)
        # the dedicated longer correlation history
        self._corr_hist.append(cur_state.copy())
        self._corr_acts.append(int(action))
        if len(self._corr_hist) > self.CORR_HIST:
            self._corr_hist.pop(0)
        if len(self._corr_acts) > self.CORR_HIST:
            self._corr_acts.pop(0)
        # PER-TRACK-ID position history (churn-free controlled discovery).
        # Record every visible track's (action, cx, cy); prune tracks no
        # longer present once they're too short to ever clear the gate.
        live_tids = set()
        for t in tracks:
            tid = t["id"]
            live_tids.add(tid)
            self._track_pos.setdefault(tid, []).append(
                (int(action), float(t.get("cx", 0.0)), float(t.get("cy", 0.0))))
            if len(self._track_pos[tid]) > self.CORR_HIST:
                self._track_pos[tid].pop(0)
        # drop dead tracks with too little history to be useful (keep short
        # recent ones in case they recur; cap the dict size).
        if len(self._track_pos) > 64:
            self._track_pos = {
                tid: h for tid, h in self._track_pos.items()
                if tid in live_tids or len(h) >= 30}
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

    # ---- prediction (one-step) ----
    def _predict(self, state, action):
        """Return (predicted ABSOLUTE next state, predicted reward) for (s, a).

        Lower level: physics_default + correction(s, a), where the correction
        is MODULATED by the higher-level regime (the GRU hidden state). The
        regime is read from the recent state history -- a longer-timescale
        signal. Used for one-step prediction (reward foresight, velocity hints);
        the multi-step ROLLOUT does NOT use this (it uses the exact physics
        default + the computed Way C bounce -- see rollout)."""
        self._lower.eval()
        feat = self._features_t(state, action)
        h = self._lower.body(feat)
        mod = torch.sigmoid(self._regime_scale(self._hx))
        h_mod = h * mod
        pred_corr = self._lower.correction(h_mod).detach().cpu().numpy()
        pred_r = float(self._lower.reward_head(h_mod).detach().reshape(()).cpu().numpy())
        default_next = self._physics_default(state)
        next_state = default_next + pred_corr
        return next_state, pred_r

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

    # ---- OPTION C: the world model IS the labeler (no separate class model).
    # "Which object am I controlling?" and "which objects behave the same
    # way?" are DYNAMICS questions, and the world model already learns each
    # object's dynamics. So we read the labels off the world model by ASKING
    # it, instead of maintaining a separate labeler with a fragile tower of
    # persistence heuristics (ghosts / revival / binding / merging) that was
    # coupled to the old association and broke when we rewired the scaffold.
    # The Alberta Plan's own move: dissolve the perception/labeling chicken-
    # and-egg -- one learned surface (the dynamics model) answers both
    # "what will happen next" and "what is each object".
    CONTROLLED_WARMUP = 300   # frames before the controlled-object query is
                              # trusted (the my-paddle's DIRECT action
                              # correlation emerges clearly by ~f=300; earlier
                              # the opp's INDIRECT correlation can spuriously
                              # win and a wrong commit gets held by hysteresis)
    CONTROLLED_CORR_GATE = 0.05  # min |corr| to NEWLY call a slot controlled
    CONTROLLED_HOLD_GATE = 0.02  # below this, an existing controlled slot is
                                 # dropped (hysteresis: commit high, hold low)
    CONTROLLED_MIN_TRACK = 8   # min frames of a LIVE track's own history
    CONTROLLED_HEIR_RADIUS = 0.25  # normalized: a churned controlled track's
                                   # heir must be within this of its last pos
    CORR_HIST = 80             # length of the dedicated correlation history

    def controlled_track(self, tracks):
        """Discover the controlled object from the world model's OWN history:
        the slot whose velocity CORRELATES most with the action over time is
        the object my action moves. Returns the TRACK ID of that slot (mapped
        back via the slot map), or None if cold / no slot clears the gate.

        This is the on-plan replacement for the class model's
        `controlled_object`: same question ('which object does my action
        affect?'), answered from the dynamics data the world model already
        aggregates (its state + action history), with NO separate labeler,
        NO ghosts, NO binding probation. The slot assignment already keeps
        per-object identity stable across re-detection, so the 'carry the
        label across churn' heuristics have no job -- the slot IS the
        persistent identity. GENERAL: 'which slot's motion correlates with my
        action' is true in any world where an action affects one object; no
        Pong vocabulary.

        HONEST DESIGN NOTE: the naive 'run M(s,UP) vs M(s,DOWN) and subtract'
        does NOT work for this architecture, because the action's effect is
        mediated through the STATE's velocity (perception measures the
        paddle's motion frame-to-frame; that velocity is what M advances via
        the physics default), not through the action INPUT to M. Same state
        -> same prediction regardless of the action input, so the
        single-frame contrast is ~0. The TIME CORRELATION over history is the
        correct query: it asks 'does this object MOVE when I act?', which is
        exactly the dynamics question, measured on the world model's own data.
        """
        if self.n_obs < self.CONTROLLED_WARMUP:
            return None   # cold -- not enough history for a correlation
        neutral = self.num_actions - 1
        def scalar(a):
            a = int(a)
            if a == neutral: return 0.0
            return +1.0 if a < neutral else -1.0
        # PER-TRACK-ID correlation (churn-free). The old per-SLOT query broke
        # because the paddle's SLOT reassigns whenever its track dies at a
        # ball-contact (the kind-gate makes the paddle REJECT the merged blob
        # -> it coasts -> dies after MAX_MISS -> restarts with a new track id
        # -> a new slot), so no single slot ever accumulated a clean paddle
        # signal and the fast, always-present BALL won the correlation
        # spuriously -- making the 'reflex' control the ball / nothing (it
        # leaked 5-8 points; the real reflex on the paddle leaks 0). Keying by
        # TRACK ID fixes it: each active paddle track accumulates its OWN
        # correlation on its stable identity; we return whichever is best NOW,
        # so track-id churn just hands off between paddle tracks instead of
        # losing the signal. GENERAL: 'the object I control is the one whose
        # motion follows my action' -- measured per persistent identity.
        live_tids = {t["id"] for t in tracks}
        best_tid, best_corr = None, 0.0
        tid_corr = {}
        # Only consider LIVE tracks (currently visible) -- returning a dead
        # track id is useless (the policy can't place it in the state) and
        # was happening when a dead paddle track's stale high corr won. A
        # freshly-churned paddle track (just restarted after a contact) has
        # little history, so the per-track minimum is LOW -- the paddle's
        # motion follows the action so clearly (it moves on +/-1 actions,
        # holds on stay) that even ~8 frames discriminate it from the ball,
        # whose motion is independent of the action.
        for tid in live_tids:
            h = self._track_pos.get(tid)
            if h is None or len(h) < self.CONTROLLED_MIN_TRACK:
                continue
            pos_x = np.array([r[1] for r in h], dtype=np.float64)
            pos_y = np.array([r[2] for r in h], dtype=np.float64)
            sc = np.array([scalar(r[0]) for r in h], dtype=np.float64)
            c_track = 0.0
            for pos in (pos_x, pos_y):
                d = np.diff(pos)            # delta[t] = pos[t+1]-pos[t]
                a = sc[1:]                   # the action that drove each delta
                d = d - d.mean(); a = a - a.mean()
                vd = float(np.sqrt((d ** 2).sum()))
                va = float(np.sqrt((a ** 2).sum()))
                if vd < 1e-9 or va < 1e-9:
                    continue
                corr = abs(float((d * a).sum() / (vd * va)))
                c_track = max(c_track, corr)
            tid_corr[tid] = c_track
            if c_track > best_corr:
                best_corr, best_tid = c_track, tid
        if best_tid is None and self._controlled_track_id is None:
            return None   # cold and nothing reacts enough to newly commit
        # ---- OBJECT-PERMANENCE HIERARCHY (the controlled object is
        # persistent; carry its identity through churn and quiet periods) ----
        held = self._controlled_track_id
        chosen = None
        if held is not None and held in live_tids:
            # (1) HOLD: the committed track is still live -> keep it. Its
            # action-correlation NATURALLY drops during quiet (mostly-stay)
            # periods -- that is NOT evidence the controlled object changed,
            # so we do NOT release on low corr (the old gate did, and lost
            # the commit the moment the reflex settled into mostly-stay).
            chosen = held
        elif held is not None and self._controlled_last_pos is not None:
            # (2) HEIR (churn bridge): the committed track DIED (e.g. the
            # paddle's track coasts out at a ball-contact and restarts with a
            # new id). The controlled object does NOT teleport and its shape
            # class persists, so the heir is the live track of the SAME shape
            # class nearest to the last known position. This re-recognizes the
            # new paddle track WITHOUT needing action variation (which quiet
            # periods don't provide). GENERAL: object permanence + kind
            # constancy -- no Pong vocabulary.
            hx, hy = self._controlled_last_pos
            hw, hh = self._controlled_shape or (0.0, 0.0)
            h_asp = (hw / hh) if hh > 1e-6 else 1.0
            h_elong = not (0.5 <= h_asp <= 2.0)
            best_d2, heir = None, None
            for t in tracks:
                tw = float(t.get("w", 0.0)); th = float(t.get("h", 0.0))
                if tw < 1e-6 or th < 1e-6:
                    continue
                t_asp = tw / th
                t_elong = not (0.5 <= t_asp <= 2.0)
                if t_elong != h_elong:
                    continue   # different shape class -> not the heir
                dx = t.get("cx", 0.0) - hx
                dy = t.get("cy", 0.0) - hy
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2, heir = d2, t["id"]
            if heir is not None and best_d2 is not None \
                    and best_d2 < (self.CONTROLLED_HEIR_RADIUS ** 2):
                chosen = heir
            else:
                # no positional heir -> newly commit via correlation if able
                if best_tid is not None and best_corr >= self.CONTROLLED_CORR_GATE:
                    chosen = best_tid
        else:
            # (3) COLD COMMIT: no held track -> newly commit on correlation
            if best_tid is not None and best_corr >= self.CONTROLLED_CORR_GATE:
                chosen = best_tid
        if chosen is None:
            return None
        self._controlled_track_id = chosen
        sl = self._slot_map.get(chosen)
        if sl is None:
            return None
        self._controlled_slot = sl
        # record the chosen track's live position + shape for next frame's
        # heir search
        for t in tracks:
            if t["id"] == chosen:
                self._controlled_last_pos = (float(t.get("cx", 0.0)),
                                             float(t.get("cy", 0.0)))
                self._controlled_shape = (float(t.get("w", 0.0)),
                                          float(t.get("h", 0.0)))
                break
        return chosen

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
        # INTERLEAVED layout: positions are at slot*STATE_PER_OBJ + (0,1);
        # velocities at + (2,3). Read them per-slot (the prior block-layout
        # read was the same bug as _physics_default).
        pos = np.array([state[s * self.STATE_PER_OBJ] for s in range(self.MAX_OBJECTS)] +
                       [state[s * self.STATE_PER_OBJ + 1] for s in range(self.MAX_OBJECTS)])
        if np.any(pos < -0.05) or np.any(pos > 1.05):
            return True, "off_screen"
        vel = np.array([state[s * self.STATE_PER_OBJ + 2] for s in range(self.MAX_OBJECTS)] +
                       [state[s * self.STATE_PER_OBJ + 3] for s in range(self.MAX_OBJECTS)])
        if np.max(np.abs(vel)) > 0.3:
            return True, "velocity_explosion"
        med, mad = self._surprise_median_mad()
        if med is not None and corr_norm > med + 12.0 * mad:
            return True, "surprise_spike"
        return False, None

    # ====================================================================
    # WAY C (ContactNets principle, adapted online): the bounce is COMPUTED,
    # not learned. A 'surface' is DISCOVERED from the stream as a slot that is
    # elongated (1d) and approximately STATIONARY along one axis -- that axis
    # is the surface NORMAL (the direction things bounce off it). GENERAL:
    # 'a surface is an object that doesn't move in the direction things bounce
    # off it' -- true for walls, floors, paddles, ceilings in any 2D world;
    # no Pong vocabulary. The reflection itself is a universal kinematic rule
    # (v -> -e*v along the normal) with a learnable restitution e per surface.
    # Nothing discontinuous is learned (the rare-event wall); only smooth /
    # every-frame things are learned (the surface geometry is discovered, e is
    # a scalar). See WORLD_MODEL_PLAN.md 'Way C / ContactNets'.
    # ====================================================================
    SURFACE_STAT_FRAMES = 30   # recent-history window for the stationarity test
    SURFACE_POS_VAR = 1e-3     # max position variance along the normal axis (~paddle-width scale)
    SURFACE_GAP = 0.05        # cluster split gap (bigger than a surface, smaller than a roam)
    SURFACE_MIN_N = 15        # min samples in a cluster to count as a surface
    def _discover_surfaces(self):
        """Return a list of (slot, normal_axis, surface_pos, half_extent) for
        each DISCOVERED surface, by POSITION CLUSTER (slot-churn + shape-label
        robust). A surface is a plane in space where SOME object sits at a
        near-constant position along one axis over the recent window. We
        collect every non-empty slot's (cx, cy) over the recent history, cluster
        the x-positions and the y-positions, and each tight cluster is a
        surface plane. Moving objects (the ball -- roams 0..1) never form a
        tight cluster; stationary-along-an-axis objects (paddles, walls) do.
        No shape/dim label needed -- pure position stationarity. GENERAL: 'a
        surface is where something sits still along an axis' -- no Pong vocab.
        The half_extent is estimated from the cluster's live slot shape (if
        available) or a small default.
        """
        out = []
        hist = self._corr_hist[-self.SURFACE_STAT_FRAMES:]
        if len(hist) < 10:
            return []
        xs_samples = []   # (cx, slot) for every non-empty slot each frame
        ys_samples = []
        for hst in hist:
            for slot in range(self.MAX_OBJECTS):
                j = slot * self.STATE_PER_OBJ
                cx, cy = hst[j], hst[j + 1]
                if abs(cx) < 1e-6 and abs(cy) < 1e-6:
                    continue
                if abs(cx) > 1e-6:
                    xs_samples.append((cx, slot))
                if abs(cy) > 1e-6:
                    ys_samples.append((cy, slot))
        out += self._cluster_surface_planes(xs_samples, normal_axis=0)
        out += self._cluster_surface_planes(ys_samples, normal_axis=1)
        return out

    def _cluster_surface_planes(self, samples, normal_axis):
        """Group samples by position along the normal axis; a tight cluster
        (var < SURFACE_POS_VAR) with >= SURFACE_STAT_FRAMES/2 samples is a
        surface plane. Returns [(live_slot, normal_axis, plane_pos,
        half_extent), ...]. half_extent from the live slot's shape or a small
        default."""
        if len(samples) < 10:
            return []
        pos = np.array([s[0] for s in samples])
        order = np.argsort(pos)
        pos_s = pos[order]
        gaps = np.diff(pos_s)
        split = np.where(gaps > self.SURFACE_GAP)[0]
        groups = np.split(order, split + 1)
        out = []
        for g in groups:
            if len(g) < self.SURFACE_MIN_N:
                continue
            gp = pos[g]
            if float(np.var(gp)) > self.SURFACE_POS_VAR:
                continue
            plane_pos = float(np.median(gp))
            live_slot = samples[g[-1]][1]
            sh = self._slot_shape.get(live_slot, (0.02, 0.02))
            half_extent = 0.5 * (sh[normal_axis] if sh[normal_axis] > 1e-6 else 0.02)
            out.append((live_slot, normal_axis, plane_pos, half_extent))
        return out

    # ==================================================================
    # The controlled object's action->motion model (DISCOVERED, for the
    # rollout). Without this, the imagined controlled object ignores the
    # imagined action, so every action imagines the same future -> a planner
    # cannot discriminate. GENERAL: 'the object I control moves along the
    # axis and at the rate my action has empirically moved it' -- discovered
    # from the action/position-delta correlation history (the SAME history
    # controlled_track uses), no Pong vocabulary.
    # ==================================================================
    def _controlled_action_model(self):
        """Return (axis, gain_per_action_scalar, neutral) for the controlled
        object, or None -- discovered from the controlled TRACK's perception
        velocity vs the action (clean: the track id has stable identity, so
        no slot-churn contamination). `axis` is 0 (x) or 1 (y); `gain` is the
        per-frame velocity per unit of the signed action scalar (the action's
        effect on the controlled object's motion), in normalized units.
        GENERAL: 'the object I control moves at the rate my action has
        empirically moved it.'"""
        if self._controlled_slot is None or self.n_obs < 200:
            return None
        hist = self._ctrl_track_vel   # list of (action, vx, vy) for the ctrl track
        if len(hist) < 60:
            return None
        neutral = self.num_actions - 1
        def scalar(a):
            a = int(a)
            if a == neutral: return 0.0
            return +1.0 if a < neutral else -1.0
        sc_all = np.array([scalar(h[0]) for h in hist], dtype=np.float64)
        if float(np.sqrt(((sc_all - sc_all.mean()) ** 2).sum())) < 1e-9:
            return None   # no action variation -> can't fit a slope
        best_axis, best_gain = None, 0.0
        for axis in (0, 1):
            # Regress the per-frame POSITION DELTA on the action scalar -- NOT
            # the reported velocity (an EMA that LAGS and underestimates the
            # true displacement; that made the imagined paddle ~5x too slow,
            # so the planner's intercept predictions were systematically
            # wrong). delta[t]=pos[t+1]-pos[t] is caused by action[t+1], so
            # align delta with sc_all[1:]. Track-id stable -> no slot churn.
            pos = np.array([h[1 + axis] for h in hist], dtype=np.float64)
            d = np.diff(pos)                 # px displacement per frame
            a = sc_all[1:]                    # the action that drove each delta
            d = d - d.mean()
            a0 = a - a.mean()
            vv = float(np.sqrt((d ** 2).sum()))
            if vv < 1e-9:
                continue
            denom = float((a0 ** 2).sum())
            if denom < 1e-12:
                continue
            gain = float((d * a0).sum() / denom)
            if abs(gain) > abs(best_gain):
                best_gain, best_axis = gain, axis
        if best_axis is None or abs(best_gain) < 1e-3:
            return None
        # gain is px/frame; normalize to the state's normalized units
        norm = self.frame_h if best_axis == 1 else self.frame_w
        return (best_axis, best_gain / norm, neutral)

    def _apply_action_effect(self, s, action, action_model):
        """In the imagination, the controlled object moves under the imagined
        action: set its position-delta this step to the action's signed
        effect (discovered gain * the action scalar), overriding the physics-
        default drift for that one slot. GENERAL: 'the object I control goes
        where my action sends it.'"""
        if action_model is None or self._controlled_slot is None:
            return s
        axis, gain, neutral = action_model
        a = int(action)
        scalar = 0.0 if a == neutral else (+1.0 if a < neutral else -1.0)
        # override the controlled slot's position advance along `axis` with
        # the action's effect (the physics default already advanced it by its
        # velocity; we ADD the action's delta on top -- a real paddle moves
        # both from its inertia and the action).
        j = self._controlled_slot * self.STATE_PER_OBJ
        s = s.copy()
        s[j + axis] += gain * scalar
        # reflect the action into the slot's velocity too (so the next physics
        # default carries it): set the velocity along the axis to the action's
        # per-step delta. This keeps the imagined paddle's motion consistent.
        s[j + 2 + axis] = gain * scalar
        return s

    def _apply_contact_reflection(self, s_before, s_after, surfaces):
        """The COMPUTED bounce (ContactNets' complementarity, made simple): if
        a moving slot CROSSES a discovered surface's plane along the normal,
        and was moving TOWARD it, reflect that slot's normal-velocity and
        clamp it to the surface (no penetration). Returns the (possibly
        reflected) s_after. e (restitution) is per-surface, learned online
        (defaults to 1.0 = elastic until learned). GENERAL -- a universal
        kinematic rule, no Pong vocabulary.
        """
        if not surfaces:
            return s_after
        s = s_after.copy()
        for (s_slot, axis, spos, half) in surfaces:
            # the surface plane is STATIONARY along its normal (that's how it
            # was discovered), so use the discovered plane position `spos`
            # directly -- not the imagined slot position (the live slot may
            # churn, and the plane doesn't move along the normal anyway).
            # elastic reflection (e=1.0; learning a per-surface restitution
            # is a future refinement -- the rare-event wall is already
            # sidestepped by computing the bounce).
            #
            # FINITE-SURFACE GATE: a surface is FINITE along the TANGENT axis
            # (a paddle has a short y-extent; only a wall spans everything).
            # It reflects a moving object ONLY if the two OVERLAP along the
            # tangent -- otherwise the object flies PAST the surface (no
            # bounce). Without this gate the paddle is an INFINITE WALL: the
            # ball always bounces at the plane whatever the paddle's y, so
            # every imagined action produces the same future and a planner
            # can never discriminate 'intercept' from 'miss'. The surface's
            # tangent position is its CURRENT imagined position (the paddle
            # MOVES along the tangent under the imagined action -- that is
            # exactly the lever the planner pulls). GENERAL: 'a finite
            # surface reflects only what actually hits it' -- no Pong vocab.
            tangent = 1 - axis
            sj = s_slot * self.STATE_PER_OBJ
            surf_t = float(s[sj + tangent])
            surf_present = (abs(s[sj]) + abs(s[sj + 1])) > 1e-6
            surf_shape = self._slot_shape.get(s_slot)
            # unknown extent -> treat as a full wall (always overlaps), so
            # we never LOSE a real bounce (e.g. W3b) to a missing shape.
            surf_half_t = (0.5 * surf_shape[tangent]
                           if surf_shape and surf_shape[tangent] > 1e-6
                           else 0.5)
            # The FINITE-SURFACE miss-gate applies ONLY to the surface I
            # CONTROL (the my-paddle): its imagined tangent position is
            # reliable (the action moves it -- that's the lever the planner
            # pulls), and its finiteness is what lets the planner discriminate
            # 'intercept' from 'miss'. OTHER surfaces (the opp paddle, walls)
            # reflect unconditionally (the old infinite-wall rule): their
            # discovered `s_slot` can be STALE under slot churn, so reading
            # their current imagined y is unreliable and would lose real
            # bounces (it regressed W3b). Robustness-driven, not Pong-specific:
            # we only trust the position of the object we ourselves move.
            finite = (s_slot == self._controlled_slot and surf_half_t < 0.5)
            for i in range(self.MAX_OBJECTS):
                if i == s_slot:
                    continue
                # only compact moving objects bounce off surfaces; skip
                # other surfaces and empty slots. Use the SHAPE aspect (robust
                # to dim-label noise) -- a near-square (aspect ~1) moving slot.
                ish = self._slot_shape.get(i)
                if ish is None:
                    continue
                iw, ih = ish
                if iw < 1e-6 or ih < 1e-6:
                    continue
                iasp = iw / ih
                if not (0.5 < iasp < 2.0):
                    continue   # elongated -> it's another surface, not a bouncer
                ij = i * self.STATE_PER_OBJ
                p_b = s_before[ij + axis]
                p_a = s_after[ij + axis]
                v = s[ij + 2 + axis]
                if abs(v) < 1e-6:
                    continue
                # crossed the surface plane (sign flip of (p - spos))?
                if (p_b - spos) * (p_a - spos) < 0:
                    # moving toward the surface (v points from before-side to
                    # the plane)
                    if ((p_b - spos) > 0 and v < 0) or ((p_b - spos) < 0 and v > 0):
                        # overlap along the tangent? (only for the finite
                        # controlled surface -- see `finite` above.)
                        obj_half_t = 0.5 * (ih if tangent == 1 else iw)
                        gap = abs(float(s[ij + tangent]) - surf_t)
                        if finite and surf_present \
                                and gap > (surf_half_t + obj_half_t):
                            continue   # no overlap -> the object flies PAST
                        # reflect normal velocity, clamp position to the surface
                        s[ij + 2 + axis] = -v
                        s[ij + axis] = spos + np.sign(p_b - spos) * (half + 1e-4)
        return s

    def rollout_from_state(self, state, first_action, action_fn, horizon=20,
                controlled_id=None, dynamic=True):
        """Imagine forward from a given state (rollout's tracks-free variant;
        the policy planner already has a state and doesn't need to rebuild it
        from tracks). Same semantics as rollout."""
        return self._rollout(state, first_action, action_fn, horizon,
                             controlled_id, dynamic)

    def rollout(self, tracks, first_action, action_fn, horizon=20,
                controlled_id=None, dynamic=True):
        """Imagine forward from the current tracks. Builds the state from
        tracks, then delegates to _rollout. See rollout_from_state for the
        full semantics."""
        s = self.state_from_tracks(tracks, controlled_id)
        return self._rollout(s, first_action, action_fn, horizon,
                             controlled_id, dynamic)

    def _rollout(self, s, first_action, action_fn, horizon,
                 controlled_id, dynamic):
        """Imagine forward from a state. Each imagined step =
        EXACT physics default (free-flight kinematics) + the COMPUTED Way C
        contact reflection (a bounce is generated by a rule, not learned --
        the rare-event wall sidestepped). The reward is predicted by the net
        (cheap, does not affect the imagined state). All in the agent's head;
        the servo never moves.

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
          run the full `horizon`.

        Returns (states, rewards, cumulative_reward, meta) where meta =
        {"stop_reason": str|None, "stopped_at": int} -- stopped_at is the
        number of imagined steps taken (== horizon if it ran the full way).
        states always includes the starting state, so len(states) ==
        stopped_at + 1."""
        states = [s.copy()]
        rewards = []
        a = int(first_action)
        stop_reason = None
        t = 0
        # WAY C: discover surfaces ONCE from the recent real-stream history
        # (they barely move, so the start-of-rollout snapshot is valid through
        # the imagined horizon; re-discovering each step is unnecessary).
        surfaces = self._discover_surfaces()
        # the controlled object's action->motion model (DISCOVERED from the
        # action/position-delta correlation history): which axis the action
        # moves it on, the sign, and the magnitude per action unit. Without
        # this, the imagined controlled object drifts at its current velocity
        # regardless of the imagined action -> the imagined future is the same
        # for every action -> a planner cannot discriminate. GENERAL: 'the
        # object I control moves along the axis and at the rate my action has
        # empirically moved it.'
        action_model = self._controlled_action_model()
        for t in range(horizon):
            s_before = s
            # physics default (exact free-flight kinematics) + reward from the
            # net (does not affect the imagined state). The learned STATE
            # correction is NOT applied in the rollout -- the physics default
            # is exact for free flight, and the correction (trained on the
            # now-fixed residual) only adds drift until it relearns.
            s = self._physics_default(s)
            # apply the imagined action's effect on the controlled object
            s = self._apply_action_effect(s, a, action_model)
            _, r = self._predict(s_before, a)
            # WAY C: the COMPUTED bounce -- reflect moving slots that cross a
            # discovered surface.
            s = self._apply_contact_reflection(s_before, s, surfaces)
            states.append(s.copy())
            rewards.append(float(r))
            if dynamic:
                bad, why = self._is_unrealistic(s, 0.0)
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
