"""
The SCAFFOLD (part 2) — a general object tracker.

Designed, domain-general. Knows NOTHING about balls, paddles, screens, or
Pong. Three evolved-style rules turn proposal candidates into persistent
objects:

  PERSISTENCE  : a region must appear MIN_AGE frames to be "confirmed"; it
                 dies after MAX_MISS frames unseen. A one-frame glare flash
                 never becomes an object.
  COHESION     : an object moves as a whole; its size stays roughly stable
                 (gated by log-area ratio).
  CONTINUITY   : an object travels on a smooth constant-velocity path; a
                 candidate too far from any track's prediction starts a NEW
                 track rather than hijacking an existing one.

DIMENSIONALITY is a scaffold CUE, not identity: each confirmed track is
tagged 0d (compact) / 1d (elongated) / 2d (planar). No names, no roles.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

DIM_COMPACT = "0d"       # roughly round/square blob  (e.g. a ball)
DIM_ELONGATED = "1d"     # tall-thin or wide-short    (e.g. a paddle)
DIM_PLANAR = "2d"        # large filled rectangle      (e.g. a screen)


# =====================================================================
# Move 2: behavioral utility scorer (the adaptive junk-rejector)
# =====================================================================
# Each track gets a score = w . features, where the features are BEHAVIORAL
# (shape stability, motion smoothness, persistence, action coupling) -- not
# brightness. Contrast-invariant by construction: a dim paddle and a bright
# paddle both have a stable shape and smooth motion. Glare / score text fail
# these and get retired no matter how salient they were at the proposal seed.
#
# The weights w are LEARNED ONLINE by IDBD (the Alberta Plan's Step-1 named
# per-weight meta-learned step-size method): each weight has its own step-size
# beta that grows when the weight is helping predict "is this track a good
# predictor of its own next position?" and shrinks when it is not. So the
# scorer self-tunes to THIS rig's particular junk (glare that happens to be
# shape-stable -> shape weight decays; jittery glare -> motion weight grows).
#
# Action coupling (feature 4) is the strongest reality+importance signal but
# only exists once the agent emits non-neutral actions (servo in the loop).
# Until then its weight stays near its prior and the score leans on the other
# three. It switches on automatically the first time an action != neutral is
# seen -- no phase switch.

class IDBD:
    """Incremental Delta-Bar-Delta: per-weight meta-learned step size.

    Maintains one step-size per weight. The step-size exponent tau is updated
    from the recent correlation between the weight's update direction and a
    decaying average of past updates (the 'bar' term): consistently useful
    directions -> larger step; flippy directions -> smaller step. This is
    Sutton's IDBD (Sutton 1992), the Alberta Plan Step-1 anti-drift tool.
    Closed-form, O(d) per step, no batches, no replay.

    We use it to learn the utility WEIGHTS online: the gradient of the
    per-track prediction-error objective w.r.t. each weight drives tau.
    """

    def __init__(self, n_weights, theta=0.01, init_step=0.1, init_weight=0.5):
        self.w = np.full(n_weights, init_weight)    # weights (positive prior)
        self.h = np.zeros(n_weights)                 # bar term (recent-update EMA)
        self.tau = np.full(n_weights, np.log(init_step))  # log step-size per weight
        self.theta = theta                           # meta step-size

    def weights(self):
        return self.w.copy()

    def step(self, grad):
        """One IDBD update. `grad` = d(objective)/d(weight) for this step.
        Positive grad = increasing this weight reduces error."""
        step = np.exp(self.tau)
        self.w -= step * grad                        # gradient descent on weights
        self.tau += self.theta * grad * self.h       # grow step where useful
        self.h = np.maximum(0.0, 1.0 - step) * self.h + step * grad  # bar term


class BehaviorUtility:
    """Scores each track by behavioral reality. Learns the weights online."""

    N_FEATS = 4  # shape_stab, motion_smooth, persistence, action_couple

    def __init__(self, retire_below=0.0, warmup=5):
        self.idbd = IDBD(self.N_FEATS, theta=0.01, init_step=0.05,
                         init_weight=0.5)
        self.retire_below = retire_below        # utility cutoff for retirement
        self.warmup = warmup                    # frames before utility is trusted
        self._action_seen = False               # flips on first non-neutral action
        self._neutral = None                    # set by Perception (STAY index)

    def set_neutral(self, neutral_idx):
        self._neutral = neutral_idx

    # ---- per-track feature extraction (all behavioral, all [0,1]-ish) ----
    @staticmethod
    def _feat_shape_stab(hist):
        # variance of log-area and aspect over the recent history. Real
        # objects keep a stable shape; score text / on-off glare jump.
        if len(hist) < 3:
            return 0.0
        la = np.log(np.maximum([h["area"] for h in hist], 1.0))
        asp = np.array([h["aspect"] for h in hist])
        # low variance -> high stability -> high feature
        s_la = 1.0 / (1.0 + 10.0 * float(np.var(la)))
        s_as = 1.0 / (1.0 + 10.0 * float(np.var(asp)))
        return 0.5 * (s_la + s_as)

    @staticmethod
    def _feat_motion_smooth(hist):
        # agreement between consecutive velocities. Real objects coast; glare
        # twitches (velocity flips / spikes).
        if len(hist) < 4:
            return 0.0
        v = np.array([[h["vx"], h["vy"]] for h in hist])
        dv = np.diff(v, axis=0)
        jitter = float(np.mean(np.linalg.norm(dv, axis=1)))
        speed = float(np.mean(np.linalg.norm(v, axis=0))) + 1e-6
        # smooth motion = low jitter relative to speed
        return 1.0 / (1.0 + 5.0 * jitter / speed)

    @staticmethod
    def _feat_persistence(track):
        # age vs missed: long-lived, rarely-lost tracks score high.
        age = max(track.age, 1)
        miss_rate = track.missed / float(age)
        return 1.0 / (1.0 + 5.0 * miss_rate) * min(1.0, age / 30.0)

    @staticmethod
    def _feat_action_couple(act_hist):
        # correlation between the agent's action and this track's vertical
        # velocity. My paddle: strong. Ball/opponent/glare: ~0. Only meaningful
        # once non-neutral actions have occurred; returns 0 otherwise.
        if not act_hist or len(act_hist) < 6:
            return 0.0
        acts = np.array([a[0] for a in act_hist], dtype=float)
        vys = np.array([a[1] for a in act_hist], dtype=float)
        if acts.std() < 1e-6:
            return 0.0  # no action variation yet -> can't measure coupling
        # pearson-ish, clipped to [0,1] (we care about magnitude of coupling)
        c = float(np.corrcoef(acts, vys)[0, 1]) if acts.std() > 0 and vys.std() > 0 else 0.0
        return float(np.clip(abs(c), 0.0, 1.0))

    def features(self, track):
        return np.array([
            self._feat_shape_stab(track.hist),
            self._feat_motion_smooth(track.hist),
            self._feat_persistence(track),
            self._feat_action_couple(track.act_hist),
        ])

    def score(self, track):
        return float(self.idbd.w @ self.features(track))

    def note_action(self, action):
        if self._neutral is not None and action != self._neutral:
            self._action_seen = True

    def update(self, track, pred_err):
        """Learn the weights online from this track's prediction error.

        Treat the utility score as a linear predictor of "is this track a
        good, learnable object?". The target is 1 - tanh(pred_err): a track
        that is easy to predict (low error -> real, coasting object) targets
        ~1; a track that is hard to predict (high error -> jittery glare)
        targets ~0. We descend the squared-error gradient
        (score - target) * features, so weights grow for features that mark
        low-error (real) tracks and shrink for features that mark high-error
        (junk) tracks. IDBD gives each weight its own meta-learned step size
        (the Alberta Plan Step-1 anti-drift tool)."""
        if len(track.hist) < self.warmup:
            return
        f = self.features(track)
        s_err = float(np.tanh(pred_err))           # bounded in (-1, 1)
        target = 1.0 - s_err                        # low err -> ~1 (real), high -> ~0 (junk)
        score = float(self.idbd.w @ f)
        grad = (score - target) * f                 # d/dw 0.5*(score-target)^2
        self.idbd.step(grad)

    def should_retire(self, track):
        # Need enough history for the shape/motion features to be meaningful;
        # don't retire young tracks before they've proven themselves.
        if track.age < max(self.warmup, 8) or len(track.hist) < 4:
            return False
        return self.score(track) < self.retire_below


def classify_dim(aspect, extent, min_dim):
    """Tag a region by dimensionality. A CUE, not an identity.
    min_dim = min(w, h) in px; a planar region must be large in BOTH dims
    (a screen), not just a small filled square (a ball)."""
    if aspect > 2.5 or aspect < 0.4:
        return DIM_ELONGATED
    if extent > 0.8 and min_dim >= 30:
        return DIM_PLANAR
    return DIM_COMPACT


class Track:
    """A single persistent object hypothesis."""

    HIST_LEN = 24        # recent per-frame shape/vel snapshots for behavior feats
    ACT_LEN = 24         # recent (action, vy) pairs for action-coupling feat

    def __init__(self, track_id, cand, frame):
        self.id = track_id
        self.cx, self.cy = cand["cx"], cand["cy"]
        self.w, self.h = cand["w"], cand["h"]
        self.area, self.aspect, self.extent = cand["area"], cand["aspect"], cand["extent"]
        self.dim = classify_dim(self.aspect, self.extent, min(self.w, self.h))
        self.vx = self.vy = 0.0
        self.age = 1
        self.missed = 0
        self.confirmed = False
        self.path = [(self.cx, self.cy)]
        # Move 2: behavior-feature buffers + utility score (set by tracker).
        self.hist = []          # recent {area, aspect, vx, vy} snapshots
        self.act_hist = []      # recent (action, vy) pairs
        self.utility = 0.0      # last behavioral-utility score
        self.last_pred_err = 0.0  # |observed - predicted position| at last match

    def _predicted(self):
        return self.cx + self.vx, self.cy + self.vy

    def _recent_speed(self):
        """A robust recent speed (the median |v| over the history) for the
        physical-plausibility velocity clamp. Median is robust to the spike
        itself."""
        if len(self.hist) < 3:
            return (self.vx ** 2 + self.vy ** 2) ** 0.5
        speeds = [(h["vx"] ** 2 + h["vy"] ** 2) ** 0.5 for h in self.hist]
        return float(np.median(speeds))

    def update(self, cand, frame, pos_ema=0.85, shape_ema=0.5, action=None):
        # prediction error = distance between this match and where we predicted
        # the object would be. Low for real coasting objects, high for jittery
        # glare. This is the signal the utility learner trains on.
        px, py = self._predicted()
        self.last_pred_err = float(((cand["cx"] - px) ** 2 +
                                    (cand["cy"] - py) ** 2) ** 0.5)
        ncx, ncy = cand["cx"], cand["cy"]
        # NO-SNAP RE-ACQUISITION (the contact/collision fix). If this track was
        # COASTING (missed > 0 -- it was occluded, typically because two blobs
        # merged at a contact/collision), then during the coast cx/cy advanced
        # by the FROZEN velocity. So (ncx - self.cx) here is the GAP-SPANNING
        # displacement over several frames, NOT a per-frame velocity. Feeding
        # that into the velocity EMA produces an unphysical SPIKE (measured:
        # ~-13px/frame when real motion is ~-2px/frame) -- the artifact that
        # hides the bounce from the world model. Instead: accept the new
        # POSITION (the candidate is where the object actually is -- reliable),
        # but KEEP the pre-coast velocity this frame and let the next few FRESH
        # frames' EMA correct it smoothly. Kills the spike; the pre-coast
        # velocity is corrected over 2-3 fresh frames instead of one huge
        # jump. This is the contained, decoupled fix -- it touches only the
        # ~3 frames around a re-acquisition, not the general velocity
        # character, so it does not regress downstream consumers (W2).
        reacquiring = self.missed > 0
        if reacquiring:
            # position: accept the fresh observation (reliable).
            self.cx = pos_ema * ncx + (1 - pos_ema) * self.cx
            self.cy = pos_ema * ncy + (1 - pos_ema) * self.cy
            # velocity: UNCHANGED this frame (no snap to the gap displacement).
            # The normal EMA below resumes on the next fresh frame.
        else:
            # position: responsive (so the per-frame displacement the model
            # learns from actually reflects the action; a heavy position EMA
            # smears the action signal across frames and the model can't learn
            # causality).
            self.vx = pos_ema * (ncx - self.cx) + (1 - pos_ema) * self.vx
            self.vy = pos_ema * (ncy - self.cy) + (1 - pos_ema) * self.vy
            self.cx = pos_ema * ncx + (1 - pos_ema) * self.cx
            self.cy = pos_ema * ncy + (1 - pos_ema) * self.cy
            # PHYSICAL-PLAUSIBILITY CLAMP: a fragment track that snaps from
            # one blob to another (a perception artifact, not real motion)
            # would otherwise produce an unphysical velocity spike that hides
            # the bounce from the world model. Cap |v| at a multiple of the
            # track's OWN recent (median) speed -- a real object's velocity
            # doesn't suddenly jump far beyond its recent norm; a fragment
            # hopping blobs does. The median is robust to the spike itself.
            # GENERAL -- 'velocity shouldn't discontinuously jump far beyond
            # this object's recent norm'; no Pong vocabulary.
            v = (self.vx ** 2 + self.vy ** 2) ** 0.5
            recent = max(self._recent_speed(), 2.0)
            vmax = 2.0 * recent
            if v > vmax and v > 1e-6:
                self.vx *= vmax / v
                self.vy *= vmax / v
        # shape: stable (used for cohesion gating + dimensionality cue; jitter
        # here would split/merge tracks spuriously).
        for a in ("w", "h", "area", "aspect", "extent"):
            setattr(self, a, shape_ema * cand[a] + (1 - shape_ema) * getattr(self, a))
        self.dim = classify_dim(self.aspect, self.extent, min(self.w, self.h))
        self.age += 1
        self.missed = 0
        self.confirmed = self.age >= ObjectTracker.MIN_AGE
        self.path.append((self.cx, self.cy))
        if len(self.path) > 64:
            self.path.pop(0)
        # Move 2: record behavior-feature snapshots + (action, vy) pairs.
        self.hist.append({"area": self.area, "aspect": self.aspect,
                          "vx": self.vx, "vy": self.vy})
        if len(self.hist) > self.HIST_LEN:
            self.hist.pop(0)
        if action is not None:
            self.act_hist.append((int(action), float(self.vy)))
            if len(self.act_hist) > self.ACT_LEN:
                self.act_hist.pop(0)

    def coast(self, velocity_hint=None):
        # MODEL-ASSISTED COAST (close the perception<->model loop). When a
        # track is occluded (its blob merged with another at a contact /
        # collision -- the canonical case is the ball touching a paddle), the
        # old behavior FREEZES the pre-bounce velocity -> the gap velocity has
        # the WRONG SIGN (the ball already bounced). If a `velocity_hint`
        # (vx, vy) is provided by the world model (a bounce-aware prediction),
        # use it instead -- so the gap velocity is plausible, not frozen-wrong.
        # If no hint (cold model / not believable), freeze as before.
        if velocity_hint is not None:
            self.vx, self.vy = float(velocity_hint[0]), float(velocity_hint[1])
        self.cx += self.vx
        self.cy += self.vy
        self.missed += 1

    def as_dict(self):
        return {"id": self.id, "cx": self.cx, "cy": self.cy,
                "vx": self.vx, "vy": self.vy, "w": self.w, "h": self.h,
                "area": self.area, "aspect": self.aspect, "extent": self.extent,
                "dim": self.dim, "age": self.age, "missed": self.missed,
                "confirmed": self.confirmed, "path": list(self.path),
                "utility": self.utility, "pred_err": self.last_pred_err}


class ObjectTracker:
    """General object tracker = the perception SCAFFOLD. See module docstring."""

    MIN_AGE = 3
    MAX_MISS = 10
    GATE_POS = 35.0
    GATE_LOGAREA = 1.0
    COST_DIM = 40.0

    def __init__(self, utility=None):
        self._tracks = []
        self._next_id = 1
        # Move 2: optional behavioral-utility scorer. If provided, tracks are
        # scored every frame, the weights learn from each track's prediction
        # error, and low-utility tracks are retired (junk rejection by
        # behavior, not brightness). If None, behavior is unchanged (legacy).
        self.utility = utility

    def reset(self):
        self._tracks = []
        self._next_id = 1

    def update(self, candidates, frame, action=None, velocity_hint_fn=None):
        if self.utility is not None and action is not None:
            self.utility.note_action(action)
        preds = [t._predicted() for t in self._tracks]
        T, C = len(self._tracks), len(candidates)
        BIG = 1e6
        cost = np.full((T, C), BIG)
        for ti, t in enumerate(self._tracks):
            px, py = preds[ti]
            gate_pos = self.GATE_POS + 2.0 * (t.vx ** 2 + t.vy ** 2) ** 0.5
            log_a = np.log(max(t.area, 1.0))
            for ci, c in enumerate(candidates):
                dpos = ((c["cx"] - px) ** 2 + (c["cy"] - py) ** 2) ** 0.5
                if dpos > gate_pos:
                    continue  # continuity
                dla = abs(np.log(max(c["area"], 1.0)) - log_a)
                if dla > self.GATE_LOGAREA:
                    continue  # cohesion
                dim_pen = (0.0 if classify_dim(c["aspect"], c["extent"],
                                                min(c["w"], c["h"])) == t.dim
                           else self.COST_DIM)
                cost[ti, ci] = dpos + 20.0 * dla + dim_pen

        matched_t, matched_c = set(), set()
        if T and C:
            rows, cols = linear_sum_assignment(cost)
            for ti, ci in zip(rows, cols):
                if cost[ti, ci] >= BIG:   # gated out (no real match)
                    continue
                self._tracks[ti].update(candidates[ci], frame, action=action)
                matched_t.add(ti)
                matched_c.add(ci)
                # Move 2: train the utility weights from this track's
                # prediction error (low for coasting real objects, high for
                # jittery glare). Done here while ti indexes the pre-append list.
                if self.utility is not None:
                    self.utility.update(self._tracks[ti],
                                        self._tracks[ti].last_pred_err)

        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                hint = None
                if velocity_hint_fn is not None:
                    hint = velocity_hint_fn(t.as_dict())
                t.coast(velocity_hint=hint)
        for ci, c in enumerate(candidates):
            if ci not in matched_c:
                self._tracks.append(Track(self._next_id, c, frame))
                self._next_id += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.MAX_MISS]

        # Move 2: score every track, learn weights from prediction error, then
        # retire low-utility junk (glare / score text / one-frame flashes that
        # happened to persist). Judged by BEHAVIOR (shape/motion/persistence/
        # action-coupling), which is contrast-invariant -- a dim paddle and a
        # bright paddle both score well; bright glare scores poorly.
        if self.utility is not None:
            for t in self._tracks:
                t.utility = self.utility.score(t)
            self._tracks = [t for t in self._tracks
                            if not self.utility.should_retire(t)]

        return [t.as_dict() for t in self._tracks if t.confirmed]

    # =====================================================================
    # TOP-DOWN / prediction-conditioned path (step 2 of the contact fix).
    # =====================================================================
    # The legacy `update` above is BOTTOM-UP: it runs the Hungarian matcher
    # over connected-component blobs, so the moment two objects touch they
    # become ONE blob and the matcher can no longer give the ball its own
    # candidate -- identity is gone before the matcher runs. The conditioned
    # path replaces the matcher for KNOWN tracks: the proposal already carved
    # out one candidate per track's prediction (Voronoi-by-prediction), so
    # track i simply takes claimed[i]. Identity is carried top-down and a
    # merge can no longer destroy it. The cohesion/continuity protection the
    # matcher gave is subsumed by the per-track search radius (a track only
    # claims pixels near its prediction -- a way-off prediction claims
    # nothing -> coast, same as a continuity gate failing). GENERAL -- no
    # Pong vocabulary.

    def predictions(self):
        """Per-track predicted pixel positions + search radius, for the
        prediction-conditioned proposal. One (px, py, radius) per track, in
        self._tracks order. The radius is the SAME continuity gate the legacy
        matcher uses (GATE_POS + a velocity margin)."""
        out = []
        for t in self._tracks:
            px, py = t._predicted()
            speed = (t.vx ** 2 + t.vy ** 2) ** 0.5
            radius = self.GATE_POS + 2.0 * speed
            out.append((px, py, radius))
        return out

    def update_conditioned(self, claimed, residual, frame, action=None,
                           velocity_hint_fn=None):
        """The top-down path. `claimed` is a list aligned with self._tracks
        (one candidate per track, carved out by the prediction-conditioned
        proposal -- or None if no salient pixels were at that track's
        prediction -> coast). `residual` is the list of new-object candidates
        (salient pixels no track claimed -> the bottom-up path, demoted to
        'new things only').

        Track i takes claimed[i] directly (no matcher -- the conditioned
        proposal already solved the assignment geometrically). Unseen tracks
        coast (with a model-assisted velocity hint if provided). Residual
        candidates start new tracks. Utility scoring + retirement run as in
        the legacy path."""
        if self.utility is not None and action is not None:
            self.utility.note_action(action)
        T = len(self._tracks)
        if len(claimed) != T:
            raise ValueError(
                f"claimed (len {len(claimed)}) must align with tracks "
                f"(len {T}) -- predictions() and update_conditioned() must "
                f"be called with no tracker mutation in between")
        for ti, t in enumerate(self._tracks):
            c = claimed[ti]
            if c is not None:
                # KIND-CONSISTENCY gate: a track only accepts a candidate whose
                # dimensionality matches its OWN kind. A 1d paddle must NOT
                # accept a 0d compact blob (that is a DIFFERENT object -- the
                # ball -- stuck to it); accepting it would hijack the ball's
                # identity (the paddle track moves to the ball's position and
                # flips to 0d, the ball track starves and dies -> no
                # continuous velocity across the contact -> the W3b blocker).
                # Reject the mismatched candidate -> the track coasts (object
                # permanence) and the ball keeps its own track. GENERAL -- 'a
                # track only owns pixels of its own object kind'; no Pong
                # vocabulary (the dimensionality cue is a general scaffold
                # property, not a Pong label).
                cand_dim = classify_dim(c["aspect"], c["extent"],
                                        min(c["w"], c["h"]))
                if t.confirmed and t.dim is not None and cand_dim != t.dim \
                        and t.dim != DIM_PLANAR:
                    c = None   # mismatched kind -> coast instead of hijack
            if c is not None:
                t.update(c, frame, action=action)
                if self.utility is not None:
                    self.utility.update(t, t.last_pred_err)
            if c is None:
                hint = None
                if velocity_hint_fn is not None:
                    hint = velocity_hint_fn(t.as_dict())
                t.coast(velocity_hint=hint)
        # residual candidates -> new tracks (the bottom-up 'new things' path)
        for c in residual:
            self._tracks.append(Track(self._next_id, c, frame))
            self._next_id += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.MAX_MISS]
        # DUPLICATE-FRAGMENT DROP: the conditioned path can persist two
        # non-overlapping halves of one elongated object (a tall paddle split
        # top+bottom). Drop the younger half; the survivor reclaims the whole
        # blob next frame. Ball-protected (see _fragments_of_one_object).
        self._drop_duplicate_fragments()
        # utility scoring + retirement (same as the legacy path)
        if self.utility is not None:
            for t in self._tracks:
                t.utility = self.utility.score(t)
            self._tracks = [t for t in self._tracks
                            if not self.utility.should_retire(t)]
        return [t.as_dict() for t in self._tracks if t.confirmed]

    def _fragments_of_one_object(self, a, b):
        # Two confirmed tracks that are NON-overlapping fragments of ONE
        # elongated object, split along its long axis: SAME dimensionality
        # (both 1d elongated), aligned on one axis, adjacent (small gap) on
        # the other, similar size, same velocity. (The conditioned path
        # bypasses the legacy matcher's one-blob-one-track rule, so a tall
        # paddle can persist as a top half + a bottom half -- the slot-
        # exhaustion blocker. The halves touch but don't overlap, so a
        # containment test misses them; this adjacency test catches them.)
        # The ball is protected by the SAME-DIM guard: a ball (0d) stuck to a
        # paddle (1d) is NOT a fragment of it, even when the paddle's carved
        # blob is briefly mis-classified 0d by the proposal -- the ball is a
        # DIFFERENT KIND of object (compact vs elongated). Two halves of one
        # paddle are BOTH 1d. GENERAL -- 'two side-by-side same-shaped same-
        # sized same-velocity pieces of one object'; no Pong vocabulary.
        if a.dim != b.dim:
            return False   # different object kinds -> never fragments of one
        ax0, ax1 = a.cx - a.w / 2, a.cx + a.w / 2
        ay0, ay1 = a.cy - a.h / 2, a.cy + a.h / 2
        bx0, bx1 = b.cx - b.w / 2, b.cx + b.w / 2
        by0, by1 = b.cy - b.h / 2, b.cy + b.h / 2
        ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        oy = max(0.0, min(ay1, by1) - max(ay0, by0))
        minw, minh = min(a.w, b.w), min(a.h, b.h)
        aligned_x = ox > 0.6 * minw
        aligned_y = oy > 0.6 * minh
        if aligned_x and not aligned_y:
            gap = max(0.0, max(ay0, by0) - min(ay1, by1))
            if gap > 0.5 * minh:
                return False
        elif aligned_y and not aligned_x:
            gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
            if gap > 0.5 * minw:
                return False
        else:
            return False
        # similar size (within 2x): a fragment is a comparable chunk; the ball
        # (~0.2x) is not.
        lo = max(min(a.area, b.area), 1.0)
        if max(a.area, b.area) > 2.0 * lo:
            return False
        # same velocity direction: a ball crossing a paddle moves
        # orthogonally to it -> different object, keep both.
        sa = (a.vx ** 2 + a.vy ** 2) ** 0.5
        sb = (b.vx ** 2 + b.vy ** 2) ** 0.5
        if sa > 1.0 and sb > 1.0:
            cosang = (a.vx * b.vx + a.vy * b.vy) / (sa * sb)
            if abs(cosang) < 0.5:
                return False
        return True

    def _drop_duplicate_fragments(self):
        tracks = self._tracks
        drop = set()
        for i in range(len(tracks)):
            if i in drop or not tracks[i].confirmed:
                continue
            for j in range(i + 1, len(tracks)):
                if j in drop or not tracks[j].confirmed:
                    continue
                if not self._fragments_of_one_object(tracks[i], tracks[j]):
                    continue
                # drop the YOUNGER (lower age; tiebreak higher id = younger);
                # the survivor's prediction claims the whole blob next frame.
                ai, bi = tracks[i], tracks[j]
                if (ai.age, -ai.id) >= (bi.age, -bi.id):
                    drop.add(j)
                else:
                    drop.add(i)
        if drop:
            self._tracks = [t for k, t in enumerate(tracks) if k not in drop]
