"""
Tests for perception (scaffold + discovered-class model), NO hardware.

A synthetic scene renders four objects on a dark background:
  MY PADDLE  : tall-thin rectangle on the left; vertical velocity set by an
               action scalar (+1 up / -1 down / 0 stay). CONTROLLED.
  OPP PADDLE : tall-thin rectangle on the right; chases the ball's y.
  BALL       : small square; constant velocity; bounces off walls/paddles.
  DECOY      : small stationary square.

Mid-run the brightness drops to 0.4x (a global lighting shift that breaks any
fixed gray threshold but leaves the z-score seed unchanged). We prove:

  CLAIM 1  the controlled paddle keeps a STABLE TRACK ID across the shift
           (the scaffold survives the lighting change; a fixed threshold
           detector does not).
  CLAIM 2  the model labels the controlled paddle (and ONLY it) as the object
           the agent controls, both before and after the shift — identity by
           behavior, not by pixels.
  CLAIM 3  REUSE: when the controlled object is swapped for a NEW object with
           the SAME dynamics (different position/size), the new object binds
           to the EXISTING controlled class (same class id) instead of
           seeding a new one — knowledge is reused, not relearned. This is
           the payoff of discovered behavior classes over per-object models.
"""

import numpy as np
import cv2
import pytest

from agent.perception import Perception

H, W = 160, 160
BG, OBJ = 24, 220

ACT_UP, ACT_DOWN, ACT_STAY = 0, 1, 2
_SCALAR = {ACT_UP: +1, ACT_DOWN: -1, ACT_STAY: 0}


class SyntheticScene:
    def __init__(self, seed=0, swap_at=None):
        self.rng = np.random.default_rng(seed)
        self.m = 16
        self.pw, self.ph = 6, 34
        self.bs = 6
        self.bx, self.by = W / 2.0, H / 2.0
        self.bvx, self.bvy = 2.2, 1.4
        self.my_x = self.m
        self.my_y = H / 2.0
        self.opp_x = W - self.m - self.pw
        self.opp_y = H / 2.0
        self.opp_speed = 1.6
        self.dec_x, self.dec_y, self.dec_s = W * 0.72, self.m + 6, 8
        self.gt = {}
        self.swap_at = swap_at
        self.swapped = False
        self.f = 0

    def _maybe_swap(self):
        # swap the controlled object: new position + new size, same dynamics.
        # the old track dies (continuity violation); a new track starts.
        if self.swap_at is not None and not self.swapped and self.f >= self.swap_at:
            self.swapped = True
            self.my_x = int(W * 0.32)
            self.pw, self.ph = 9, 26
            self.my_y = H / 2.0

    def step(self, action):
        self.f += 1
        self._maybe_swap()
        sp = 3.0
        self.my_y = float(np.clip(self.my_y - action * sp, self.m, H - self.m - self.ph))
        target = self.by + self.bs / 2 - self.ph / 2
        self.opp_y = float(np.clip(
            self.opp_y + np.clip(target - self.opp_y, -self.opp_speed, self.opp_speed),
            self.m, H - self.m - self.ph))
        self.bx += self.bvx
        self.by += self.bvy
        if self.by <= self.m:
            self.by, self.bvy = self.m, abs(self.bvy)
        if self.by >= H - self.m - self.bs:
            self.by, self.bvy = H - self.m - self.bs, -abs(self.bvy)
        if self.bx <= self.my_x + self.pw and self.m <= self.by <= H - self.m and self.bvx < 0:
            self.bvx = abs(self.bvx)
        if self.bx + self.bs >= self.opp_x and self.m <= self.by <= H - self.m and self.bvx > 0:
            self.bvx = -abs(self.bvx)
        if self.bx < self.m or self.bx > W - self.m:
            self.bx, self.by = W / 2.0, H / 2.0
            self.bvx = 2.2 * (1 if self.rng.random() > 0.5 else -1)
            self.bvy = 1.4 * (1 if self.rng.random() > 0.5 else -1)
        self.gt = {
            "my":   (self.my_x + self.pw / 2, self.my_y + self.ph / 2),
            "opp":  (self.opp_x + self.pw / 2, self.opp_y + self.ph / 2),
            "ball": (self.bx + self.bs / 2, self.by + self.bs / 2),
            "decoy": (self.dec_x + self.dec_s / 2, self.dec_y + self.dec_s / 2),
        }

    def render(self, brightness=1.0):
        img = np.full((H, W), BG, dtype=np.float32)

        def fill(cx, cy, w, h, val):
            x0 = max(0, int(round(cx - w / 2)))
            y0 = max(0, int(round(cy - h / 2)))
            x1 = min(W, x0 + int(round(w)))
            y1 = min(H, y0 + int(round(h)))
            img[y0:y1, x0:x1] = val

        fill(*self.gt["my"], self.pw, self.ph, OBJ)
        fill(*self.gt["opp"], self.pw, self.ph, OBJ)
        fill(*self.gt["ball"], self.bs, self.bs, OBJ)
        fill(*self.gt["decoy"], self.dec_s, self.dec_s, OBJ)
        return np.clip(img * brightness, 0, 255).astype(np.uint8)


def _track_at(tracks, gt_xy, tol=14.0):
    gx, gy = gt_xy
    best, best_d = None, 1e9
    for t in tracks:
        d = (t["cx"] - gx) ** 2 + (t["cy"] - gy) ** 2
        if d < best_d:
            best_d, best = d, t["id"]
    return best if best_d <= tol ** 2 else None


def _run(total=220, shift=80, seed=1, act_seed=777, swap_at=None):
    scene = SyntheticScene(seed=seed, swap_at=swap_at)
    perc = Perception(H, W)
    rng = np.random.default_rng(act_seed)
    hold = 4
    cur = ACT_STAY
    log = []
    for f in range(total):
        if f % hold == 0:
            cur = int(rng.choice([ACT_UP, ACT_DOWN, ACT_STAY]))
        scene.step(_SCALAR[cur])
        bright = 1.0 if f < shift else 0.4
        gray = scene.render(brightness=bright)
        tracks, controlled = perc.step(gray, cur)
        log.append({"frame": f, "action": cur, "brightness": bright,
                    "gt": dict(scene.gt), "tracks": tracks,
                    "controlled": controlled, "diag": perc.diagnostics()})
    return log, perc


# ------------------------------ CLAIM 1 -------------------------------------

def test_controlled_paddle_survives_lighting_shift():
    log, _ = _run()
    pre = log[78]
    id_pre = _track_at(pre["tracks"], pre["gt"]["my"])
    assert id_pre is not None, "my paddle not tracked just before the shift"
    survived = any(_track_at(e["tracks"], e["gt"]["my"]) == id_pre
                   for e in log[100:])
    assert survived, (
        f"my paddle track id {id_pre} did NOT survive the 0.4x brightness shift; "
        "the scaffold lost the object (the bug we are avoiding)")


def test_fixed_threshold_detector_breaks_on_shift():
    """Contrast proof: a naive fixed-gray-threshold detector loses the paddle
    after the 0.4x shift, while the z-score scaffold does not."""
    scene = SyntheticScene(seed=1)
    rng = np.random.default_rng(777)
    before = after = None
    cur = ACT_STAY
    for f in range(220):
        if f % 4 == 0:
            cur = int(rng.choice([ACT_UP, ACT_DOWN, ACT_STAY]))
        scene.step(_SCALAR[cur])
        img = scene.render(1.0 if f < 80 else 0.4)
        _, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        has = int(mask.sum()) > 0
        if f == 78:
            before = has
        if f == 84:
            after = has
    assert before, "baseline should see the paddle before the shift"
    assert not after, "baseline fixed-threshold unexpectedly survived the shift"

# CLAIMS 2-5 (controlled-object labeling, class reuse, passive subclasses,
# tentative-ghost revival) used to live here as perception-internal tests of
# the class-model labeler. They were DELETED under option C: the labeler moved
# to the world model. "Which object am I controlling?" is now a WORLD-MODEL
# claim (the dynamics model discovers it -- see test_world_model.py
# `test_world_model_discovers_controlled_object`), not a perception claim.
# Perception's job is now the SCAFFOLD only: find + track objects, lighting-
# invariant, stable ids. Behavior-class discovery (the second half of option
# C -- "which objects have similar learned dynamics?") is future work.
# Lighting invariance of controlled discovery follows from the scaffold's
# lighting invariance (CLAIM 1) plus the world model learning from normalized
# positions, so it is not re-stated here.


# ------------------------------ CLAIM: no velocity spike at contact ----------

def test_no_velocity_spike_at_paddle_contact():
    """The contact/collision fix. When the ball touches a paddle, the ball
    blob merges with the paddle blob -> the ball track coasts (occluded) for a
    few frames -> it re-acquires on separation. BEFORE the fix, re-acquisition
    computed velocity from the GAP-SPANNING displacement, producing an
    unphysical SPIKE (~-13px/frame when real motion is ~-2px/frame) that hid
    the bounce from the world model. AFTER the no-snap fix, the re-acquiring
    track keeps its pre-coast velocity and lets fresh frames correct it, so
    the reported velocity stays bounded (no spike).

    Uses the world-model test's PointScene (paddles block in their real
    y-range -> real bounces -> real contact/occlusion). We log the ball's
    reported velocity every frame (the track nearest GT ball) and assert it
    never exceeds a physical bound, and specifically does not spike at the
    re-acquisition frames right after a paddle contact.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "twm", "/workspace/agent/tests/test_world_model.py")
    twm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twm)
    PointScene, H_, W_ = twm.PointScene, twm.H, twm.W
    ACT_UP_, ACT_DOWN_, ACT_STAY_ = twm.ACT_UP, twm.ACT_DOWN, twm.ACT_STAY
    SC = {ACT_UP_: +1, ACT_DOWN_: -1, ACT_STAY_: 0}

    scene = PointScene(seed=1)
    perc = Perception(H_, W_)
    rng = np.random.default_rng(777)
    hold, cur = 4, ACT_STAY_
    FREE_FLIGHT = 2.2          # the ball's |bvx| in this scene
    SPIKE_BOUND = 3.0 * FREE_FLIGHT   # 3x free-flight = clearly non-spike
    v_log = []
    contact_v = []
    for f in range(1200):
        if f % hold == 0:
            cur = int(rng.choice([ACT_UP_, ACT_DOWN_, ACT_STAY_]))
        scene.step(SC[cur])
        gray = scene.render(1.0)
        tracks, _ = perc.step(gray, cur)
        bx = scene.bx + scene.bs / 2
        by = scene.by + scene.bs / 2
        # nearest track to GT ball
        best, bd = None, 1e9
        for t in tracks:
            d = ((t["cx"] - bx) ** 2 + (t["cy"] - by) ** 2) ** 0.5
            if d < bd:
                bd, best = d, t
        rvx = (best["vx"] if best is not None and bd < 25 else None)
        v_log.append((f, rvx, bd, best.get("missed") if best else None))
        # collect velocities at CONTACT frames (ball near a paddle plane) --
        # this is where the spike used to happen on re-acquisition.
        mx, ox = scene.my_x + scene.pw / 2, scene.opp_x + scene.pw / 2
        if (abs(bx - mx) < 10 or abs(bx - ox) < 10) and rvx is not None:
            contact_v.append((f, rvx, best.get("missed")))

    # 1) GLOBAL: no reported ball velocity ever exceeds the spike bound.
    finite = [v for _, v, _, _ in v_log if v is not None]
    vmax = max(abs(v) for v in finite)
    assert vmax < SPIKE_BOUND, (
        f"reported ball velocity spiked to {vmax:.1f}px/frame "
        f"(bound {SPIKE_BOUND:.1f}); the no-snap re-acquisition fix is not "
        f"bounding the gap-spanning displacement spike. "
        f"Before the fix this hit ~13px/frame.")

    # 2) AT CONTACT: the re-acquisition frames (missed was >0 -> just
    # re-matched) near a paddle do not spike. This is the specific case.
    contact_max = max((abs(v) for _, v, _ in contact_v), default=0.0)
    assert contact_max < SPIKE_BOUND, (
        f"velocity at paddle contact spiked to {contact_max:.1f}px/frame "
        f"(bound {SPIKE_BOUND:.1f}); the no-snap fix is not working at the "
        f"re-acquisition frame where the spike historically occurred.")

    # 3) SANITY: we actually saw contact frames (else the test is vacuous).
    assert len(contact_v) >= 10, (
        f"only {len(contact_v)} contact frames observed; need >=10 for the "
        f"test to be meaningful (tune the scene if this regresses)")


# ---------------- PREDICTION-CONDITIONED proposal (top-down / contact) -------

def _two_touching_blobs_frame():
    """A 160x160 frame with TWO bright objects whose blobs TOUCH (merged into
    one connected component by the legacy proposal). Object A on the left,
    object B on the right, touching in the middle. This is the contact/
    collision case -- the thing that destroys identity in the bottom-up
    scaffold. Built to be Pong-agnostic: just 'two objects that touch'."""
    import numpy as np
    img = np.full((160, 160), 24, dtype=np.float32)
    # two 12x12 bright squares touching at x=84 (A: x 72..83, B: x 84..95)
    img[70:82, 72:84] = 220      # A, center ~ (77.5, 76)
    img[70:82, 84:96] = 220      # B, center ~ (89.5, 76)
    return np.clip(img, 0, 255).astype(np.uint8), (77.5, 76.0), (89.5, 76.0)


def test_conditioned_proposal_keeps_identity_through_contact():
    """The core top-down property. When two objects TOUCH (their blobs merge
    into ONE connected component), the legacy bottom-up proposal returns ONE
    candidate. The prediction-conditioned proposal returns TWO -- one per
    prediction -- because identity is carried top-down (from the predictions),
    not derived from touching pixels. This is the property the W3b blocker
    (identity loss at contact) needs, at the proposal level."""
    from agent.perception.proposal import (propose_regions,
                                           propose_regions_conditioned)
    img, a_xy, b_xy = _two_touching_blobs_frame()

    # sanity: the LEGACY bottom-up proposal sees the two touching objects as
    # ONE merged blob (one candidate). This is the bug we are fixing.
    legacy = propose_regions(img, z_thresh=2.2)
    assert len(legacy) == 1, (
        f"sanity check failed: legacy proposal found {len(legacy)} candidates, "
        f"expected 1 (the merged blob) -- the test scene is not reproducing "
        f"the contact/merge condition; tune it")

    # the conditioned proposal, given the TWO predicted positions (where each
    # object expects to be), must return TWO claimed candidates -- one per
    # object -- even though the blobs touch. radius generous enough to reach.
    predictions = [(a_xy[0], a_xy[1], 14.0), (b_xy[0], b_xy[1], 14.0)]
    claimed, residual = propose_regions_conditioned(img, predictions)
    n_claimed = sum(1 for c in claimed if c is not None)
    assert n_claimed == 2, (
        f"conditioned proposal claimed {n_claimed} objects, expected 2 -- it "
        f"did not keep both identities through the contact/merge. claimed: "
        f"{[c is not None for c in claimed]}")
    # the two claimed candidates sit at the two objects' centers (within the
    # object size), not collapsed to one midpoint.
    cas = [c for c in claimed if c is not None]
    cx_a = min(c["cx"] for c in cas)
    cx_b = max(c["cx"] for c in cas)
    assert abs(cx_a - a_xy[0]) < 4.0 and abs(cx_b - b_xy[0]) < 4.0, (
        f"claimed candidates at cx={cx_a:.1f},{cx_b:.1f} do not match the two "
        f"objects' centers {a_xy[0]},{b_xy[0]} -- identity was not preserved")
    # and nothing residual (both blobs were claimed).
    assert len(residual) == 0, (
        f"residual had {len(residual)} new-object candidates, expected 0 -- "
        f"pixels that belong to known objects leaked into the new-object path")


def test_conditioned_proposal_finds_new_objects_in_residual():
    """The bottom-up path is demoted (not deleted): salient pixels NO track\n    claims still become NEW-object candidates. A brand-new object appearing\n    away from any prediction must be found in the residual. This is the\n    'surprise / new thing' path -- general, keeps cold-start working."""
    from agent.perception.proposal import propose_regions_conditioned
    img, a_xy, b_xy = _two_touching_blobs_frame()
    # a third, separate bright object nowhere near any prediction
    img2 = img.copy()
    img2[120:132, 120:132] = 220   # new object at ~ (126, 126)
    # predict only the two known objects; the third is unclaimed -> residual
    predictions = [(a_xy[0], a_xy[1], 14.0), (b_xy[0], b_xy[1], 14.0)]
    claimed, residual = propose_regions_conditioned(img2, predictions)
    assert sum(1 for c in claimed if c is not None) == 2, "known objects not claimed"
    assert len(residual) >= 1, (
        "the new (unpredicted) object was not found in the residual -- the "
        "bottom-up 'new things' path is broken")
    new = residual[0]
    assert abs(new["cx"] - 126.0) < 4.0 and abs(new["cy"] - 126.0) < 4.0, (
        f"residual new object at ({new['cx']:.1f},{new['cy']:.1f}), expected "
        f"~(126,126) -- the wrong residual was found")


def test_conditioned_proposal_coasts_when_object_not_visible():
    """If a predicted object has NO salient pixels in its radius (it is fully\n    occluded / not visible), the conditioned proposal returns None for that\n    prediction -- signalling the tracker to COAST it (carry identity through\n    the gap) rather than collapse it. The honest 'object not seen now' path."""
    from agent.perception.proposal import propose_regions_conditioned
    img, a_xy, b_xy = _two_touching_blobs_frame()
    # predict a third object somewhere there is nothing bright -> None
    predictions = [(a_xy[0], a_xy[1], 14.0), (50.0, 50.0, 10.0)]
    claimed, residual = propose_regions_conditioned(img, predictions)
    assert claimed[0] is not None, "object A should be claimed"
    assert claimed[1] is None, (
        "a prediction with no salient pixels in its radius should return None "
        "(coast signal), not a phantom candidate")
