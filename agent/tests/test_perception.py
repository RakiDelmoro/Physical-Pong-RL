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


# ------------------------------ CLAIM 2 -------------------------------------

def test_identity_by_behavior_labels_controlled_only():
    log, _ = _run(total=240)
    final = log[-1]
    controlled = final["controlled"]
    assert controlled is not None, "model failed to identify a controlled object"
    id_my = _track_at(final["tracks"], final["gt"]["my"])
    assert id_my is not None, "my paddle not tracked at end"
    assert controlled == id_my, (
        f"model labeled track {controlled} controlled, but my paddle is {id_my}")
    for role in ("decoy", "opp", "ball"):
        rid = _track_at(final["tracks"], final["gt"][role])
        assert controlled != rid, f"the {role} was labeled controlled"


def test_label_holds_after_lighting_shift():
    log, _ = _run(total=240)
    held = False
    for e in log[180:]:
        c = e["controlled"]
        if c is None:
            continue
        if _track_at(e["tracks"], e["gt"]["my"]) == c:
            held = True
            break
    assert held, "identity by behavior failed after the lighting shift"


# ------------------------------ CLAIM 3 -------------------------------------

def test_reuse_class_survives_object_swap():
    """A new controlled object (same dynamics, different pos/size) binds to the
    EXISTING controlled class instead of seeding a new one."""
    # no lighting shift here; isolate the reuse behavior.
    log, perc = _run(total=320, shift=10_000, seed=1, swap_at=120)

    # find the controlled class id established BEFORE the swap (use the
    # per-frame diag snapshot, since the pre-swap object is retired by the end)
    class_before = None
    for e in log[100:118]:
        c = e["controlled"]
        if c is None:
            continue
        objs = e["diag"]["objects"]
        if c in objs and objs[c]["bound"] is not None:
            class_before = objs[c]["bound"]
            break
    assert class_before is not None, "controlled class not established before swap"

    # after the swap, the NEW controlled object must bind to the SAME class
    reused = False
    for e in log[200:]:
        c = e["controlled"]
        if c is None:
            continue
        objs = e["diag"]["objects"]
        if c in objs and objs[c]["bound"] == class_before:
            reused = True
            break
    assert reused, (
        f"swapped object did not reuse controlled class {class_before}; "
        "discovered classes did not transfer knowledge")


if __name__ == "__main__":
    log, perc = _run(total=240)
    f = log[-1]
    print("frames:", len(log))
    print("controlled:", f["controlled"])
    import json
    print("classes:", {k: v for k, v in f["diag"]["classes"].items()})
    print("objects:", {k: v for k, v in f["diag"]["objects"].items()})
    print("tracks:", [(t["id"], t["dim"], round(t["cx"]), round(t["cy"]))
                      for t in f["tracks"]])
