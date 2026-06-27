"""
Tests for the world model (`M(state, action) -> (next_state, reward)`), NO
hardware. Servo-free core (Steps 0-2 of WORLD_MODEL_PLAN.md): the reward
channel.

We use a synthetic Pong scene where paddles block only in their ACTUAL
y-range, so the ball can MISS and escape -> a point. Reward: +1 when the ball
passes the opponent (right) paddle, -1 when it passes my (left) paddle. My
paddle is action-controlled (action -> my_y), so reward depends on state AND
action -- the full M, including the action channel (which is servo-blocked on
the real rig but works in sim).

CLAIM W1 -- the reward predictor trains and learns the base rate:
  after enough frames, the mean predicted reward approximates the empirical
  reward-per-frame rate (the RLS bias term captures the long-run average).

CLAIM W2 -- FORESIGHT (the keystone): the predicted reward at the frame
  BEFORE a point has the same SIGN as the actual point, well above chance.
  This is the proof the agent can foresee a score event from ball+paddle
  geometry BEFORE the ball crosses the line -- exactly what planning needs.
"""

import numpy as np
import pytest

from agent.perception import Perception
from agent.world_model import WorldModel

H, W = 160, 160
BG, OBJ = 24, 220
ACT_UP, ACT_DOWN, ACT_STAY = 0, 1, 2
_SCALAR = {ACT_UP: +1, ACT_DOWN: -1, ACT_STAY: 0}


class PointScene:
    """Synthetic Pong where paddles block only in their real y-range, so the
    ball can miss and escape -> a point. My paddle is action-controlled."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.m = 16
        self.pw, self.ph, self.bs = 6, 30, 6
        self.bx, self.by = W / 2.0, H / 2.0
        self.bvx, self.bvy = 2.2, 1.4
        self.my_x = self.m
        self.my_y = H / 2.0
        self.opp_x = W - self.m - self.pw
        self.opp_y = H / 2.0
        self.opp_speed = 1.4
        self.gt = {}

    def step(self, action):
        sp = 3.0
        self.my_y = float(np.clip(self.my_y - action * sp,
                                  self.m, H - self.m - self.ph))
        target = self.by + self.bs / 2 - self.ph / 2
        self.opp_y = float(np.clip(
            self.opp_y + np.clip(target - self.opp_y,
                                 -self.opp_speed, self.opp_speed),
            self.m, H - self.m - self.ph))
        self.bx += self.bvx
        self.by += self.bvy
        if self.by <= self.m:
            self.by, self.bvy = self.m, abs(self.bvy)
        if self.by >= H - self.m - self.bs:
            self.by, self.bvy = H - self.m - self.bs, -abs(self.bvy)
        # bounce off my paddle ONLY if the ball is in the paddle's y-range
        if self.bx <= self.my_x + self.pw and self.bvx < 0:
            if self.my_y <= self.by <= self.my_y + self.ph:
                self.bvx = abs(self.bvx)
        if self.bx + self.bs >= self.opp_x and self.bvx > 0:
            if self.opp_y <= self.by <= self.opp_y + self.ph:
                self.bvx = -abs(self.bvx)
        r = 0
        if self.bx < self.m:
            r = -1
            self._reset()
        elif self.bx + self.bs > W - self.m:
            r = +1
            self._reset()
        self.gt = {
            "my": (self.my_x + self.pw / 2, self.my_y + self.ph / 2),
            "opp": (self.opp_x + self.pw / 2, self.opp_y + self.ph / 2),
            "ball": (self.bx + self.bs / 2, self.by + self.bs / 2),
        }
        return r

    def _reset(self):
        self.bx, self.by = W / 2.0, H / 2.0
        self.bvx = 2.2 * (1 if self.rng.random() > 0.5 else -1)
        self.bvy = 1.4 * (1 if self.rng.random() > 0.5 else -1)

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
        return np.clip(img * brightness, 0, 255).astype(np.uint8)


def _run(total=2000, seed=1, act_seed=777, n_features=1000, gamma=0.5):
    scene = PointScene(seed=seed)
    perc = Perception(H, W)
    wm = WorldModel(H, W, num_actions=3, n_features=n_features, gamma=gamma)
    rng = np.random.default_rng(act_seed)
    hold = 4
    cur = ACT_STAY
    preds, rewards = [], []
    for f in range(total):
        if f % hold == 0:
            cur = int(rng.choice([ACT_UP, ACT_DOWN, ACT_STAY]))
        r = scene.step(_SCALAR[cur])
        gray = scene.render(1.0)
        tracks, controlled = perc.step(gray, cur)
        preds.append(wm.predict_reward(tracks, cur, controlled))
        rewards.append(r)
        wm.step(tracks, cur, r, controlled)
    return np.array(preds), np.array(rewards), wm


# ------------------------------ CLAIM W1 ------------------------------------

def test_world_model_trains_and_learns_base_rate():
    preds, rewards, wm = _run(total=1200, n_features=1000)
    assert wm.n_obs > 500, "world model did not train enough"
    assert np.all(np.isfinite(preds)), "predictions are not finite"
    assert float(preds[300:].std()) > 0.0, "predictions are constant (no learning)"
    # the RLS bias should capture the long-run reward-per-frame rate
    rate = float(rewards.mean())
    mean_pred = float(preds[300:].mean())
    assert abs(mean_pred - rate) < 0.02, (
        f"learned base rate {mean_pred:.4f} != empirical rate {rate:.4f}; "
        "the reward predictor did not learn the average reward")


# ------------------------------ CLAIM W2 ------------------------------------

def test_world_model_foresees_reward():
    """The keystone: predicted reward at the frame BEFORE a point has the
    same sign as the actual point, well above chance (0.5)."""
    preds, rewards, wm = _run(total=1200, n_features=1000)
    nonzero = np.where(rewards != 0)[0]
    assert len(nonzero) >= 10, (
        f"only {len(nonzero)} points in the run; need >=10 for a stable "
        "foresight measurement (tune the scene if this regresses)")
    correct = 0
    total = 0
    for idx in nonzero:
        j = idx - 1                       # the frame right before the point
        if j < 200:                       # skip the warm-up
            continue
        total += 1
        if preds[j] != 0 and np.sign(preds[j]) == np.sign(rewards[idx]):
            correct += 1
    acc = correct / float(total)
    assert total >= 10, "not enough post-warmup points to measure foresight"
    assert acc > 0.70, (
        f"foresight accuracy {acc:.2f} ({correct}/{total}) is not above 0.70; "
        "the world model is not foreseeing point events from ball+paddle "
        "geometry before the ball crosses the line")


def _run_with_log(total=1200, n_features=1000, gamma=0.5,
                      act_seed=777, seed=1):
    """Train the world model AND log, each frame: the perception tracks (so we
    can build the state later), the action, and the ground-truth ball x /
    velocity (so a rollout can be checked against reality). Used by the
    rollout/bounce test."""
    scene = PointScene(seed=seed)
    perc = Perception(H, W)
    wm = WorldModel(H, W, num_actions=3, n_features=n_features, gamma=gamma)
    rng = np.random.default_rng(act_seed)
    hold = 4
    cur = ACT_STAY
    log = []
    for f in range(total):
        if f % hold == 0:
            cur = int(rng.choice([ACT_UP, ACT_DOWN, ACT_STAY]))
        r = scene.step(_SCALAR[cur])
        gray = scene.render(1.0)
        tracks, controlled = perc.step(gray, cur)
        log.append({"frame": f, "tracks": tracks, "action": cur,
                    "controlled": controlled,
                    "gt_bx": scene.bx, "gt_bvx": scene.bvx,
                    "gt_opp_x": scene.opp_x, "gt_opp_y": scene.opp_y,
                    "gt_my_x": scene.my_x, "gt_my_y": scene.my_y})
        wm.step(tracks, cur, r, controlled)
    return wm, log, scene, perc


def _state_ball_xy(state, slot=0):
    """Read the (cx, cy) of slot `slot` from a joint state vector."""
    j = slot * WorldModel.STATE_PER_OBJ
    return state[j], state[j + 1]


def _state_ball_vx(state, slot=0):
    return state[slot * WorldModel.STATE_PER_OBJ + 2]


# ------------------------------ CLAIM W3 ------------------------------------

@pytest.mark.xfail(
    reason=(
        "PAUSED known-gap: the ball's velocity estimate is wrong through a "
        "paddle bounce (freezes during the coast, spikes at re-acquisition), "
        "so the bounce is not a clean training signal and the world model "
        "cannot learn a sharp velocity flip. Blocker is in PERCEPTION "
        "(occlusion-velocity quality), not world-model architecture. "
        "See WORLD_MODEL_PLAN.md 'W3 PAUSED'."),
    strict=True,
)
def test_world_model_rollout_bounces_and_does_not_pass_through():
    """CLAIM W3 (the rollout/coupling test): an imagined multi-frame rollout
    stays physically sane -- the ball BOUNCES off a paddle (its x-velocity
    flips sign when it reaches the paddle's x-plane) and does NOT fly through
    the paddle to the far side. This is the property the frozen-nonlinear
    (random Fourier) design is supposed to give us that a linear model cannot:
    a bounce is a sharp piecewise event, and only a nonlinear basis can
    represent it well enough to survive the error-compounding of a multi-step
    rollout. One-frame foresight (W2) does NOT test this -- errors compound
    when each imagined state is fed back as the next input, and a smeared flip
    becomes a pass-through over several steps.

    We train, then find a frame where the ball is heading TOWARD the opponent
    paddle (bvx > 0) and not too close to a wall, run a 25-step rollout with a
    'stay' action, and check the imagined ball: its x crosses the opp paddle's
    x-plane at some step, and at that crossing its vx has FLIPPED to negative
    (it bounced), AND it never ends up well beyond the paddle plane (it didn't
    pass through)."""
    wm, log, scene, perc = _run_with_log(total=1200, n_features=1000)
    assert wm.n_obs > 500, "world model did not train enough"

    # find a good approach frame: ball heading right (toward opp), in mid-field,
    # and a few frames of training have happened so the model is warm.
    cand = None
    for e in log[300:]:
        if e["gt_bvx"] > 0 and 0.35 * W < e["gt_bx"] < 0.6 * W:
            cand = e
            break
    assert cand is not None, "no suitable ball-approach frame in the run"

    # opp paddle x-plane in NORMALIZED coords (the world model works normalized)
    opp_x_norm = (cand["gt_opp_x"]) / float(W)
    stay = ACT_STAY

    states, rewards, _ = wm.rollout(cand["tracks"], first_action=stay,
                                    action_fn=lambda s, t: stay, horizon=25,
                                    controlled_id=cand.get("controlled"))

    # extract imagined ball (cx, vx) per step. Slot 0 is the first object after
    # the (cy, cx) sort; not guaranteed to be the ball, so search all slots for
    # the one whose cx crosses the opp plane -- that's the bouncing object.
    n_slots = WorldModel.MAX_OBJECTS
    found_bounce = False
    passed_through = False
    for slot in range(n_slots):
        cxs = [s[slot * WorldModel.STATE_PER_OBJ] for s in states]
        vxs = [s[slot * WorldModel.STATE_PER_OBJ + 2] for s in states]
        # does this slot's x actually approach & cross the opp plane?
        if not any(cx >= opp_x_norm for cx in cxs):
            continue
        if cxs[0] > opp_x_norm:  # already past -- not a clean approach
            continue
        # find the first step where it reaches the plane
        cross_step = next((i for i, cx in enumerate(cxs) if cx >= opp_x_norm),
                          None)
        if cross_step is None or cross_step < 1:
            continue
        # at the crossing, the vx should have flipped to NEGATIVE (bounced)
        vx_after = vxs[cross_step]
        if vx_after < 0:
            found_bounce = True
        # and it should not keep going far past the plane (pass-through)
        max_past = max(cx - opp_x_norm for cx in cxs[cross_step:])
        if max_past > 0.15:   # >15% of frame width past the plane = flew through
            passed_through = True

    assert found_bounce, (
        "rollout did NOT bounce the ball off the opponent paddle: no slot "
        "showed an x-velocity flip to negative at the paddle plane. The "
        "imagined trajectory passed through the paddle (the linear-model "
        "failure the nonlinear design was meant to fix).")
    assert not passed_through, (
        "rollout bounced BUT the imagined ball kept going well past the "
        "paddle plane -- a smeared/partial bounce. Errors compounded into a "
        "pass-through over the rollout horizon.")


if __name__ == "__main__":
    preds, rewards, wm = _run(total=1200, n_features=1000)
    nz = int((rewards != 0).sum())
    print(f"n_obs={wm.n_obs} points={nz} rate={rewards.mean():.4f} "
          f"mean_pred={preds[200:].mean():.4f}")
    nonzero = np.where(rewards != 0)[0]
    correct = sum(1 for idx in nonzero if idx - 1 >= 200 and preds[idx - 1] != 0
                  and np.sign(preds[idx - 1]) == np.sign(rewards[idx]))
    total = sum(1 for idx in nonzero if idx - 1 >= 200)
    print(f"foresight: {correct}/{total} = {correct / total:.2f}")
