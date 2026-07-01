"""A HARDER Pong scene where the simple reflex STRUGGLES, so the model-based
planner has something to beat.

Why the existing PointScene can't test the planner: the reflex leaks 0 points
there (paddle is tall ph=30/160 and the ball is slow), so 'beat the reflex'
has no headroom -- you can't do better than 0. This scene makes the reflex
miss by:

  - a SHORTER paddle (ph=10 vs 30): an aim error now lets the ball past.
  - a FASTER, more-vertical ball (bvy=2.6 vs 1.4, bvx=2.6 vs 2.2): the ball
    wall-bounces MORE during its approach to my paddle. The reflex aims by a
    STRAIGHT line (arrive_y = by + bvy*t, ignoring wall bounces), so each
    bounce it ignores produces an aim error -- and the short paddle can't
    cover it. The planner's imagination KNOWS about wall bounces (the Way C
    reflection discovers the walls as surfaces), so it should predict the
    real bounced path and intercept where the reflex aims wrong.

Everything else mirrors PointScene (the agent's code is scene-agnostic).
"""
import numpy as np

from agent.tests.test_world_model import PointScene, H, W


class HarderScene(PointScene):
    """PointScene with a short paddle + a fast, wall-bouncy ball."""

    def __init__(self, seed=0, ph=10, bvx=2.6, bvy=2.6, opp_speed=2.0):
        super().__init__(seed=seed)
        self.ph = ph              # short paddle (was 30)
        self.bvx, self.bvy = bvx, bvy   # faster, more vertical (was 2.2, 1.4)
        self.opp_speed = opp_speed
        # re-center paddles for the new height
        self.my_y = H / 2.0 - ph / 2.0
        self.opp_y = H / 2.0 - ph / 2.0

    def _reset(self):
        self.bx, self.by = W / 2.0, H / 2.0
        self.bvx = 2.6 * (1 if self.rng.random() > 0.5 else -1)
        self.bvy = 2.6 * (1 if self.rng.random() > 0.5 else -1)
