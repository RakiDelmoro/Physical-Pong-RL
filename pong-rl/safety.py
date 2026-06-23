"""
Hardware safety filter for the servo.

The planner can think whatever it wants; this decides what is physically safe
to send to the SG90. Servos break from fast large reversals (UP<->DOWN is a
70-degree slam). This module enforces:

  1. NO immediate reversals: after UP you may only do UP or STAY; after DOWN
     only DOWN or STAY. A reversal must go through STAY first.
  2. DWELL: each action is held for at least `dwell` frames (~150ms at 30fps
     with dwell=5) before a change is allowed.
  3. RATE LIMIT: at most one action change per `dwell` frames (same as dwell).
  4. CONFIDENCE GATE: when the planner's confidence is below `conf_thresh`,
     we override to STAY (the safe neutral). Early on, the planner is shaky;
     this keeps the servo parked until it has evidence.
  5. SOFT RETURN: going UP->STAY or DOWN->STAY is always allowed (it's
     easing toward neutral, not a reversal).

These are NOT training hacks. They are a permanent filter between the brain
and the motor. The KARC agent's chosen action passes through here every frame
before reaching env.step().
"""

import numpy as np

UP, DOWN, STAY = 0, 1, 2


class SafetyFilter:
    def __init__(self, dwell=5, conf_thresh=0.4):
        self.dwell = dwell
        self.conf_thresh = conf_thresh
        self._last_action = STAY
        self._held_for = 999  # frames we've held the current action

    def reset(self):
        self._last_action = STAY
        self._held_for = 999

    def filter(self, desired_action, confidence=1.0):
        """
        desired_action: int the planner wants (0/1/2)
        confidence:     planner confidence 0..1 (low -> default to STAY)
        returns:        the action actually allowed (0/1/2)
        """
        # 1. confidence gate: if unsure, hold neutral
        if confidence < self.conf_thresh:
            desired = STAY
        else:
            desired = int(desired_action)

        # 2. no immediate reversals: UP<->DOWN forbidden
        if self._last_action == UP and desired == DOWN:
            desired = STAY
        elif self._last_action == DOWN and desired == UP:
            desired = STAY

        # 3. dwell: enforce min hold time before any change
        if desired != self._last_action and self._held_for < self.dwell:
            desired = self._last_action  # not allowed to change yet

        # commit
        if desired == self._last_action:
            self._held_for += 1
        else:
            self._last_action = desired
            self._held_for = 1
        return self._last_action

    def is_safe_transition(self, a, b):
        """True if a->b is allowed by the reversal rule (debug/test)."""
        if a == UP and b == DOWN:
            return False
        if a == DOWN and b == UP:
            return False
        return True
