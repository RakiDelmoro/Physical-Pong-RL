"""
KARC State Extractor.

Turns the warped 128x128 RGB observation (from env.RealEnv.get_observation())
into the 4-number low-dimensional state KARC needs:

    state = [ball_x, ball_y, my_y, opp_y]   (all normalized 0..1)

All coordinates are the CENTROID of the relevant white blob on a binary
(Otsu-thresholded) image. Pong is black background + white elements, so
blob tracking is the whole perception pipeline. No neural net.

Why a 4-number state:
  - ball_x, ball_y : where the ball is (its motion is what we forecast)
  - my_y           : my paddle center (the ball bounces off it, so the
                     forecaster must see it; and it is what we control)
  - opp_y          : opponent paddle center (the ball bounces off it too)

Velocity is NOT extracted. KARC reconstructs it from the delay embedding
(looking at the last k states) -- that's its native strength and avoids a
noisy finite-difference step.

Robustness:
  - The ball can briefly touch a paddle or vanish during a bounce. We track
    by nearest-centroid across frames and fall back to the last known
    position on loss (flagged lost=True so the caller can skip the update).
  - Paddles are the tall white bars at the far-left / far-right margins.
  - Everything is bounded 0..1 to match KARC's basis-function domain.
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Geometry (mirrors sense-pong/main/pong.c, normalized by 480)
# ---------------------------------------------------------------------------
# We detect at the native 128x128 obs resolution. Firmware geometry is 480x480,
# so we scale the firmware constants by 128/480 = 1/3.75 to get 128-space bands.
SCALE = 128.0 / 480.0
DISP = 128                       # detection resolution
PADDLE_W = 12 * SCALE            # ~3.2 px
PADDLE_H = 80 * SCALE            # ~21 px
PADDLE_MARGIN = 24 * SCALE       # ~6.4 px
# Margins for "is this blob a paddle" (it must be tall and at the edge)
MIN_PADDLE_H = PADDLE_H * 0.5    # at least half the paddle visible
PADDLE_X_BAND = (PADDLE_MARGIN + PADDLE_W + 6 * SCALE)  # px from edge = paddle zone
# Ball: smallish blob, not at the extreme edges. We rely on nearest-centroid
# tracking to distinguish the ball from net dashes / score text, so the area
# bounds are generous (just reject tiny noise and huge non-game blobs).
MIN_BALL_AREA = 3
MAX_BALL_AREA = 500


def to_binary(rgb):
    """RGB uint8 (H,W,3) -> Otsu binary uint8 (H,W) with 255=game elements.
    Operates at the native obs resolution (no upscale)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _blob_centroids(binary):
    """Return list of (cx, cy, w, h) for all white blobs in the image."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 1 and h <= 1:
            continue
        out.append((x + w / 2.0, y + h / 2.0, float(w), float(h)))
    return out


class StateExtractor:
    """
    Extract [ball_x, ball_y, my_y, opp_y] from a warped 128x128 RGB frame.

    Stateful: remembers the last ball position so it can track the ball by
    nearest-centroid when there are several candidate blobs (e.g. the ball
    momentarily overlapping a paddle). On total loss, returns the last known
    state with lost=True so KARC can skip that update.
    """

    def __init__(self):
        self._last_state = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        self._last_ball = np.array([0.5, 0.5], dtype=np.float64)
        self._has_last = False

    def reset_ball(self, center=0.5):
        """Tell the tracker the ball was just teleported to center by the
        firmware (a point was scored -> reset_ball() ran on the ESP32).
        Resets the held position and clears the 'has seen' memory so the
        next frame's nearest-centroid search starts fresh from center,
        instead of chasing a ghost at the edge where the ball escaped."""
        self._last_ball = np.array([center, center], dtype=np.float64)
        self._has_last = True

    def extract(self, rgb):
        """
        rgb: (H,W,3) uint8 RGB (the warped obs from env).
        returns: (state np.float64[4] in 0..1, found dict)
          found = {
            'ball': True if the ball was actually seen this frame,
            'my':   True if my paddle was actually seen this frame,
            'opp':  True if the opponent paddle was actually seen this frame,
          }
          When an element is not found, its coordinate is HELD at the last
          known value (so the state vector stays valid) but the caller MUST
          check the flag before training a model on it -- held values are
          stale, not real observations.
        """
        b = to_binary(rgb)
        blobs = _blob_centroids(b)

        # --- classify blobs by SHAPE, then by position ---
        # Paddles are TALL and THIN (height >> width). The ball is roughly
        # SQUARE. Score text / the dashed net are WIDE (width >> height).
        # Using aspect ratio is far more robust than x-position alone, because
        # the warp can shift the paddles a few px off the nominal edge band.
        my_y = self._last_state[2]
        opp_y = self._last_state[3]
        found_my = False
        found_opp = False
        paddle_blobs = []      # (cx, cy, w, h) for tall-thin blobs
        ball_candidates = []   # (cx, cy) for square-ish small blobs
        for (cx, cy, w, h) in blobs:
            if w <= 0 or h <= 0:
                continue
            aspect = h / float(w)
            area = w * h
            if aspect >= 1.8 and h >= MIN_PADDLE_H:
                # tall & thin -> paddle (left or right decided by x below)
                paddle_blobs.append((cx, cy, w, h))
            elif 0.4 <= aspect <= 2.5 and MIN_BALL_AREA <= area <= MAX_BALL_AREA:
                # roughly square & small -> ball candidate
                ball_candidates.append((cx, cy))
        # split paddles into left (mine) / right (opponent) by x midpoint
        if paddle_blobs:
            mid_x = DISP / 2.0
            left_pdls = [p for p in paddle_blobs if p[0] < mid_x]
            right_pdls = [p for p in paddle_blobs if p[0] >= mid_x]
            if left_pdls:
                my_y = max(left_pdls, key=lambda t: t[3])[1] / (DISP - 1)
                found_my = True
            if right_pdls:
                opp_y = max(right_pdls, key=lambda t: t[3])[1] / (DISP - 1)
                found_opp = True
        ball_found = False
        if ball_candidates:
            if self._has_last:
                # pick the candidate closest to where we last saw the ball
                last = self._last_ball * DISP
                ball = min(ball_candidates,
                           key=lambda p: (p[0] - last[0]) ** 2
                                         + (p[1] - last[1]) ** 2)
            else:
                ball = ball_candidates[0]
            bx, by = ball
            self._last_ball = np.array([bx / (DISP - 1), by / (DISP - 1)],
                                       dtype=np.float64)
            self._has_last = True
            ball_found = True
        # else: ball lost this frame -> hold _last_ball (stale), do NOT update it

        state = np.array([
            self._last_ball[0],  # ball_x (held on loss)
            self._last_ball[1],  # ball_y
            np.clip(my_y, 0.0, 1.0),
            np.clip(opp_y, 0.0, 1.0),
        ], dtype=np.float64)
        self._last_state = state.copy()
        found = {'ball': ball_found, 'my': found_my, 'opp': found_opp}
        return state, found
