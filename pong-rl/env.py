"""
Real-hardware Pong environment for the container.

Talks to the Windows-host bridge over persistent sockets via
host.docker.internal:
  - Camera + reward: one request byte -> host returns
        [u32 frame_len][JPEG][i32 reward_delta][i32 score_l][i32 score_r]
    (atomic obs+reward per step)
  - Servo: persistent TCP socket, send "<angle>\n" per action.

Observation pipeline: JPEG decode -> detect the SenseCAP screen as a
bright rectangle (Otsu threshold + largest 4-sided contour, classical CV
not a neural net) -> warp the FULL screen to 128x128 -> RGB uint8. No
AprilTags, no manual config; the NN sees only the game.

Action mapping (locked by user):
    UP=0 -> 40, DOWN=1 -> 110, STAY=2 -> 90

Episode semantics:
  terminated = always False (cannot reset the physical game)
  truncated  = True when no reward for max_frames_without_reward frames
  reset()    = internal counters only (does NOT touch the game)
"""

import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

HOST = os.environ.get("BRIDGE_HOST", "host.docker.internal")
CAM_PORT = int(os.environ.get("CAMERA_PORT", "8000"))   # camera+reward packet
SERVO_PORT = int(os.environ.get("SERVO_PORT", "8001"))  # servo angle

OBS_SIZE = 128  # NN input size

# locked servo angles
SERVO_ANGLES = {0: 70, 1: 120, 2: 90}  # UP, DOWN, STAY


def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge closed the connection")
        buf.extend(chunk)
    return bytes(buf)


class _ServoSock:
    """One persistent TCP connection to the host servo bridge."""

    def __init__(self, host, port):
        self._lock = threading.Lock()
        self._sock = socket.create_connection((host, port), timeout=5)

    def send_angle(self, angle):
        with self._lock:
            self._sock.sendall(f"{int(angle)}\n".encode())

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


def _order_corners(pts):
    """
    Order 4 points as TL, TR, BR, BL (top-left, top-right, bottom-right,
    bottom-left). Standard sum/difference trick:
      TL: min(x+y), BR: max(x+y), TR: max(x-y), BL: min(x-y).
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)        # x + y
    d = pts[:, 0] - pts[:, 1]  # x - y
    tl = pts[np.argmin(s)]     # min(x+y) -> top-left
    br = pts[np.argmax(s)]     # max(x+y) -> bottom-right
    tr = pts[np.argmax(d)]     # max(x-y) -> top-right
    bl = pts[np.argmin(d)]     # min(x-y) -> bottom-left
    return np.array([tl, tr, br, bl], dtype=np.float32)


class ScreenWarper:
    """
    Lighting-robust detection + warp of the SenseCAP screen to OBS_SIZE.

    Classical computer vision (no neural net), made robust to lighting by
    using RELATIVE contrast (interior vs a bezel ring) instead of absolute
    gray thresholds, plus temporal stability:
      1. CLAHE on the detection grayscale flattens illumination gradients.
      2. Otsu threshold -> dark mask -> contours.
      3. Aspect + area + border-touch filters (as before).
      4. For each candidate, compare INTERIOR mean to the mean of a RING
         just outside the contour (the bezel). Accept by relative contrast
         (polarity-agnostic), NOT an absolute `mean > 110` cutoff. This is
         what makes it work in bright rooms where the display's black bg
         reflects ambient light and exceeds 110 in absolute terms.
      5. Corners via approxPolyDP (multi-epsilon) with minAreaRect fallback
         (always 4 corners; immune to ragged/glare-broken edges).
      6. Order corners TL/TR/BR/BL; EMA-smooth across frames to kill jitter.
      7. Homography -> warp to (OBS_SIZE, OBS_SIZE) from the ORIGINAL frame.

    Resilience (Layer 4): on a transient detection failure, reuse the last
    good corners for up to `coast_frames` frames (decaying confidence) so a
    single bad frame (hand passing, auto-exposure transient, flicker) no
    longer throws. warp() NEVER raises; it returns (warped, confidence) and
    the trainer can gate on confidence if it wants.
    """

    def __init__(self, aspect_min=0.7, aspect_max=1.3, min_area_frac=0.05,
                 contrast_margin=30.0, contrast_ratio=0.75,
                 coast_frames=30, ema_alpha=0.3):
        self._aspect_min = aspect_min
        self._aspect_max = aspect_max
        self._min_area_frac = min_area_frac  # screen must fill >=5% of frame
        # relative-contrast gate (Layer 2): interior vs surround ring must
        # differ by >= contrast_margin gray levels AND have a ratio <=
        # contrast_ratio. Both are relative -> hold from dim to bright rooms.
        # Tune on your rig if needed.
        self._contrast_margin = contrast_margin
        self._contrast_ratio = contrast_ratio
        # temporal stability (Layer 4): coast this many frames on transient
        # detection failure before giving up; EMA-smooth corners across frames.
        self._coast_frames = coast_frames
        self._ema_alpha = ema_alpha
        self._confidence = 0.0
        self._last_corners_smoothed = None  # for EMA + coasting
        self._last_warped = None            # cached good warp (coast fallback)
        self._coast_left = 0

    def _detect_corners(self, bgr):
        """
        Lighting-robust screen corner detection. Returns 4 corners
        (np.float32 Nx2) or None.

        Layer 1 -- CLAHE on the detection grayscale flattens uneven
        illumination so Otsu has a stable bimodal histogram regardless of
        room brightness gradients. The warp later uses the ORIGINAL frame,
        so the NN input is unchanged.

        Layer 2 -- Each square-ish non-border candidate is accepted by
        RELATIVE contrast: its interior mean vs the mean of a band just
        OUTSIDE the contour (the bezel ring). Polarity-agnostic (dark-in-
        bright OR bright-in-dark). This replaces the old absolute
        `mean_gray > 110` reject that threw out the display in bright rooms
        where its black bg reflects ambient light and climbs above 110.

        Layer 3 -- Corners: try approxPolyDP at a few epsilons for a clean
        fit; fall back to minAreaRect, which ALWAYS returns a rectangle and
        is immune to ragged/glare-broken edges that made the old code
        hard-fail on corner count.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Layer 1: CLAHE flattens illumination gradients for a stable split.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        _, bright = cv2.threshold(gray_eq, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = cv2.bitwise_not(bright)  # display bg + room = 255
        contours, _ = cv2.findContours(dark, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = None
        best_ring_contrast = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area_frac * (h * w):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / float(bh)
            if not (self._aspect_min <= ar <= self._aspect_max):
                continue  # not square-ish
            # reject blobs touching the image border (room/desk, not display)
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            # Layer 2: RELATIVE contrast ring. Interior mean vs the mean of a
            # band just OUTSIDE the contour (the bezel). Means are taken on
            # the ORIGINAL gray so they reflect the real scene, not the
            # equalized image used only to find the mask.
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            interior_mean = float(cv2.mean(gray, mask=mask)[0])
            ring = (cv2.dilate(mask, np.ones((7, 7), np.uint8),
                               iterations=2) - mask)
            surround_mean = float(cv2.mean(gray, mask=ring)[0])
            ring_contrast = abs(surround_mean - interior_mean)
            if ring_contrast < self._contrast_margin:
                continue  # no strong bezel-like enclosure
            lo = min(interior_mean, surround_mean)
            hi = max(interior_mean, surround_mean)
            if hi <= 0 or (lo / hi) > self._contrast_ratio:
                continue  # inner and surround too similar -> not a bezel
            # pick the candidate with the strongest bezel-like contrast ring
            if ring_contrast > best_ring_contrast:
                best_ring_contrast = ring_contrast
                best = c
        if best is None:
            return None
        # Layer 3: robust 4-corner extraction.
        peri = cv2.arcLength(best, True)
        for eps_frac in (0.02, 0.04, 0.06):
            approx = cv2.approxPolyDP(best, eps_frac * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        # minAreaRect always returns a rectangle -> 4 corners. Fallback when
        # approxPolyDP can't clean up a ragged/glare-broken contour.
        return cv2.boxPoints(cv2.minAreaRect(best)).astype(np.float32)

    def warp(self, bgr_frame):
        """
        Never raises. Returns (warped_obs, confidence):
          confidence 1.0  -> fresh detection succeeded.
          0.0 < c < 1.0   -> coasting on last good corners (transient
                             detection failure); obs is the CURRENT frame
                             warped with slightly stale corners.
          0.0            -> coast exhausted or never detected; obs is the
                             last good warped frame (stale) or a zero frame.
        The trainer can gate on confidence; at minimum it no longer crashes
        on a single bad frame.
        """
        corners = self._detect_corners(bgr_frame)
        dst = np.array([[0, 0],
                        [OBS_SIZE - 1, 0],
                        [OBS_SIZE - 1, OBS_SIZE - 1],
                        [0, OBS_SIZE - 1]], dtype=np.float32)
        if corners is not None:
            ordered_new = _order_corners(corners)
            if self._last_corners_smoothed is not None:
                # EMA smoothing to kill per-frame homography jitter; also a
                # safety net against an occasional wrong candidate (70% old).
                smoothed = (self._ema_alpha * ordered_new
                            + (1.0 - self._ema_alpha) * self._last_corners_smoothed)
            else:
                smoothed = ordered_new
            self._last_corners_smoothed = smoothed
            self._coast_left = self._coast_frames
            H, _ = cv2.findHomography(smoothed, dst, method=0)
            out = cv2.warpPerspective(bgr_frame, H, (OBS_SIZE, OBS_SIZE),
                                      flags=cv2.INTER_AREA)
            self._last_warped = out
            self._confidence = 1.0
            return out, self._confidence
        # detection failed -> coast on last good corners (Layer 4)
        if self._last_corners_smoothed is not None and self._coast_left > 0:
            self._coast_left -= 1
            H, _ = cv2.findHomography(self._last_corners_smoothed, dst,
                                      method=0)
            out = cv2.warpPerspective(bgr_frame, H, (OBS_SIZE, OBS_SIZE),
                                      flags=cv2.INTER_AREA)
            # decaying confidence: trainer CAN gate, obs is still current-ish
            self._confidence = 0.4 * (self._coast_left
                                      / float(max(1, self._coast_frames)))
            return out, self._confidence
        # coast exhausted or never detected: return last good warp (stale) so
        # shapes stay valid; confidence 0 so the trainer knows it's garbage.
        self._confidence = 0.0
        if self._last_warped is not None:
            return self._last_warped, 0.0
        return np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8), 0.0


class RealEnv:
    """Real-hardware Pong env implementing the interface the trainer expects."""

    def __init__(self, max_frames_without_reward=1800):
        self._cam = socket.create_connection((HOST, CAM_PORT), timeout=5)
        self._servo = _ServoSock(HOST, SERVO_PORT)
        self._warper = ScreenWarper()

        self.max_frames_without_reward = max_frames_without_reward

        # cached last-step results
        self._obs = None
        self._raw_frame = None  # raw decoded camera frame (BGR)
        self._reward = 0
        self._score_l = 0
        self._score_r = 0
        self._terminated = False
        self._truncated = False

        self._frames_since_reward = 0

        # Throttle: detect repeated JPEGs from the bridge. The camera runs at
        # ~30fps but the loop can spin faster; repeated perceive() calls then
        # return the SAME latest JPEG. Training on duplicates corrupts KARC's
        # time/velocity estimates. We remember the last JPEG's hash and let
        # the caller know whether this frame is genuinely new.
        self._last_jpeg_hash = None
        self._last_frame_was_new = True

    # ---- core I/O ----
    def _request_frame_packet(self):
        # one request byte -> host returns the atomic frame+reward packet
        self._cam.sendall(b"\x01")
        hdr = _recv_exactly(self._cam, 4)
        (flen,) = struct.unpack("<I", hdr)
        jpeg = _recv_exactly(self._cam, flen)
        tail = _recv_exactly(self._cam, 12)
        reward_delta, score_l, score_r = struct.unpack("<iii", tail)
        return jpeg, reward_delta, score_l, score_r

    def perceive(self):
        jpeg, reward_delta, score_l, score_r = self._request_frame_packet()
        # Throttle: is this a genuinely new camera frame, or a repeat of the
        # last JPEG the bridge had cached? (Bridge serves whatever is current,
        # so a fast loop gets repeats.) We flag repeats so the trainer can
        # skip training/acting on them -- only real new frames advance time.
        import hashlib
        h = hashlib.md5(jpeg).digest()
        self._last_frame_was_new = (h != self._last_jpeg_hash)
        self._last_jpeg_hash = h
        arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError("failed to decode JPEG from bridge")
        self._raw_frame = arr
        warped, conf = self._warper.warp(arr)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        self._obs = rgb
        self._reward = reward_delta
        self._score_l = score_l
        self._score_r = score_r

        if reward_delta != 0:
            self._frames_since_reward = 0
        else:
            self._frames_since_reward += 1
        self._terminated = False  # physical game can't be force-reset
        self._truncated = (self._frames_since_reward
                           >= self.max_frames_without_reward)
        if self._truncated:
            # reset the truncation timer so it doesn't immediately re-trigger
            self._frames_since_reward = 0

    def step(self, action_index):
        angle = SERVO_ANGLES[int(action_index)]
        self._servo.send_angle(angle)

    def reset(self):
        # Internal counters only. The physical game keeps running; we do NOT
        # reset it. Just clear the truncation/reward bookkeeping.
        self._frames_since_reward = 0
        self._truncated = False
        self._reward = 0

    # ---- accessors ----
    def get_observation(self):
        return self._obs

    def get_raw_frame(self):
        """The raw decoded camera frame (BGR), before any warping."""
        return self._raw_frame

    def get_reward(self):
        return self._reward

    def get_terminated(self):
        return self._terminated

    def get_truncated(self):
        return self._truncated

    def get_num_actions(self):
        return 3

    def get_scores(self):
        return self._score_l, self._score_r

    def get_confidence(self):
        return self._warper._confidence

    def frame_is_new(self):
        """True if the last perceive() returned a genuinely new camera frame
        (not a repeat of the cached JPEG). The trainer should skip training
        and acting on repeats so KARC's time/velocity estimates stay honest."""
        return self._last_frame_was_new

    def shutdown(self):
        # park the servo in the neutral (stay) position before leaving
        try:
            self._servo.send_angle(SERVO_ANGLES[2])
        except Exception:
            pass
        self._servo.close()
        try:
            self._cam.close()
        except OSError:
            pass


if __name__ == "__main__":
    # quick connectivity check (needs the host bridge running)
    env = RealEnv()
    print(f"bridge host: {HOST}")
    env.perceive()
    print(f"obs shape: {env.get_observation().shape} reward: {env.get_reward()} "
          f"scores: L{env.get_scores()[0]}-R{env.get_scores()[1]} "
          f"conf: {env.get_confidence():.2f}")
    for a, name in [(0, "UP"), (1, "DOWN"), (2, "STAY")]:
        env.step(a)
        print(f"sent {name} ({SERVO_ANGLES[a]})")
        time.sleep(0.3)
    env.shutdown()
