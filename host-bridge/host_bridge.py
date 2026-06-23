"""
Low-latency host bridge for physical Pong RL. Runs ON THE WINDOWS HOST.

Replaces the old HTTP/MJPEG bridge. Uses persistent sockets + an atomic
per-step frame+reward packet so the container's RL loop sees bounded,
consistent latency (no per-frame TCP handshake, no per-frame re-encode
overhead, no reward/frame timing skew).

Two services, both reached from the container via host.docker.internal:

  Port 8000 (camera + reward, request/response):
    container sends 1 byte (any data) -> host replies with:
        [u32 frame_len][JPEG bytes][i32 reward_delta][i32 score_l][i32 score_r]
    reward_delta is the sum of all point events received from the SenseCAP
    since the last frame request (+1 per "P L", -1 per "P R"). The host then
    zeroes its accumulator. Atomic: obs and reward always arrive together.

  Port 8001 (servo, persistent):
    container sends "<angle>\\n" per action (e.g. "40\\n"). Host writes the
    integer to the Arduino's serial port (0..180 -> myServo.write). One TCP
    connection stays open for the whole session.

Reward source: the SenseCAP Indicator's USB-CDC console (the same COM port
idf.py monitor uses). A background thread reads lines; any line matching
`P ([LR]) (\\d+) (\\d+)` is counted as a point:
    "P L <l> <r>" -> +1  (agent / left paddle scored)
    "P R <l> <r>" -> -1  (CPU / right paddle scored; agent missed)

Install on Windows (once):
    pip install opencv-python pyserial

Run:
    python host_bridge.py --camera 0 --arduino COM5 --arduino-baud 9600 --sensecap COM10

Find COM ports in Device Manager. --sensecap is the port idf.py monitor used
(COM10 in your logs). --arduino is the Arduino's COM port.

Flags:
    --no-arduino       skip opening the Arduino (test camera+reward only)
    --no-sensecap      skip opening the SenseCAP (test camera+servo only)
    --camera WxH@fps   e.g. 1280x720@30 (defaults to letting OpenCV pick)
    --cam-port 8000    camera+reward TCP port
    --servo-port 8001  servo TCP port
"""

import argparse
import re
import socket
import socketserver
import struct
import threading
import time

import cv2
import numpy as np
import serial

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_reward_lock = threading.Lock()
_reward_since_last_frame = 0      # accumulated +1/-1 since last frame request
_score_left = 0
_score_right = 0

_cam_lock = threading.Lock()
_latest_jpeg = b""                 # most recent JPEG, kept by the camera thread
_latest_jpeg_event = threading.Event()

_arduino_lock = threading.Lock()
_arduino = None                    # serial.Serial for the Arduino, or None
_show_preview = False              # set in main() if --show-preview


def detect_screen(frame):
    """
    Mirror of the container's ScreenWarper._detect_corners for the live
    preview. Lighting-robust version (Layers 1-3): CLAHE + relative
    contrast ring (polarity-agnostic) + minAreaRect corner fallback.
    Returns the 4 display corners (np.float32 Nx2) or None. Pure OpenCV.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Layer 1: CLAHE flattens illumination gradients for a stable Otsu split.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    _, bright = cv2.threshold(gray_eq, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(bright)  # display bg + room = 255
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contrast_margin = 30.0
    contrast_ratio = 0.75
    best = None
    best_ring_contrast = 0.0
    for c in contours:
        if cv2.contourArea(c) < 0.05 * (h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh)
        if not (0.7 <= ar <= 1.3):
            continue
        # reject blobs touching the image border (room/desk, not display)
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        # Layer 2: RELATIVE contrast ring (interior vs bezel surround).
        # Means on the ORIGINAL gray reflect the real scene, not the
        # equalized image used only to find the mask. Polarity-agnostic.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        interior_mean = float(cv2.mean(gray, mask=mask)[0])
        ring = (cv2.dilate(mask, np.ones((7, 7), np.uint8),
                           iterations=2) - mask)
        surround_mean = float(cv2.mean(gray, mask=ring)[0])
        ring_contrast = abs(surround_mean - interior_mean)
        if ring_contrast < contrast_margin:
            continue  # no strong bezel-like enclosure
        lo = min(interior_mean, surround_mean)
        hi = max(interior_mean, surround_mean)
        if hi <= 0 or (lo / hi) > contrast_ratio:
            continue  # inner and surround too similar -> not a bezel
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


# ---------------------------------------------------------------------------
# SenseCAP reward reader (background thread)
# ---------------------------------------------------------------------------
_POINT_RE = re.compile(rb"P\s+([LR])\s+(\d+)\s+(\d+)")


def _sensecap_reader(sense_port, baud):
    global _reward_since_last_frame, _score_left, _score_right
    print(f"[sensecap] opening {sense_port} @ {baud} for reward lines")
    while True:
        try:
            with serial.Serial(sense_port, baud, timeout=1) as s:
                print(f"[sensecap] listening on {sense_port}")
                while True:
                    line = s.readline()
                    if not line:
                        continue
                    m = _POINT_RE.search(line)
                    if not m:
                        continue
                    side = m.group(1)
                    sl = int(m.group(2))
                    sr = int(m.group(3))
                    delta = 1 if side == b"L" else -1
                    with _reward_lock:
                        _reward_since_last_frame += delta
                        _score_left = sl
                        _score_right = sr
                    print(f"[sensecap] point side={side.decode()} "
                          f"score L{sl}-R{sr} delta={delta:+d}")
        except Exception as e:
            print(f"[sensecap] error: {e}; reconnecting in 2s")
            time.sleep(2)


def _consume_reward():
    """Atomically read + zero the accumulated reward delta and current scores."""
    global _reward_since_last_frame
    with _reward_lock:
        d = _reward_since_last_frame
        sl = _score_left
        sr = _score_right
        _reward_since_last_frame = 0
    return d, sl, sr


# ---------------------------------------------------------------------------
# Camera capture (background thread keeps only the latest JPEG)
# ---------------------------------------------------------------------------
def _camera_thread(cam_index, width, height, fps, focus, autofocus=False, backend=None):
    # Try several backends in order. DSHOW and MSMF are the Windows ones;
    # fall back to the default (None) which lets OpenCV pick.
    if backend == "dshow":
        backends = [cv2.CAP_DSHOW]
    elif backend == "msmf":
        backends = [cv2.CAP_MSMF]
    elif backend == "any":
        backends = [cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backend_names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF",
                     cv2.CAP_ANY: "default"}
    cap = None
    for b in backends:
        cap = cv2.VideoCapture(cam_index, b)
        if cap.isOpened():
            print(f"[camera] opened index {cam_index} via {backend_names[b]}")
            break
        cap.release()
    if cap is None or not cap.isOpened():
        print(f"[camera] FAILED to open index {cam_index} with any backend")
        print(f"[camera]   - is another process holding the camera? "
              f"(Zoom/Teams/Discord/browser/OBS, or a leftover python.exe)")
        print(f"[camera]   - try a different index with --camera 1")
        return
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    # Camera config policy: by default touch NOTHING. Focus, autofocus and
    # exposure are all configured in the camera's own app (Elgato Camera Hub,
    # LogiTune, Logi Capture, Windows Camera settings), which persist across
    # OpenCV opens. The script just opens the camera and reads frames.
    #
    # The flags below are OPT-IN overrides for debugging only. Passing none of
    # them leaves the camera exactly as its own app left it.
    #   --focus N     -> disable AF, set manual focus to N
    #   --autofocus   -> explicitly ENABLE AF (CAP_PROP_AUTOFOCUS=1). The old
    #                   code did `pass` here = no-op; on cams started in manual
    #                   mode by the driver (common with MSMF on UVC cams) AF
    #                   stayed OFF and the lens sat at its last position ->
    #                   blurry. So we must actually send AUTOFOCUS=1.
    touched_focus = False
    if focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        touched_focus = True
    elif autofocus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1 = continuous AF on UVC
        touched_focus = True
    if touched_focus:
        af_after = cap.get(cv2.CAP_PROP_AUTOFOCUS)
        focus_after = cap.get(cv2.CAP_PROP_FOCUS)
        print(f"[camera] autofocus reads back {af_after!r} (1=ON, 0=manual); "
              f"focus reads {focus_after!r}")
        print("[camera] NOTE: CAP_PROP_AUTOFOCUS/FOCUS are UVC controls, "
              "backend+driver dependent. They usually work with DSHOW, often "
              "NOT with MSMF. If the readback is None/0.0, set focus in the "
              "camera's own app (Elgato Camera Hub, LogiTune, Windows Camera "
              "settings) -- those persist. Or force DSHOW: "
              "--camera-backend dshow.")
    print(f"[camera] open idx {cam_index} "
          f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}@"
          f"{cap.get(cv2.CAP_PROP_FPS):.0f}fps "
          f"(focus/exposure left untouched -- set them in your camera's app)")
    global _latest_jpeg
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        # JPEG encode params: quality 85 is plenty for Pong and keeps size down
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        with _cam_lock:
            _latest_jpeg = buf.tobytes()
        _latest_jpeg_event.set()
        if _show_preview:
            try:
                disp = frame.copy()
                corners = detect_screen(frame)
                if corners is not None:
                    corners_i = corners.astype(np.int32)
                    cv2.polylines(disp, [corners_i], True, (0, 255, 0), 3)
                    cv2.putText(disp, "SCREEN DETECTED",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 3)
                    # order corners TL/TR/BR/BL and warp to 128x128 -- this
                    # is exactly what the container's ScreenWarper does, so
                    # the preview shows what the NN will see.
                    pts = corners.astype(np.float32)
                    s = pts.sum(axis=1)
                    d = pts[:, 0] - pts[:, 1]
                    tl = pts[np.argmin(s)]
                    br = pts[np.argmax(s)]
                    tr = pts[np.argmax(d)]
                    bl = pts[np.argmin(d)]
                    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
                    dst = np.array([[0, 0], [127, 0], [127, 127], [0, 127]],
                                   dtype=np.float32)
                    H, _ = cv2.findHomography(ordered, dst, method=0)
                    warped = cv2.warpPerspective(frame, H, (128, 128),
                                                 flags=cv2.INTER_AREA)
                    grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    # The NN's actual input: Otsu binarized greyscale
                    # (matches agent.py _preprocess). Pure black/white --
                    # glare (grey) becomes black, game elements stay white.
                    # This is EXACTLY what the NN trains on.
                    grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(grey, 0, 255,
                                              cv2.THRESH_BINARY
                                              + cv2.THRESH_OTSU)
                    nn_view = cv2.resize(binary, (480, 480),
                                         interpolation=cv2.INTER_NEAREST)
                    cv2.putText(nn_view, "WHAT THE NN SEES (binary 128x128)",
                                (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                255, 1)
                    cv2.imshow("NN view", nn_view)
                else:
                    cv2.putText(disp, "NO SCREEN (reposition)",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)
                cv2.imshow("host_bridge camera preview", disp)
                cv2.waitKey(1)
            except Exception:
                pass


def _get_latest_jpeg(timeout_s=2.0):
    _latest_jpeg_event.wait(timeout=timeout_s)
    with _cam_lock:
        return _latest_jpeg


# ---------------------------------------------------------------------------
# Camera + reward server (port 8000): request/response, atomic packet
# ---------------------------------------------------------------------------
class CameraRewardHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # One persistent connection; container sends 1 byte per perceive().
        while True:
            try:
                req = self.request.recv(1)
            except ConnectionResetError:
                break
            if not req:
                break
            jpeg = _get_latest_jpeg()
            if not jpeg:
                # no frame yet; send an empty packet so the container can
                # decide (it'll retry). Shouldn't happen once the camera warms up.
                jpeg = b""
            delta, sl, sr = _consume_reward()
            packet = (struct.pack("<I", len(jpeg)) + jpeg
                      + struct.pack("<iii", delta, sl, sr))
            try:
                self.request.sendall(packet)
            except (BrokenPipeError, ConnectionResetError):
                break


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


# ---------------------------------------------------------------------------
# Servo server (port 8001): persistent, line-based angle commands
# ---------------------------------------------------------------------------
class ServoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        buf = b""
        while True:
            try:
                data = self.request.recv(64)
            except ConnectionResetError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    angle = int(line)
                except ValueError:
                    continue
                if 0 <= angle <= 180 and _arduino is not None:
                    with _arduino_lock:
                        _arduino.write(f"{angle}\n".encode())
                # else: silently ignore out-of-range / no-arduino


# ---------------------------------------------------------------------------
# Viewer receiver (port 8002) + display HTTP server (port 8088)
# ---------------------------------------------------------------------------
# The container PUSHES (raw_jpeg, obs_jpeg, stats) here; we cache them and
# serve them to a browser at http://localhost:8088. Works because the
# container can reach the host (host.docker.internal); the browser hits
# localhost on the host, so no firewall/port-publishing hassle.

_viewer_lock = threading.Lock()
_viewer = {
    "raw_jpg": b"",
    "obs_jpg": b"",
    "reward": 0,
    "score_l": 0,
    "score_r": 0,
    "confidence": 0.0,
    "agent_steps": None,
    "agent_epsilon": None,
    "last_update": 0.0,
}


def _recv_exact_sock(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError("closed")
        buf.extend(c)
    return bytes(buf)


class ViewerPushHandler(socketserver.BaseRequestHandler):
    """Receive pushed viewer packets from the container (persistent socket)."""

    def handle(self):
        while True:
            try:
                hdr = _recv_exact_sock(self.request, 4)
                (raw_len,) = struct.unpack("<I", hdr)
                raw_jpg = _recv_exact_sock(self.request, raw_len)
                hdr = _recv_exact_sock(self.request, 4)
                (obs_len,) = struct.unpack("<I", hdr)
                obs_jpg = _recv_exact_sock(self.request, obs_len)
                tail = _recv_exact_sock(self.request, 24)
                reward, sl, sr = struct.unpack("<iii", tail[:12])
                conf = struct.unpack("<f", tail[12:16])[0]
                a_steps_u = struct.unpack("<I", tail[16:20])[0]
                a_eps = struct.unpack("<f", tail[20:24])[0]
            except (ConnectionError, OSError):
                break
            with _viewer_lock:
                _viewer["raw_jpg"] = raw_jpg
                _viewer["obs_jpg"] = obs_jpg
                _viewer["reward"] = reward
                _viewer["score_l"] = sl
                _viewer["score_r"] = sr
                _viewer["confidence"] = conf
                _viewer["agent_steps"] = (None if a_steps_u == 0xFFFFFFFF
                                          else a_steps_u)
                _viewer["agent_epsilon"] = (None if a_eps < 0 else a_eps)
                _viewer["last_update"] = time.time()


from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ViewerHTTPHandler(BaseHTTPRequestHandler):
    """Serve the browser: HTML page + two JPEGs + stats JSON."""

    def log_message(self, *a):
        pass  # quieter logs

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = self._html()
            self._send(body, "text/html; charset=utf-8")
        elif self.path == "/raw.jpg":
            # serve the host's OWN latest camera frame — always available,
            # no need to round-trip it through the container.
            jpg = _get_latest_jpeg(timeout_s=0.5)
            self._send(jpg if jpg else b"", "image/jpeg", ok=bool(jpg))
        elif self.path == "/obs.jpg":
            with _viewer_lock:
                jpg = _viewer["obs_jpg"]
            self._send(jpg if jpg else b"", "image/jpeg", ok=bool(jpg))
        elif self.path == "/stats.json":
            import json
            with _viewer_lock:
                data = {
                    "reward": _viewer["reward"],
                    "score_l": _viewer["score_l"],
                    "score_r": _viewer["score_r"],
                    "confidence": round(_viewer["confidence"], 3),
                    "agent_steps": _viewer["agent_steps"],
                    "agent_epsilon": (None if _viewer["agent_epsilon"] is None
                                       else round(_viewer["agent_epsilon"], 4)),
                    "age_s": (round(time.time() - _viewer["last_update"], 1)
                              if _viewer["last_update"] else None),
                }
            self._send(json.dumps(data).encode(), "application/json")
        else:
            self._send(b"404 not found\n", "text/plain", code=404)

    def _send(self, body, ctype, code=200, ok=True):
        if not ok:
            self.send_response(503)
        else:
            self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    @staticmethod
    def _html():
        return """<!doctype html>
<html><head><meta charset="utf-8"><title>Physical Pong RL Viewer</title>
<style>
  body { font-family: system-ui, sans-serif; background:#111; color:#eee; margin:20px; }
  h1 { font-size: 1.3em; }
  .row { display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap; }
  .panel { background:#1c1c1c; padding:12px; border-radius:8px; }
  .panel h2 { font-size:1em; margin:0 0 8px; color:#9bd; }
  img { display:block; border:2px solid #333; border-radius:4px; background:#000; }
  .stats { font-size:0.95em; line-height:1.6; min-width:220px; }
  .stats span { color:#7d7; font-weight:bold; }
  .reward-pos { color:#7d7; }
  .reward-neg { color:#f77; }
  .conf-low { color:#f93; }
  .stale { color:#888; }
</style></head>
<body>
<h1>Physical Pong RL - live viewer</h1>
<div class="row">
  <div class="panel">
    <h2>Raw camera (before preprocessing)</h2>
    <img src="/raw.jpg" width="480" id="raw">
  </div>
  <div class="panel">
    <h2>Warped 128x128 observation (NN input)</h2>
    <img src="/obs.jpg" width="256" id="obs" style="image-rendering:pixelated;">
  </div>
  <div class="panel stats">
    <h2>State</h2>
    <div>Last reward: <span id="reward">--</span></div>
    <div>Score: <span id="score">L0-R0</span></div>
    <div>AprilTag confidence: <span id="conf">--</span></div>
    <div>Agent steps: <span id="steps">--</span></div>
    <div>Agent epsilon: <span id="eps">--</span></div>
    <div>Last update: <span id="age">--</span>s ago</div>
    <p style="color:#888;font-size:0.85em;">
      Read-only viewer. The container computes both views (using the same
      AprilTag detector the agent uses) and pushes them here. Safe to run
      alongside training.
    </p>
  </div>
</div>
<script>
function poll(){
  fetch('/stats.json').then(r=>r.json()).then(d=>{
    let r=document.getElementById('reward');
    r.textContent=d.reward; r.className=d.reward>0?'reward-pos':(d.reward<0?'reward-neg':'');
    document.getElementById('score').textContent='L'+d.score_l+'-R'+d.score_r;
    let c=document.getElementById('conf'); c.textContent=d.confidence;
    c.className=d.confidence<0.5?'conf-low':'';
    document.getElementById('steps').textContent=d.agent_steps!=null?d.agent_steps:'--';
    document.getElementById('eps').textContent=d.agent_epsilon!=null?d.agent_epsilon:'--';
    let a=document.getElementById('age');
    if(d.age_s==null){a.textContent='no data yet';a.className='stale';}
    else{a.textContent=d.age_s;a.className=d.age_s>2?'stale':'';}
  }).catch(()=>{});
  document.getElementById('raw').src='/raw.jpg?t='+Date.now();
  document.getElementById('obs').src='/obs.jpg?t='+Date.now();
}
setInterval(poll,500); poll();
</script>
</body></html>""".encode("utf-8")


# ---------------------------------------------------------------------------
# Arduino open
# ---------------------------------------------------------------------------
def _open_arduino(port, baud):
    global _arduino
    try:
        s = serial.Serial(port, baud, timeout=1)
        time.sleep(1.5)  # Arduino resets on serial open; let the bootloader pass
        _arduino = s
        print(f"[arduino] opened {port} @ {baud}")
    except Exception as e:
        print(f"[arduino] could not open {port}: {e} (servo commands ignored)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--camera-backend", default="",
                    choices=["", "dshow", "msmf", "any"],
                    help="force a capture backend. dshow usually required "
                         "for CAP_PROP_AUTOFOCUS/FOCUS to take effect on UVC cams")
    ap.add_argument("--camera-res", default="",
                    help="WxH@fps e.g. 1280x720@30 (optional)")
    ap.add_argument("--focus", type=int, default=None,
                    help="opt-in manual focus value (disables autofocus). "
                         "Default: leave the camera alone.")
    ap.add_argument("--autofocus", action="store_true",
                    help="opt-in: explicitly enable autofocus "
                         "(CAP_PROP_AUTOFOCUS=1). Default: leave the camera "
                         "alone -- set focus in your camera's own app.")
    ap.add_argument("--show-preview", action="store_true",
                    help="show a live camera preview window (for positioning)")
    ap.add_argument("--arduino", default="COM5")
    ap.add_argument("--arduino-baud", type=int, default=9600)
    ap.add_argument("--no-arduino", action="store_true")
    ap.add_argument("--sensecap", default="COM10")
    ap.add_argument("--sensecap-baud", type=int, default=115200)
    ap.add_argument("--no-sensecap", action="store_true")
    ap.add_argument("--cam-port", type=int, default=8000)
    ap.add_argument("--servo-port", type=int, default=8001)
    ap.add_argument("--viewer-port", type=int, default=8002,
                    help="port the container pushes viewer frames to")
    ap.add_argument("--http-port", type=int, default=8088,
                    help="port the browser opens")
    args = ap.parse_args()

    w = h = fps = None
    if args.camera_res:
        wh, _, fps_s = args.camera_res.partition("@")
        w, _, h = wh.partition("x")
        w = int(w) if w else None
        h = int(h) if h else None
        fps = int(fps_s) if fps_s else None

    global _show_preview
    _show_preview = args.show_preview

    # start background threads
    threading.Thread(target=_camera_thread,
                     args=(args.camera, w, h, fps, args.focus,
                           args.autofocus, args.camera_backend),
                     daemon=True).start()
    if not args.no_sensecap:
        threading.Thread(target=_sensecap_reader,
                         args=(args.sensecap, args.sensecap_baud),
                         daemon=True).start()
    if not args.no_arduino:
        _open_arduino(args.arduino, args.arduino_baud)

    cam_srv = ThreadingTCPServer(("0.0.0.0", args.cam_port),
                                 CameraRewardHandler)
    servo_srv = ThreadingTCPServer(("0.0.0.0", args.servo_port),
                                   ServoHandler)
    viewer_srv = ThreadingTCPServer(("0.0.0.0", args.viewer_port),
                                    ViewerPushHandler)
    http_srv = ThreadingHTTPServer(("0.0.0.0", args.http_port),
                                  ViewerHTTPHandler)
    print(f"[bridge] camera+reward on tcp 0.0.0.0:{args.cam_port}")
    print(f"[bridge] servo        on tcp 0.0.0.0:{args.servo_port}")
    print(f"[bridge] viewer push  on tcp 0.0.0.0:{args.viewer_port}")
    print(f"[bridge] viewer HTTP  on port {args.http_port} (open http://127.0.0.1:{args.http_port} in a browser)")
    print("[bridge] ready. from the container run: python train.py")

    threading.Thread(target=cam_srv.serve_forever, daemon=True).start()
    threading.Thread(target=viewer_srv.serve_forever, daemon=True).start()
    threading.Thread(target=http_srv.serve_forever, daemon=True).start()
    try:
        servo_srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] shutting down")
        cam_srv.shutdown()
        servo_srv.shutdown()
        viewer_srv.shutdown()
        http_srv.shutdown()


if __name__ == "__main__":
    main()
