"""
Pull live frames from the running host_bridge (port 8000) and render a PNG
showing exactly what the agent's Perception sees.

Two modes:

  --warp-mode screen (default)
      Detect the Pong display, warp it to 128x128 (the designed scaffold:
      crop to the game + normalize geometry so position-invariant class
      binding works), and run perception on that. Annotates tracked objects,
      discovered behavior classes, and the controlled object.

  --warp-mode raw
      Run perception DIRECTLY on the raw camera (downscaled grayscale), with
      NO screen warp. This shows what perception sees when it gets the WHOLE
      camera -- room, desk, hands, reflections AND the game -- so you can
      compare against the screen-warp mode and see why the warp exists.

Run while host_bridge.py is running on the Windows host:

    python visualize_perception.py --out /workspace/perception_view.png

Tune --z-thresh for the real rig (real video is lower-contrast / noisier than
the synthetic scene; start ~3.2-3.6 and adjust until the warped view has a
small clean set of tracks, not dozens).
"""

import argparse
import socket
import struct
import sys
import time

import cv2
import numpy as np

from agent.perception import Perception


# --------------------------------------------------------------------------- #
# Bridge I/O (port 8000 atomic packet)                                        #
# --------------------------------------------------------------------------- #
def _recv(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError("bridge closed the connection")
        buf.extend(c)
    return bytes(buf)


def pull_frame(host="host.docker.internal", port=8000, timeout=5.0):
    """One atomic (frame, reward_delta, score_l, score_r) pull from the host."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        s.sendall(b"\x00")
        (flen,) = struct.unpack("<I", _recv(s, 4))
        jpg = _recv(s, flen) if flen else b""
        delta, sl, sr = struct.unpack("<iii", _recv(s, 12))
    finally:
        s.close()
    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img, delta, sl, sr


# --------------------------------------------------------------------------- #
# Screen detection + warp (mirrors the host preview / ScreenWarper)          #
# --------------------------------------------------------------------------- #
def detect_screen_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    _, bright = cv2.threshold(gray_eq, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(bright)
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best, best_ring = None, 0.0
    for c in contours:
        if cv2.contourArea(c) < 0.05 * (h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if not (0.7 <= bw / float(bh) <= 1.3):
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        interior = float(cv2.mean(gray, mask=mask)[0])
        ring = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2) - mask
        surround = float(cv2.mean(gray, mask=ring)[0])
        rc = abs(surround - interior)
        if rc < 30.0:
            continue
        lo, hi = min(interior, surround), max(interior, surround)
        if hi <= 0 or (lo / hi) > 0.75:
            continue
        if rc > best_ring:
            best_ring, best = rc, c
    if best is None:
        return None
    peri = cv2.arcLength(best, True)
    for eps_frac in (0.02, 0.04, 0.06):
        approx = cv2.approxPolyDP(best, eps_frac * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            break
    else:
        corners = cv2.boxPoints(cv2.minAreaRect(best)).astype(np.float32)
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmax(d)]
    bl = corners[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_screen(frame, corners, out=128):
    dst = np.array([[0, 0], [out - 1, 0], [out - 1, out - 1], [0, out - 1]],
                   dtype=np.float32)
    H, _ = cv2.findHomography(corners, dst, method=0)
    warped = cv2.warpPerspective(frame, H, (out, out), flags=cv2.INTER_AREA)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), H


def to_warped_xy(cx, cy, H):
    """Map a point in warped-image coords back to ORIGINAL frame coords."""
    p = np.array([[[cx, cy]]], dtype=np.float32)
    inv = cv2.invert(H)[1]
    out = cv2.perspectiveTransform(p, inv)[0, 0]
    return float(out[0]), float(out[1])


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def class_color(cid):
    rng = np.random.default_rng(int(cid) * 2654435761 % 2**31)
    return tuple(int(x) for x in (rng.integers(80, 255), rng.integers(80, 255),
                                  rng.integers(80, 255)))


def _class_label(cid, classes, BRACKET=0.005):
    if cid is None:
        return "unbound", (160, 160, 160)
    c = classes.get(cid)
    if c is None:
        return f"cls{cid}?", (200, 200, 200)
    if c["action_effect"] > BRACKET:
        return f"cls{cid} CONTROLLED ae={c['action_effect']:.3f}", (0, 165, 255)
    fm = c["free_motion"]
    tag = ("decoy-ish" if fm < 0.03
           else "mover(ball?)" if fm > 0.10
           else "tracker(opp?)")
    return f"cls{cid} passive {tag} fm={fm:.3f}", class_color(cid)


def _draw_tracks(canvas, tracks, controlled_id, diag, S):
    objs = diag["objects"]
    classes = diag["classes"]
    for t in tracks:
        cid = objs.get(t["id"], {}).get("bound")
        label, color = _class_label(cid, classes)
        x, y, w, h = t["cx"], t["cy"], t["w"], t["h"]
        x0 = int((x - w / 2) * S); y0 = int((y - h / 2) * S)
        x1 = int((x + w / 2) * S); y1 = int((y + h / 2) * S)
        thick = 4 if t["id"] == controlled_id else 2
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thick)
        cv2.putText(canvas, f"id{t['id']} {t['dim']}", (x0, max(15, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        # show behavioral utility score (Move 2) if present -- high = real,
        # low = junk that survived the seed. pred_err = how jittery it is.
        util = t.get("utility")
        if util is not None:
            label = f"{label}  U={util:.2f}"
        cv2.putText(canvas, label, (x0, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        cv2.circle(canvas, (int(x * S), int(y * S)), 3, color, -1)
    return canvas


def render_screen(warped_gray, tracks, controlled_id, diag, H, raw_frame, corners):
    """Warp mode: left = warped 128x128 with boxes; right = raw cam + polygon."""
    S = 6
    big = cv2.resize(warped_gray, (128 * S, 128 * S),
                     interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    canvas = _draw_tracks(canvas, tracks, controlled_id, diag, S=S)
    title = f"WHAT PERCEPTION SEES (warped 128x128)  tracks={len(tracks)}"
    title += f"  controlled=id{controlled_id}" if controlled_id is not None \
        else "  controlled=(none yet)"
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

    raw = raw_frame.copy()
    if corners is not None:
        cv2.polylines(raw, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        cv2.putText(raw, "raw camera (green=detected screen)",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        objs = diag["objects"]; classes = diag["classes"]
        for t in tracks:
            cid = objs.get(t["id"], {}).get("bound")
            _, color = _class_label(cid, classes)
            rx, ry = to_warped_xy(t["cx"], t["cy"], H)
            cv2.circle(raw, (int(rx), int(ry)), 7, color, -1)
            cv2.putText(raw, f"id{t['id']}", (int(rx) + 8, int(ry) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    h0, w0 = raw.shape[:2]
    scale = (128 * S) / float(w0)
    raw = cv2.resize(raw, (int(w0 * scale), int(h0 * scale)))
    hL, wL = canvas.shape[:2]; hR, wR = raw.shape[:2]
    if hR < hL:
        raw = cv2.copyMakeBorder(raw, 0, hL - hR, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.hstack([canvas, raw])


def _annotate_one(color_perc, tracks, controlled_id, diag, panel_w):
    """Render ONE perception frame (color) at panel_w wide with boxes/ids/classes."""
    gh, gw = color_perc.shape[:2]
    S = max(1, panel_w // gw)
    big = cv2.resize(color_perc, (gw * S, gh * S), interpolation=cv2.INTER_NEAREST)
    canvas = big.copy()
    canvas = _draw_tracks(canvas, tracks, controlled_id, diag, S=S)
    return canvas


def render_raw_montage(history, raw_frame=None, tile_w=480, cols=3):
    """Raw mode: a MONTAGE of the last N perception frames (color, no raw
    camera panel) so you can watch the tracker ADAPT -- new tracks appear,
    ids persist, classes mature, junk (eventually) retires.

    All tiles are the SAME size (no tiny/big split), arranged in a grid of
    `cols` columns that wraps. Each tile shows frame number + track count +
    boxes/ids/classes/utility. Bigger tiles than before so it's readable.
    """
    N = len(history)
    if N == 0:
        return np.zeros((10, 10, 3), np.uint8)
    tiles = []
    for h in history:
        c = _annotate_one(h["color"], h["tracks"], h["controlled"],
                          h["diag"], tile_w)
        cv2.putText(c, f"frame {h['frame']}  tracks={len(h['tracks'])}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        tiles.append(c)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, max(0, th - t.shape[0]),
                                0, max(0, tw - t.shape[1]),
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for t in tiles]
    # pad to a full grid (fill last row with black tiles)
    while len(tiles) % cols != 0:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out = np.vstack(rows)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="host.docker.internal")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="/workspace/perception_view.png")
    ap.add_argument("--warp-mode", choices=["screen", "raw"], default="screen",
                    help="screen: detect+warp the display to 128x128 (the "
                         "designed scaffold). raw: run perception directly on "
                         "the raw camera (no warp) -- shows the whole room.")
    ap.add_argument("--frames", type=int, default=160,
                    help="frames to feed perception before rendering (so "
                         "behavior classes mature; GATE_OBS=80)")
    ap.add_argument("--z-thresh", type=float, default=3.2)
    ap.add_argument("--top-k", type=int, default=None,
                    help="Move-1 top-K seed: keep the K most salient regions "
                         "per frame (rank-based, brightness-invariant) "
                         "instead of a fixed z-thresh. If set, --z-thresh "
                         "is ignored.")
    ap.add_argument("--permissive-z", type=float, default=1.5,
                    help="low z used only to form connected components in "
                         "top-K mode")
    ap.add_argument("--warp", type=int, default=128)
    ap.add_argument("--raw-width", type=int, default=320,
                    help="downscale width for raw mode (keeps aspect). The "
                         "agent's perception normalizes by frame size, so any "
                         "size works; smaller = fewer noise tracks + faster.")
    ap.add_argument("--action", type=int, default=2,
                    help="action index fed to perception each frame (the "
                         "neutral/stay action; no servo so we do nothing)")
    ap.add_argument("--no-utility", action="store_true",
                    help="disable Move 2 (behavioral junk-rejection) for "
                         "A/B comparison")
    ap.add_argument("--montage", type=int, default=6,
                    help="raw mode: number of recent frames to show in the "
                         "montage (frame 0..N-1) so you can watch the "
                         "tracker adapt and new tracks appear")
    ap.add_argument("--tile-w", type=int, default=480,
                    help="raw mode: width of each montage tile (all tiles "
                         "same size; bigger = more readable)")
    ap.add_argument("--cols", type=int, default=3,
                    help="raw mode: montage grid columns")
    ap.add_argument("--fps-cap", type=float, default=30.0)
    args = ap.parse_args()

    if args.warp_mode == "raw":
        # perception will be (re)made once we know the raw frame size
        perc = None
    else:
        perc = Perception(args.warp, args.warp, z_thresh=args.z_thresh,
                          use_utility=not args.no_utility,
                          top_k=args.top_k, permissive_z=args.permissive_z)
    last_render = None
    history = []   # rolling buffer of recent (color, tracks, controlled, diag) for the montage

    print(f"pulling {args.frames} frames from {args.host}:{args.port} "
          f"(mode={args.warp_mode}, z_thresh={args.z_thresh}) ...")
    got = 0
    corners_fixed = None
    for f in range(args.frames):
        try:
            img, delta, sl, sr = pull_frame(args.host, args.port)
        except Exception as e:
            print(f"  frame {f}: pull failed: {e}; retrying in 0.5s")
            time.sleep(0.5)
            continue
        if img is None:
            time.sleep(0.05)
            continue
        got += 1

        if args.warp_mode == "screen":
            if corners_fixed is None:
                corners_fixed = detect_screen_corners(img)
                if corners_fixed is None:
                    print(f"  frame {f}: no screen detected yet")
                    time.sleep(0.05)
                    continue
            gray, H = warp_screen(img, corners_fixed, out=args.warp)
            tracks, controlled = perc.step(gray, args.action)
            last_render = ("screen", gray, tracks, controlled,
                           perc.diagnostics(), H, img, corners_fixed)
        else:  # raw
            h0, w0 = img.shape[:2]
            rw = args.raw_width
            rh = max(1, int(h0 * rw / float(w0)))
            gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                              (rw, rh), interpolation=cv2.INTER_AREA)
            # color version at the SAME resolution, for the visualization only
            color_perc = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
            if perc is None:
                perc = Perception(rh, rw, z_thresh=args.z_thresh,
                                  use_utility=not args.no_utility,
                                  top_k=args.top_k,
                                  permissive_z=args.permissive_z)
            tracks, controlled = perc.step(gray, args.action)
            last_render = ("raw", color_perc, tracks, controlled,
                           perc.diagnostics())
            history.append({"frame": f, "color": color_perc,
                            "tracks": tracks, "controlled": controlled,
                            "diag": perc.diagnostics()})
            if len(history) > args.montage:
                history.pop(0)

        if got % 20 == 0:
            nc = len(last_render[4]["classes"]) if args.warp_mode == "screen" \
                else len(last_render[4]["classes"])
            print(f"  frame {f}: tracks={len(last_render[2])} "
                  f"classes={nc} controlled={last_render[3]}")

        time.sleep(max(0.0, 1.0 / args.fps_cap - 0.005))

    if last_render is None:
        print("ERROR: never got a renderable frame. Is host_bridge running "
              "and the camera pointed at the Pong screen?")
        sys.exit(1)

    if last_render[0] == "screen":
        _, gray, tracks, controlled, diag, H, raw, corners = last_render
        out = render_screen(gray, tracks, controlled, diag, H, raw, corners)
    else:
        out = render_raw_montage(history, tile_w=args.tile_w, cols=args.cols)
    cv2.imwrite(args.out, out)

    # pull the final frame's state for the console summary
    if args.warp_mode == "screen":
        _, _, tracks, controlled, diag, _, _, _ = last_render
    else:
        h = history[-1]
        tracks, controlled, diag = h["tracks"], h["controlled"], h["diag"]

    print("\n=== FINAL ===")
    print(f"mode: {args.warp_mode}   tracks: {len(tracks)}   "
          f"controlled: {controlled}")
    print("classes:")
    for cid, c in diag["classes"].items():
        tag = "CONTROLLED" if c["action_effect"] > 0.005 else "passive"
        print(f"  cls{cid}: {tag} ae={c['action_effect']:.4f} "
              f"fm={c['free_motion']:.4f} n_obs={c['n_obs']} "
              f"n_bound={c['n_bound']}")
    print("objects:")
    for tid, o in diag["objects"].items():
        print(f"  id{tid}: bound=cls{o['bound']} confident={o['confident']} "
              f"n_obs={o['n_obs']}")
    print(f"\nPNG written to {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
