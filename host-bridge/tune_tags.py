"""
Live camera positioning tuner. Run ON YOUR WINDOWS HOST.

No AprilTag library needed -- pure OpenCV. Shows the live camera feed plus,
in the 4 corners of the SCREEN (where your firmware draws the AprilTags),
a brightness/contrast readout. You adjust the camera until all 4 corners are
bright with high contrast (std), meaning the tags are evenly lit and visible.

Goal: all 4 corner boxes show high mean AND high std, and you can see crisp
black/white tag patterns in each when you look at the feed.

    python tune_tags.py --camera 0

It also lets you drag the 4 corner boxes to where the SenseCAP screen's
corners actually are in the camera view, so the stats target the right pixels.

Keys:
  q / Esc : quit
  r       : reset corners to default (image corners)
  1/2/3/4 : select corner TL/TR/BR/BL to move with mouse
"""

import argparse
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--focus", type=int, default=None,
                    help="manual focus value (try your Elgato Camera Hub value)")
    ap.add_argument("--autofocus", action="store_true",
                    help="leave autofocus ON (default: off unless --focus given)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Only touch focus if the user asked. Otherwise leave the camera's
    # current (e.g. Elgato Camera Hub) setting alone so it stays sharp.
    if args.focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, args.focus)
    elif not args.autofocus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # else: autofocus left on

    cv2.namedWindow("tune", cv2.WINDOW_NORMAL)

    state = {"sel": -1, "drag": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # find nearest corner
            best, bd = -1, 1e9
            for i, (cx, cy) in enumerate(corners):
                d = (cx - x) ** 2 + (cy - y) ** 2
                if d < bd:
                    bd, best = d, i
            if bd < 60 * 60:
                state["sel"] = best
                state["drag"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] and state["sel"] >= 0:
            corners[state["sel"]] = (max(0, min(x, args.width - 1)),
                                     max(0, min(y, args.height - 1)))
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = False

    cv2.setMouseCallback("tune", on_mouse)

    # default corners: spread around the image
    W, H = args.width, args.height
    corners = [(int(W * 0.15), int(H * 0.15)),   # TL
               (int(W * 0.85), int(H * 0.15)),   # TR
               (int(W * 0.85), int(H * 0.85)),   # BR
               (int(W * 0.15), int(H * 0.85))]   # BL
    names = ["TL", "TR", "BR", "BL"]
    box = 60  # half-size of the analysis box at each corner

    print("drag the 4 corner boxes onto the 4 corners of the SenseCAP screen")
    print("adjust camera/lighting until all 4 boxes are bright (high mean) AND")
    print("have high contrast (high std) -- meaning the tags are crisp + visible")
    print("press q/Esc to quit, r to reset corners")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        all_good = True
        for i, (cx, cy) in enumerate(corners):
            x0 = max(0, cx - box); x1 = min(W, cx + box)
            y0 = max(0, cy - box); y1 = min(H, cy + box)
            crop = gray[y0:y1, x0:x1]
            m, s = (float(crop.mean()), float(crop.std())) if crop.size else (0, 0)
            good = m > 80 and s > 40
            all_good = all_good and good
            color = (0, 255, 0) if good else (0, 180, 255) if m > 80 else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            label = f"{names[i]} m={m:.0f} s={s:.0f}"
            cv2.putText(frame, label, (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if state["sel"] == i:
                cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2),
                              (255, 255, 0), 2)

        # overall status
        status = "ALL GOOD - ready!" if all_good else "keep adjusting"
        scolor = (0, 255, 0) if all_good else (0, 180, 255)
        cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(frame, status, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, scolor, 2)

        cv2.imshow("tune", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            corners = [(int(W * 0.15), int(H * 0.15)), (int(W * 0.85), int(H * 0.15)),
                       (int(W * 0.85), int(H * 0.85)), (int(W * 0.15), int(H * 0.85))]
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            state["sel"] = int(chr(key)) - 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
