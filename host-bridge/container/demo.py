"""
Demo run INSIDE the Docker container.

Requires the host bridge to be running on your Windows host:
    python host_bridge.py --camera 0 --port COM5 --baud 9600

This script:
  1. Fetches a single JPEG frame from the host camera and saves it to disk.
  2. Sweeps the servo 0 -> 180 -> 0 a couple of times through the serial bridge.

If OpenCV works in this container it will also decode the frame to a numpy
array and print its shape; otherwise it just saves the JPEG.
"""

import time

import bridge


def main():
    # --- Camera -----------------------------------------------------------
    jpg = bridge.get_frame()
    out = "frame.jpg"
    with open(out, "wb") as f:
        f.write(jpg)
    print(f"saved {out} ({len(jpg)} bytes)")

    arr = bridge.get_frame_as_array()
    try:
        arr.shape  # numpy arrays have .shape; raw bytes do not
        print(f"decoded frame shape: {arr.shape}  dtype: {arr.dtype}")
    except AttributeError:
        print("OpenCV not usable here; saved raw JPEG only.")

    # --- Servo ------------------------------------------------------------
    with bridge.ServoStream() as servo:
        print("sweeping servo 0 -> 180 -> 0 ...")
        for _ in range(2):
            for a in range(0, 181, 10):
                servo.move(a)
                time.sleep(0.05)
            for a in range(180, -1, -10):
                servo.move(a)
                time.sleep(0.05)
    print("done")


if __name__ == "__main__":
    main()
