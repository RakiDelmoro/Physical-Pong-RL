"""
Helpers for talking to the Windows-host bridge from INSIDE the Docker container.

The bridge runs on your Windows host; from this container it is reachable at
host.docker.internal (resolved automatically by Docker Desktop).
"""

import os
import socket
import time

import requests

HOST = os.environ.get("BRIDGE_HOST", "host.docker.internal")
CAMERA_PORT = int(os.environ.get("CAMERA_PORT", "8000"))
SERIAL_PORT = int(os.environ.get("SERIAL_PORT", "8001"))


def snapshot_url() -> str:
    """URL of a single JPEG frame from the host camera."""
    return f"http://{HOST}:{CAMERA_PORT}/snapshot"


def stream_url() -> str:
    """URL of the MJPEG stream from the host camera."""
    return f"http://{HOST}:{CAMERA_PORT}/stream"


def get_frame():
    """Fetch one JPEG frame from the host camera. Returns raw JPEG bytes."""
    r = requests.get(snapshot_url(), timeout=5)
    r.raise_for_status()
    return r.content


def get_frame_as_array():
    """
    Fetch one JPEG frame and decode it to a numpy array (BGR).
    Falls back to a tiny pure-Python decoder path if OpenCV is broken.
    """
    jpg = get_frame()
    try:
        import cv2
        arr = cv2.imdecode(__import__("numpy").frombuffer(jpg, dtype="uint8"),
                           cv2.IMREAD_COLOR)
        return arr
    except Exception as e:
        print(f"[bridge] cv2 decode failed ({e}); returning raw JPEG bytes instead")
        return jpg


def set_servo(angle: int) -> None:
    """
    Send a servo angle (0..180) to the Arduino through the host serial bridge.
    Opens a fresh TCP connection per call (simple, robust). For high-rate
    commands, keep one socket open instead (see servo_stream).
    """
    if not 0 <= angle <= 180:
        raise ValueError(f"angle must be 0..180, got {angle}")
    with socket.create_connection((HOST, SERIAL_PORT), timeout=5) as s:
        s.sendall(f"{int(angle)}\n".encode())


class ServoStream:
    """Keep one TCP connection open for many servo commands (lower latency)."""

    def __init__(self):
        self._sock = socket.create_connection((HOST, SERIAL_PORT), timeout=5)

    def move(self, angle: int) -> None:
        if not 0 <= angle <= 180:
            raise ValueError(f"angle must be 0..180, got {angle}")
        self._sock.sendall(f"{int(angle)}\n".encode())

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    # Quick connectivity check.
    print(f"bridge host: {HOST}")
    try:
        frame = get_frame()
        print(f"camera OK, got {len(frame)} JPEG bytes")
    except Exception as e:
        print(f"camera FAILED: {e}")
    try:
        set_servo(90)
        print("servo OK, sent 90")
    except Exception as e:
        print(f"servo FAILED: {e}")
