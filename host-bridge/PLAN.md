# Plan: physical-atari-rlc in this workspace container

Goal: camera frames in → train a neural policy on real (physical) Atari, with
bounded latency so the RL loop stays real-time. Repo:
https://github.com/Keen-Technologies/physical-atari-rlc

## How the repo's loop works (and its latency budget)

```
Devbox (RPi + Waveshare screen, PhysicalALE, shows game + AprilTags)
   │  camera observes screen                          ▲ joystick
   ▼                                                  │
Agent machine (this container, ideally):              │
   Camera.grab()  → AprilTag detect → warp to 128x128x3 obs
   policy.forward(obs) → action
   Robotroller.act(action) → Dynamixel goal-position writes ──┘
   repeat @ 30 fps  →  ~33 ms budget per step
```

The I/O is a C++/pybind11 module `robotroller` (classes `Camera`,
`Robotroller`, `PhysicalAtariEnv`) that talks to hardware *locally*:
OpenCV `VideoCapture` on `camera.device` (e.g. /dev/video0) and the Robotis
Dynamixel SDK on `robot.serial_port` (e.g. /dev/serial/by-id/...) at 1 Mbps.

Reference config (`robotroller.default.json`):
  camera: 1280x720, target_fps 60 / fps 30, fixed focus/exposure
  robot:  Dynamixel XC330, IDs 50(fire)/51(LR)/52(UD), baud 1,000,000,
          P_gain 4500, servo position values for dpad/button deflections

## Why the current HTTP/MJPEG bridge is wrong for RL

`host-bridge/host_bridge.py` + `container/bridge.py` (already built):
- Camera: one HTTP GET per frame → TCP handshake + JPEG re-encode per call.
- Servo: one TCP connect per `set_servo()`.

At 30 fps each step must stay under ~33 ms. HTTP-per-frame over the Docker
Desktop NAT adds 30–80 ms of jitter; the policy then trains on stale,
mis-timed frames. RL needs **bounded, consistent** latency, not just a low
average. So for real hardware we use one of Path B or Path C below. (The HTTP
bridge stays useful only for the single-frame + single-servo demo, not RL.)

## Hardware reality checks (must-haves for "real" env)

1. **Devbox**: a Raspberry Pi running PhysicalALE that displays the game plus
   AprilTags on a Waveshare screen. The camera observes *that screen*. No
   devbox → no AprilTags → no observation warp → no reward. Non-negotiable for
   the physical env.
2. **Servos**: the repo drives **Dynamixel XC330** servos (IDs 50/51/52,
   1 Mbps, Dynamixel protocol) via the Robotis SDK in the C++ module — NOT the
   SG90 hobby servo + `Serial.parseInt()` sketch. The current Arduino sketch
   cannot be driven by `Robotroller`.

This container today: Quadro RTX 4000 (8 GB), CUDA 12.8, torch 2.1.0a0+CUDA
verified. Missing build deps only (cmake, python3-pybind11, libopencv-dev,
libapriltag-dev) — apt-installable.

## Path A — Sim training in this container  (DO FIRST, no hardware)

Validates the whole training stack on the RTX 4000 with zero latency issues.

1. apt install build deps: `cmake python3-pybind11 libopencv-dev
   libapriltag-dev build-essential`.
2. Clone the repo + DynamixelSDK (C++), build the `robotroller` module into
   this Python env (cmake/make/make install per input_output_cpp_library
   README).
3. `pip install -r agent_code/requirements.txt` (need ale-py >= 0.12).
4. Run sim: `python learn_policy.py --config
   experiment_configs/agent_action_input_sim.json --run 0` (cuda).

Exit criteria: a sim run trains and writes checkpoints. This proves the
neural net, optimizer, and env glue all work on this box before any hardware.

## Path B — Real hardware via native USB passthrough  (best latency)

The right architecture for real-time: get the USB devices *into* the container
so the `robotroller` C++ module talks to them directly — no network hop,
bounded latency identical to a bare-metal agent machine.

1. On Windows: `winget install usbipd-win`.
2. `usbipd list` → find the Dynamixel USB-serial adapter and the webcam.
3. `usbipd bind --busid <id>` then `usbipd attach --wsl --busid <id>` for each.
   They appear in WSL2 as `/dev/ttyUSBx` (or `/dev/serial/by-id/...`) and
   `/dev/video0`.
4. **Restart this workspace container** with
   `--device /dev/serial/by-id/... --device /dev/video0 --privileged` so the
   devices show up here (the harness-managed container cannot be restarted by
   us with custom flags — this is the one manual step that needs the harness/
   Docker Desktop config). Optionally `--group-add video,dialout`.
5. Edit `physical_atari_sample_config.json`:
   `camera.device` and `robot.serial_port` to the in-container device paths.
6. Run: `python learn_policy.py --config
   experiment_configs/agent_action_input_real.json --run 0`.

Notes:
- Dynamixel over usbipd → WSL2 works well (the protocol tolerates USB-serial
  latency; 1 Mbps is fine).
- Webcam over usbipd → WSL2 is historically **flaky** (UVC isoctranfers). If it
  drops frames, fall back to Path C for the camera only and keep the Dynamixel
  on native passthrough.

## Path C — Low-latency bridge fallback  (camera only, no container restart)

If usbipd webcam passthrough is unstable and we can't restart the container
with `--device /dev/video0`, keep a persistent-socket camera bridge:

- Replace the HTTP/MJPEG server with a **single persistent TCP socket** that
  pushes frames continuously: `[uint32 len][JPEG bytes]`, host keeps the
  connection open, container keeps reading. No per-frame handshake.
- JPEG 1280x720 ≈ 80–150 KB → ~4 MB/s @ 30 fps, trivial for the Docker NAT.
  Encode/decode ~2 ms each side on any modern CPU.
- Added latency ≈ JPEG encode (~2 ms) + socket transfer (~1–5 ms) + JPEG
  decode (~2 ms) ≈ **~10 ms** (plus the inherent 33 ms capture cadence).
  Bounded and consistent, unlike the HTTP version.
- Implement as a subclass/drop-in for the repo's `Camera` (or monkey-patch
  `robotroller.Camera`) so `PhysicalAtariEnv` is unchanged. Servos stay on
  native USB passthrough (Path B) for action latency.

This is only worth building if Path B's webcam step fails. The Dynamixel bus
should always go native — never bridge a 1 Mbps half-duplex servo bus over
TCP, it adds action latency and risks dropped writes.

## Path D (optional) — Hobby-servo shim for a minimal real shake-out

If you don't have Dynamixels yet but want to exercise the camera → policy →
servo path on real hardware with just the SG90 + Arduino you already have:

- Write a small `Robotroller`-compatible shim that translates the repo's
  3-servo action API (fire / LR / UD) into a single hobby-servo angle on the
  existing bridge: only the **fire-button** axis maps meaningfully (press =
  angle A, release = angle B). LR/UD are ignored.
- Lets you verify the full camera→NN→actuation loop physically with one axis
  before buying Dynamixels. Not a real Atari controller (can't move the dpad).

## Servo protocol mismatch — the SG90 sketch vs. the repo

| Aspect           | Your Arduino sketch        | physical-atari `Robotroller`      |
|------------------|----------------------------|------------------------------------|
| Servo            | SG90 hobby (PWM, pin 9)    | Dynamixel XC330 (TTL, IDs 50/51/52)|
| Transport        | USB-CDC serial, 9600 baud  | USB-serial, 1,000,000 baud         |
| Protocol         | `Serial.parseInt()` 0–180  | Robotis Dynamixel protocol v2     |
| Control          | 1 angle                    | 3 goal positions + PID gains       |

The existing `host_bridge.py` servo path matches the **SG90 sketch** only.
For the repo you either (a) buy Dynamixels and use Path B, or (b) use the
Path D shim with the SG90 for a one-axis shake-out.

## Recommended order

1. **Path A now** — I can build deps + the module + run a sim training here
   today; no hardware needed and it de-risks everything.
2. **Path B when you have the Devbox + Robotroller + Dynamixels** — native
   passthrough, best latency, this is the real target.
3. **Path C only if** usbipd webcam passthrough proves flaky in step 2.
4. **Path D only if** you want a quick physical shake-out before buying
   Dynamixels.

## What I need from you to proceed

- Confirm Path A: may I apt-install the build deps and clone the repo + build
  the `robotroller` module here and run a sim training?
- For Path B/C/D: do you have (a) the Devbox (RPi + Waveshare screen) running
  PhysicalALE, (b) the Robotroller + 3× Dynamixel XC330, or (c) just the
  SG90 + Arduino for now? The answer decides B vs C vs D.
- Webcam model (for usbipd feasibility in Path B / JPEG tuning in Path C).
