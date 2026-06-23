# Host bridge (low-latency, persistent sockets)

Run this ON YOUR WINDOWS HOST. It is the bridge between the physical hardware
(camera, Arduino servo, SenseCAP reward) and the RL container.

Replaces the old HTTP/MJPEG bridge. The RL loop needs bounded, consistent
latency, so this version uses persistent sockets + an atomic per-step
frame+reward packet.

## Install on Windows (once)

```bat
pip install opencv-python pyserial
```

## Run

```bat
python host_bridge.py --camera 0 --arduino COM5 --arduino-baud 9600 --sensecap COM10
```

Find the COM ports in Device Manager:
- `--sensecap` is the port `idf.py monitor` used (COM10 in your logs). The
  bridge reads `P L <l> <r>` / `P R <l> <r>` reward lines off this console.
- `--arduino` is the Arduino's COM port. The bridge writes `<angle>\n` (0..180)
  to it, matching your SG90 sketch (`Serial.parseInt` @9600).
- `--camera` is the webcam index (0 usually).

Optional flags:
- `--no-arduino` / `--no-sensecap` to skip opening that device (test subsets).
- `--camera-res 1280x720@30` to pin resolution/fps.
- `--focus N` / `--exposure N` to lock webcam focus/exposure (recommended with
  AprilTags — see below).
- `--cam-port 8000` / `--servo-port 8001` to change ports.

## What it exposes (reached from the container via host.docker.internal)

### Port 8000 — camera + reward (request/response, persistent)

Container sends 1 byte per `perceive()`. Host replies with:

```
[u32 frame_len][JPEG bytes][i32 reward_delta][i32 score_l][i32 score_r]
```

- `frame_len` = number of JPEG bytes following.
- `reward_delta` = sum of point events since the last frame request
  (+1 per `P L`, -1 per `P R`), then zeroed. Atomic with the frame: obs and
  reward always arrive together, no timing skew, no lost points.
- `score_l` / `score_r` = current running totals (display/debug only).

### Port 8001 — servo (persistent, line-based)

Container sends `<angle>\n` per action (e.g. `40\n`). Host writes the integer
to the Arduino's serial port. One connection stays open for the whole session.

Locked action mapping (in the container):
- UP   (0) -> 40
- DOWN (1) -> 110
- STAY (2) -> 90

## Windows Firewall

On first run, allow Python through the firewall on Private networks (the
prompt may appear automatically). If the container can't connect, add a rule
manually (as Administrator):

```bat
netsh advfirewall firewall add rule name="pong_bridge" dir=in action=allow protocol=TCP localport=8000,8001
```

You can test locally on the host first: with the bridge running, from another
shell `telnet 127.0.0.1 8001` and type `90<enter>` — the servo should move.

## Reward parsing

A background thread reads lines from the SenseCAP COM port. Any line matching
`P ([LR]) (\d+) (\d+)` is a point event:
- `P L <l> <r>` -> +1 (agent / left paddle scored)
- `P R <l> <r>` -> -1 (CPU / right paddle scored; agent missed)

Other lines (ESP log lines `I (...) tag:`, joystick status, etc.) are ignored.

## Verify it's working

Start the bridge, then from the container:

```bash
cd /workspace/pong-rl
python env.py
```

You should see an obs shape `(128, 128, 3)`, the current reward, scores, and
the servo cycle UP/DOWN/STAY. Then:

```bash
python train.py
```

## AprilTags

The bridge itself does NOT do AprilTag detection — that happens in the
container (`env.py`) so it stays tunable next to the NN. The bridge just
relays the raw webcam JPEG. But because AprilTag detection is exposure-
sensitive, lock the webcam focus + exposure here:

```bat
python host_bridge.py --camera 0 --focus 400 --exposure 20 --arduino COM5 --sensecap COM10
```

(Find good values by trial; the reference physical-atari config used focus 370,
exposure 20.) The tags (4x tag36h11 IDs 0-3) go around the SenseCAP bezel —
see /workspace/host-bridge/DESIGN.md.
