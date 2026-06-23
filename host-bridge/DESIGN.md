# Physical Pong RL — locked design

Train a neural policy that plays Pong **physically**: a SenseCAP Indicator
running a custom Pong firmware is observed by a webcam; an Arduino-driven SG90
servo pushes a KY-023 joystick to move the player paddle. The neural network
runs in this Docker container (Quadro RTX 4000). The Windows host bridges the
camera, the SenseCAP reward, and the Arduino servo to the container over
persistent sockets via `host.docker.internal`.

This is a "physical Atari" loop in the spirit of
github.com/Keen-Technologies/physical-atari-rlc, but with hacky/accessible
hardware (Arduino + hobby servo + ESP32 Pong) instead of Dynamixel servos and
PhysicalALE. The reference repo's training harness and Q-learning agent are
reused; its C++ `robotroller` I/O module is NOT used.

## Physical setup

- **SenseCAP Indicator** (ESP32-S3, **480x480 square** LCD) running a fork of
  github.com/RakiDelmoro/sense-pong (development branch). Pong on the screen.
  Left paddle = velocity-controlled by a KY-023 joystick (stick deflection ->
  paddle speed, deadzone 0.12, low-pass smoothed). Right paddle = CPU chase.
  The RP2040 reads the joystick ADC at ~100 Hz over UART to the ESP32; the
  game ticks at ~60 Hz (16 ms).
- **Webcam** on the Windows host, fixed focus + fixed exposure, observes the
  SenseCAP screen. ~30 fps. Clamped fixed, but treated as possibly moving
  (real-world robustness via AprilTags).
- **AprilTags**: 4 x tag36h11 (IDs 0-3) printed on a rigid card fixed around
  the SenseCAP bezel, one per corner of the 480x480 square. Tag centers map
  to display coords (0,0),(480,0),(480,480),(0,480). Screen stays pure game
  pixels; tags are a rigid physical frame the camera locks onto.
- **Arduino + SG90 servo**: the servo mechanically pushes the KY-023 stick.
  Reuse the existing sketch (0..180 -> myServo.write, Serial.parseInt @9600).
  The servo REPLACES the human hand on the joystick.

## Data channels (all over persistent sockets via host.docker.internal)

| Channel | Direction | Content | Mechanism | Latency |
|---|---|---|---|---|
| Observation | host->container | JPEG of the screen | persistent TCP; request/response | ~5 ms + ~2 ms decode |
| Reward | host->container | score delta since last frame | bundled INTO the obs packet (atomic) | ~1 ms |
| Action | container->host | servo angle 0..180 | persistent TCP (one connection, reused) | ~2 ms (9600 baud) |

The earlier HTTP/MJPEG bridge (host-bridge/host_bridge.py) is WRONG for RL: one
TCP handshake + JPEG re-encode per frame adds 30-80 ms jitter. It is replaced
by a persistent-socket bridge. RL needs BOUNDED, CONSISTENT latency, not a low
average.

## Per-step packet contract (core design)

Host keeps a `reward_since_last_frame` accumulator fed by the SenseCAP's
USB-CDC serial line. On each container frame request, the host returns:

    [u32 frame_len][JPEG bytes][i32 reward_delta][i32 score_l][i32 score_r]

and zeroes the accumulator. This attributes every point that happened between
two frames to the later frame -> atomic obs+reward, no timestamp skew, no lost
points. This contract is what everything else keys off.

## Reward (Option A - locked)

Firmware patch to sense-pong `pong.c` (currently line ~240: "Ball escaped a
side: just reset, no score"):

1. Add `score_left` / `score_right`.
2. Ball escapes RIGHT (past CPU paddle) -> score_left++, reward +1 for the
   agent (left paddle), emit `P L <score_l> <score_r>\n`.
3. Ball escapes LEFT (past agent paddle) -> score_right++, reward -1, emit
   `P R <score_l> <score_r>\n`.
4. Draw two score labels on screen (nice to watch; enables OCR fallback later).
5. Emit on the ESP32-S3 native USB-CDC - the SAME COM port idf.py monitor uses.
   Reward lines are prefixed `P `, ESP log lines start with `I(`/`W(` etc., so
   the host parses by line prefix. No second CDC interface needed.

Why A: exact, deterministic, ~1 ms, immune to lighting/OCR/clipping. Faithful
to the reference design (physical-atari-rlc gets reward from the game's
internal score, never from the camera). Firmware flashes on the Windows host;
the container never touches the firmware - I hand the user the exact patch to
apply/flash.

Reward shape: +1 opponent misses, -1 agent misses. Sparse +-1 is fine -
agent_action_input already does multistep returns.

## Episode semantics (locked - important)

- `terminated` = ALWAYS False on real hardware. The physical game cannot be
  force-reset by env.reset() - it runs autonomously and self-resets on escape.
  So we never claim termination.
- `truncated` = True when no reward for `max_frames_without_reward` frames
  (the repo's existing mechanism). This is exactly what the repo's RealEnv
  does.
- `env.reset()` on real hardware resets the env's INTERNAL counters (truncation
  timer, cached obs/reward) ONLY, not the game. The game keeps running.

This matches the reference design and avoids a physical impossibility.

## Action space (locked)

3 actions {UP, DOWN, HOLD}, one-hot, num_actions=3, action_encoding=0 (the
repo's agent_action_input supports one-hot when num_actions != 18).

- HOLD = servo at a_center (stick in deadzone 0.12 -> paddle stops).
- UP/DOWN = servo at a_up/a_down (fixed deflection -> paddle moves at fixed
  speed, matching sense-pong's velocity control).

One-time calibration finds a_center/a_up/a_down by watching the paddle. The
existing Arduino sketch (0..180 -> myServo.write, Serial.parseInt @9600) is
reused unchanged.

Risk flagged: servo transit lag. A 9600-baud SG90 takes ~0.1 s to swing 60 deg.
At 30 fps the servo may not reach its target before the next command, so paddle
motion lags the policy by a frame or two. Bounded but real. Mitigations if it
hurts: smaller deflection angles (less travel), or faster servo/baud. Measure
once running. A digital bypass (container -> ESP32 USB-CDC joystick-Y,
skipping servo+stick+RP2040) is a debug-only fallback, NOT the target - the
physical actuation is the whole point.

## Observation / AprilTags (locked)

- 4 x tag36h11 IDs 0-3 printed around the SenseCAP, one per square corner.
- Detection in the CONTAINER (full JPEG crosses the wire, ~3 MB/s @ 30 fps,
  trivial for Docker NAT; keeping the warp in Python next to the NN is easy to
  tune). Host stays a dumb relay.
- Library: `dt-apriltags` (pip; Python bindings to the fast C lib). No
  libapriltag-dev / cmake build - we do NOT use the repo's C++ robotroller
  module at all.
- Rate caching: detect ~every 0.5 s (or on confidence drop), CACHE the 3x3
  homography H, apply H every frame (~1 ms). AprilTags affordable in the 33 ms
  loop. For a fixed-ish rig a 0.5 s stale H is fine; if a tag is occluded the
  cached H carries the warp until detection reacquires.
- Fallback on detection failure: reuse last good H, flag obs low-confidence,
  KEEP GOING. Never drop frames in an RL loop (policy needs a continuous
  stream); a slightly-stale warp beats a gap.
- Warp target: 128x128x3 (matches repo obs_size/obs_channels and
  agent_action_input's CNN). 480x480 square -> 128x128, pure downscale +
  perspective correction, no aspect distortion.
- Lock webcam focus + exposure (manual cap.set) so the tag detector and NN are
  not fighting brightness drift. Fixed exposure matters MORE with AprilTags
  (tag detection is exposure-sensitive).

AprilTags from the start (user decision): real-world robustness against
vibration, bumps, autofocus hunting, the SenseCAP not sitting flat after a
power cycle, webcam sag over time. Static one-time calibration is a fragile
assumption and is NOT used. AprilTags turn a brittle assumption into a measured
fact every frame.

## Training stack reuse (no C++ build)

- `learn_policy.py` stays as-is - it is env-agnostic (only calls
  reset/perceive/get_observation/get_reward/get_terminated/get_truncated/
  step/get_num_actions/shutdown).
- `agent_action_input.py` (Q-learning CNN policy) stays essentially as-is; set
  num_actions=3, action_encoding=0. Already tuned (EMA observations, multistep
  returns, target net, SplitOpt). Reusing it saves reproducing RL that works.
- Write ONE new file: a Python `RealEnv` (same interface as the repo's
  real_env.py) that, per perceive(), pulls (frame, reward) from the bridge,
  runs AprilTag warp -> 128x128x3, caches obs/reward/terminated/truncated; per
  step(action), maps the action to a servo angle and sends it. Replaces
  robotroller.PhysicalAtariEnv entirely. No cmake, no libapriltag-dev, no
  Dynamixel SDK.

## Sim -> real (locked flow)

There is no sim for sense-pong in either repo, so build a tiny Python Pong sim
that mirrors sense-pong physics (480x480, ball speed 4, paddle speed 6,
velocity joystick control, deadzone 0.12, CPU-chase opponent) and emits the
same 128x128x3 observation + score reward.

1. Pretrain agent_action_input in this sim on the RTX 4000 (fast, no latency,
   no hardware). This is where the NN actually learns Pong.
2. Fine-tune on the real SenseCAP via the bridge + custom RealEnv (the repo's
   finetune_policy.py already supports --env real).

Matches the reference repo's sim->real flow (agent_action_input_sim.json ->
finetune_policy.py --env real) and de-risks everything: working trained NN in
sim first, then adapt to camera+servo reality.

## Latency budget (per step, 33 ms @ 30 fps)

| Stage | Latency |
|---|---|
| Host->container JPEG | ~5 ms |
| JPEG decode | ~2 ms |
| AprilTag detect (amortized) | ~0 most frames, ~5 on detect frames |
| Warp | ~1 ms |
| NN forward (RTX 4000) | ~3-5 ms |
| Servo command + 9600 baud write | ~2 ms |
| Servo transit + RP2040 100 Hz read + 60 Hz game tick | up to ~16 ms (inherent) |

Total bounded well under 33 ms. Dominant irreducible term is the 10-16 ms
actuation chain, fine at 30 fps.

## Build order (NOT STARTED)

1. Sim pretraining - needs NONE of the above; runs on the RTX 4000 today;
   gives a trained NN fast. Lowest-risk first.
2. Persistent-socket bridge - camera (latest-frame + reward bundle) + servo,
   host relay for SenseCAP USB-CDC reward. Replaces host-bridge/host_bridge.py.
3. Python RealEnv - AprilTag warp + 3-action servo mapping + atomic
   obs/reward/truncation.
4. Fine-tune on real - adapt sim-trained policy to webcam+servo via
   finetune_policy.py --env real.
5. Iterate: exposure, detection rate, action rate, servo transit lag.

## Open confirms before build (non-blocking for Phase 1)

- Firmware patch: user flashes on Windows host; I hand the exact pong.c patch
  (scoring + score labels + P L/R ... USB-CDC lines). User confirmed OK to be
  the one to flash.
- Display dims confirmed: 480x480 square.
- AprilTag printing: user prints/mounts 4 x tag36h11 IDs 0-3 around the bezel.
