# World Model — the plan (`M(state, action) → (next_state, reward)`)

Reference: Alberta Plan Step 8 (Prototype-AI I). The keystone that unblocks
planning, the Horde, and control.

## Goal (one sentence)

Build a predictor that takes the current game state + my action and outputs
**where everything will be next** and **whether I'll score/lose** — so the
agent can *imagine* futures and later pick the best one.

## What we already have (the half that's done)

Perception's per-object models already predict **position**:

    predict(object's recent positions, action) -> (Δcx, Δcy)   # next move

Online RLS (closed-form, per-frame, no training phase) on a Fourier+Chebyshev
reservoir. Position prediction per object EXISTS. The world model generalizes
it — we are not starting from zero.

## What we're adding (two pieces)

### Piece A — the reward channel (the keystone, servo-free)

Perception's models predict position only. We add a **reward predictor**:
given the game state, predict the next reward. This is the crucial piece —
planning needs to *imagine reward* to decide which future is good.

Servo-free: reward = "did the ball cross my line" = a function of the ball's
trajectory, which perception already tracks. The **SenseCAP already feeds
ground-truth reward** ("score L0-R335"). Train from real reward signal today.

### Piece B — coupling the objects (servo-free)

Right now each object has its OWN independent predictor — the ball's model
doesn't know where the paddles are. Wrong for Pong: the ball bounces off
paddles, so its next position depends on paddle positions, and reward depends
on ball+paddle geometry. We couple them into one joint state.

## Steps (front-loaded: hard part in its simplest form first)

- **Step 0 — thread reward into the loop.** `Perception.step` / the loop must
  see `reward_delta` (the bridge already delivers it; perception ignores it).
- **Step 1 — define the joint state.** `s_t = [for each object: cx, cy, vx, vy]`
  (normalized), padded to a fixed max object count. Built from perception's
  tracks — no new sensing.
- **Step 2 — the reward predictor (keystone, servo-free).** A new `OnlineRLS`:
  input = `s_t + action`, output = scalar `reward_delta`, trained every frame
  from the SenseCAP. Test: can it foresee a point event BEFORE the ball crosses?
- **Step 3 — couple position prediction (servo-free).** Augment the ball's
  position model with paddle positions (relative geometry) so it learns bounces.
- **Step 4 — imagination rollouts (planning primitive, servo-free).** Roll `M`
  forward `k` steps with a candidate action sequence; sum predicted reward = a
  value estimate for that imagined future. Seed of Dyna/planning.
- **Step 5 — my-paddle dynamics (servo-blocked).** The one channel that needs
  the servo: "my paddle moves when I emit an action." Fills in automatically
  when the servo arrives — same model, same training, no rewrite.

## Servo split (honest)

| channel                         | needs servo? | now? |
|---------------------------------|:---:|:---:|
| ball dynamics + wall bounces    | no  | yes |
| opponent dynamics (chases ball) | no  | yes |
| **reward prediction** (keystone)| no (SenseCAP) | yes |
| ball-paddle coupling (bounces)  | no  | yes |
| imagination rollouts            | no  | yes |
| my-paddle dynamics (action→motion) | YES | later (plugs in) |

5 of 6 channels are servo-free.

## What it is NOT (scope guard)

- Not a new perception — perception stays as-is (top-K + tracker + classes).
- Not a neural net — same online RLS we already use, generalized. Closed-form,
  per-frame, no training phases, no replay.
- Not a policy/controller yet — that's the next phase (actor-critic, Step 4).
  The world model is the PREDICTOR the controller will later plan with.
- Not the Horde/Dyna — those come AFTER `M` exists (they use `M` to imagine).

## Tests (how we know it works)

1. **Reward foresight (sim, servo-free):** the reward predictor forecasts a
   point event N frames before the ball crosses the line — sign matches ahead
   of time.
2. **Coupling:** an imagined rollout respects bounces (ball doesn't pass
   through a paddle in imagination).
3. **Rollout:** a 20-step imagination from a real state yields a plausible
   predicted trajectory + cumulative reward, not garbage.
4. **No regression:** the 7 perception tests still pass (world model is
   additive).

## First commit  [DONE]

Step 0 + Step 2: thread reward + build the reward predictor in
`agent/world_model.py`. Smallest thing that proves the core idea ("the agent
can foresee reward from ball trajectory"), fully servo-free.

Architecture: FROZEN NONLINEAR (random Fourier features, the multivariate
Fourier family that mixes all coordinates -- the form needed so the model can
represent cross-object interactions like bounces) + a TWO-HEAD linear RLS
readout (next-state + reward), reusing perception's closed-form OnlineRLS.
Defaults tuned by sweep: M=1000 features, gamma=0.5. No GPU -- O(M^2) per
update is sub-ms at this M for one update/frame; LMS+Autostep is the scaling
path if M ever grows huge.

A REAL BUG was found and fixed while building the rollout test: the world
model was assigning objects to slots by SORTING their position each frame, so
"slot 2" was sometimes the ball, sometimes a paddle -- the readout could never
learn consistent per-slot dynamics. Fixed: each track id now maps to a FIXED
slot for the whole session (ID-based slots). This is a genuine perception->
world-model interface fix.

Implemented in `agent/world_model.py` (`WorldModel` + `RandomFourierFeatures`)
and `agent/tests/test_world_model.py` (2 passed, 1 honest xfail):
  - W1 trains and learns the base reward rate.
  - W2 FORESIGHT: predicted reward the frame before a point matches the actual
    sign at ~0.71 (M=1000, gamma=0.5, ID-based slots). The agent foresees a
    score event from ball+paddle geometry BEFORE the ball crosses the line.
  - W3 ROLLOUT/BOUNCE (xfail, KNOWN GAP): an imagined 25-step rollout does NOT
    yet bounce the ball off a paddle -- the imagined ball drifts / passes
    through. Diagnosis: one-step ball-position error (~0.05) is ~4x the real
    per-frame movement (~0.014); direction agreement ~65% (barely above
    chance). Foresight (coarse) works; fine position dynamics (precise) does
    not. Marked xfail with a strict reason so we do NOT pretend rollouts work;
    it is the target for the rollout/planning phase (more capacity / better
    gamma / longer training / a smarter prediction target).

KEY LESSON: reward foresight (one step, coarse) is NOT rollout-able dynamics
(many steps, compounded, sharp bounce events). The W3 test was the thing that
proved this -- without it, "rollouts work" would have been an untested claim.

All tests green except the documented xfail (7 perception + 2 world model +
1 xfail). No perception regression.

## Second pass: slot stability + kind-speed merging  [DONE]

The W3 xfail was re-diagnosed and partially fixed. The root cause was NOT
what the first pass assumed (one-step position error). It was a SLOT-
LABELING bug: the ball occupied 8 distinct slots over the run because its
track id churned on every reset (the ball scores, exits, and a brand-new
ball appears at center). No slot ever saw enough bounces to learn the flip.

Three findings, measured directly:
  1. ALL ball track-id churns are RESETS (ball exits -> new ball at center),
     not bounces. A rally (~70 frames) is shorter than GATE_OBS (80), so the
     ball never completes probation in one rally -> never binds -> no class.
  2. The reset breaks geometric continuity (teleport to center, ~0.4 away)
     AND direction continuity (direction randomizes on reset). Neither
     position nor the existing direction-based behavior signature can link
     the old and new ball.
  3. The ball's tentative action-effect is noisy (~40% false-controlled at
     GATE_OBS), so it sometimes binds to the paddle class -- putting two
     different objects in one class. The model-predicted free-motion is also
     noisy (paddle > ball, backwards). The ONLY clean direction-invariant
     kind signal is the OBSERVED speed (ball ~0.016, opp ~0.009, decoy ~0).

Fix (in the world model, not perception): slot assignment by track id +
KIND-SPEED merging. A new track id merges into the slot of an existing object
with the same observed speed (direction-invariant, stable across resets). The
controlled object's slot is EXCLUDED (its speed overlaps the ball's but it's
a different kind). A consolidation pass merges duplicate same-kind slots.
Perception's class_id is deliberately NOT used (it's noisy for the ball).

Result: the ball is now in ONE slot 99% of the time (down from 8). The
learned vx-correction at the opponent paddle plane now goes NEGATIVE
(-0.005, was +0.005) -- the model IS learning the bounce direction for the
first time. But the correction magnitude is ~40% of the -0.014 needed to flip
the velocity, so the imagined ball still passes through (it decelerates but
doesn't bounce). This is now a BASIS-CAPACITY issue (smooth Fourier features
represent a sharp bounce as a smeared deceleration), NOT a labeling bug.

Perception also gained class-binding persistence (Spelke persistence for the
learned label + learner): geometric ghost inheritance for re-detection
without teleport, tentative-ghost revival + coasting adoption to carry the
learner across brief tracking gaps, and kind-speed-based tentative adoption
to carry the learner across resets. 2 new perception tests (CLAIM 5).

## What remains for W3 (the basis-capacity gap)

The ball is in a stable slot and the model learns the bounce DIRECTION, but
the smooth Fourier basis can't produce a sharp enough velocity flip. Options:
  - A sharper basis (higher gamma, more features) -- marginal (39% at 2x).
  - A smarter prediction target: predict the bounce as a discrete EVENT and
    apply it, instead of a continuous residual correction.
  - LMS+Autostep scaling (the Alberta Plan's meta-learning path) for more
    capacity at higher M without the O(M^2) cost.
This is the open next step for the rollout/planning phase.

## Naming

Module: `agent/world_model.py`. Class: `WorldModel`. The Alberta Plan calls
this the "transition model"; we call it the world model (same thing).

## Third pass: DPC world model + event-vs-motion discovery  [DONE; W3 PAUSED]

The world model was rebuilt as a DYNAMIC PREDICTIVE CODING model (Jiang &
Rao 2024 -- the minimal slice): a lower-level MLP predicts the residual
correction to the physics default; a higher-level GRU over the recent state
history outputs a regime vector that MODULATES the lower level (the DPC
trick -- a slower brain reads the approach pattern). Trained online by
backprop, end-to-end. Same interface + signature as the prior RLS model, so
the W1/W2/W3 tests run unchanged. RLS is gone (no fallback).

Why the rebuild: four prior attempts at W3 (RLS discrete-event, LSQ velocity,
DPC-on-RLS, Autostep meta-learning) all hit the SAME wall -- the bounce is a
TEMPORALLY-SPARSE rare event (1-2 frames in a sea of ~zeros), and any
learner/gate trained FRAME-BY-FRAME against that ~zero target collapses and
misses the rare event. The DPC higher level got CLOSEST (its correction hit
-0.013 vs the needed -0.0138) but its RLS-trained gate couldn't learn when to
fire. Real DPC fixes exactly that: the modulation/gate is learned END-TO-END
by backprop over the sequence, not per-frame. That is the tool for the wall.

Result: W1 ✅, W2 ✅ (foresight ~0.86, matching RLS). The DPC model is a
genuine upgrade and stays. W3 still fails -- see the diagnosis below.

### The event-vs-motion discovery (Alberta-Plan "learn your world's structure")

A key insight (from the user): the ball teleporting to center on a point is
not "fake physics to ignore" -- it is a REAL, LEARNABLE regularity of this
world (an episode boundary / a point event). The agent should DISCOVER it
from the stream, not be handed a `reset` flag. This is the Alberta Plan's
core (learn every concept from ordinary experience) applied to events.

Implemented: the world model maintains a running histogram of its own
RESIDUAL norm (the surprise). The distribution is bimodal -- most frames are
tiny (real motion), a few are huge (teleports / non-physical discontinuities).
A frame whose residual norm exceeds median + 12*MAD is flagged an EVENT. The
MOTION correction head does NOT train on event frames (that would teach fake
physics -- "objects teleport to center"); the REWARD head still trains (a
point event co-occurs with the teleport -- that signal is real and wanted).
No hand threshold, no "this is a reset" flag: the agent discovers the event
cluster from its own surprise statistics. General (any world with episode
resets has learnable event structure). Validated: ~22 events detected vs ~17
GT teleports; W1/W2 stay green.

This is the seed of a GVF/Horde predictor for "is an episode boundary about
to happen?" -- the on-plan, continual, experience-only version of handling
episodes.

### W3 PAUSED -- the remaining blocker (honest)

W3 (sharp multi-step rollout that respects paddle bounces) is PAUSED as a
documented known-gap. The precise remaining blocker, measured directly:

  - The ball DOES survive paddle bounces (it coasts through the overlap,
    missed climbing 1->7, then re-acquires the SAME id). The earlier
    "ball vanishes into the paddle blob" diagnosis was WRONG -- re-traced
    frame by frame, the ball is not lost at bounces. The churn is at RESETS
    (a real new ball), not overlaps.
  - But the ball's VELOCITY estimate is wrong through the bounce: it FREEZES
    at the old direction during the coast, then SPIKES to an unphysical
    value (-0.085 vs real -0.014) at re-acquisition. So the residual the
    world model trains on does NOT contain a clean "velocity reversed" event
    at the bounce -- the signal is entangled with the coast/spike artifact.
  - An LSQ velocity fix was tried (responsive, glitch-robust) but REGRESSED
    W2 (foresight 0.86 -> 0.64) -- the changed velocity ripples through the
    whole pipeline. Reverted.

So W3's remaining blocker is a PERCEPTION problem: a responsive, non-spiking
velocity estimate for fast objects during occlusion/coast. It is NOT a
world-model-architecture problem (DPC, event-skip, and the slot fix are all
done and correct). The world model cannot learn a bounce that isn't a clean
signal in its training data, and the velocity artifact hides the bounce.

Next attempts at W3 should target the occlusion-velocity quality in
perception (a responsive estimator that doesn't regress W2), NOT more
world-model architecture. This is parked until that lands.

## What's unblocked next: the Horde (Step 3)  [DONE]

The Horde is built and validated (`agent/horde.py`, `agent/tests/test_horde.py`).
A GVF = (cumulant, gamma) + a linear readout on shared features, learned by
TD(0) (`delta = cumulant + gamma*V(s') - V(s)`). The Horde = many GVFs sharing
one feature representation (the world model's frozen Fourier basis over
(state, action)). Each GVF is an OnlineRLS readout (closed-form, online, no
training phase -- the Alberta Plan's "learn on every step").

Three claims validated (all sim/servo-free):
  H1  A GVF LEARNS to forecast its event: the 'my_side' GVF (cumulant = ball
      exited my wall) rises BEFORE the crossing and ~0 after -- learned by TD
      from the stream, not hardcoded.
  H2  The reward GVF reproduces the world model's W2 foresight (predicts a
      point's sign before it happens, >0.65) -- the Horde is a genuine
      generalization of the working W2 predictor, not a fake. The shared
      features carry enough signal for TD learning.
  H3  The Horde is a POPULATION: two GVFs with the same cumulant but different
      gammas (horizons) learn DISTINCT predictions -- the long-horizon GVF
      rises earlier (15-20 frames out) than the short-horizon one. Same
      features, different predictors -- not one shared scalar.

The cumulants are stated in DISCOVERED terms: 'the ball' = the fast free-
moving slot (learned by the world model's kind-speed slot assignment), 'my
side' = the ball exiting the left wall region (no hardcoded 'the ball is
slot 0' or 'my wall is x=0'). On-goal.

HONEST FINDING (the temporal-sparsity wall, again): a GVF whose cumulant
fires RARELY (e.g. 'opponent scores' -- only ~2-4 events in 4000 frames in
this scene) cannot learn from so few TD examples. The Horde inherits the
same rare-event limitation as the world model: frequent cumulants (my-side
crossings, reward) learn well; rare ones (opp points) don't get enough
signal. This is the same wall W3 hit -- rare events are hard for ANY online
learner. The fix is more frames (the rig streams forever) or a smarter TD
(Emphatic TD weights, the plan's Step 5) -- not architecture.

## What stands now

  - Perception: 9/9 (real-camera validated).
  - World model (DPC): W1 ✅, W2 ✅, W3 xfail (PAUSED -- perception occlusion-
    velocity blocker, documented).
  - Event-vs-motion discovery: the agent learns episode-boundary discontinuities
    from its own surprise statistics (no hand-flag). Validated.
  - Horde (Step 3): H1 ✅, H2 ✅, H3 ✅. A population of GVFs on shared
    features, learned online by TD.

Suite: 14 passed, 1 xfailed. No regressions, no dead code.

## Next idea: DYNAMIC rollout (stop when imagination becomes unrealistic)

INSIGHT: our W3 test demands a FIXED 25-step rollout, no matter what. Even
if the imagined ball goes off-rails at frame 5 (flies through the paddle,
zooms to x=3.0), the rollout KEEPS GOING to 25 -- piling nonsense on
nonsense -- and the test then fails on the compounded garbage. A human
imagining the future does NOT do this: "the ball flies straight... straight...
reaches the paddle... and now I'm not sure, so I STOP." Imagination should
be DYNAMIC -- the length is EARNED by realism, not fixed.

WHY THIS IS RIGHT (not just a test tweak):
  1. It matches how imagination is useful. A short, TRUSTWORTHY imagination
     ("the ball reaches my paddle in ~6 frames") is more useful than a long,
     garbage one. The Horde's forecasts care about the near, reliable future
     most. A dynamic rollout gives exactly the horizon we can trust.
  2. It sidesteps the W3 wall we kept hitting: every W3 failure was about
     COMPOUNDING ERROR over 25 steps. Stop early when imagination goes off-
     rails and we never reach the compounded-garbage regime -- we get a SHORT
     HONEST rollout instead of a LONG DISHONEST one. The bounce doesn't need
     to survive 25 steps; it needs to be real WHERE IT HAPPENS, then we stop.
  3. It's how the Alberta Plan thinks about Dyna (Step 7): imagine UNTIL the
     model's confidence runs out, prioritizing where imagination is reliable.
     "Stop when unrealistic" is the natural stopping rule, not a fixed N.

HOW 'UNREALISTIC' IS DETECTED (no Pong knowledge):
  The world model already computes a SURPRISE = how far the imagined state
  deviates from the physics default (the same signal used to detect real-
  stream teleports/events). In imagination: when the imagined surprise crosses
  a threshold (the imagined ball jumps, its velocity explodes, it goes off-
  screen), STOP. Same mechanism, applied to imagined frames. General: "stop
  imagining when your own prediction says 'I don't believe this anymore.'"
  No Pong geometry; the detector is the model's own surprise statistic.

WHAT CHANGES:
  - rollout returns a VARIABLE-LENGTH trajectory: 6 frames on one run, 12 on
    another, 25 only if the imagination stays realistic the whole way.
  - W3 is rewritten: instead of "a 25-step rollout bounces," it becomes "the
    rollout CONTINUES THROUGH a bounce (does not stop before it) AND the
    bounce is real AT the step it happens." A more honest test of
    'imagination works' than '25 steps and a flip.'

HONEST CAVEAT: the underlying problem (the ball's velocity is wrong at the
bounce, so the bounce signal is weak) does NOT go away -- the dynamic rollout
just stops asking the model to be reliable PAST its breaking point. So it is a
better TEST/USE of the rollout, and it may turn W3 green, but it is not a
deeper fix. Frame it as: W3 was OVER-STRICT; a dynamic rollout is the honest
version, and we should find out what it shows. The deeper fix (occlusion-
velocity quality in perception) is still the real unblocker for sharp bounces.

DO WE EVEN NEED ROLLOUT? For the core loop right now: NO. The Horde learns
its forecasts directly from the live stream by TD (one step at a time, no
imagination). W1/W2 are one-step. Rollout is only needed for the LATER
boost -- Dyna (Step 7) / prioritized sweeping (Step 9), where the agent runs
IMAGINED transitions to update the Horde FASTER than real time. That is a
speedup, not a requirement for the core loop or for the next step (control,
Step 4). So W3/dynamic-rollout is not blocking; it's a future boost we can do
honestly with the dynamic version when we get to Dyna.

