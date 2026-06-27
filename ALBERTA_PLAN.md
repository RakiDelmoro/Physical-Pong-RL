# The Alberta Plan mapped onto Physical Pong — a research architecture

Reference: Sutton, Bowling & Pilarski, *The Alberta Plan for AI Research*
(arXiv:2208.11173, 2022).

This doc is the conceptual map for our agent: how the Alberta Plan's vision,
base agent, and 12-step roadmap connect to a real, continuing,
partially-observable, hardware-in-the-loop agent (a physical Pong rig), and
how the published pieces (GVFs/Horde, options, option keyboard, Dyna,
prioritized sweeping, average-reward, IDBD/Autostep, STOMP, Oak) compose into
*our own* agent. We are doing research: each piece is already published; our
contribution is connecting them on this rig.

## Status at a glance

- ✅ **Perception** (scaffold + learned model with discovered behavior classes)
  — built, tested in sim (5/5), validated on the real camera. See §4, §7.
- ⬜ **Transition model / Horde / planning / control** — the remaining work,
  front-loaded by the Alberta Plan roadmap. See §8.

---

## 0. The single most important recognition

**An Alberta-Plan base agent has four components** (the paper's Fig. 2), and we
will build all four deliberately rather than inherit a prior shape:

| Alberta Plan base-agent component | What it does |
|---|---|
| **Perception** | turns observation into a recurrently-updated *state* `s` |
| **Transition model** | `M(s,a) → s', r'`; used to imagine outcomes |
| **Value functions** | evaluate states/actions; multiple, one per policy |
| **Reactive policies** | `π(s) → a`; the primary policy, plus option-like sub-policies |

The paper gives the **vocabulary, the discipline, and the roadmap** to make a
design rigorous and to push it toward a complete continual model-based agent.

---

## 1. The four distinguishing features of the Alberta-Plan vision, applied here

### 1a. Grounded in ordinary experience only
> "Only experience is available to the agent... the environment is known only as a source and sink for these signals."

Every constant the agent "knows" must be learned from (observation, action,
reward) or be a permanent motor/perception prior (like the servo's existence).
Nothing internal to the environment crosses the experience boundary.

### 1b. Temporal uniformity
> "All times are the same... there are no special training periods... if the agent learns or plans, then it learns or plans on every time step."

No phase switches. A continual meta-process blends exploration and exploitation
on every step, with the blend driven by a model-uncertainty / prediction-error
signal. "Increase speed of learning when they start to change."

### 1c. Cognizance of computation (the bitter lesson)
> Prioritize methods that "scale extensively with computer power" — search and learning — over human insight.

Push further into *search* (more imagination steps, prioritized sweeping,
option-level search) rather than into hand-engineering planner heuristics. The
magic numbers a human tunes are exactly what should eventually be *learned*.

### 1d. Other agents / Intelligence Amplification
> Step 12: a "computational exo-cerebellum" built on prediction and continual feature construction; a "computational exo-cortex" using planning to enhance a partnered agent.

The servo-actuated joystick is literally an IA channel. The far goal: the
agent's *predictions* (GVFs) signal to a human partner ("the ball will beat
you on the left in 0.3 s"). Each GVF should be a *useful signal to someone*,
not just an internal variable.

---

## 1½. The system workflow — host holds the sensors, container holds the brain

The agent and the world are **physically separated**: sensors/actuators live on
the Windows host; the brain (perception + model + decision) lives in this
container. They talk over persistent sockets via `host.docker.internal`. This
is the **experience boundary** the Alberta Plan wants: the agent only sees
(observation, reward) in and emits `action` out; it never touches the
environment's internals.

```
┌──────────────────── WINDOWS HOST ────────────────────┐    ┌──────────── CONTAINER (the brain) ────────────┐
│                                                      │    │                                               │
│  host_bridge.py                                      │    │  perception  →  model  →  decision (TBD)     │
│  --camera 0 --arduino COM7 --arduino-baud 9600       │    │                                               │
│  --sensecap COM10 --show-preview                     │    │  in:  frame (JPEG) + reward (+ scores)        │
│                                                      │    │  out: action index                            │
│  webcam ─┐                                           │    │                                               │
│  sensecap┼──► :8000  [u32 len][JPEG][i32 r][i32 sL][i32 sR] ──► perceive      │
│  arduino ◄── :8001  "<angle>\n"  ◄──────── action─map ───────  decide        │
│          :8002 viewer push (optional)                │    │                                               │
│          :8088 viewer HTTP  (optional)               │    │  (the agent NEVER opens the camera / servo /  │
│                                                      │    │   sensecap directly — only via the bridge)    │
└──────────────────────────────────────────────────────┘    └───────────────────────────────────────────────┘
```

### Per-step loop (one tick of the agent)

1. **Container → Host `:8000`**: send 1 byte → host replies with the atomic
   packet `[u32 frame_len][JPEG][i32 reward_delta][i32 score_l][i32 score_r]`.
   `reward_delta` is the sum of all SenseCAP point events since the last
   request; the host then zeroes its accumulator. This gives **atomic
   obs+reward, no timestamp skew, no lost points** — exactly what RL needs.
2. **Container**: run **perception** on the warped game (find objects → learn
   dynamics → discover classes → identify the controlled object), then the
   **decision layer** turns that into an action index. (Decision layer not yet
   built — see §8.)
3. **Container → Host `:8001`**: send `"<angle>\n"` → host writes the integer
   to the Arduino → SG90 → joystick → paddle. One persistent connection for
   the whole session. The container must map `action_index → servo_angle`
   (a tiny, learned-or-calibrated table) before sending.

### Division of responsibility (locked)

| Layer | Where | Job |
|---|---|---|
| **Sensors / actuators** | Host | camera capture, SenseCAP reward parse, servo write |
| **I/O bridge** | Host (`host_bridge.py`) | relays frame+reward and servo commands over sockets |
| **The brain** | Container | perception → model → action. **Never touches hardware directly.** |

### How perception fits this workflow today

Perception already consumes step 1 (the warped grayscale frame from `:8000`)
and produces **tracks + a controlled-object id**. It does **not** yet emit an
action — there is no decision layer yet, and we are intentionally not building
it while we focus on perception. The action path (step 3) is stubbed: the live
viewer sends a constant STAY, and on the real rig `controlled=None` because no
servo means no action→motion relationship to learn. Both are expected.

---

## 2. The base agent, redrawn for this rig

(The container-side view. The host side is §1½.)

```
            ┌─────────────────── the agent (container) ───────────────────┐
            │                                                             │
 camera ──► │  PERCEPTION                                                 │
 (JPEG)     │   ScreenWarper ──► RGB ──► StateExtractor(learned)         │
            │      └─► normalized coords ──► delay-embedding ──► state s │
            │                          │                                  │
            │        ┌─────────────────┼──────────────────┐              │
            │        ▼                 ▼                  ▼              │
            │  TRANSITION MODEL   VALUE FUNCTIONS    REACTIVE POLICIES   │
            │   M(s,a)→s',r'       (a Horde of GVFs)   primary π +       │
            │   unified model       Q-model, reach,     options           │
            │   (ball+paddle+rwd)  arrive, point-prob   (move-to-y)       │
            │        │                 ▲                  │              │
            │        └──► PLANNING ◄───┘                  │              │
            │           (Dyna + prioritized sweeping,     │              │
            │            background, average-reward)      │              │
            │                                           action a         │
            │                              SafetyFilter (motor bound)   │
            └─────────────────────────────│─────────────────────────────┘
                                          ▼
                                   servo angle → Arduino → SG90 → joystick
                                          ▲
 reward (USB-CDC) ◄──────────────────────┘
```

Everything updates **every step in the foreground**; planning runs an
**asynchronous background** loop (the paper's Systems 1 / 2 split, after
Kahneman).

---

## 3. How the 12 roadmap steps map (and what to do at each)

The roadmap is "front-loaded": meet the hard issues in their simplest setting
first.

**Step 1 — Representation I: continual supervised learning with given features.**
Fixed Chebyshev/Fourier features + a linear readout, online, non-stationary.
The asks: (a) per-weight meta-learned step-sizes and (b) **online
normalization** (running µ_i, σ_i per feature).

**Step 2 — Representation II: supervised feature finding.**
A budgeted loop: *generate* candidate features (interactions), *test* them by
utility, *replace* low-utility ones. The seed of STOMP (Step 10): high-ranked
features become subtasks.

**Step 3 — Prediction I: continual GVF prediction learning.**
"Explicitly addresses the question of constructing state, the perception part."
Reframe every predictor as a **GVF**: a (cumulant, policy, termination) triple
and a γ. Many value functions sharing one feature representation — a **Horde**.
This is where learned perception enters (see §4).

**Step 4 — Control I: continual actor-critic control.**
Actor + critic, continual, robust, off-policy. Keep a safe prior (physics) and
let a learned actor be shaped residually by the critic, with a trust-gate as
the continual-learning stability mechanism.

**Step 5 — Prediction II: average-reward GVF learning.**
Continuing environment → average-reward: predict the *differential* value and
learn the average reward rate `r̄`. Maintain `r̄`, predict `G_t − r̄`, optimize
the differential return. The biggest conceptual upgrade for a continuing game.

**Step 6 — Control II: continuing control problems.**
Physical Pong *is* the continuing testbed (no reset, self-resetting rallies).
Treat it as our genuine continuing domain.

**Step 7 — Planning I: planning with average reward.**
Asynchronous dynamic programming on the model for the differential value, in
the background. Dyna for the average-reward case.

**Step 8 — Prototype-AI I: one-step model-based RL with continual function approximation.**
The first *complete* prototype: perception + one-step model + feature finding +
planning + search control, *without* temporal abstraction. The keystone:
unify ball/paddle into one **control-affine model `M(s,a) → (s', r')` that
also predicts reward**, so planning can *imagine reward* and produce real value
backups.

**Step 9 — Planning II: search control and exploration.**
"Viewed most generally, search control enables planning to radically change —
from MCTS to classical heuristic search." Add **prioritized sweeping**: rank
imagined backups by prediction-error / value-change magnitude. Replace
uniform-horizon rollout with error-prioritized imagination.

**Step 10 — Prototype-AI II: the STOMP progression (SubTask → Option → Model → Planning).**
Take high-ranked features → make each a *reward-respecting subtask* with a
terminal value → solve to an *option* (policy + termination) → learn the
option's *model* → add to planning. Make options first-class.

**Step 11 — Prototype-AI III: Oak + the option keyboard.**
Reference options by a real-valued vector `w` (one component per
subtask/height). A "chord" (multiple nonzero components) blends options. The
transition model learns about *whatever chord is played*, treating `w` as a
non-interpreted descriptor. Continuous, composable skill space.

**Step 12 — Prototype-IA: intelligence amplification.**
Emit the Horde's GVF predictions as signals to a human partner
(exo-cerebellum); eventually let a planning agent *multiply* a partner's
decisions (exo-cortex).

---

## 4. Perception — designed scaffold, learned surface

Perception is the hardest open problem in the Alberta Plan. The paper itself
says so:

> "The perception component is perhaps the least well understood... how
> perception should be learned, or meta-learned, to maximally support the other
> components remains an open research question."

### 4a. Is human perception designed or learned? Both — at different layers.

Human perception is **evolved scaffolds tuned by lifelong learning**:

- Babies are born with **scaffolding**: the brain already expects faces, edges,
  motion, objects (Spelke's core knowledge: cohesion, continuity, contact).
  Evolution built that over geological time.
- A lifetime then **tunes** that scaffolding: your exact face recognition, your
  exact lighting, your exact world.

Nature did the *design* (slow, over millions of years); nurture does the
*tuning* (fast, during one life). The false dichotomy is "designed or learned."
The real question is always: **which layer is designed, and which layer is
tuned by learning?**

And note: the thing that *designed* human perception — natural selection — is
itself a **search process at vast scale**, not a human writing rules. That is
exactly what Sutton's bitter lesson says wins. So "designed by evolution"
doesn't refute the bitter lesson; it's the bitter lesson playing out over
geological time.

### 4b. The split, in one picture

```
┌────────────────────────────────────────────────────────────┐
│  THE SCAFFOLD (design it — like evolution did)             │
│                                                            │
│  Objects exist; they persist; they move as wholes          │
│  (cohesion); they travel on smooth paths (continuity).    │
│  Dimensionality is a scaffold cue: 0D-ish blob (ball),    │
│  1D-ish tall-thin (paddle), 2D plane (screen).            │
│                                                            │
│  Output: a set of tracked objects, each with a stable ID.  │
│  NO appearance rule says which is the ball. Lighting        │
│  doesn't enter identity.                                   │
├────────────────────────────────────────────────────────────┤
│  THE SURFACE (tune it — like a lifetime)                   │
│                                                            │
│  For each tracked object, learn online:                    │
│    - "does it respond to MY action?"  → it's my paddle     │
│    - "does it bounce elastically?"     → it's the ball     │
│    - "does it chase the ball?"         → it's the opponent │
│    - "where does it go next?"          → the motion model   │
│                                                            │
│  Identity-by-behavior, not identity-by-pixels.             │
└────────────────────────────────────────────────────────────┘
```

- **Scaffold** = the big truths that don't change. Design it. It's fine.
- **Surface** = the details that drift with lighting, wear, camera angle.
  Don't hardcode these. Let them be tuned every frame.

### 4c. Why this is lighting-invariant

Appearance is *demoted from identity to seed*. Appearance gives loose,
low-bar proposals ("here are some candidate moving regions"). Then
**persistence + cohesion + continuity** (the scaffold) filter and stabilize the
proposals. Then **behavior** (the surface) gives final identity and the motion
model.

Identity ends up defined by **behavior**, which is lighting-invariant by
definition: "the object that responds to my servo = my paddle." Lighting could
be terrible and the paddle would still be identified, *because it still responds
to my action*. The agent's own action becomes the probe that labels objects —
deeply Alberta-Plan: only the agent's experience (action → observation) is used.

### 4d. Biology supports the dimensionality cue

The visual system evolved in a 3D world, so it has strong 3D priors (depth,
occlusion, persistence behind things). When it meets a 2D thing (a screen, a
paddle on a screen) it still treats it as an *object* — because "object" is the
scaffold, and screen content rides on top of that machinery. Edges / lines
(1D-ish) are detected so early and so robustly (V1 orientation columns are
largely innate) that they're the textbook example of "designed, not learned."
Dimensionality is a legitimate, biology-backed scaffold cue.

### 4e. The one-liner

The scaffold is "there are objects of certain dimensions that persist and
move"; the surface is "learn how each one behaves." Track objects by
persistence+cohesion+dimensionality (lighting-invariant), then label them by
behavior — "the one that obeys my servo is my paddle, the one that bounces is
the ball." Identity by behavior, not by pixels.

---

## 5. Build order (research-ordered, front-loaded) — with status

Front-load = meet the hard issue in its simplest form first, per the paper's
strategy. ✅ = done, ⬜ = remaining.

1. ✅ **Perception — scaffold + learned model + discovered behavior classes
   (incl. passive sub-classes)** (§4, §7). Object tracker
   (persistence/cohesion/continuity/dimensionality) + per-object
   reservoir+RLS model that *discovers* behavior classes via a two-part
   signature: action-contrast (controlled vs passive) + a bracket-aware
   autonomous-dynamics (free-motion) term that splits the passive bucket
   into ball / opponent / decoy. Lighting-invariant by construction. Tested
   in sim (7/7) and validated on the real camera.
2. ⬜ **Unify the transition model** `M(s,a)→(s',r')` with reward predicted.
   (Alberta Plan Step 8b.) The keystone that unblocks real Dyna.
3. ⬜ **Reframe predictors as a Horde of GVFs** on shared features.
   (Step 3.)
4. ⬜ **Background Dyna loop + prioritized sweeping** using the unified model;
   imagination ranked by prediction error. (Steps 7, 9.)
5. ⬜ **Average-reward / differential value** for the continuing game; maintain
   `r̄`. (Step 5.)
6. ⬜ **Options first-class** (move-to-y with termination + option-models) and
   the **option keyboard** for continuous skill composition. (Steps 10, 11.)
7. ⬜ **Online normalization + Autostep step-sizes** in the feature pipeline.
   (Step 1.)
8. ⬜ **Feature-finding loop** with utility ranking and a budget. (Step 2.)
9. ⬜ **Make temporal uniformity real**: replace phase switches with a continual
   uncertainty-driven blend. (vision §1b.)
10. ⬜ *(far)* **Learned perception**: replace the designed object tracker with a
    learned recurrent state-updater. (Step 3 open problem.)
11. ⬜ *(far)* **IA demo**: emit GVF predictions as human-facing signals.
    (Step 12.)

---

## 6. Conscious departures to track

The paper says departures are sometimes fine but must be conscious. Track them:

- Hand-coded appearance rules at the identity layer — the thing we explicitly
  *did not* do this time (appearance is a seed only; identity is by behavior).
- The reservoir features (Fourier/Chebyshev) are **designed** — a human bias,
  acceptable for Step 1; becomes learned in Step 2 (feature-finding).
- The scaffold's object rules (persistence/cohesion/continuity) are **designed**
  — that is the point (the evolved-scaffold analogue).
- Perception → model is **one-directional** so far. The feedback loop (model
  predictions helping the tracker when detection drops) is a later step.
- Hard phase switches (WATCH/PROBE/PLAN) — should become continual blends.
- Foreground (synchronous) planning only — should become background Dyna.
- ±1 episodic credit assignment — should become average-reward differential.
- Hand-tuned planner thresholds — should become learned/meta-tuned.

None of these are bugs; they are the frontier the Alberta Plan points at.

---

## 7. What "perception" is concretely (built)

One package, `agent/perception/`, two halves:

- **Scaffold (designed, general, never trains):**
  - `proposal.py` — the seed. Multi-scale local-contrast z-score
    (`|g − μ_local|/σ_local`): polarity-agnostic, scale-relative, and
    **brightness-invariant by construction** (a global `g→k·g` leaves the
    z-score unchanged, so a lighting shift that breaks any fixed gray
    threshold does not break this). Appearance is a *seed*, never identity.
  - `tracker.py` — the object tracker. **Persistence** (MIN_AGE to confirm,
    MAX_MISS to retire), **cohesion** (size stays stable, gated by log-area),
    **continuity** (constant-velocity prediction; a too-far candidate starts a
    new track instead of hijacking). Each track gets a **dimensionality cue**
    (0d compact / 1d elongated / 2d planar) — a cue, not an identity.
- **Model (learned, per object, online):** `model.py`
  - Per object: a fixed **Kolmogorov-flavored reservoir** (Fourier + Chebyshev
    bases over a delay embedding of recent positions) + a **learned linear
    readout** updated every frame by **Recursive Least Squares** (closed-form,
    exact, no GD/replay/learning-rate). Reservoir is designed; only the readout
    trains (Alberta Plan Step 1).
  - Predicts **displacement** Δ = next − current (position-invariant dynamics:
    two same-behavior objects at different locations predict the same Δ).
  - **Discovered behavior classes** (the research move): each new object trains
    a tentative readout; once confident, its **action-contrast signature**
    (`predict(h,a) − predict(h,neutral)`, position-invariant) is compared to
    existing classes. Same bracket (controlled vs passive) + close → **bind**
    (share the class readout, reuse knowledge). Far → **promote** a new class.
    Classes that grow close **merge**. **No human names the classes.**
  - **Identity by behavior:** the class whose predicted Δcy varies most with the
    action is "controlled"; any object bound to it is a controlled object.
    Lighting-invariant by construction (behavior, not pixels).
- **Top level:** `perception.py` — `Perception.step(gray, action) →
  (tracks, controlled_id)`. Scaffold feeds model; `z_thresh` is tunable for the
  real rig (higher than the synthetic default, because real video is
  lower-contrast/noisier).

### Proven in sim (`agent/tests/test_perception.py`, 5/5)

1. Controlled paddle keeps a **stable track ID** across a 0.4× brightness shift.
2. A naive fixed-threshold detector **loses** the paddle on the same shift
   (contrast proof — the win is real).
3. The model labels the controlled paddle (and **only** it — not ball/opponent/
   decoy).
4. Identity-by-behavior **holds after** the lighting shift.
5. **Reuse:** when the controlled object is swapped for a new one (different
   position/size, same dynamics), it **binds to the existing controlled class**
   — knowledge transfers, not relearned.

### Validated on the real camera

Connected to the live bridge; screen detected every frame; the left paddle is
stably tracked with a persistent ID; track count cleaned up (18 → 6) by tuning
`z_thresh` for real video. `controlled=None` on the rig is **expected** — there
is no servo in the loop yet, so no action→motion relationship to learn.

### Known limitations of the current perception (honest)

- **Synthetic scene is clean; the real rig is messier.** Validated, not
  bulletproof: partial glare (not a global brightness multiply) is untested.
- **Finer passive classes (done; tested in sim).** It now separates the
  passive bucket into distinct classes for ball / opponent / decoy, not just
  *controlled vs passive*. The behavior signature gained a second,
  unit-normalized **autonomous-dynamics (free-motion) half** — how an object
  moves under the do-nothing action — which is the only signal that
  discriminates passive sub-classes (action-contrast is ~0 for all of them).
  The distance is **bracket-aware**: free motion participates only in
  passive-vs-passive comparisons, never controlled-vs-controlled (a
  controlled paddle's spurious free-motion is noise comparable to real
  passive motion and would break controlled-class reuse). Proven in sim
  (CLAIM 4: 3 distinct passive classes, each role in its own class, holds
  across the 0.4× brightness shift). Still synthetic-only; real-rig tuning
  (esp. faint ball / right paddle) remains open.
- **Right paddle / ball detection on the real rig is weaker** than the left
  paddle (lower contrast on that side / small faint ball). A real-camera
  tuning issue, not a tracker bug.
- **No test through the real actuation chain** (servo→joystick→firmware lag).
  The delay embedding *should* absorb it, but it is untested on hardware.

---

## 8. The remaining work — what to build next, and why

Perception is the foundation (the paper's hardest, least-understood piece).
Everything below builds **on top of the objects + classes perception produces**.
Each item is a published idea we connect to our rig; the ordering follows the
Alberta Plan's front-loading (simplest setting for each hard issue first).

### 8a. The unified transition model  `M(s, a) → (s', r')`  (Step 8 — keystone)

**What:** one model that, given the current object-states `s` and an action
`a`, predicts the **next** object-states `s'` **and the next reward `r'`**.

**Why it's the keystone:** today perception's per-object models predict
*displacement* only (no reward). Planning (Dyna) needs to **imagine reward** to
produce real value backups — without a reward-predicting model, imagination
can't update the value function. This is the single change that unblocks
everything downstream (Horde, Dyna, control).

**How it connects to what we have:** perception already learns each object's
dynamics (the reservoir+RLS readouts). The unified model *adds a reward channel*
to that same structure and *couples* the objects (the ball's next state depends
on the paddles, the paddle's next state depends on the action). It is a
control-affine, online, RLS-trained model — the natural generalization of the
per-object reservoir to a joint world model.

**Research claim to test:** can a single online RLS model, trained only from
experience, predict the next reward well enough that imagined rollouts improve
the policy? (This is the Alberta Plan's Prototype-AI I, minus temporal
abstraction.)

### 8b. A Horde of General Value Functions  (Step 3)

**What:** instead of one "value function", keep **many** predictors, each a GVF
`= (cumulant, policy, termination, γ)`, all sharing the same feature
representation. Examples on this rig:
- *point-probability* (the real value function; cumulant = reward);
- *ball-reaches-my-side-within-k* (cumulant = 1 if ball crosses my x-line);
- *opponent-reachability* (cumulant = a reach-margin; the old SafeGuess as a
  learned GVF, not hardcoded physics);
- *paddle-arrives-at-target-in-time* (cumulant = 1 if |my_y − target| < tol).

**Why:** the Horde is how the Alberta Plan does prediction at scale — many
value functions, one representation, all online. It also makes each prediction
a *useful signal to someone* (the IA orientation, §1d). And it dissolves the
perception/value chicken-and-egg: the **GVF vector itself can be the state**
(proto-value-functions / successor features).

**How it connects:** each GVF is just another RLS readout on the same reservoir
features the perception model already computes. Code-structure win, not new
math — but conceptually it turns "one model" into "a population of predictors",
which is the paper's actual stance.

### 8c. Background Dyna + prioritized sweeping  (Steps 7, 9)

**What:** an **asynchronous background loop** that, every step, runs `k`
*imagined* transitions through `M` and updates the Horde. **Prioritized
sweeping** ranks those imagined updates by prediction-error / value-change
magnitude (we get prediction error for free from RLS), so planning focuses on
where it matters instead of uniform rollout.

**Why:** this is the Systems-1/System-2 split (Kahneman, via the paper):
perception + the reactive policy run in the **foreground** every step; planning
runs in the **background** whenever it can. It is the paper's definition of
planning: "planning would typically not be complete in a single time step … an
ongoing process that operates asynchronously."

**How it connects:** perception gives objects; 8a gives `M`; 8b gives the Horde;
this is the loop that uses both to improve the policy without freezing the
agent. The 33 ms loop budget is the constraint (foreground must stay fast;
background uses spare cycles).

### 8d. Average-reward / differential value  (Step 5)

**What:** our environment is **continuing** (`terminated` always False — the
physical game can't be force-reset). The Alberta Plan insists on the
**average-reward** objective for continuing problems: maintain the long-run
reward rate `r̄`, predict the **differential** value `G_t − r̄`, and optimize
the differential return (R-learning), not ±1 episodic credit assignment.

**Why:** ±1-per-point is episodic thinking; Pong never ends. Average-reward is
the correct framing for a never-resetting game and is the single biggest
conceptual upgrade for the control layer.

**How it connects:** it changes the *target* of the value GVFs and the
*objective* of the policy — a modification of 8b/8c, not a new module.

### 8e. Control: actor-critic over a safe prior  (Step 4)

**What:** the **decision layer** that turns perception + predictions into an
action. Keep a **safe prior** (a closed-form physics guess: "move toward where
the ball will be") and let a learned **actor** be shaped **residually** by a
**critic** (one of the Horde GVFs), with a **trust-gate**: the learned actor is
not *acted on* until it has empirically beaten the bare safe prior over a
window of points.

**Why:** the trust-gate is our continual-learning stability mechanism (the old
bias-trap problem: a self-induced degenerate data slice corrupts the learner).
The safe prior means worst-case = the physics guess. This is continual,
off-policy, robust — exactly Step 4.

**How it connects:** this is the missing **decision** box in the §1½ workflow.
It emits the `action` that step 3 sends to the servo.

### 8f. Options + the option keyboard  (Steps 10, 11 — STOMP / Oak)

**What:** make **temporal abstraction** first-class. Take high-ranked features →
make each a *reward-respecting subtask* with a terminal value → solve to an
**option** (policy + termination) → learn the option's **model** → add to
planning (the **STOMP** progression). Then the **option keyboard**: reference
options by a real-valued vector `w`; a "chord" (multiple nonzero components)
blends options; the model learns about *whatever chord is played*, treating `w`
as a non-interpreted descriptor.

**Why:** this turns our discrete 3-action space into a **continuous, composable
skill space** — a genuine research contribution on this rig, and the Alberta
Plan's route to temporally-abstract cognitive structure.

**How it connects:** perception's "move-to-y" is a proto-option; the candidate
heights are proto-subtasks. Options become first-class objects the planner
(8c) searches over.

### 8g. Representation upgrades  (Steps 1, 2)

**What:** (1) **online normalization** (running µ_i, σ_i per feature) +
**Autostep/IDBD per-weight step-sizes** in the feature pipeline — the paper's
named anti-drift mechanism, and a direct aid to lighting/contrast robustness.
(2) **Feature-finding**: a budgeted loop that *generates* candidate features
(interaction terms), *tests* them by utility, *replaces* low-utility ones —
the seed of STOMP (high-ranked features become subtasks).

**Why:** the reservoir features are *designed* today (our chosen bases). The
Alberta Plan says eventually the agent should *find* its own features. This is
the move from "designed representation" (Step 1) to "learned representation"
(Step 2).

**How it connects:** it upgrades the reservoir inside `model.py` from fixed to
self-improving, without changing the readout/RLS machinery above it.

### 8h. (far) Learned perception & Intelligence Amplification  (Steps 3, 12)

- **Learned perception:** replace the designed object tracker with a **learned
  recurrent state-updater** `s_t = f_θ(s_{t-1}, obs_t)` trained to predict
  future obs-features + reward. This is the paper's explicitly *least
  understood* open problem ("how perception should be learned … remains an open
  research question"). Our designed scaffold is the deliberate precursor.
- **IA demo:** emit the Horde's GVF predictions as **signals to a human partner**
  ("the ball will beat you on the left in 0.3 s") — the exo-cerebellum. The
  servo-actuated joystick is the physical IA channel. This is the long north
  star that shapes how we structure predictions (each GVF useful to someone).

### Suggested next move

**8a — the unified transition model with reward.** It is the keystone (unblocks
Dyna, the Horde's value GVFs, and control), it builds directly on the
per-object reservoirs we already have, and it is the Alberta Plan's own
Prototype-AI I. Everything in §8b–8f either uses it or is a modification of it.
We start there when we leave perception.
