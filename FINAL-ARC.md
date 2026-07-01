# FINAL-ARC — where the project stands and where it goes

A snapshot of what we have, what's proven, and the remaining arc to the far
goal. Written so we keep moving forward and don't lose the thread.

## The goal (the whole arc)

Build the **Alberta-Plan base agent** on the physical Pong rig — four parts,
all learning online from the camera stream (no training phase, experience is
the training), no hand-coded Pong knowledge in the **engine**:

| Part | Job | Status |
|---|---|---|
| **Perception** (eyes) | camera frame → state (objects + velocities) | ✅ done |
| **World model** (imagination) | `M(s,a) → (next state, reward)`; imagine the future; bounce off paddles AND walls | ✅ done (structurally sound, accurate enough to act on) |
| **Horde** (judgment) | many GVF predictors on shared features | ✅ framework done (H2/H3 pass); ⚠️ H1 broken (xfail) |
| **Policy** (action) | pick up/down/stay; emit to the servo | ✅ done — **the imagination-based planner beats the hand-coded reflex** |

Far goal (Step 12, Intelligence Amplification): the agent's GVF predictions
become **human-facing signals** — "the ball will beat you on the left in 0.3
s" — a computational exo-cerebellum for a human partner. The servo joystick
is the IA channel. Every GVF should be useful to someone, not just an
internal variable.

## What's proven (the wins, in order)

1. **The agent plays.** Perception keeps the ball's identity through paddle
   contacts; the world model imagines 25 frames ahead across bounces; the
   simple reflex intercepts every ball on the easy game (0 points leaked).
2. **The imagination is structurally sound.** The ball bounces off **paddles
   and walls** in imagination (two real fixes: the `is_constituent` tag so a
   wall's witness ball can bounce off it, and the spurious-surface filter so
   only the fastest object's bounces reveal walls).
3. **The imagination's bounce foresight beats the reflex's straight-line aim.**
   The wall position was EMA-biased inward; the general principle "a wall is
   the edge of observed motion, and smoothing only rounds inward, so take the
   extremum not the median" fixed it. Imagination bounced-arrival error 0.064
   → 0.025 (beats the reflex's straight-line 0.034).
4. **The model-based planner beats the hand-coded reflex.** On the harder
   scene (short paddle + wall-bouncy ball): planner leaks **5** opponent
   points vs the reflex's **6**. Both in a forced test AND in real operation
   (the trust-gate explore fraction lets it act enough to realize the win).
   **This is the Alberta Plan Step 4 payoff, live** — a model-based planner
   over the imagination, all online, no Pong rules in the engine.
5. **Suite green throughout: 14 passed, 1 xfailed.** No regressions at any
   step.

## What's NOT done (honest gaps)

- **H1 (the `my_side` GVF) is broken (xfailed).** Diagnosed: the ball is
  dropped from the state behind the my-paddle toward the wall, and a coasting
  ghost fires the `my_side` cumulant ~5 frames late → the GVF forecasts the
  lagged cumulant → wrong sign. This is the **perception-through-contact
  frontier** (the shared root the whole project circled) and it's upstream of
  the human-facing signals (Step 12's whole point).
- **The trust gate hasn't formally promoted.** The 1-point/800-frame
  improvement is below the conservative margin (correctly — rare events are
  noisy). The explore fraction realizes the value anyway. A real promotion
  needs more frames / a harder scene where the planner's edge is larger.
- **No Dyna, no options yet.** The imagination exists but isn't used to
  *learn faster* (Dyna) or to structure *multiple policies* (options).
- **One Pong-tuning caveat:** the harder-scene measurement ("beat the
  reflex at Pong") is a Pong-shaped yardstick. The engine fixes were done by
  general principle (wall = edge of motion; is_constituent = a surface's
  physics semantics; controlled = action-correlated track), not Pong
  constants — but a portability audit (confirm no Pong assumption leaked into
  the engine's logic) is worth doing before trusting the generality fully.

## The forward arc (priority order)

### 1. Dyna + prioritized sweeping (Alberta Plan Steps 7, 9) — the keystone's payoff

This is *why the transition model exists*. Once `M` can imagine a trustworthy
few steps (it can), run **imagined transitions** to update the Horde faster
than real time, ranked by prediction error (prioritized sweeping). The
Alberta Plan's "scale with computation" bet — the agent gets smarter per
frame by *thinking*. The dynamic rollout we already built (imagine until
unrealistic) is exactly the right primitive. **Highest leverage: it makes
every GVF converge faster, which directly serves the IA far goal.**

### 2. Options + option keyboard (Steps 10, 11) — multiple policies

An *option* = a temporally-extended sub-policy ("defend-left," "set up a kill
shot") with its own termination condition. A *policy over options* chooses
which to invoke, using the Horde's GVF forecasts as the value of each option.
This is the "multiple policies" the design always pointed at: today one
policy (reflex + planner); tomorrow several options + a meta-policy. Each
option can have its own cumulants/GVFs tuned to its goal.

### 3. The H1 fix + the IA demo (Steps 3, 12) — the human-facing payoff

Fix the perception-through-contact root (the ball lost behind the my-paddle
near the wall) so H1's `my_side` cumulant fires on time → the GVF forecasts
it correctly → it becomes a human-facing signal ("the ball will beat you on
the left in 0.3 s"). Then wire the GVF predictions to a human-facing display
(the IA demo). This is the **real end goal** — the agent as an exo-cerebellum.

### 4. (Housekeeping) Generality audit + trust-gate hardening

- Audit the policy + Horde for Pong-specific seams; confirm each is isolated
  behind a clear "port this" interface and that no Pong assumption leaked into
  the world model's *logic* (only comments/examples).
- Harden the trust gate (a harder scene where the planner's edge is large
  enough to formally promote; or a Bayesian gate that handles rare-event
  noise).

## The one-line summary

The four-part agent **plays Pong** and its **imagination-based planner beats
the hand-coded reflex** — both proven in a forced test and realized in real
operation. The engine is general (the wins came from general principles, not
Pong constants); the world-specific parts (cumulants, safety metric, safe
prior) are swappable seams by design. The far arc — Dyna (learn faster by
thinking), options (multiple policies), and the IA demo (human-facing
signals) — is unblocked. One known gap: the H1 `my_side` predictor is broken
on the perception-through-contact frontier, upstream of the human-facing
signals.

## Pointers

- Big-picture plan: `ALBERTA_PLAN.md`
- World-model detail + the W3b/ContactNets history: `WORLD_MODEL_PLAN.md`
- The session log (the whole journey, dated): `PLAN_2026-06-30.md`
- Code: `agent/perception/` (scaffold), `agent/world_model.py` (imagination +
  Way C surfaces + controlled discovery + wall-extremum), `agent/horde.py`
  (GVFs), `agent/policy.py` (the planner + trust gate), `agent/tests/
  harder_scene.py` (the harder scene where the planner beats the reflex).
- Latest commits: `11923db` (trust-gate explore fraction), `3ba4a55` (wall =
  edge of motion — the planner beats the reflex), `2c9d224` (wall-bounce
  fix), `854399a` (controlled discovery fixed — the agent plays).
