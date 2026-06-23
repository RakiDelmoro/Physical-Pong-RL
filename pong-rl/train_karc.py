"""
Real-time training loop for the KARC physical-Pong agent.

Pure online from frame 1. Same env I/O as the host bridge (camera + reward + servo
via host_bridge), but the brain is the KARC agent (no neural net, no GPU).

Loop every frame:
  env.perceive()                -> warped 128x128 obs + reward_delta
  state = agent.observe(obs)    -> extract 4-num state, train both KARC models
  action = agent.act(state)     -> phase policy + planner + safety filter
  env.step(action)              -> servo -> joystick -> paddle
  log reward/scores

Run (with host_bridge.py running on the Windows host):
    python train_karc.py            # auto: resume if checkpoint exists, else fresh
    python train_karc.py --fresh    # start over from scratch
    python train_karc.py --resume   # resume from checkpoint (fails to fresh if none)
    python train_karc.py --steps 50000 --fresh
"""

import argparse
import os
import time

import numpy as np

import env as env_mod
from karc_agent import KARCAgent, STAY


def smooth(values, window=50):
    if len(values) < window:
        return float(np.mean(values)) if values else 0.0
    return float(np.mean(values[-window:]))


def main():
    ap = argparse.ArgumentParser(
        description="KARC real-time Pong training. By default, resumes from "
        "the latest checkpoint if one exists; use --fresh to start over.")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--save-every", type=int, default=2000)
    # Resume vs fresh: mutually exclusive. If neither is passed, we AUTO-
    # resume when a checkpoint exists (so you don't accidentally throw away
    # hours of learning by forgetting a flag), and start fresh otherwise.
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--resume", action="store_true",
                     help="resume from the checkpoint in --run-dir (fails if none)")
    grp.add_argument("--fresh", action="store_true",
                     help="start training from scratch (ignores/overwrites checkpoint)")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_path = os.path.join(args.run_dir, "karc.pt")

    # decide whether to load the checkpoint
    load_ckpt = False
    if args.resume:
        load_ckpt = True          # caller explicitly wants to resume
    elif args.fresh:
        load_ckpt = False         # caller explicitly wants to start over
    else:
        # auto: resume if a checkpoint exists, else fresh
        import os as _os
        load_ckpt = _os.path.isfile(ckpt_path + ".npz")

    env = env_mod.RealEnv()
    agent = KARCAgent()
    if load_ckpt:
        agent.load(ckpt_path)
        if agent.steps == 0:
            # load() found no file (it printed 'starting fresh'). Only
            # noteworthy if --resume was explicitly requested.
            if args.resume:
                print("[train] --resume requested but no checkpoint found; "
                      "starting fresh anyway.")
        else:
            print(f"[train] resuming from step {agent.steps}")
    else:
        if args.fresh:
            print("[train] --fresh: starting from scratch (checkpoint will be "
                  "overwritten on next save).")
        else:
            print("[train] no checkpoint found; starting fresh.")

    rewards = []
    pts_for = 0
    pts_against = 0
    # per-action counters (how many times each was actually sent to servo)
    act_counts = [0, 0, 0]   # UP, DOWN, STAY
    # running tally of the planner's last decision (for the log)
    last_intercept = None
    t0 = time.time()
    try:
        print("=== KARC real-time training (pure online, frame 1) ===")
        print(f"phase1 WATCH  (frames 1..100):    servo PARKED, train ball model")
        print(f"phase2 PROBE  (100..300):         gentle cycling, train servo model")
        print(f"phase3 PLAN   (300+):             planner + safety filter, both models refine")
        print()

        s = 0
        new_frames = 0          # genuinely-new camera frames seen (real time)
        last_log_new = 0
        while s < args.steps:
            env.perceive()
            # THROTTLE: only advance on genuinely new camera frames. The
            # bridge serves its latest cached JPEG on every request, so a
            # fast loop would see the same frame 3-4x. We busy-wait briefly
            # for a new frame instead of training on repeats (which would
            # corrupt KARC's velocity/time estimates by the repeat factor).
            if not env.frame_is_new():
                time.sleep(0.002)
                continue
            new_frames += 1
            obs = env.get_observation()
            reward = env.get_reward()
            sl, sr = env.get_scores()
            conf = env.get_confidence()  # screen-detector confidence (skip low ones)

            state = agent.observe(obs, screen_conf=conf, reward=reward)
            action = agent.act(state, train=True)
            env.step(action)
            act_counts[int(action)] += 1

            if reward > 0:
                pts_for += 1
            elif reward < 0:
                pts_against += 1
            rewards.append(reward)
            s += 1

            if s % 100 == 0:
                dt = time.time() - t0
                real_fps = new_frames / dt   # new-frame rate (should be ~30)
                avg_r = smooth(rewards)
                phase = agent._phase()
                pname = {1: "WATCH", 2: "PROBE", 3: "PLAN"}[phase]
                bupd = agent.ball.rls.n_updates
                supd = agent.servo.rls.n_updates
                flag = "FROZEN" if agent._frozen else (
                       f"ball_lost_{agent._ball_loss_streak}"
                       if agent._ball_loss_streak > 0 else "ok")
                # action mix: how many UP/DOWN/STAY were sent this run.
                # If STAY dominates in PLAN phase the paddle is stuck and we
                # know the planner isn't committing. If UP/DOWN are firing the
                # servo IS moving and the problem is elsewhere (timing/aim).
                print(f"[KARC/{pname}] step {s} fps {real_fps:.1f} "
                      f"avg_r {avg_r:+.4f} score L{sl}-R{sr} "
                      f"pts_for {pts_for} pts_against {pts_against} "
                      f"ball_upd {bupd} servo_upd {supd} "
                      f"acts U{act_counts[0]}/D{act_counts[1]}/S{act_counts[2]} "
                      f"pad {agent._last_my_y:.2f} ball {agent._last_intercept_y:.2f} "
                      f"t_ball {agent._last_intercept_t} t_pad {agent._last_arrive_t} "
                      f"[{agent._last_decision}] "
                      f"scr_conf {conf:.2f} {flag}", flush=True)

            if s % args.save_every == 0:
                agent.save(ckpt_path)
                print(f"[checkpoint] saved {ckpt_path} at step {s}", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted, saving...")
    finally:
        agent.save(ckpt_path)
        print(f"saved {ckpt_path}")
        env.shutdown()


if __name__ == "__main__":
    main()
