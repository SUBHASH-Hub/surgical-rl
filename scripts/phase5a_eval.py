#!/usr/bin/env python3
"""
Phase 5A — Systematic PPO Agent Evaluation (30 Episodes)
=========================================================
Project : Autonomous Tissue Retraction via Safe RL
Author  : Subhash Arockiadoss
Date    : June 2026

Follows exact same patterns as scripts/eval_agent.py (Phase 2).
Environment wrapped in SafeRewardWrapper — same as training.
Model loaded via DummyVecEnv — same as eval_agent.py.

Usage:
  source ~/surgical_robot_lapgym_ws/activate.sh
  cd ~/surgical_robot_lapgym_ws/surgical-rl
  python scripts/phase5a_eval.py

Output:
  results/phase5a_ppo_eval.csv          per-episode metrics
  results/phase5a_summary.txt           summary table for paper
  results/phase5a_comparison.csv        PPO vs baseline side-by-side

Runtime estimate on GTX 1650:
  ~142 steps x ~65ms/step = ~9s/episode
  30 episodes = ~5-6 minutes total
"""

import os
import sys
import csv
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Compatibility shim — same as eval_agent.py ────────────────────────────────
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
sys.path.insert(0, '.')

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT = (
    "logs/checkpoints/"
    "phase2_ppo_tissue_retraction_20260409_211946/"
    "ppo_tissue_final"
)

N_EPISODES = 30
SEED       = 42
MAX_STEPS  = 300   # same as training

# SafeRewardWrapper params — same as eval_agent.py
LAMBDA_FORCE     = 0.5
LAMBDA_COLLISION = 0.5
FORCE_THRESHOLD  = 0.5
STEP_PENALTY     = 0.01

OUTPUT_DIR   = "results"
CSV_PATH     = os.path.join(OUTPUT_DIR, "phase5a_ppo_eval.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "phase5a_summary.txt")
COMPARE_PATH = os.path.join(OUTPUT_DIR, "phase5a_comparison.csv")

# ── Phase 1 scripted baseline (docs/baseline_metrics.md) ─────────────────────
BASELINE = {
    'steps_mean'     : 247.0,
    'steps_std'      : 8.2,
    'reward_mean'    : -165.54,
    'reward_std'     : 12.3,
    'goal_rate'      : 1.00,
    'collision_mean' : 49.0,
    'collision_std'  : 4.1,
}

# ── Imports — same order as eval_agent.py ────────────────────────────────────
print("\nLoading environment and model...")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from envs import TissueRetractionV2
    from envs.safe_reward import SafeRewardWrapper
except ImportError as e:
    print(f"ERROR: Cannot import environment: {e}")
    sys.exit(1)

# ── Validate checkpoint ───────────────────────────────────────────────────────
if not Path(str(CHECKPOINT) + ".zip").exists():
    print(f"ERROR: Checkpoint not found: {CHECKPOINT}.zip")
    sys.exit(1)

# ── Single episode runner — same logic as eval_agent.py ──────────────────────
def run_episode(env, model, episode_id):
    """Run one deterministic episode. Returns dict of metrics."""
    obs, info = env.reset()

    steps           = 0
    total_reward    = 0.0
    r_task_total    = 0.0
    r_coll_total    = 0.0
    collision_steps = 0
    force_viol      = 0
    goal_reached    = False
    inference_times = []

    ep_start = time.perf_counter()

    while True:
        # ── Policy inference — timed ──────────────────────────────────
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        inference_times.append((time.perf_counter() - t0) * 1000.0)

        if action.ndim == 2:
            action = action[0]
        action = action.astype(np.float32)

        # ── Environment step ──────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)
        steps        += 1
        total_reward += reward
        r_task_total += info.get('r_task', reward)
        r_coll_total += info.get('r_coll', 0.0)

        # ── Collision tracking — same as eval_agent.py ────────────────
        if info.get('in_collision', False) or info.get('r_coll', 0.0) < 0:
            collision_steps += 1

        # ── Force violation tracking ──────────────────────────────────
        if info.get('force_viol', 0.0) > 0:
            force_viol += 1

        # ── Goal tracking ─────────────────────────────────────────────
        if info.get('goal_reached', False) or info.get('is_success', False):
            goal_reached = True

        # ── Termination ───────────────────────────────────────────────
        if terminated or truncated or steps >= MAX_STEPS:
            if info.get('goal_reached', False):
                goal_reached = True
            break

    wall_time    = time.perf_counter() - ep_start
    mean_inf_ms  = float(np.mean(inference_times)) if inference_times else 0.0

    return {
        'episode_id'     : episode_id,
        'steps'          : steps,
        'total_reward'   : round(float(total_reward), 4),
        'r_task'         : round(float(r_task_total), 4),
        'r_coll'         : round(float(r_coll_total), 4),
        'goal_reached'   : bool(goal_reached),
        'collision_steps': collision_steps,
        'force_viol'     : force_viol,
        'inference_ms'   : round(mean_inf_ms, 3),
        'wall_time_s'    : round(wall_time, 2),
    }

# ── Summary statistics ────────────────────────────────────────────────────────
def stats(results, key):
    v = [r[key] for r in results]
    return {
        'mean': round(float(np.mean(v)),  3),
        'std' : round(float(np.std(v)),   3),
        'min' : round(float(np.min(v)),   3),
        'max' : round(float(np.max(v)),   3),
    }

# ── Save per-episode CSV ──────────────────────────────────────────────────────
def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(results[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  [saved] {path}")

# ── Save comparison CSV (Table 1 for paper) ───────────────────────────────────
def save_comparison_csv(summary, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            'Method'         : 'Scripted Baseline (Phase 1)',
            'N_episodes'     : 3,
            'Steps_mean'     : BASELINE['steps_mean'],
            'Steps_std'      : BASELINE['steps_std'],
            'Reward_mean'    : BASELINE['reward_mean'],
            'Reward_std'     : BASELINE['reward_std'],
            'Goal_rate_pct'  : BASELINE['goal_rate'] * 100,
            'Collision_mean' : BASELINE['collision_mean'],
            'Collision_std'  : BASELINE['collision_std'],
            'Inference_ms'   : 0.0,
        },
        {
            'Method'         : 'PPO Phase 2D (ours)',
            'N_episodes'     : N_EPISODES,
            'Steps_mean'     : summary['steps']['mean'],
            'Steps_std'      : summary['steps']['std'],
            'Reward_mean'    : summary['reward']['mean'],
            'Reward_std'     : summary['reward']['std'],
            'Goal_rate_pct'  : summary['goal_rate'] * 100,
            'Collision_mean' : summary['collision']['mean'],
            'Collision_std'  : summary['collision']['std'],
            'Inference_ms'   : summary['inference_ms']['mean'],
        },
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {path}")

# ── Format summary text ───────────────────────────────────────────────────────
def format_summary(results, summary):
    s   = summary
    sep = "=" * 65
    n_goal = sum(1 for r in results if r['goal_reached'])

    ep_imp  = (BASELINE['steps_mean'] - s['steps']['mean']) / BASELINE['steps_mean'] * 100
    rew_imp = (s['reward']['mean'] - BASELINE['reward_mean']) / abs(BASELINE['reward_mean']) * 100

    lines = [
        sep,
        "  PHASE 5A — PPO EVALUATION SUMMARY (30 EPISODES)",
        f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Episodes  : {N_EPISODES}",
        f"  Seed      : {SEED}  (deterministic=True)",
        f"  Checkpoint: {CHECKPOINT}",
        sep,
        "",
        "  CORE METRICS               Mean ± Std        Min     Max",
        "  " + "-" * 58,
        f"  Episode steps          {s['steps']['mean']:>8.1f} ± {s['steps']['std']:<7.1f} {s['steps']['min']:>6.0f}  {s['steps']['max']:>6.0f}",
        f"  Total reward           {s['reward']['mean']:>8.2f} ± {s['reward']['std']:<7.2f} {s['reward']['min']:>6.2f}  {s['reward']['max']:>6.2f}",
        f"  Task reward            {s['r_task']['mean']:>8.2f} ± {s['r_task']['std']:<7.2f} {s['r_task']['min']:>6.2f}  {s['r_task']['max']:>6.2f}",
        f"  Collision steps/ep     {s['collision']['mean']:>8.1f} ± {s['collision']['std']:<7.1f} {s['collision']['min']:>6.0f}  {s['collision']['max']:>6.0f}",
        f"  Inference time (ms)    {s['inference_ms']['mean']:>8.3f} ± {s['inference_ms']['std']:<7.3f}",
        "",
        f"  Goal success rate      {s['goal_rate']*100:.1f}%  ({n_goal}/{N_EPISODES} episodes)",
        "",
        sep,
        "  VS SCRIPTED BASELINE (Phase 1, N=3)",
        "  " + "-" * 58,
        f"  Steps  : PPO {s['steps']['mean']:.1f} ± {s['steps']['std']:.1f}  vs  Baseline {BASELINE['steps_mean']:.1f} ± {BASELINE['steps_std']:.1f}  →  {ep_imp:+.1f}%",
        f"  Reward : PPO {s['reward']['mean']:.2f} ± {s['reward']['std']:.2f}  vs  Baseline {BASELINE['reward_mean']:.2f} ± {BASELINE['reward_std']:.2f}  →  {rew_imp:+.1f}%",
        f"  Goal   : PPO {s['goal_rate']*100:.1f}%  vs  Baseline {BASELINE['goal_rate']*100:.1f}%",
        f"  Coll.  : PPO {s['collision']['mean']:.1f} ± {s['collision']['std']:.1f}  vs  Baseline {BASELINE['collision_mean']:.1f} ± {BASELINE['collision_std']:.1f}",
        "",
        sep,
        "  PER-EPISODE DETAIL",
        "  " + "-" * 58,
        "  Ep  Steps   Reward   R_task  Coll  Goal  Time(s)",
        "  " + "-" * 58,
    ]

    for r in results:
        g = "✓" if r['goal_reached'] else "✗"
        lines.append(
            f"  {r['episode_id']+1:>2}  {r['steps']:>5}  "
            f"{r['total_reward']:>8.2f}  {r['r_task']:>7.2f}  "
            f"{r['collision_steps']:>4}  {g:>4}  {r['wall_time_s']:>6.1f}"
        )

    lines += ["", sep, ""]
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("=" * 65)
    print("  PHASE 5A — PPO AGENT SYSTEMATIC EVALUATION")
    print(f"  {N_EPISODES} episodes · deterministic=True · seed={SEED}")
    print("=" * 65)

    # ── 1. Load model with DummyVecEnv — same as eval_agent.py ───────────
    print("\n  [1/3] Loading PPO model...")
    try:
        _dummy = SafeRewardWrapper(
            TissueRetractionV2(),
            lambda_force=LAMBDA_FORCE,
            lambda_collision=LAMBDA_COLLISION,
            force_threshold=FORCE_THRESHOLD,
            step_penalty=STEP_PENALTY
        )
        _vec = DummyVecEnv([lambda: _dummy])
        model = PPO.load(CHECKPOINT, env=_vec)
        n_params = sum(p.numel() for p in model.policy.parameters())
        print(f"        OK — {n_params:,} parameters")
        _vec.close()
    except Exception as e:
        print(f"        FAIL: {e}")
        sys.exit(1)

    # ── 2. Create evaluation environment ──────────────────────────────────
    print("\n  [2/3] Creating evaluation environment...")
    try:
        eval_env = SafeRewardWrapper(
            TissueRetractionV2(),
            lambda_force=LAMBDA_FORCE,
            lambda_collision=LAMBDA_COLLISION,
            force_threshold=FORCE_THRESHOLD,
            step_penalty=STEP_PENALTY
        )
        print(f"        OK — obs space: {eval_env.observation_space}")
        print(f"             act space: {eval_env.action_space}")
    except Exception as e:
        print(f"        FAIL: {e}")
        sys.exit(1)

    # ── 3. Run 30 episodes ────────────────────────────────────────────────
    print(f"\n  [3/3] Running {N_EPISODES} episodes...")
    print()
    print(f"  {'Ep':>3}  {'Steps':>5}  {'Reward':>8}  "
          f"{'Coll':>5}  {'Goal':>4}  {'Time':>6}  Running%")
    print("  " + "-" * 58)

    results     = []
    total_start = time.perf_counter()
    np.random.seed(SEED)

    for ep in range(N_EPISODES):
        result = run_episode(eval_env, model, ep)
        results.append(result)

        running_goal_rate = sum(
            1 for r in results if r['goal_reached']) / len(results)
        g = "✓" if result['goal_reached'] else "✗"

        print(
            f"  {ep+1:>3}  {result['steps']:>5}  "
            f"{result['total_reward']:>8.2f}  "
            f"{result['collision_steps']:>5}  "
            f"{g:>4}  {result['wall_time_s']:>5.1f}s"
            f"  [{running_goal_rate:.0%}]"
        )

    total_time = time.perf_counter() - total_start
    eval_env.close()

    print()
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── 4. Compute summary ────────────────────────────────────────────────
    summary = {
        'n_episodes'  : N_EPISODES,
        'goal_rate'   : sum(1 for r in results if r['goal_reached']) / N_EPISODES,
        'steps'       : stats(results, 'steps'),
        'reward'      : stats(results, 'total_reward'),
        'r_task'      : stats(results, 'r_task'),
        'r_coll'      : stats(results, 'r_coll'),
        'collision'   : stats(results, 'collision_steps'),
        'force_viol'  : stats(results, 'force_viol'),
        'inference_ms': stats(results, 'inference_ms'),
    }

    summary_text = format_summary(results, summary)
    print()
    print(summary_text)

    # ── 5. Save outputs ───────────────────────────────────────────────────
    print("  Saving outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_csv(results, CSV_PATH)
    save_comparison_csv(summary, COMPARE_PATH)

    with open(SUMMARY_PATH, 'w') as f:
        f.write(summary_text)
    print(f"  [saved] {SUMMARY_PATH}")

    # Save JSON (same as eval_agent.py pattern)
    json_path = os.path.join(OUTPUT_DIR, "phase5a_summary.json")
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)
    print(f"  [saved] {json_path}")

    print()
    print("  Phase 5A complete. Results in: results/")
    print()

    os._exit(0)

if __name__ == "__main__":
    main()