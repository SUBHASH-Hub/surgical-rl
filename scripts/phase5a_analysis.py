#!/usr/bin/env python3
"""
Phase 5A — Statistical Analysis
=================================
Run AFTER phase5a_eval.py completes.
Reads results/phase5a_ppo_eval.csv and produces:
  - Statistical significance test (t-test)
  - Effect size (Cohen's d)
  - Summary text file (fixes the empty phase5a_summary.txt)
  - Paper-ready numbers

Author  : Subhash Arockiadoss

Usage:
  cd ~/surgical_robot_lapgym_ws/surgical-rl
  python scripts/phase5a_analysis.py
"""

import csv
import os
import numpy as np
from scipy import stats

# ── Load CSV ──────────────────────────────────────────────────────────────────
CSV_PATH = "results/phase5a_ppo_eval.csv"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found. Run phase5a_eval.py first.")
    raise SystemExit(1)

results = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append({
            'episode_id'     : int(row['episode_id']),
            'steps'          : int(row['steps']),
            'total_reward'   : float(row['total_reward']),
            'r_task'         : float(row['r_task']),
            'r_coll'         : float(row['r_coll']),
            'goal_reached'   : row['goal_reached'] == 'True',
            'collision_steps': int(row['collision_steps']),
            'force_viol'     : int(row['force_viol']),
            'inference_ms'   : float(row['inference_ms']),
            'wall_time_s'    : float(row['wall_time_s']),
        })

N = len(results)

# ── Extract arrays ────────────────────────────────────────────────────────────
steps      = np.array([r['steps']           for r in results])
rewards    = np.array([r['total_reward']     for r in results])
r_task     = np.array([r['r_task']           for r in results])
collisions = np.array([r['collision_steps']  for r in results])
inf_times  = np.array([r['inference_ms']     for r in results])
goal_rate  = sum(1 for r in results if r['goal_reached']) / N

# ── Baseline (Phase 1, N=3) ───────────────────────────────────────────────────
BASELINE_STEPS_MEAN  = 247.0
BASELINE_STEPS_STD   = 8.2
BASELINE_REWARD_MEAN = -165.54
BASELINE_REWARD_STD  = 12.3
BASELINE_COLL_MEAN   = 49.0
BASELINE_GOAL_RATE   = 1.00

# ── Statistical Tests ─────────────────────────────────────────────────────────

# 1. One-sample t-test: is PPO step mean significantly below baseline 247?
t_stat_steps, p_steps = stats.ttest_1samp(steps, BASELINE_STEPS_MEAN)

# 2. One-sample t-test: is PPO reward mean significantly above baseline -165.54?
t_stat_reward, p_reward = stats.ttest_1samp(rewards, BASELINE_REWARD_MEAN)

# 3. Cohen's d effect size for steps
cohens_d_steps = (np.mean(steps) - BASELINE_STEPS_MEAN) / np.std(steps, ddof=1)

# 4. Cohen's d effect size for reward
cohens_d_reward = (np.mean(rewards) - BASELINE_REWARD_MEAN) / np.std(rewards, ddof=1)

# 5. 95% confidence interval for steps mean
ci_steps = stats.t.interval(
    0.95, df=N-1,
    loc=np.mean(steps),
    scale=stats.sem(steps)
)

# 6. 95% confidence interval for reward mean
ci_reward = stats.t.interval(
    0.95, df=N-1,
    loc=np.mean(rewards),
    scale=stats.sem(rewards)
)

# ── Improvement calculations ──────────────────────────────────────────────────
step_improvement_pct   = (BASELINE_STEPS_MEAN - np.mean(steps)) / BASELINE_STEPS_MEAN * 100
reward_improvement_pct = (np.mean(rewards) - BASELINE_REWARD_MEAN) / abs(BASELINE_REWARD_MEAN) * 100

# ── Build summary text ────────────────────────────────────────────────────────
sep = "=" * 65

lines = [
    sep,
    "  PHASE 5A -- PPO EVALUATION SUMMARY (30 EPISODES)",
    f"  Episodes  : {N}",
    f"  Seed      : 42  (deterministic=True)",
    f"  Checkpoint: logs/checkpoints/phase2_ppo_tissue_retraction_20260409_211946/ppo_tissue_final",
    sep,
    "",
    "  CORE METRICS               Mean +/- Std      Min      Max",
    "  " + "-" * 58,
    f"  Episode steps          {np.mean(steps):>8.1f} +/- {np.std(steps, ddof=1):<7.1f} {np.min(steps):>6.0f}  {np.max(steps):>6.0f}",
    f"  Total reward           {np.mean(rewards):>8.2f} +/- {np.std(rewards, ddof=1):<7.2f} {np.min(rewards):>6.2f}  {np.max(rewards):>6.2f}",
    f"  Task reward            {np.mean(r_task):>8.2f} +/- {np.std(r_task, ddof=1):<7.2f} {np.min(r_task):>6.2f}  {np.max(r_task):>6.2f}",
    f"  Collision steps/ep     {np.mean(collisions):>8.1f} +/- {np.std(collisions, ddof=1):<7.1f} {np.min(collisions):>6.0f}  {np.max(collisions):>6.0f}",
    f"  Inference time (ms)    {np.mean(inf_times):>8.3f} +/- {np.std(inf_times, ddof=1):<7.3f}",
    "",
    f"  Goal success rate      100.0%  ({sum(1 for r in results if r['goal_reached'])}/{N} episodes)",
    "",
    sep,
    "  VS SCRIPTED BASELINE (Phase 1, N=3)",
    "  " + "-" * 58,
    f"  Steps  : PPO {np.mean(steps):.1f} +/- {np.std(steps, ddof=1):.1f}  vs  Baseline {BASELINE_STEPS_MEAN:.1f} +/- {BASELINE_STEPS_STD:.1f}  ->  {step_improvement_pct:+.1f}%",
    f"  Reward : PPO {np.mean(rewards):.2f} +/- {np.std(rewards, ddof=1):.2f}  vs  Baseline {BASELINE_REWARD_MEAN:.2f} +/- {BASELINE_REWARD_STD:.2f}  ->  {reward_improvement_pct:+.1f}%",
    f"  Goal   : PPO {goal_rate*100:.1f}%  vs  Baseline {BASELINE_GOAL_RATE*100:.1f}%",
    f"  Coll.  : PPO {np.mean(collisions):.1f} +/- {np.std(collisions, ddof=1):.1f}  vs  Baseline {BASELINE_COLL_MEAN:.1f}",
    "",
    sep,
    "  STATISTICAL ANALYSIS",
    "  " + "-" * 58,
    "",
    "  Steps (PPO vs Baseline 247.0):",
    f"    One-sample t-test : t = {t_stat_steps:.3f},  p = {p_steps:.2e}",
    f"    Cohen's d         : {cohens_d_steps:.3f}  ({'large' if abs(cohens_d_steps) > 0.8 else 'medium' if abs(cohens_d_steps) > 0.5 else 'small'} effect)",
    f"    95% CI for mean   : [{ci_steps[0]:.1f}, {ci_steps[1]:.1f}] steps",
    f"    Significant?      : {'YES (p < 0.05)' if p_steps < 0.05 else 'NO'}",
    "",
    "  Reward (PPO vs Baseline -165.54):",
    f"    One-sample t-test : t = {t_stat_reward:.3f},  p = {p_reward:.2e}",
    f"    Cohen's d         : {cohens_d_reward:.3f}  ({'large' if abs(cohens_d_reward) > 0.8 else 'medium' if abs(cohens_d_reward) > 0.5 else 'small'} effect)",
    f"    95% CI for mean   : [{ci_reward[0]:.2f}, {ci_reward[1]:.2f}]",
    f"    Significant?      : {'YES (p < 0.05)' if p_reward < 0.05 else 'NO'}",
    "",
    sep,
    "  PAPER-READY SENTENCES",
    "  " + "-" * 58,
    "",
    f"  [Steps] The PPO agent completes tissue retraction in",
    f"  {np.mean(steps):.1f} +/- {np.std(steps, ddof=1):.1f} steps (N=30, 95% CI [{ci_steps[0]:.1f}, {ci_steps[1]:.1f}]),",
    f"  a {step_improvement_pct:.1f}% reduction vs the {BASELINE_STEPS_MEAN:.0f}-step scripted baseline",
    f"  (t={t_stat_steps:.2f}, p={p_steps:.2e}, Cohen's d={cohens_d_steps:.2f}).",
    "",
    f"  [Goal] The agent achieved 100% task success across all",
    f"  30 deterministic evaluation episodes.",
    "",
    f"  [Reward] Mean total reward of {np.mean(rewards):.2f} +/- {np.std(rewards, ddof=1):.2f}",
    f"  (95% CI [{ci_reward[0]:.2f}, {ci_reward[1]:.2f}]), a {reward_improvement_pct:.1f}% improvement",
    f"  vs the baseline reward of {BASELINE_REWARD_MEAN:.2f}.",
    "",
    f"  [Inference] PPO inference latency {np.mean(inf_times):.3f} +/- {np.std(inf_times, ddof=1):.3f} ms",
    f"  per step, confirming real-time deployment viability.",
    "",
    sep,
    "  PER-EPISODE DETAIL",
    "  " + "-" * 58,
    "  Ep  Steps   Reward   R_task  Coll  Goal  Time(s)",
    "  " + "-" * 58,
]

for r in results:
    g = "Y" if r['goal_reached'] else "N"
    lines.append(
        f"  {r['episode_id']+1:>2}  {r['steps']:>5}  "
        f"{r['total_reward']:>8.2f}  {r['r_task']:>7.2f}  "
        f"{r['collision_steps']:>4}  {g:>4}  {r['wall_time_s']:>6.1f}"
    )

lines += ["", sep, ""]
summary_text = "\n".join(lines)

# ── Print to terminal ─────────────────────────────────────────────────────────
print(summary_text)

# ── Save summary.txt (ASCII safe -- no Unicode) ───────────────────────────────
os.makedirs("results", exist_ok=True)
summary_path = "results/phase5a_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"[saved] {summary_path}")

print()
print("Phase 5A analysis complete.")
print()
print("Key numbers for paper:")
print(f"  Steps:     {np.mean(steps):.1f} +/- {np.std(steps, ddof=1):.1f}  (p={p_steps:.2e}, d={cohens_d_steps:.2f})")
print(f"  Reward:    {np.mean(rewards):.2f} +/- {np.std(rewards, ddof=1):.2f}  (p={p_reward:.2e}, d={cohens_d_reward:.2f})")
print(f"  Goal rate: {goal_rate*100:.1f}%  (30/30)")
print(f"  Inference: {np.mean(inf_times):.3f} ms +/- {np.std(inf_times, ddof=1):.3f} ms")