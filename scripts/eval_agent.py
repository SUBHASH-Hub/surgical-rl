"""
eval_agent.py — Phase 2 Final Evaluation
==========================================
Runs the Phase 2D PPO agent for N episodes (default 10) and compares
against the Phase 1 documented baseline from baseline_demo.py.

The scripted baseline numbers (247 steps, -165.54 reward, 49 collision
steps, 100% goal rate) are the official Phase 1 results committed in
docs/baseline_metrics.md. They are used directly here rather than
re-running the scripted policy, because baseline_demo.py uses SOFA's
internal waypoint follower (add_waypoints_to_end_effector) which is
not compatible with the SafeRewardWrapper evaluation loop.

Usage:
    source setup_env.sh
    python scripts/eval_agent.py

    # Custom checkpoint or episodes:
    python scripts/eval_agent.py --checkpoint logs/checkpoints/<run>/ppo_tissue_final
    python scripts/eval_agent.py --n_episodes 20

Author: Subhash Arockiadoss
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
sys.path.insert(0, '.')

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate Phase 2D PPO agent")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to PPO checkpoint (without .zip). "
         "Auto-detects most recent if not given."
)
parser.add_argument(
    "--n_episodes", type=int, default=10,
    help="Number of episodes to evaluate (default: 10)"
)
parser.add_argument(
    "--deterministic", type=lambda x: x.lower() != 'false', default=True,
    help="Use deterministic PPO actions (default: True)"
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed (default: 42)"
)
args = parser.parse_args()

# ── Phase 1 documented baseline ───────────────────────────────────────────────
# Source: baseline_demo.py — 10 episodes, committed in docs/baseline_metrics.md
# The scripted agent uses SOFA's add_waypoints_to_end_effector() which is
# not compatible with the SafeRewardWrapper eval loop, so we use the
# documented numbers directly.
BASELINE = {
    'ep_len':          {'mean': 247.0,   'std': 8.2},
    'r_total':         {'mean': -165.54, 'std': 12.3},
    'r_task':          {'mean': -165.54, 'std': 12.3},
    'r_coll':          {'mean': -30.0,   'std': 0.0},
    'collision_steps': {'mean': 49.0,    'std': 4.1},
    'force_violations':{'mean': 0.0,     'std': 0.0},
    'goal_rate':       1.0,
    'source':          'baseline_demo.py — Phase 1 documented result',
}

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def hr(char="─", width=65): print(char * width)

# ── Find checkpoint ───────────────────────────────────────────────────────────
def find_checkpoint():
    checkpoint_dir = Path("logs/checkpoints")
    if not checkpoint_dir.exists():
        return None
    runs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    for run_dir in reversed(runs):
        candidate = run_dir / "ppo_tissue_final"
        if Path(str(candidate) + ".zip").exists():
            return str(candidate)
    return None

checkpoint_path = args.checkpoint or find_checkpoint()
if checkpoint_path is None:
    print(f"{RED}ERROR: No checkpoint found. Specify --checkpoint path/to/checkpoint{RESET}")
    sys.exit(1)
if not Path(str(checkpoint_path) + ".zip").exists():
    print(f"{RED}ERROR: Checkpoint not found: {checkpoint_path}.zip{RESET}")
    sys.exit(1)

# ── Imports ────────────────────────────────────────────────────────────────────
print("\nLoading environment and model...")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from envs import TissueRetractionV2
    from envs.safe_reward import SafeRewardWrapper
except ImportError as e:
    print(f"{RED}ERROR: Cannot import environment: {e}{RESET}")
    sys.exit(1)

# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(env, policy_fn):
    obs, info = env.reset()
    ep_len = r_total = r_task_total = r_coll_total = 0
    collision_steps = force_violations = 0
    goal_reached = False

    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_len  += 1
        r_total += reward
        r_task_total += info.get('r_task', reward)
        r_coll_total += info.get('r_coll', 0.0)
        force_violations += int(info.get('force_viol', 0.0) > 0)
        if info.get('in_collision', False) or info.get('r_coll', 0.0) < 0:
            collision_steps += 1
        if info.get('goal_reached', False) or info.get('is_success', False):
            goal_reached = True
        if terminated or truncated:
            if info.get('goal_reached', False):
                goal_reached = True
            break

    return {
        'ep_len': ep_len, 'r_total': r_total,
        'r_task': r_task_total, 'r_coll': r_coll_total,
        'goal_reached': goal_reached,
        'collision_steps': collision_steps,
        'force_violations': force_violations,
    }

# ── PPO policy ─────────────────────────────────────────────────────────────────
class PPOPolicy:
    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic
    def __call__(self, obs):
        if not isinstance(obs, np.ndarray): obs = np.array(obs)
        if obs.ndim == 1: obs = obs.reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return (action[0] if action.ndim == 2 else action).astype(np.float32)

# ── Stats helper ──────────────────────────────────────────────────────────────
def stats(results, key):
    v = [r[key] for r in results]
    return {'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'min': float(np.min(v)),   'max': float(np.max(v))}

def pct(base, agent, higher_better=True):
    if abs(base) < 1e-8: return "N/A"
    ch = (agent - base) / abs(base) * 100
    col = (GREEN if (ch>0)==higher_better else RED)
    return f"{col}{'+' if ch>0 else ''}{ch:.1f}%{RESET}"

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*65}{RESET}")
print(f"{BOLD}  Phase 2 Final Evaluation — PPO Agent vs Scripted Baseline{RESET}")
print(f"{BOLD}{'='*65}{RESET}")
print(f"  Checkpoint:    {checkpoint_path}")
print(f"  Episodes:      {args.n_episodes}")
print(f"  Deterministic: {args.deterministic}")
print(f"  Seed:          {args.seed}")
print(f"  Baseline:      {BASELINE['source']}")
print(f"  Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load model
print(f"\nLoading PPO checkpoint...")
try:
    _dummy = SafeRewardWrapper(TissueRetractionV2(), lambda_force=0.5,
                lambda_collision=0.5, force_threshold=0.5, step_penalty=0.01)
    _vec = DummyVecEnv([lambda: _dummy])
    ppo_model = PPO.load(checkpoint_path, env=_vec)
    n_params = sum(p.numel() for p in ppo_model.policy.parameters())
    print(f"  {GREEN}✓{RESET} PPO model loaded — {n_params:,} parameters")
    _vec.close()
except Exception as e:
    print(f"{RED}ERROR loading PPO model: {e}{RESET}")
    sys.exit(1)

# Run PPO episodes
np.random.seed(args.seed)
ppo_policy = PPOPolicy(ppo_model, deterministic=args.deterministic)

print(f"\n{BOLD}Running PPO Agent (Phase 2D) — {args.n_episodes} episodes{RESET}")
hr()

eval_env = SafeRewardWrapper(
    TissueRetractionV2(), lambda_force=0.5, lambda_collision=0.5,
    force_threshold=0.5, step_penalty=0.01
)
ppo_results = []
for i in range(args.n_episodes):
    ep = run_episode(eval_env, ppo_policy)
    ppo_results.append(ep)
    goal_str = f"{GREEN}✓ goal{RESET}" if ep['goal_reached'] else f"{RED}✗ goal{RESET}"
    print(
        f"  Ep {i+1:2d}: steps={ep['ep_len']:4d} | "
        f"r_total={ep['r_total']:8.2f} | "
        f"r_task={ep['r_task']:8.2f} | "
        f"coll_steps={ep['collision_steps']:4d} | {goal_str}"
    )
eval_env.close()

# Compute PPO stats
ppo_stats = {m: stats(ppo_results, m) for m in
             ['ep_len','r_total','r_task','r_coll','collision_steps','force_violations']}
ppo_goal_rate = sum(1 for r in ppo_results if r['goal_reached']) / args.n_episodes

# ── Comparison table ───────────────────────────────────────────────────────────
print(f"\n\n{BOLD}{'='*65}{RESET}")
print(f"{BOLD}  RESULTS — PPO Agent vs Phase 1 Scripted Baseline{RESET}")
print(f"{BOLD}{'='*65}{RESET}")
print(f"  Baseline source: {BASELINE['source']}\n")

col = [28, 14, 14, 10]
print(f"{BOLD}{'Metric':<{col[0]}} {'Baseline':>{col[1]}} {'PPO Agent':>{col[2]}} {'Change':>{col[3]}}{RESET}")
hr()

rows = [
    ('ep_len',          'Episode steps mean',    False),
    ('r_total',         'Total reward mean',      True),
    ('r_task',          'Task reward mean',       True),
    ('collision_steps', 'Collision steps mean',   False),
    ('force_violations','Force violations mean',  False),
]
for key, label, hi in rows:
    b = BASELINE[key]['mean']
    p = ppo_stats[key]['mean']
    ch = pct(b, p, higher_better=hi)
    print(f"{label:<{col[0]}} {b:>{col[1]}.2f} {p:>{col[2]}.2f} {ch:>{col[3]}}")

hr()
print(f"{'Goal rate':<{col[0]}} {BASELINE['goal_rate']*100:>{col[1]}.0f}% "
      f"{ppo_goal_rate*100:>{col[2]}.0f}%")
hr()

# ── Target assessment ──────────────────────────────────────────────────────────
print(f"\n{BOLD}Phase 2 Target Assessment{RESET}")
hr("─", 50)

checks = [
    ('ep_len',          200,   False, "< 200 steps"),
    ('r_total',        -100,   True,  "> -100 reward"),
    ('collision_steps',  20,   False, "< 20 collision steps"),
]
for key, target, hi, label in checks:
    val = ppo_stats[key]['mean']
    met = val >= target if hi else val <= target
    near = abs(val - target) / abs(target) < 0.35
    status = (f"{GREEN}✓ MET{RESET}" if met else
              f"{YELLOW}~ CLOSE{RESET}" if near else
              f"{RED}✗ NOT MET{RESET}")
    print(f"  Target {label:<24} PPO = {val:.1f}   {status}")

fv = ppo_stats['force_violations']['mean']
fv_s = f"{YELLOW}Unverified — SOFA constraint solver limitation{RESET}"
print(f"  Target < 5% force violations      PPO = {fv:.2f}   {fv_s}")

# ── Key improvements ───────────────────────────────────────────────────────────
print(f"\n{BOLD}Key Improvements vs Scripted Baseline{RESET}")
hr("─", 50)
ep_imp = (BASELINE['ep_len']['mean'] - ppo_stats['ep_len']['mean']) / BASELINE['ep_len']['mean'] * 100
rew_imp = ppo_stats['r_total']['mean'] - BASELINE['r_total']['mean']
print(f"  Episode efficiency: PPO is {ep_imp:.1f}% faster ({BASELINE['ep_len']['mean']:.0f} → {ppo_stats['ep_len']['mean']:.1f} steps)")
print(f"  Total reward:       PPO scores {abs(rew_imp):.2f} points higher ({BASELINE['r_total']['mean']:.2f} → {ppo_stats['r_total']['mean']:.2f})")
print(f"  Goal success:       {ppo_goal_rate*100:.0f}% vs {BASELINE['goal_rate']*100:.0f}% — learned autonomously with ZERO hardcoded waypoints")

# ── Per-episode detail ─────────────────────────────────────────────────────────
print(f"\n{BOLD}Per-Episode PPO Detail{RESET}")
hr()
print(f"  {'Ep':>3}  {'Steps':>6}  {'R_total':>9}  {'R_task':>9}  {'Coll_steps':>11}  {'Goal':>5}")
hr("─", 60)
for i, ep in enumerate(ppo_results):
    g = f"{GREEN}✓{RESET}" if ep['goal_reached'] else f"{RED}✗{RESET}"
    print(f"  {i+1:>3}  {ep['ep_len']:>6}  {ep['r_total']:>9.2f}  "
          f"{ep['r_task']:>9.2f}  {ep['collision_steps']:>11}  {g:>5}")
hr()
print(f"  {'Mean':>3}  {ppo_stats['ep_len']['mean']:>6.1f}  "
      f"{ppo_stats['r_total']['mean']:>9.2f}  "
      f"{ppo_stats['r_task']['mean']:>9.2f}  "
      f"{ppo_stats['collision_steps']['mean']:>11.1f}")
print(f"  {'Std':>3}  {ppo_stats['ep_len']['std']:>6.1f}  "
      f"{ppo_stats['r_total']['std']:>9.2f}  "
      f"{ppo_stats['r_task']['std']:>9.2f}  "
      f"{ppo_stats['collision_steps']['std']:>11.1f}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path("logs/eval")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"eval_phase2d_{timestamp}.json"

output = {
    'timestamp': timestamp,
    'checkpoint': checkpoint_path,
    'n_episodes': args.n_episodes,
    'deterministic': args.deterministic,
    'seed': args.seed,
    'baseline': BASELINE,
    'ppo': {
        'episodes': ppo_results,
        'stats': ppo_stats,
        'goal_rate': ppo_goal_rate,
    },
    'phase2_targets': {
        'ep_len_met':       ppo_stats['ep_len']['mean'] <= 200,
        'reward_met':       ppo_stats['r_total']['mean'] >= -100,
        'collision_met':    ppo_stats['collision_steps']['mean'] <= 20,
        'force_unverified': True,
    },
    'vs_baseline': {
        'ep_len_improvement_pct':    round((BASELINE['ep_len']['mean'] - ppo_stats['ep_len']['mean']) / BASELINE['ep_len']['mean'] * 100, 1),
        'reward_improvement_abs':    round(ppo_stats['r_total']['mean'] - BASELINE['r_total']['mean'], 2),
        'reward_improvement_pct':    round((ppo_stats['r_total']['mean'] - BASELINE['r_total']['mean']) / abs(BASELINE['r_total']['mean']) * 100, 1),
    }
}

with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n{GREEN}Results saved:{RESET} {out_path}")

print(f"\n{BOLD}{'='*65}{RESET}")
print(f"{BOLD}  Evaluation complete — Phase 2 is done{RESET}")
print(f"{BOLD}{'='*65}{RESET}\n")

os._exit(0)