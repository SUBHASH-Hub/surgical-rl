# Phase 1 Baseline Metrics

## Experimental Setup
- Environment: TissueRetractionEnv (LapGym + SOFA v25.12)
- Mode: Headless, STATE observations, scripted waypoints
- Platform: Ubuntu 22.04, GTX 1650, Python 3.10
- Date: March 2026

## Results (3 runs)

| Run | Steps | Total Reward | Grasping Steps | Retraction Steps | Collision Steps |
|-----|-------|-------------|----------------|------------------|-----------------|
| 1   | 254   | -177.53     | 205            | 49               | 51              |
| 2   | 251   | -168.17     | 203            | 48               | 49              |
| 3   | 237   | -150.93     | 188            | 49               | 48              |
| **Mean** | **247** | **-165.54** | **199** | **49** | **49** |

All 3 runs: Goal reached = True, Mean FPS = 15-17

## Key Observations
- Retraction phase is consistent (48-49 steps) — direct path once grasped
- Grasping phase varies (188-205 steps) — sensitive to physics non-determinism
- 49 mean collision steps — scripted approach passes through tissue
- No force constraints in reward — Phase 2 addition

## Phase 2 PPO Targets (must beat these)
- Total steps: below 200 (currently 247 mean)
- Collision steps: below 20 (currently 49 mean)
- Total reward: above -100 (currently -165.54 mean)
- Force constraint violations: below 5% (new metric, not in baseline)