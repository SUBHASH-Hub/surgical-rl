# Surgical RL — Project Overview

## Clinical North Star
Autonomous tissue retraction via safe, force-bounded reinforcement 
learning in laparoscopic cholecystectomy simulation.

## The Clinical Problem
During laparoscopic cholecystectomy (gallbladder removal), safe 
exposure of Calot's triangle requires sustained tissue retraction 
within a force window of 0.5–3.0 N. Too little force and the 
anatomy is not visible. Too much force risks tearing the cystic 
artery. This project trains an RL agent to perform this autonomously.

## Stack
- Simulation: SOFA v25.12.00 + LapGym
- RL: PPO/SAC via Stable-Baselines3  
- Middleware: ROS 2 Humble
- Perception: OpenCV + MONAI
- Task Planning: Finite State Machine (py_trees_ros)
- Safety: Force-constrained reward + emergency stop watchdog

## Architecture
SOFA inner loop (physics) → ROS 2 middleware → RL policy + FSM planner

## Phases
- Phase 1: Simulation Foundation + ROS2 Infrastructure (current)
- Phase 2: RL Core — Safe Force-Bounded Retraction
- Phase 3: Perception — Endoscopic Vision Integration
- Phase 4: ROS2 Task Orchestration + FSM Planner
- Phase 5: Evaluation, Safety Analysis, Portfolio

## Platform
- Ubuntu 22.04 LTS
- NVIDIA GTX 1650, 16GB RAM
- Python 3.10, SOFA v25.12.00 pre-built binary
