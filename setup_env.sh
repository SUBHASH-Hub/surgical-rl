#!/bin/bash
# =============================================================================
# Surgical RL — Project Environment Setup
# Source this file at the start of every development session:
#   source setup_env.sh
# =============================================================================

# --- 1. Activate Python virtual environment ---
# The venv contains all pip packages: stable-baselines3, gymnasium, torch, etc.
source ~/surgical_robot_lapgym_ws/sofa_venv/bin/activate

# --- 2. Connect Python to SOFA bindings ---
# SOFA's Python bindings (import Sofa) are compiled .so files that live
# outside the venv. PYTHONPATH tells Python where to find them.
export PYTHONPATH=/home/ubuntu/surgical_robot_lapgym_ws/sofa_install/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH

# --- 3. Tell SOFA where its own installation lives ---
# SOFA uses these to find its plugins and configuration at runtime
export SOFA_ROOT=/home/ubuntu/surgical_robot_lapgym_ws/sofa_install
export SOFA_PLUGIN_PATH=/home/ubuntu/surgical_robot_lapgym_ws/sofa_install/plugins
export SOFAPYTHON3_ROOT=/home/ubuntu/surgical_robot_lapgym_ws/sofa_install/plugins/SofaPython3

# --- 4. Confirm everything is active ---
echo "========================================"
echo " Surgical RL environment activated"
echo " Python : $(which python)"
echo " Version: $(python --version)"
echo " SOFA   : $SOFA_ROOT"
echo "========================================"