"""
Phase 4G: surgical_system.launch.py — Hybrid C++/Python architecture

Complete supervised autonomy surgical system — one launch command.

Architecture:
  C++ control layer (no GIL — true parallel threads):
    approach_policy_server_cpp  /approach_policy_cpp
    hold_policy_server_cpp      /hold_policy_cpp
    retract_policy_server_cpp   /retract_policy_cpp

  Python ML/physics layer (GIL acceptable — GPU or no blocking):
    sofa_step_service     /sofa_step      SOFA FEM physics
    ppo_predict_service   /ppo_predict    PPO inference (PyTorch/GPU)
    bridge_node           SOFA bridge + /tissue_force_proxy
    safety_watchdog_node  IEC 62304 independent force monitor
    surgical_bt_node      Behaviour tree orchestrator
    surgeon_console       Human-in-the-loop terminal UI

Nodes started:
  1.  bridge_node              SOFA simulation bridge
  2.  sofa_step_service        Python SOFA physics service
  3.  ppo_predict_service      Python PPO inference service
  4.  approach_policy_server_cpp  C++ approach controller
  5.  retract_policy_server_cpp   C++ PPO retract controller
  6.  hold_policy_server_cpp      C++ zero-action hold
  7.  safety_watchdog_node     IEC 62304 independent watchdog
  8.  surgeon_console          Terminal dashboard S/R/E/Q
  9.  surgical_bt_node         BT orchestrator (delayed 20s)

Author: Subhash Arockiadoss
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    render_arg = DeclareLaunchArgument(
        'render_mode',
        default_value='headless',
        description='SOFA render mode: headless or human')
    render_mode = LaunchConfiguration('render_mode')

    # -- Node 1: SOFA bridge -----------------------------------------------
    # Publishes /tissue_force_proxy, /joint_states at 50Hz
    # Must stay Python — SOFA bindings are SofaPython3
    bridge = Node(
        package='lapgym_ros2_bridge',
        executable='bridge_node',
        name='sofa_bridge_node',
        output='screen',
        parameters=[{'render_mode': render_mode}])

    # -- Node 2: SOFA step service -----------------------------------------
    # Python service wrapping env.step() for C++ action servers
    # C++ servers call /sofa_step instead of calling SOFA directly
    sofa_step = Node(
        package='lapgym_ros2_bridge',
        executable='sofa_step_service',
        name='sofa_step_service',
        output='screen')

    # -- Node 3: PPO predict service ---------------------------------------
    # Python service wrapping policy.predict() for C++ retract server
    # PyTorch releases GIL for GPU — inference does not block callbacks
    ppo_predict = Node(
        package='lapgym_ros2_bridge',
        executable='ppo_predict_service',
        name='ppo_predict_service',
        output='screen')

    # -- Node 4: C++ approach server ---------------------------------------
    # rclcpp action server — proportional controller
    # std::atomic stop flags — no GIL — true parallel threads
    approach = Node(
        package='lapgym_ros2_bridge_cpp',
        executable='approach_policy_server_cpp',
        name='approach_policy_server_cpp',
        output='screen')

    # -- Node 5: C++ retract server ----------------------------------------
    # rclcpp action server — PPO policy via /ppo_predict service
    # 4 callback groups: action + sofa + ppo + stop
    retract = Node(
        package='lapgym_ros2_bridge_cpp',
        executable='retract_policy_server_cpp',
        name='retract_policy_server_cpp',
        output='screen')

    # -- Node 6: C++ hold server -------------------------------------------
    # rclcpp action server — zero-action position hold
    # Simplest C++ node — no ML, pure control logic
    hold = Node(
        package='lapgym_ros2_bridge_cpp',
        executable='hold_policy_server_cpp',
        name='hold_policy_server_cpp',
        output='screen')

    # -- Node 7: Safety watchdog -------------------------------------------
    # Stays Python — already meets 60ms target (no blocking calls)
    # Independent process — IEC 62304 Class C requirement
    watchdog = Node(
        package='lapgym_ros2_bridge',
        executable='safety_watchdog_node',
        name='safety_watchdog_node',
        output='screen')

    # -- Node 8: Surgeon console -------------------------------------------
    # Stays Python — curses UI appropriate for terminal interface
    console = Node(
        package='lapgym_ros2_bridge',
        executable='surgeon_console',
        name='surgeon_console',
        output='screen',
        prefix='xterm -e')

    # -- Node 9: Behaviour Tree (delayed 20s) ------------------------------
    # Delayed 20s — C++ servers + Python services need time to init
    # (was 15s for Python-only — increased for SOFA + PPO loading)
    bt = TimerAction(
        period=20.0,
        actions=[Node(
            package='lapgym_ros2_bridge',
            executable='surgical_bt_node',
            name='surgical_bt_node',
            output='screen')])

    return LaunchDescription([
        render_arg,
        bridge,
        sofa_step,
        ppo_predict,
        approach,
        retract,
        hold,
        watchdog,
        console,
        bt,
    ])