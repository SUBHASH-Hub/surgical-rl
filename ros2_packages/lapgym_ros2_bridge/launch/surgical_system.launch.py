"""
Phase 4C/4E: surgical_system.launch.py

Launches the complete supervised autonomy surgical system with one command:
  ros2 launch lapgym_ros2_bridge surgical_system.launch.py

Nodes started:
  1. bridge_node            -- SOFA simulation + /tissue_force_proxy
  2. approach_policy_server -- proportional controller to grasping zone
  3. retract_policy_server  -- Phase 2D PPO tissue retraction
  4. hold_policy_server     -- zero-action position hold
  5. safety_watchdog_node   -- independent IEC 62304 force monitor
  6. surgeon_console        -- terminal dashboard with keyboard control
  7. surgical_bt_node       -- Behaviour Tree orchestrator (delayed 15s)

All nodes start simultaneously except BT which waits 15s for servers.

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

    # -- Node 1: SOFA bridge --------------------------------------------------
    bridge = Node(
        package='lapgym_ros2_bridge',
        executable='bridge_node',
        name='sofa_bridge_node',
        output='screen',
        parameters=[{'render_mode': render_mode}])

    # -- Node 2: Approach policy server ---------------------------------------
    approach = Node(
        package='lapgym_ros2_bridge',
        executable='approach_policy_server',
        name='approach_policy_server',
        output='screen')

    # -- Node 3: Retract policy server ----------------------------------------
    retract = Node(
        package='lapgym_ros2_bridge',
        executable='retract_policy_server',
        name='retract_policy_server',
        output='screen')

    # -- Node 4: Hold policy server -------------------------------------------
    hold = Node(
        package='lapgym_ros2_bridge',
        executable='hold_policy_server',
        name='hold_policy_server',
        output='screen')

    # -- Node 5: Safety watchdog (independent process) ------------------------
    watchdog = Node(
        package='lapgym_ros2_bridge',
        executable='safety_watchdog_node',
        name='safety_watchdog_node',
        output='screen')
    
    # -- Node 6: Surgeon console ----------------------------------------------
    console = Node(
        package='lapgym_ros2_bridge',
        executable='surgeon_console',
        name='surgeon_console',
        output='screen',
        prefix='xterm -e')

    # -- Node 7: Behaviour Tree (delayed 15s to allow servers to init) --------
    bt = TimerAction(
        period=15.0,
        actions=[Node(
            package='lapgym_ros2_bridge',
            executable='surgical_bt_node',
            name='surgical_bt_node',
            output='screen')])

    return LaunchDescription([
        render_arg,
        bridge,
        approach,
        retract,
        hold,
        watchdog,
        console,
        bt,
    ])