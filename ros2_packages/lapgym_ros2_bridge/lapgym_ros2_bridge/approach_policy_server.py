"""
Phase 4B: ApproachPolicyServer -- navigates instrument to grasping zone.

Uses a proportional controller (no PPO needed) to move the instrument
from its current position toward the fixed grasping target position.
Implements is_preempted() every step for safe cancellation.

Grasping target (world): [-0.0486, 0.0085, 0.0356] metres
Approach threshold: 0.015 metres (15mm) -- close enough for PPO to take over

Author: Subhash Arockiadoss
"""

# -- Gymnasium shim MUST be first ---------------------------------------------
import sys
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
# -----------------------------------------------------------------------------

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from lapgym_interfaces.action import Retract

# Fixed grasping target in world metres (confirmed from Phase 4A analysis)
GRASPING_TARGET = np.array([-0.0485583, 0.0085, 0.0356076], dtype=np.float32)
APPROACH_THRESHOLD = 0.015   # 15mm -- hand off to RetractPolicyServer
DEFAULT_MAX_STEPS = 200
APPROACH_GAIN = 2.0          # proportional gain for velocity command


class ApproachPolicyServer(Node):
    """Navigates instrument to grasping zone using proportional controller."""

    def __init__(self):
        super().__init__('approach_policy_server')

        self._env = None
        self._load_env()

        self._action_server = ActionServer(
            self,
            Retract,
            'approach_policy',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info('ApproachPolicyServer ready')

    def _load_env(self):
        try:
            from sofa_env.scenes.tissue_retraction.tissue_retraction_env import RenderMode
            from envs.tissue_retraction_v2 import TissueRetractionV2
            self._env = TissueRetractionV2(
                env_kwargs={'render_mode': RenderMode.HEADLESS})
            obs, _ = self._env.reset()
            self._obs = obs
            self.get_logger().info('Environment ready')
        except Exception as e:
            self.get_logger().error(f'Failed to load env: {e}')
            self._env = None

    def _goal_cb(self, goal_request):
        if self._env is None:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().info('Cancel request received -- accepting')
        return CancelResponse.ACCEPT

    async def _execute_cb(self, goal_handle):
        self.get_logger().info('Executing approach policy')

        goal = goal_handle.request
        max_steps = int(goal.max_steps) if goal.max_steps > 0 else DEFAULT_MAX_STEPS

        obs, _ = self._env.reset()
        feedback_msg = Retract.Feedback()
        step = 0
        termination = 'timeout'
        final_distance = 0.0

        while step < max_steps:

            # -- Preemption check FIRST ---------------------------------------
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info(f'Approach preempted at step {step}')
                result = Retract.Result()
                result.success = False
                result.steps_taken = step
                result.final_distance = final_distance
                result.termination = 'preempted'
                return result

            # -- Get current tool world position ------------------------------
            try:
                tool_world = np.array(
                    self._env._env.end_effector.gripper
                    .motion_target_mechanical_object
                    .position.array()[0][:3], dtype=np.float32)
            except Exception:
                tool_world = np.zeros(3, dtype=np.float32)

            # -- Compute distance to grasping target --------------------------
            error = GRASPING_TARGET - tool_world
            dist = float(np.linalg.norm(error))
            final_distance = dist

            # -- Check if close enough ----------------------------------------
            if dist < APPROACH_THRESHOLD:
                termination = 'goal_reached'
                self.get_logger().info(
                    f'Approach complete at step {step} dist={dist*1000:.1f}mm')
                break

            # -- Proportional controller --------------------------------------
            # Normalise direction and scale to action space [-3, 3]
            direction = error / (dist + 1e-8)
            action = np.clip(direction * APPROACH_GAIN, -3.0, 3.0).astype(np.float32)

            obs, reward, terminated, truncated, info = self._env.step(action)
            step += 1

            # -- Feedback every 5 steps ---------------------------------------
            if step % 5 == 0:
                in_collision = bool(info.get('in_collision', False))
                collision_cost = abs(float(info.get('collision_cost', 0.0) or 0.0))
                feedback_msg.distance_to_goal = dist
                feedback_msg.distance_mm = dist * 1000.0
                feedback_msg.step = step
                feedback_msg.in_collision = in_collision
                feedback_msg.collision_cost = collision_cost
                goal_handle.publish_feedback(feedback_msg)
                self.get_logger().info(
                    f'Approach step {step:3d} | Dist: {dist*1000:.1f}mm')

            if terminated:
                termination = 'collision'
                break

        success = termination == 'goal_reached'
        goal_handle.succeed()

        result = Retract.Result()
        result.success = success
        result.steps_taken = step
        result.final_distance = final_distance
        result.termination = termination
        self.get_logger().info(
            f'Approach complete: {termination} steps={step}')
        return result

    def destroy_node(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ApproachPolicyServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)