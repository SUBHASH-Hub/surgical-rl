"""
Phase 4B: HoldPolicyServer -- holds instrument at current position.

After RetractPolicyServer completes, this server keeps the instrument
stationary by publishing zero-delta actions. The surgeon can inspect
the retracted tissue. Runs until cancelled by the Behaviour Tree.

Author: Subhash Arockiadoss
"""

# -- Gymnasium shim MUST be first ---------------------------------------------
import sys
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
# -----------------------------------------------------------------------------

import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from lapgym_interfaces.action import Retract

DEFAULT_MAX_STEPS = 500   # hold for up to 500 steps (~55 seconds)
HOLD_STEP_DELAY = 0.1     # 10 Hz hold rate -- no need for 50Hz


class HoldPolicyServer(Node):
    """Holds instrument at current position by publishing zero actions."""

    def __init__(self):
        super().__init__('hold_policy_server')

        self._env = None
        self._load_env()

        self._action_server = ActionServer(
            self,
            Retract,
            'hold_policy',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info('HoldPolicyServer ready')

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
        self.get_logger().info('Hold cancel received -- accepting')
        return CancelResponse.ACCEPT

    async def _execute_cb(self, goal_handle):
        self.get_logger().info('Hold policy active -- holding position')

        goal = goal_handle.request
        max_steps = int(goal.max_steps) if goal.max_steps > 0 else DEFAULT_MAX_STEPS

        # Zero action -- hold current position
        zero_action = np.zeros(3, dtype=np.float32)
        feedback_msg = Retract.Feedback()
        step = 0

        while step < max_steps:

            # -- Preemption check FIRST ---------------------------------------
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info(f'Hold preempted at step {step}')
                result = Retract.Result()
                result.success = True   # hold is always "successful"
                result.steps_taken = step
                result.final_distance = 0.0
                result.termination = 'preempted'
                return result

            # -- Publish zero action to hold position -------------------------
            obs, reward, terminated, truncated, info = self._env.step(zero_action)
            step += 1

            # -- Feedback every 10 steps --------------------------------------
            if step % 10 == 0:
                in_collision = bool(info.get('in_collision', False))
                collision_cost = abs(float(info.get('collision_cost', 0.0) or 0.0))
                feedback_msg.distance_to_goal = 0.0
                feedback_msg.distance_mm = 0.0
                feedback_msg.step = step
                feedback_msg.in_collision = in_collision
                feedback_msg.collision_cost = collision_cost
                goal_handle.publish_feedback(feedback_msg)
                self.get_logger().info(
                    f'Holding step {step:3d} | '
                    f'{"COL" if in_collision else "SAFE"}')

            time.sleep(HOLD_STEP_DELAY)

        goal_handle.succeed()
        result = Retract.Result()
        result.success = True
        result.steps_taken = step
        result.final_distance = 0.0
        result.termination = 'timeout'
        self.get_logger().info('Hold timeout reached')
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
    node = HoldPolicyServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)