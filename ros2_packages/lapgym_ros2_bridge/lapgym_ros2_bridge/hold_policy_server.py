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
import threading
import time
import numpy as np
import rclpy
from std_msgs.msg import Bool
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

        # # Surgeon stop -- dedicated background node + thread
        # # env.step() blocks the main executor so we need a separate thread
        self._surgeon_stopped = False
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._emergency = False

        self._stop_context = rclpy.Context()
        self._stop_context.init()

        self._stop_node = rclpy.create_node(
            '_surgeon_stop_hold',
            context=self._stop_context,
            enable_rosout=False
        )
        self._stop_node.create_subscription(
            Bool, '/surgeon_stop', self._surgeon_cb, 10)
        self._stop_node.create_subscription(
            Bool, '/emergency_stop', self._emergency_cb, 10)

        self._stop_executor = rclpy.executors.SingleThreadedExecutor(
            context=self._stop_context)
        self._stop_executor.add_node(self._stop_node)

        self._stop_thread = threading.Thread(
            target=self._spin_stop_node, daemon=True)
        self._stop_thread.start()


        self.get_logger().info('HoldPolicyServer ready')

    def _spin_stop_node(self):
        while self._stop_context.ok():
            try:
                self._stop_executor.spin_once(timeout_sec=0.01)
            except Exception:
                pass

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
    
    def _emergency_cb(self, msg: Bool):
        if msg.data and not self._emergency:
            self._emergency = True
            self.get_logger().error(
                'Hold server: EMERGENCY STOP received -- halting')
            
    # add callback
    def _surgeon_cb(self, msg: Bool):
        if msg.data:
            self._surgeon_stopped = True
            self._stop_event.clear()
            self.get_logger().warn('Hold: SURGEON STOP received')
        else:
            self._surgeon_stopped = False
            self._stop_event.set()
            self.get_logger().info('Hold: SURGEON RESUME received')
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
            
            # -- Emergency stop check -----------------------------------------
            if self._emergency:
                self.get_logger().error(
                    f'Hold emergency stop at step {step}')
                try:
                    goal_handle.canceled()
                except Exception:
                    goal_handle.abort()
                result = Retract.Result()
                result.success = False
                result.steps_taken = step
                result.final_distance = 0.0
                result.termination = 'emergency_stop'
                return result
            
            # -- Surgeon stop: freeze until resumed ---------------------------
            while self._surgeon_stopped and not self._emergency:
                if goal_handle.is_cancel_requested:
                    break
                self._stop_event.wait(timeout=0.05)
            
            # -- Publish zero action to hold position -------------------------
            obs, reward, terminated, truncated, info = self._env.step(zero_action)
            step += 1

            # --Immediate Surgeon stop: freeze until resumed ---------------------------
            while self._surgeon_stopped and not self._emergency:
                if goal_handle.is_cancel_requested:
                    break
                self._stop_event.wait(timeout=0.05)

            # -- Feedback every step (for accurate console display)
            
            in_collision = bool(info.get('in_collision', False))
            collision_cost = abs(float(info.get('collision_cost', 0.0) or 0.0))
            feedback_msg.distance_to_goal = 0.0
            feedback_msg.distance_mm = 0.0
            feedback_msg.step = step
            feedback_msg.in_collision = in_collision
            feedback_msg.collision_cost = collision_cost
            goal_handle.publish_feedback(feedback_msg)
    
            if step % 5 == 0:
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
        self._stop_executor.shutdown()
        self._stop_node.destroy_node()
        self._stop_context.shutdown()
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