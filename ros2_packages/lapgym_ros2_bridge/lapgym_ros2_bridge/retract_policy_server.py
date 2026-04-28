"""
Phase 4B: RetractPolicyServer -- ROS 2 action server wrapping Phase 2D PPO.

Action: lapgym_interfaces/action/Retract
  Goal:     max_steps (float32), render (bool)
  Feedback: distance_to_goal, distance_mm, step, in_collision, collision_cost
  Result:   success, steps_taken, final_distance, termination

The server loads the Phase 2D PPO checkpoint and runs policy.predict(obs)
every step. It checks is_preempted() after every step so the Behaviour Tree
(Phase 4C) can cancel the goal at any time safely.

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
import numpy as np
import rclpy
from std_msgs.msg import Bool
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from lapgym_interfaces.action import Retract

# PPO checkpoint path relative to surgical-rl repo root
CHECKPOINT_PATH = 'logs/checkpoints/phase2_ppo_tissue_retraction_20260409_211946/ppo_tissue_final'
DEFAULT_MAX_STEPS = 300


class RetractPolicyServer(Node):
    """ROS 2 action server that runs the Phase 2D PPO policy."""

    def __init__(self):
        super().__init__('retract_policy_server')

        # -- Load PPO policy --------------------------------------------------
        self._policy = None
        self._env = None
        self._load_policy()

        # -- Action server ----------------------------------------------------
        self._action_server = ActionServer(
            self,
            Retract,
            'retract_policy',
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
            '_surgeon_stop_retract',
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
        
        self.get_logger().info('RetractPolicyServer ready')


    def _spin_stop_node(self):
        while self._stop_context.ok():
            try:
                self._stop_executor.spin_once(timeout_sec=0.01)
            except Exception:
                pass


    # -- Policy loading -------------------------------------------------------

    def _load_policy(self):
        """Load Phase 2D PPO checkpoint and create environment."""
        try:
            from stable_baselines3 import PPO
            from sofa_env.scenes.tissue_retraction.tissue_retraction_env import RenderMode
            from envs.tissue_retraction_v2 import TissueRetractionV2

            self.get_logger().info(f'Loading PPO checkpoint: {CHECKPOINT_PATH}')
            self._policy = PPO.load(CHECKPOINT_PATH)
            self.get_logger().info('PPO checkpoint loaded successfully')

            # Create environment in headless mode by default
            self._env = TissueRetractionV2(
                env_kwargs={'render_mode': RenderMode.HUMAN})
            obs, _ = self._env.reset()
            self._obs = obs
            self.get_logger().info('TissueRetractionV2 environment ready')

        except Exception as e:
            self.get_logger().error(f'Failed to load policy: {e}')
            self._policy = None
            self._env = None

    # -- Action callbacks -----------------------------------------------------

    def _goal_cb(self, goal_request):
        """Accept or reject incoming goal."""
        self.get_logger().info(
            f'Received goal: max_steps={goal_request.max_steps} '
            f'render={goal_request.render}')
        if self._policy is None or self._env is None:
            self.get_logger().error('Policy not loaded -- rejecting goal')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        """Accept cancellation requests -- always allow preemption."""
        self.get_logger().info('Cancel request received -- accepting')
        return CancelResponse.ACCEPT
    
    def _surgeon_cb(self, msg: Bool):
        if msg.data:
            self._surgeon_stopped = True
            self._stop_event.clear()
            self.get_logger().warn('Retract: SURGEON STOP received')
        else:
            self._surgeon_stopped = False
            self._stop_event.set()
            self.get_logger().info('Retract: SURGEON RESUME received')

    def _emergency_cb(self, msg: Bool):
        if msg.data and not self._emergency:
            self._emergency = True
            self.get_logger().error('Retract: EMERGENCY STOP received')

    async def _execute_cb(self, goal_handle):
        """Main policy execution loop."""
        self.get_logger().info('Executing retract policy')

        goal = goal_handle.request
        max_steps = int(goal.max_steps) if goal.max_steps > 0 else DEFAULT_MAX_STEPS

        # Reset environment for fresh episode
        obs, _ = self._env.reset()
        self._obs = obs

        feedback_msg = Retract.Feedback()
        step = 0
        termination = 'timeout'
        final_distance = 0.0

        while step < max_steps:

            # -- Check preemption FIRST every step ----------------------------
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info(
                    f'Goal preempted at step {step}')
                result = Retract.Result()
                result.success = False
                result.steps_taken = step
                result.final_distance = final_distance
                result.termination = 'preempted'
                return result
            
            # -- Emergency stop check -----------------------------------------
            if self._emergency:
                try:
                    goal_handle.canceled()
                except Exception:
                    goal_handle.abort()
                result = Retract.Result()
                result.success = False
                result.steps_taken = step
                result.final_distance = final_distance
                result.termination = 'emergency_stop'
                return result

            # -- Surgeon stop: freeze until resumed ---------------------------
            while self._surgeon_stopped and not self._emergency:
                if goal_handle.is_cancel_requested:
                    break
                self._stop_event.wait(timeout=0.05)

            # -- Policy inference ---------------------------------------------
            action, _ = self._policy.predict(obs, deterministic=True)

            # -- Step environment ---------------------------------------------
            obs, reward, terminated, truncated, info = self._env.step(action)
            step += 1

            # -- Immediate surgeon stop check after every step ----------------
            while self._surgeon_stopped and not self._emergency:
                if goal_handle.is_cancel_requested:
                    break
                self._stop_event.wait(timeout=0.05)


            # -- Extract guidance info ----------------------------------------
            dist_world = float(
                info.get('distance_to_grasping_position', 0.0) or 0.0)
            in_collision = bool(info.get('in_collision', False))
            collision_cost = abs(
                float(info.get('collision_cost', 0.0) or 0.0))
            final_distance = dist_world

            # -- Feedback every step (for accurate console display) -----------    
            feedback_msg.distance_to_goal = dist_world
            feedback_msg.distance_mm = dist_world * 1000.0
            feedback_msg.step = step
            feedback_msg.in_collision = in_collision
            feedback_msg.collision_cost = collision_cost
            goal_handle.publish_feedback(feedback_msg)
            if step % 5 == 0:
                self.get_logger().info(
                    f'Step {step:3d} | Dist: {dist_world*1000:.1f}mm '
                    f'| {"COL" if in_collision else "SAFE"}')
            # -- Check termination --------------------------------------------
            if terminated:
                goal_reached = bool(info.get('goal_reached', False))
                if goal_reached:
                    termination = 'goal_reached'
                    self.get_logger().info(
                        f'Goal reached at step {step}!')
                else:
                    termination = 'collision'
                    self.get_logger().warn(
                        f'Episode terminated (collision) at step {step}')
                break

        # -- Build result -----------------------------------------------------
        success = termination == 'goal_reached'
        goal_handle.succeed()

        result = Retract.Result()
        result.success = success
        result.steps_taken = step
        result.final_distance = final_distance
        result.termination = termination

        self.get_logger().info(
            f'Episode complete: {termination} | '
            f'steps={step} | dist={final_distance*1000:.1f}mm')
        return result

    # -- Cleanup --------------------------------------------------------------

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
    node = RetractPolicyServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)