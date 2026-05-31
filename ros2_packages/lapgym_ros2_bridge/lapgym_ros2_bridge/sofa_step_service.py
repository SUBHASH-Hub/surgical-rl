"""
Phase 4F: SofaStepService -- Python ROS 2 service server wrapping SOFA env.step().

The C++ approach_policy_server calls this service to execute one physics step
in the SOFA simulation. This pattern keeps all SOFA/Python bindings in one
Python node while the action server logic lives in C++.

Service: /sofa_step (lapgym_interfaces/srv/SofaStep)

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

from lapgym_interfaces.srv import SofaStep


class SofaStepService(Node):
    """Python service server that owns the SOFA environment.
    
    The C++ approach_policy_server sends action commands here.
    This node calls env.step() and returns the full physics result.
    """

    def __init__(self):
        super().__init__('sofa_step_service')

        self._env = None
        self._load_env()

        self._service = self.create_service(
            SofaStep,
            '/sofa_step',
            self._handle_step
        )

        self.get_logger().info('SofaStepService ready on /sofa_step')

    def _load_env(self):
        """Load TissueRetractionV2 environment."""
        try:
            from sofa_env.scenes.tissue_retraction.tissue_retraction_env \
                import RenderMode
            from envs.tissue_retraction_v2 import TissueRetractionV2

            self._env = TissueRetractionV2(
                env_kwargs={'render_mode': RenderMode.HEADLESS})
            obs, _ = self._env.reset()
            self._obs = obs
            self.get_logger().info('TissueRetractionV2 ready')

        except Exception as e:
            self.get_logger().error(f'Failed to load env: {e}')
            self._env = None

    def _handle_step(self, request, response):
        """Handle one SofaStep service call from C++ action server.
        
        Args:
            request.action  -- float32[3] xyz delta [-3, 3]
            request.reset   -- bool True = reset env before step
            
        Returns:
            response with full physics state
        """
        if self._env is None:
            self.get_logger().error('Environment not loaded')
            return response

        # -- Reset if requested (start of new episode) ----------------------
        if request.reset:
            obs, _ = self._env.reset()
            self._obs = obs
            self.get_logger().info('Environment reset by C++ client')

        # -- Build action from request --------------------------------------
        action = np.array(
            [request.action[0],
             request.action[1],
             request.action[2]],
            dtype=np.float32
        )
        action = np.clip(action, -3.0, 3.0)

        # -- Step SOFA physics ----------------------------------------------
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._obs = obs

        # -- Get tool world position from SOFA scene graph ------------------
        try:
            tool_world = np.array(
                self._env._env.end_effector.gripper
                .motion_target_mechanical_object
                .position.array()[0][:3],
                dtype=np.float32
            )
        except Exception:
            tool_world = np.zeros(3, dtype=np.float32)

        # -- Build response -------------------------------------------------
        response.observation = [float(x) for x in obs]
        response.reward = float(reward)
        response.terminated = bool(terminated)
        response.truncated = bool(truncated)
        response.distance_to_grasping_position = float(
            info.get('distance_to_grasping_position', 0.0) or 0.0)
        response.distance_to_end_position = float(
            info.get('distance_to_end_position', 0.0) or 0.0)
        response.in_collision = bool(info.get('in_collision', False))
        response.collision_cost = abs(float(
            info.get('collision_cost', 0.0) or 0.0))
        response.tool_world_x = float(tool_world[0])
        response.tool_world_y = float(tool_world[1])
        response.tool_world_z = float(tool_world[2])

        return response

    def destroy_node(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SofaStepService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)
