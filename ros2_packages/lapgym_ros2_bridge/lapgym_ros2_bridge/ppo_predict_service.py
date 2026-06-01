"""
Phase 4G: PPOPredictService -- Python ROS 2 service server wrapping PPO inference.

The C++ retract_policy_server sends observations here.
This node runs policy.predict() and returns the action.

PyTorch releases the GIL before GPU computation — so PPO inference
does NOT block other Python threads. This is why PPO stays in Python
while the action server logic moves to C++.

Service: /ppo_predict (lapgym_interfaces/srv/PPOPredict)

Author: Subhash Arockiadoss
"""

import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node

from lapgym_interfaces.srv import PPOPredict

CHECKPOINT_PATH = (
    'logs/checkpoints/'
    'phase2_ppo_tissue_retraction_20260409_211946/ppo_tissue_final'
)


class PPOPredictService(Node):
    """Python service server that owns the PPO policy.

    C++ retract server sends obs[7] here.
    This node calls policy.predict(obs, deterministic=True).
    Returns action[3] to C++ server.
    """

    def __init__(self):
        super().__init__('ppo_predict_service')

        self._policy = None
        self._load_policy()

        self._service = self.create_service(
            PPOPredict,
            '/ppo_predict',
            self._handle_predict
        )

        self.get_logger().info('PPOPredictService ready on /ppo_predict')

    def _load_policy(self):
        """Load Phase 2D PPO checkpoint."""
        try:
            # Gymnasium shim required for stable-baselines3
            import gymnasium
            sys.modules['gym'] = gymnasium
            sys.modules['gym.spaces'] = gymnasium.spaces

            from stable_baselines3 import PPO

            self.get_logger().info(
                f'Loading PPO checkpoint: {CHECKPOINT_PATH}')
            self._policy = PPO.load(CHECKPOINT_PATH)
            self.get_logger().info('PPO checkpoint loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load policy: {e}')
            self._policy = None

    def _handle_predict(self, request, response):
        """Handle one PPOPredict service call from C++ retract server.

        Args:
            request.observation -- float32[7] current observation

        Returns:
            response.action     -- float32[3] predicted action
            response.success    -- False if policy not loaded
        """
        if self._policy is None:
            self.get_logger().error('Policy not loaded')
            response.action  = [0.0, 0.0, 0.0]
            response.success = False
            return response

        # Build numpy observation from request
        obs = np.array(request.observation, dtype=np.float32)

        # policy.predict() — PyTorch RELEASES GIL before GPU computation
        # This means other Python threads can run during inference
        # deterministic=True — no exploration noise at deployment
        action, _ = self._policy.predict(obs, deterministic=True)

        response.action  = [float(action[0]),
                            float(action[1]),
                            float(action[2])]
        response.success = True
        return response

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PPOPredictService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)