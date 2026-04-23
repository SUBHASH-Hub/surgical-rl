"""
Phase 4B: PolicyTestClient -- simple action client to test RetractPolicyServer.

Sends a single Retract goal to the action server, prints live feedback,
and reports the final result. Used to verify the PPO policy runs correctly
via the ROS 2 action protocol before connecting to the Behaviour Tree.

Usage:
  ros2 run lapgym_ros2_bridge policy_test_client

Author: Subhash Arockiadoss
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from lapgym_interfaces.action import Retract


class PolicyTestClient(Node):

    def __init__(self):
        super().__init__('policy_test_client')
        self._client = ActionClient(self, Retract, 'retract_policy')

    def send_goal(self, max_steps: int = 300, render: bool = False):
        """Send a retract goal and wait for result."""
        self.get_logger().info('Waiting for retract_policy action server...')
        self._client.wait_for_server()
        self.get_logger().info('Action server found -- sending goal')

        goal_msg = Retract.Goal()
        goal_msg.max_steps = float(max_steps)
        goal_msg.render = render

        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb)
        send_goal_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by server')
            return
        self.get_logger().info('Goal accepted -- policy running')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        status = 'COL' if fb.in_collision else 'SAFE'
        self.get_logger().info(
            f'Step {fb.step:3d} | '
            f'Dist: {fb.distance_mm:.1f}mm | '
            f'{status} | '
            f'col_cost: {fb.collision_cost:.3f}')

    def _result_cb(self, future):
        result = future.result().result
        self.get_logger().info('--- RESULT ---')
        self.get_logger().info(f'Success:      {result.success}')
        self.get_logger().info(f'Termination:  {result.termination}')
        self.get_logger().info(f'Steps taken:  {result.steps_taken}')
        self.get_logger().info(
            f'Final dist:   {result.final_distance * 1000:.1f}mm')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyTestClient()
    node.send_goal(max_steps=300, render=False)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        os._exit(0)