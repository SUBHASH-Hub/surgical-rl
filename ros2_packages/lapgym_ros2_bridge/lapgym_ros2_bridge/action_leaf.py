"""
Phase 4C: ActionLeaf -- reusable py_trees leaf that wraps a ROS 2 action server.

Each leaf connects to one action server (approach, retract, or hold).
On first tick it sends a goal. On subsequent ticks it returns RUNNING
while the server executes. When the server returns a result it returns
SUCCESS or FAILURE. If the BT cancels this leaf it sends cancel_goal().

Author: Subhash Arockiadoss
"""

import py_trees
import rclpy
from rclpy.action import ActionClient
from lapgym_interfaces.action import Retract


class ActionLeaf(py_trees.behaviour.Behaviour):
    """
    A py_trees leaf that sends a goal to a ROS 2 action server.

    Parameters
    ----------
    name        : display name in the BT (e.g. 'Approach', 'Retract', 'Hold')
    action_name : ROS 2 action server name (e.g. '/approach_policy')
    node        : the rclpy node that owns this leaf
    max_steps   : goal parameter -- max steps for the action server
    """

    def __init__(self, name: str, action_name: str, node, max_steps: float = 300.0):
        super().__init__(name=name)
        self._action_name = action_name
        self._node = node
        self._max_steps = max_steps

        self._client = ActionClient(node, Retract, action_name)
        self._goal_handle = None
        self._result = None
        self._goal_sent = False
        self._done = False

    def setup(self, **kwargs):
        """Wait for action server to be available."""
        self._node.get_logger().info(
            f'[{self.name}] Waiting for action server {self._action_name}')
        self._client.wait_for_server(timeout_sec=10.0)
        self._node.get_logger().info(
            f'[{self.name}] Action server {self._action_name} ready')

    def initialise(self):
        """Called when leaf transitions from idle to running -- send goal."""
        self._goal_handle = None
        self._result = None
        self._goal_sent = False
        self._done = False

        goal_msg = Retract.Goal()
        goal_msg.max_steps = self._max_steps
        goal_msg.render = False

        self._node.get_logger().info(
            f'[{self.name}] Sending goal to {self._action_name}')

        future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb)
        future.add_done_callback(self._goal_response_cb)
        self._goal_sent = True

    def update(self) -> py_trees.common.Status:
        """Called every BT tick -- check if action is done."""
        if not self._goal_sent:
            return py_trees.common.Status.RUNNING

        if not self._done:
            # Spin once to process any pending callbacks
            rclpy.spin_once(self._node, timeout_sec=0.0)
            return py_trees.common.Status.RUNNING

        # Action finished -- check result
        if self._result is not None:
            termination = self._result.termination
            success = self._result.success

            self._node.get_logger().info(
                f'[{self.name}] Result: {termination} '
                f'steps={self._result.steps_taken} '
                f'dist={self._result.final_distance*1000:.1f}mm')

            if termination == 'goal_reached':
                return py_trees.common.Status.SUCCESS
            elif termination == 'preempted':
                return py_trees.common.Status.FAILURE
            elif termination == 'timeout':
                # timeout on hold is expected success
                if 'hold' in self._action_name.lower():
                    return py_trees.common.Status.SUCCESS
                return py_trees.common.Status.FAILURE
            else:
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.FAILURE

    def terminate(self, new_status: py_trees.common.Status):
        """Called when BT cancels or preempts this leaf -- cancel the goal."""
        if self._goal_handle is not None and not self._done:
            self._node.get_logger().warn(
                f'[{self.name}] BT terminating leaf -- cancelling goal')
            self._goal_handle.cancel_goal_async()

    # -- Callbacks ------------------------------------------------------------

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().error(
                f'[{self.name}] Goal rejected')
            self._done = True
            return
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self._node.get_logger().info(
            f'[{self.name}] Step {fb.step:3d} | '
            f'Dist: {fb.distance_mm:.1f}mm | '
            f'{"COL" if fb.in_collision else "SAFE"}',
            throttle_duration_sec=2.0)

    def _result_cb(self, future):
        self._result = future.result().result
        self._done = True