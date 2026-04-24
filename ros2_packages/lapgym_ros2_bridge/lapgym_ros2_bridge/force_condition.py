"""
Phase 4C: ForceCondition -- py_trees condition leaf that monitors tissue force.

Subscribes to /tissue_force_proxy (std_msgs/Float32).
Returns SUCCESS if force is below alert threshold.
Returns FAILURE if force exceeds threshold for 3 consecutive readings.

When this node returns FAILURE the parallel safety node fails which
triggers cancel_goal() on the active action server -- stopping the
policy within one simulation step.

Thresholds (from bridge_node.py):
  FORCE_ALERT_THRESHOLD = 0.35  -- yellow warning
  FORCE_STOP_THRESHOLD  = 1.0   -- red stop

Author: Subhash Arockiadoss
"""

import py_trees
from std_msgs.msg import Float32


FORCE_ALERT_THRESHOLD = 0.35
CONSECUTIVE_READINGS_TO_FAIL = 3


class ForceCondition(py_trees.behaviour.Behaviour):
    """
    Monitors /tissue_force_proxy and fails if force is persistently high.

    Uses consecutive readings to avoid false positives from transient
    force spikes -- the instrument must exceed threshold for 3 ticks
    in a row before the condition fails.
    """

    def __init__(self, name: str, node):
        super().__init__(name=name)
        self._node = node
        self._current_force = 0.0
        self._consecutive_high = 0
        self._subscription = None

    def setup(self, **kwargs):
        """Create subscriber to force proxy topic."""
        self._subscription = self._node.create_subscription(
            Float32,
            '/tissue_force_proxy',
            self._force_cb,
            10)
        self._node.get_logger().info(
            '[ForceCondition] Subscribed to /tissue_force_proxy')

    def update(self) -> py_trees.common.Status:
        """Called every BT tick -- check current force level."""
        if self._current_force > FORCE_ALERT_THRESHOLD:
            self._consecutive_high += 1
            self._node.get_logger().warn(
                f'[ForceCondition] High force: {self._current_force:.3f} '
                f'({self._consecutive_high}/{CONSECUTIVE_READINGS_TO_FAIL})')
        else:
            self._consecutive_high = 0

        if self._consecutive_high >= CONSECUTIVE_READINGS_TO_FAIL:
            self._node.get_logger().error(
                f'[ForceCondition] SAFETY STOP -- force {self._current_force:.3f} '
                f'exceeded threshold {FORCE_ALERT_THRESHOLD} for '
                f'{CONSECUTIVE_READINGS_TO_FAIL} consecutive ticks')
            self._consecutive_high = 0  # reset for next attempt
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status):
        """Reset consecutive counter on any termination."""
        self._consecutive_high = 0

    def _force_cb(self, msg: Float32):
        """Update current force from topic."""
        self._current_force = float(msg.data)