"""
Phase 4C: SurgicalBTNode -- Behaviour Tree orchestrating all three action servers.

Tree structure:
  Root [Sequence]
    SafetyMonitor [Parallel - fail if watchdog fails]
      SurgicalSequence [Sequence]
        Approach  [ActionLeaf -> /approach_policy]
        Retract   [ActionLeaf -> /retract_policy]
        Hold      [ActionLeaf -> /hold_policy]
      ForceWatchdog [ForceCondition -> /tissue_force_proxy]

The BT ticks at 10 Hz. If ForceCondition returns FAILURE:
  - Parallel node fails
  - Active ActionLeaf.terminate() called
  - cancel_goal() sent to active action server
  - Instrument holds current position

Author: Subhash Arockiadoss
"""

import os
import rclpy
from rclpy.node import Node
import py_trees
import py_trees_ros

from std_msgs.msg import String
from lapgym_ros2_bridge.action_leaf import ActionLeaf
from lapgym_ros2_bridge.force_condition import ForceCondition


BT_TICK_HZ = 10


def create_surgical_tree(node) -> py_trees.trees.BehaviourTree:
    """Build and return the surgical behaviour tree."""

    # -- Leaf nodes -----------------------------------------------------------

    approach = ActionLeaf(
        name='Approach',
        action_name='approach_policy',
        node=node,
        max_steps=400.0)

    retract = ActionLeaf(
        name='Retract',
        action_name='retract_policy',
        node=node,
        max_steps=300.0)

    hold = ActionLeaf(
        name='Hold',
        action_name='hold_policy',
        node=node,
        max_steps=200.0)

    force_watchdog = ForceCondition(
        name='ForceWatchdog',
        node=node)

    # -- Surgical sequence: approach then retract then hold -------------------
    surgical_sequence = py_trees.composites.Sequence(
        name='SurgicalSequence',
        memory=True)
    surgical_sequence.add_children([approach, retract, hold])

    # -- Parallel safety: surgical sequence AND force watchdog ----------------
    # policy=py_trees.common.ParallelPolicy.SuccessOnAll means
    # the parallel node succeeds only when ALL children succeed.
    # If force watchdog fails, parallel fails immediately.
    safety_monitor = py_trees.composites.Parallel(
        name='SafetyMonitor',
        policy=py_trees.common.ParallelPolicy.SuccessOnSelected(
            children=[surgical_sequence],
            synchronise=False))
    safety_monitor.add_children([surgical_sequence, force_watchdog])

    # -- Root sequence --------------------------------------------------------
    root = py_trees.composites.Sequence(
        name='Root',
        memory=True)
    root.add_child(safety_monitor)

    # -- Build tree -----------------------------------------------------------
    tree = py_trees.trees.BehaviourTree(root=root)
    return tree


class SurgicalBTNode(Node):
    """ROS 2 node that runs the surgical behaviour tree at BT_TICK_HZ."""

    def __init__(self):
        super().__init__('surgical_bt_node')

        self.get_logger().info('Building surgical behaviour tree')
        self._tree = create_surgical_tree(self)

        # Setup all leaves (connects to action servers, creates subscribers)
        self._tree.setup(timeout=30)
        self.get_logger().info('Behaviour tree setup complete')

        # Console feedback publisher
        self._pub_console = self.create_publisher(
            String, '/console_feedback', 10)
        self._current_phase = 'IDLE'
        self._current_step = 0
        self._current_max = 300
        self._current_dist = 0.0
        self._bt_state = 'WAITING'

        # Print tree structure to terminal
        print(py_trees.display.ascii_tree(self._tree.root))

        # Tick timer at BT_TICK_HZ
        self._tick_timer = self.create_timer(
            1.0 / BT_TICK_HZ, self._tick)

        self.get_logger().info(
            f'SurgicalBTNode started -- ticking at {BT_TICK_HZ} Hz')

    def _tick(self):
        """Tick the behaviour tree once."""
        self._tree.tick()

        # -- Publish console feedback -----------------------------------------
        # Detect active phase from tree leaf statuses
        leaves = self._tree.root.iterate()
        for leaf in leaves:
            if leaf.status == py_trees.common.Status.RUNNING:
                if leaf.name == 'Approach':
                    self._current_phase = 'APPROACH'
                    if hasattr(leaf, 'current_step'):
                        self._current_step = leaf.current_step
                        self._current_dist = leaf.current_distance
                        self._current_max = 400
                elif leaf.name == 'Retract':
                    self._current_phase = 'RETRACT'
                    if hasattr(leaf, 'current_step'):
                        self._current_step = leaf.current_step
                        self._current_dist = leaf.current_distance
                        self._current_max = 300
                elif leaf.name == 'Hold':
                    self._current_phase = 'HOLD'
                    if hasattr(leaf, 'current_step'):
                        self._current_step = leaf.current_step
                        self._current_dist = leaf.current_distance
                        self._current_max = 200
                        
        root_status = self._tree.root.status
        if root_status == py_trees.common.Status.RUNNING:
            self._bt_state = 'RUNNING'
        elif root_status == py_trees.common.Status.SUCCESS:
            self._bt_state = 'SUCCESS'
            self._current_phase = 'COMPLETE'
        elif root_status == py_trees.common.Status.FAILURE:
            self._bt_state = 'FAILED'

        feedback = String()
        feedback.data = (
            f'{self._current_phase}|'
            f'{self._current_step}|'
            f'{self._current_max}|'
            f'{self._current_dist:.1f}|'
            f'{self._bt_state}')
        self._pub_console.publish(feedback)


        # Check root status
        root_status = self._tree.root.status
        if root_status == py_trees.common.Status.SUCCESS:
            self.get_logger().info(
                'Surgical procedure complete -- SUCCESS')
            self._tick_timer.cancel()
        elif root_status == py_trees.common.Status.FAILURE:
            self.get_logger().error(
                'Surgical procedure FAILED -- check logs')
            self._tick_timer.cancel()

    def destroy_node(self):
        self._tree.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SurgicalBTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)