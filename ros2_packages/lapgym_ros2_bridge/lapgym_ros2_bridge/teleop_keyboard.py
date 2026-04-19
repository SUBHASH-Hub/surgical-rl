"""
Phase 4A: Keyboard teleoperation node.
Uses pynput for reliable cross-terminal keyboard capture.

Key map:
  W/S  -> +/- Y axis (forward/back)
  A/D  -> +/- X axis (left/right)
  Q/E  -> +/- Z axis (up/down)
  Space -> publish zero (hold position)
  Esc  -> emergency stop + exit

Author: Subhash Arockiadoss
"""

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from pynput import keyboard

STEP = 1.0

BINDINGS = {
    'w': ( 0.0,  STEP,  0.0),
    's': ( 0.0, -STEP,  0.0),
    'a': (-STEP,  0.0,  0.0),
    'd': ( STEP,  0.0,  0.0),
    'q': ( 0.0,  0.0,  STEP),
    'e': ( 0.0,  0.0, -STEP),
}

HELP = """
-------- LapGym Keyboard Teleop --------
  W/S  : +/- Y  (forward / back)
  A/D  : +/- X  (left / right)
  Q/E  : +/- Z  (up / down)
  Space: hold    (publish zero delta)
  Esc  : emergency stop + quit
  Ctrl-C: quit
----------------------------------------
"""


class TeleopKeyboardNode(Node):

    PUBLISH_HZ = 20

    def __init__(self):
        super().__init__('teleop_keyboard_node')

        self._pub = self.create_publisher(
            Vector3, '/joint_target', 10)
        self._pub_estop = self.create_publisher(
            Bool, '/emergency_stop', 10)

        self._delta = (0.0, 0.0, 0.0)
        self._running = True

        # pynput listener runs in its own thread
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

        self._timer = self.create_timer(
            1.0 / self.PUBLISH_HZ, self._publish)

        print(HELP)
        self.get_logger().info(
            f'TeleopKeyboardNode started at {self.PUBLISH_HZ} Hz')

    def _on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            ch = None

        if key == keyboard.Key.esc:
            estop = Bool()
            estop.data = True
            self._pub_estop.publish(estop)
            self.get_logger().warn('Emergency stop sent -- exiting teleop')
            self._running = False
            rclpy.shutdown()
            os._exit(0)

        if ch in BINDINGS:
            self._delta = BINDINGS[ch]
        else:
            self._delta = (0.0, 0.0, 0.0)

    def _publish(self):
        if not self._running:
            return
        msg = Vector3()
        msg.x = float(self._delta[0])
        msg.y = float(self._delta[1])
        msg.z = float(self._delta[2])
        self._pub.publish(msg)

        if any(d != 0.0 for d in self._delta):
            self.get_logger().info(
                f'cmd x={msg.x:+.1f} y={msg.y:+.1f} z={msg.z:+.1f}',
                throttle_duration_sec=0.1)

        # reset after publishing so one keypress = one delta pulse
        self._delta = (0.0, 0.0, 0.0)

    def destroy_node(self):
        self._listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)