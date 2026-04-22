"""
Phase 4A: Keyboard teleoperation node with position HUD.
Uses pynput for reliable cross-terminal keyboard capture.

Key map:
  W/S  -> +/- Y axis (forward/back)
  A/D  -> +/- X axis (left/right)
  Q/E  -> +/- Z axis (up/down)
  Esc  -> emergency stop + exit

Author: Subhash Arockiadoss  
"""

import os
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from pynput import keyboard
# Suppress terminal echo so keypresses don't pollute HUD
import termios
import tty


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
  Esc  : emergency stop + quit
  Ctrl-C: quit
----------------------------------------
"""


class TeleopKeyboardNode(Node):

    PUBLISH_HZ = 20
    HUD_HZ = 2   # HUD refresh rate -- 2 Hz is enough, not spammy

    def __init__(self):
        super().__init__('teleop_keyboard_node')

        self._pub = self.create_publisher(
            Vector3, '/joint_target', 10)
        self._pub_estop = self.create_publisher(
            Bool, '/emergency_stop', 10)

        # Subscribe to bridge topics for HUD
        self.create_subscription(
            JointState, '/joint_states', self._cb_joint_states, 10)
        self.create_subscription(
            JointState, '/guidance', self._cb_guidance, 10)

        self._delta = (0.0, 0.0, 0.0)
        self._running = True

        # HUD state
        self._tool_xyz = [0.0, 0.0, 0.0]
        self._goal_xyz = [0.0, 0.0, 0.0]
        self._distance = 0.0
        self._phase = 0.0
        self._hud_active = False
        self._collision = 0.0
        self._steps_in_collision = 0
        self._tool_world = [0.0, 0.0, 0.0]

        # pynput listener
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

        self._timer = self.create_timer(
            1.0 / self.PUBLISH_HZ, self._publish)
        self._hud_timer = self.create_timer(
            1.0 / self.HUD_HZ, self._print_hud)

        print(HELP)
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.get_logger().info(
            f'TeleopKeyboardNode started at {self.PUBLISH_HZ} Hz')

    # -- Subscribers ----------------------------------------------------------

    def _cb_joint_states(self, msg: JointState):
        if len(msg.position) >= 4:
            self._tool_xyz = list(msg.position[:3])
            self._phase = msg.position[3]

    def _cb_guidance(self, msg: JointState):
        if len(msg.position) >= 6:
            self._goal_xyz = list(msg.position[:3])    # world metres
            self._distance = msg.position[3]            # world metres
            self._collision = msg.position[4] if len(msg.position) >= 5 else 0.0
            self._steps_in_collision = int(msg.position[5]) if len(msg.position) >= 6 else 0
            self._tool_world = list(msg.position[6:9]) if len(msg.position) >= 9 else [0.0, 0.0, 0.0]
            self._hud_active = True
    # -- HUD ------------------------------------------------------------------

    def _print_hud(self):
        if not self._hud_active:
            return

        # World-space positions in mm for readability
        gx, gy, gz = [v * 1000 for v in self._goal_xyz]   # mm
        tx, ty, tz = [v * 1000 for v in self._tool_world]  # mm
        dist = self._distance   # world metres
        phase = 'GRASPING' if self._phase == 0.0 else 'RETRACTING'
        reset = '\033[0m'

        # Distance in mm -- grasping triggers at 3mm
        dist_mm = dist * 1000
        if dist_mm < 5.0:
            dist_colour = '\033[92m'   # green -- grasp imminent
        elif dist_mm < 20.0:
            dist_colour = '\033[93m'   # yellow -- close
        else:
            dist_colour = '\033[91m'   # red -- still navigating

        # Per-axis world-space guidance in mm
        dx = gx - tx
        dy = gy - ty
        dz = gz - tz

        def axis_str(label, delta_mm, key_pos, key_neg):
            if abs(delta_mm) < 3.0:
                return f'\033[92m{label}:OK{reset}'
            elif delta_mm > 0:
                return f'\033[93m{label}:{delta_mm:+.0f}({key_pos}){reset}'
            else:
                return f'\033[93m{label}:{delta_mm:+.0f}({key_neg}){reset}'

        x_str = axis_str('X', dx, 'D', 'A')
        y_str = axis_str('Y', dy, 'W', 'S')
        z_str = axis_str('Z', dz, 'Q', 'E')

        # Collision indicator
        if self._collision > 0.0:
            col_str = f'\033[91mCOL:{self._collision:.2f}(s:{self._steps_in_collision}){reset}'
        else:
            col_str = f'\033[92mSAFE{reset}'

        print(
            f'\r\033[K'
            f'[{phase}] '
            f'G({gx:+.0f},{gy:+.0f},{gz:+.0f}mm) '
            f'T({tx:+.0f},{ty:+.0f},{tz:+.0f}mm) '
            f'D:{dist_colour}{dist_mm:.0f}mm{reset} '
            f'{x_str} {y_str} {z_str} '
            f'{col_str}',
            end='', flush=True)
    # -- Keyboard -------------------------------------------------------------

    def _on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            ch = None

        if key == keyboard.Key.esc:
            estop = Bool()
            estop.data = True
            self._pub_estop.publish(estop)
            print('\nEmergency stop sent -- exiting teleop')
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
        self._delta = (0.0, 0.0, 0.0)

    def destroy_node(self):
        # Restore terminal settings
        import termios
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass
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
        os._exit(0)