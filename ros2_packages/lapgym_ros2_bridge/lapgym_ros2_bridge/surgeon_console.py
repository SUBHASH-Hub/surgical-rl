"""
Phase 4E: SurgeonConsole -- terminal dashboard with keyboard control.

Replicates the physical surgeon console interface of real surgical robots.
Displays live system status and accepts keyboard commands.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │         SURGICAL ROBOT CONSOLE  v1.0                │
  │─────────────────────────────────────────────────────│
  │  PHASE:    RETRACT          STEP:  045 / 300        │
  │  DISTANCE: 38.3mm           FORCE: 0.021            │
  │  WATCHDOG: ● NOMINAL        ESTOP: CLEAR            │
  │  BT STATE: RUNNING                                  │
  │─────────────────────────────────────────────────────│
  │  [S] STOP    [R] RESUME    [E] EMERGENCY    [Q] QUIT│
  └─────────────────────────────────────────────────────┘

Key bindings:
  S -- surgeon stop: publishes /surgeon_stop=True
       BT ForceCondition monitors this and cancels active leaf
       Hold activates automatically via BT sequence
  R -- resume: publishes /surgeon_stop=False to clear stop
  E -- emergency stop: publishes /emergency_stop=True directly
  Q -- quit console (does not stop the surgical system)

Topics subscribed:
  /watchdog_status    (String)  -- NOMINAL | ALERT | STOP
  /watchdog_heartbeat (Bool)    -- watchdog alive indicator
  /tissue_force_proxy (Float32) -- live force reading
  /emergency_stop     (Bool)    -- current stop state
  /console_feedback   (String)  -- phase, step, distance from BT

Topics published:
  /surgeon_stop   (Bool) -- surgeon manual stop signal
  /emergency_stop (Bool) -- direct emergency stop

Author: Subhash Arockiadoss
"""

import os
import curses
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class SurgeonConsole(Node):
    """Terminal dashboard surgeon console node."""

    def __init__(self):
        super().__init__('surgeon_console')

        # -- State ------------------------------------------------------------
        self._watchdog_status = 'NOMINAL'
        self._watchdog_alive = False
        self._force = 0.0
        self._emergency = False
        self._surgeon_stopped = False
        self._phase = 'IDLE'
        self._step = 0
        self._max_steps = 0
        self._distance = 0.0
        self._bt_state = 'WAITING'
        self._last_heartbeat = time.time()
        self._log_messages = []

        # -- Publishers -------------------------------------------------------
        self._pub_surgeon_stop = self.create_publisher(
            Bool, '/surgeon_stop', 10)
        self._pub_emergency = self.create_publisher(
            Bool, '/emergency_stop', 10)
        
        # Republish surgeon stop at 10Hz to avoid race conditions
        self.create_timer(0.1, self._republish_surgeon_stop)

        # -- Subscribers ------------------------------------------------------
        self.create_subscription(
            String, '/watchdog_status', self._cb_watchdog_status, 10)
        self.create_subscription(
            Bool, '/watchdog_heartbeat', self._cb_watchdog_heartbeat, 10)
        self.create_subscription(
            Float32, '/tissue_force_proxy', self._cb_force, 10)
        self.create_subscription(
            Bool, '/emergency_stop', self._cb_emergency, 10)
        self.create_subscription(
            String, '/console_feedback', self._cb_feedback, 10)

        self.get_logger().info('SurgeonConsole started')

    # -- Callbacks ------------------------------------------------------------

    def _cb_watchdog_status(self, msg: String):
        self._watchdog_status = msg.data

    def _cb_watchdog_heartbeat(self, msg: Bool):
        self._watchdog_alive = msg.data
        self._last_heartbeat = time.time()

    def _cb_force(self, msg: Float32):
        self._force = float(msg.data)

    def _cb_emergency(self, msg: Bool):
        self._emergency = msg.data

    def _cb_feedback(self, msg: String):
        """Parse feedback string: PHASE|STEP|MAX_STEPS|DISTANCE|BT_STATE"""
        try:
            parts = msg.data.split('|')
            if len(parts) >= 5:
                self._phase = parts[0]
                self._step = int(parts[1])
                self._max_steps = int(parts[2])
                self._distance = float(parts[3])
                self._bt_state = parts[4]
        except Exception:
            pass

    # -- Commands -------------------------------------------------------------

    def surgeon_stop(self):
        self._surgeon_stopped = True
        msg = Bool()
        msg.data = True
        self._pub_surgeon_stop.publish(msg)
        self._add_log(
            f'[S] STOP  phase={self._phase} '
            f'step={self._step:03d} dist={self._distance:.1f}mm')

    def surgeon_resume(self):
        if self._emergency:
            self._add_log('[R] BLOCKED -- emergency active, restart required')
            return
        self._surgeon_stopped = False
        msg = Bool()
        msg.data = False
        self._pub_surgeon_stop.publish(msg)
        self._add_log(
            f'[R] RESUME phase={self._phase} '
            f'step={self._step:03d} dist={self._distance:.1f}mm')

    def emergency_stop(self):
        self._emergency = True
        msg = Bool()
        msg.data = True
        self._pub_emergency.publish(msg)
        self._add_log(
            f'[E] ESTOP  phase={self._phase} '
            f'step={self._step:03d} dist={self._distance:.1f}mm')
        
    def _republish_surgeon_stop(self):
        if self._surgeon_stopped:
            msg = Bool()
            msg.data = True
            self._pub_surgeon_stop.publish(msg)

    def _add_log(self, message: str):
        timestamp = time.strftime('%H:%M:%S')
        self._log_messages.append(f'[{timestamp}] {message}')
        


def draw_console(stdscr, node: SurgeonConsole):
    """Curses draw loop -- updates display at 10Hz."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # 10Hz refresh

    # Colours
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN,  curses.COLOR_BLACK)  # nominal
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # alert
    curses.init_pair(3, curses.COLOR_RED,    curses.COLOR_BLACK)  # stop/error
    curses.init_pair(4, curses.COLOR_CYAN,   curses.COLOR_BLACK)  # header
    curses.init_pair(5, curses.COLOR_WHITE,  curses.COLOR_BLACK)  # normal

    GREEN  = curses.color_pair(1)
    YELLOW = curses.color_pair(2)
    RED    = curses.color_pair(3)
    CYAN   = curses.color_pair(4)
    WHITE  = curses.color_pair(5)

    WIDTH = 60

    while rclpy.ok():
        try:
            stdscr.erase()
            row = 0

            # -- Header -------------------------------------------------------
            stdscr.addstr(row, 0, '=' * WIDTH, CYAN)
            row += 1
            title = 'SURGICAL ROBOT CONSOLE  v1.0'
            stdscr.addstr(row, (WIDTH - len(title)) // 2, title,
                          CYAN | curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, '=' * WIDTH, CYAN)
            row += 1

            # -- Phase and step -----------------------------------------------
            phase_str = f'PHASE:    {node._phase:<12}'
            step_str  = f'STEP:  {node._step:03d} / {node._max_steps:03d}'
            stdscr.addstr(row, 2,  phase_str, WHITE)
            stdscr.addstr(row, 32, step_str,  WHITE)
            row += 1

            # -- Distance and force -------------------------------------------
            dist_str  = f'DISTANCE: {node._distance:6.1f}mm'
            force_val = node._force
            force_col = GREEN if force_val < 0.35 else (
                        YELLOW if force_val < 1.0 else RED)
            stdscr.addstr(row, 2,  dist_str, WHITE)
            stdscr.addstr(row, 32, f'FORCE:  {force_val:.3f}', force_col)
            row += 1

            # -- Watchdog status ----------------------------------------------
            wd_status = node._watchdog_status
            heartbeat_age = time.time() - node._last_heartbeat
            wd_alive = heartbeat_age < 3.0
            wd_col = GREEN if wd_status == 'NOMINAL' else (
                     YELLOW if wd_status == 'ALERT' else RED)
            wd_dot = '●' if wd_alive else '○'
            wd_str = f'WATCHDOG: {wd_dot} {wd_status:<8}'
            estop_col = RED if node._emergency else GREEN
            estop_str = 'ESTOP: ACTIVE' if node._emergency else 'ESTOP: CLEAR'
            stdscr.addstr(row, 2,  wd_str,   wd_col)
            stdscr.addstr(row, 32, estop_str, estop_col)
            row += 1

            # -- BT state -----------------------------------------------------
            bt_col = GREEN if node._bt_state == 'RUNNING' else (
                     RED if node._bt_state == 'FAILED' else WHITE)
            bt_str = f'BT STATE: {node._bt_state}'
            if node._surgeon_stopped:
                bt_str += '  [SURGEON STOPPED]'
            stdscr.addstr(row, 2, bt_str, bt_col)
            row += 1

            # -- Divider ------------------------------------------------------
            stdscr.addstr(row, 0, '-' * WIDTH, CYAN)
            row += 1

            # -- Controls -----------------------------------------------------
            controls = '[S] STOP    [R] RESUME    [E] EMERGENCY    [Q] QUIT'
            stdscr.addstr(row, (WIDTH - len(controls)) // 2,
                          controls, WHITE | curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, '=' * WIDTH, CYAN)
            row += 1

            # -- Log ----------------------------------------------------------
            stdscr.addstr(row, 2, 'Event log:', WHITE)
            row += 1
            max_rows = curses.LINES - row - 1
            visible = node._log_messages[-max_rows:] if max_rows > 0 else []
            for msg in visible:
                if row >= curses.LINES - 1:
                    break
                stdscr.addstr(row, 2, msg[:WIDTH - 4], YELLOW)
                row += 1

            stdscr.refresh()

            # -- Keyboard input -----------------------------------------------
            key = stdscr.getch()
            if key in (ord('s'), ord('S')):
                node.surgeon_stop()
            elif key in (ord('r'), ord('R')):
                node.surgeon_resume()
            elif key in (ord('e'), ord('E')):
                node.emergency_stop()
            elif key in (ord('q'), ord('Q')):
                break

            # -- Spin ROS 2 callbacks -----------------------------------------
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0)
        except curses.error:
            pass

    return


def main(args=None):
    rclpy.init(args=args)
    node = SurgeonConsole()

    try:
        curses.wrapper(draw_console, node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)