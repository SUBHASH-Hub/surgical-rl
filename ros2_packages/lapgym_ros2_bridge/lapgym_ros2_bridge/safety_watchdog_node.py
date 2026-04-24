"""
Phase 4D: SafetyWatchdogNode -- independent tissue force safety monitor.

Runs as a completely independent ROS 2 process (separate PID) from the
Behaviour Tree and action servers. Implements IEC 62304 Class B safety
software architectural independence requirement.

Three-tier alert system:
  NOMINAL : force < ALERT_THRESHOLD (0.35)  -- normal operation
  ALERT   : force >= ALERT_THRESHOLD         -- warning, log only
  STOP    : force >= STOP_THRESHOLD (1.0)    -- publish /emergency_stop

Publishes:
  /emergency_stop     (std_msgs/Bool)    -- True triggers system halt
  /watchdog_status    (std_msgs/String)  -- NOMINAL | ALERT | STOP
  /watchdog_heartbeat (std_msgs/Bool)    -- True every 1s, proves alive

Subscribes:
  /tissue_force_proxy (std_msgs/Float32) -- optical flow force estimate

Check rate: 50 Hz (independent of BT tick rate)
Consecutive readings before STOP: 3 (60ms at 50Hz, avoids noise triggers)

Author: Subhash Arockiadoss
"""

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


ALERT_THRESHOLD = 0.35   # yellow warning -- log and publish ALERT status
STOP_THRESHOLD  = 1.0    # red stop -- publish /emergency_stop after N readings
CONSECUTIVE_TO_STOP = 3  # 3 readings at 50Hz = 60ms sustained force
CHECK_HZ = 50            # independent check rate
HEARTBEAT_HZ = 1         # heartbeat publish rate


class SafetyWatchdogNode(Node):
    """
    Independent safety watchdog -- separate process from BT and action servers.

    This node has no knowledge of the BT, PPO policy, or action servers.
    It only knows about force and the emergency stop topic.
    This is the IEC 62304 architectural independence requirement.
    """

    def __init__(self):
        super().__init__('safety_watchdog_node')

        # -- State ------------------------------------------------------------
        self._current_force = 0.0
        self._state = 'NOMINAL'
        self._consecutive_stop = 0
        self._emergency_sent = False
        self._total_alerts = 0
        self._total_stops = 0
        self._start_time = self.get_clock().now()

        # -- Publishers -------------------------------------------------------
        self._pub_estop = self.create_publisher(
            Bool, '/emergency_stop', 10)
        self._pub_status = self.create_publisher(
            String, '/watchdog_status', 10)
        self._pub_heartbeat = self.create_publisher(
            Bool, '/watchdog_heartbeat', 10)

        # -- Subscriber -------------------------------------------------------
        self.create_subscription(
            Float32,
            '/tissue_force_proxy',
            self._force_cb,
            10)

        # -- Timers -----------------------------------------------------------
        self.create_timer(1.0 / CHECK_HZ, self._check)
        self.create_timer(1.0 / HEARTBEAT_HZ, self._heartbeat)

        self.get_logger().info(
            f'SafetyWatchdogNode started -- '
            f'ALERT={ALERT_THRESHOLD} STOP={STOP_THRESHOLD} '
            f'at {CHECK_HZ}Hz')
        self.get_logger().info(
            'IEC 62304 independent safety layer ACTIVE')

    # -- Force callback -------------------------------------------------------

    def _force_cb(self, msg: Float32):
        self._current_force = float(msg.data)

    # -- Main safety check ----------------------------------------------------

    def _check(self):
        """50 Hz safety check -- the core watchdog loop."""
        force = self._current_force

        if force >= STOP_THRESHOLD:
            self._consecutive_stop += 1

            if self._consecutive_stop >= CONSECUTIVE_TO_STOP:
                self._state = 'STOP'
                if not self._emergency_sent:
                    # Only trigger once per sustained event
                    self._total_stops += 1
                    self._trigger_emergency_stop(force)

        elif force >= ALERT_THRESHOLD:
            self._consecutive_stop = 0
            # Only reset emergency_sent when force drops below STOP threshold
            # but stays in ALERT -- do NOT reset here
            self._total_alerts += 1

            if self._state != 'ALERT' and self._state != 'STOP':
                self._state = 'ALERT'
                self.get_logger().warn(
                    f'[WATCHDOG ALERT] force={force:.3f} '
                    f'>= threshold={ALERT_THRESHOLD}')

        else:
            # Force fully normalised below ALERT threshold
            if self._state == 'STOP':
                self.get_logger().info(
                    f'[WATCHDOG] Force normalised: {force:.3f} '
                    f'-- returning to NOMINAL')
            self._consecutive_stop = 0
            self._emergency_sent = False  # safe to reset only when truly NOMINAL
            self._state = 'NOMINAL'

        # Publish status every check
        status_msg = String()
        status_msg.data = self._state
        self._pub_status.publish(status_msg)

    # -- Emergency stop trigger -----------------------------------------------

    def _trigger_emergency_stop(self, force: float):
        """Publish emergency stop -- called only when STOP threshold sustained."""
        self._emergency_sent = True

        estop = Bool()
        estop.data = True
        self._pub_estop.publish(estop)

        self.get_logger().error(
            f'[WATCHDOG EMERGENCY STOP] '
            f'force={force:.3f} >= {STOP_THRESHOLD} '
            f'for {CONSECUTIVE_TO_STOP} consecutive readings '
            f'({CONSECUTIVE_TO_STOP / CHECK_HZ * 1000:.0f}ms) '
            f'-- /emergency_stop published')

    # -- Heartbeat ------------------------------------------------------------

    def _heartbeat(self):
        """Publish heartbeat every second -- proves watchdog process is alive."""
        hb = Bool()
        hb.data = True
        self._pub_heartbeat.publish(hb)

        # Log status summary every 10 seconds
        uptime = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        if int(uptime) % 10 == 0 and int(uptime) > 0:
            self.get_logger().info(
                f'[WATCHDOG] uptime={uptime:.0f}s '
                f'state={self._state} '
                f'force={self._current_force:.3f} '
                f'alerts={self._total_alerts} '
                f'stops={self._total_stops}')


def main(args=None):
    rclpy.init(args=args)
    node = SafetyWatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)