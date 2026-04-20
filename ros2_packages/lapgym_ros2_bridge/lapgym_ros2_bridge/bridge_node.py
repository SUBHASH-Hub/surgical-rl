"""
Phase 4A: SOFA -> ROS 2 bridge node.

Publishes:
  /joint_states          (sensor_msgs/JointState)   -- instrument xyz + phase
  /tissue_force_proxy    (std_msgs/Float32)          -- optical-flow force estimate
  /camera/image_raw      (sensor_msgs/Image)         -- 480x480 RGB from SOFA

Subscribes:
  /joint_target          (geometry_msgs/Vector3)     -- xyz delta from teleop
  /emergency_stop        (std_msgs/Bool)             -- safety halt

Spin rate: 50 Hz

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

from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Vector3
from builtin_interfaces.msg import Time

SOFA_AVAILABLE = False
try:
    from stable_baselines3 import PPO
    from envs.tissue_retraction_v2 import TissueRetractionV2
    SOFA_AVAILABLE = True
except ImportError as e:
    print(f"[bridge_node] WARNING: SOFA imports unavailable: {e}")
    print("[bridge_node] Running in stub mode -- topics will publish zeros")

class SofaBridgeNode(Node):
    """ROS 2 node that drives the SOFA simulation and relays state to topics."""

    SPIN_HZ = 50
    DELTA_SCALE = 0.002
    FORCE_ALERT_THRESHOLD = 0.35
    FORCE_STOP_THRESHOLD = 1.0

    def __init__(self):
        super().__init__('sofa_bridge_node')

        # -- Publishers -------------------------------------------------------
        self._pub_joint = self.create_publisher(JointState, '/joint_states', 10)
        self._pub_force = self.create_publisher(Float32, '/tissue_force_proxy', 10)
        self._pub_image = self.create_publisher(Image, '/camera/image_raw', 10)

        # -- Subscribers ------------------------------------------------------
        self.create_subscription(
            Vector3, '/joint_target', self._cb_joint_target, 10)
        self.create_subscription(
            Bool, '/emergency_stop', self._cb_emergency_stop, 10)

        # -- State ------------------------------------------------------------
        self._pending_delta = np.zeros(3, dtype=np.float64)
        self._emergency = False
        self._last_rgb = None
        self._step_count = 0
        self._env = None
        self._obs = None

        # -- SOFA env ---------------------------------------------------------
        if SOFA_AVAILABLE:
            self._init_env()
        else:
            self.get_logger().warning(
                'SOFA not available -- running in stub mode (publishing zeros)')

        # -- 50 Hz timer ------------------------------------------------------
        self._timer = self.create_timer(1.0 / self.SPIN_HZ, self._step)
        self.get_logger().info(f'SofaBridgeNode started at {self.SPIN_HZ} Hz')

    # -- Initialisation -------------------------------------------------------

    def _init_env(self):
        import importlib
        from sofa_env.scenes.tissue_retraction.tissue_retraction_env import RenderMode

        self.declare_parameter('env_class',
                'envs.tissue_retraction_v2.TissueRetractionV2')
        self.declare_parameter('render_mode', 'headless')

        env_class_path = self.get_parameter('env_class').value
        render_mode_str = self.get_parameter('render_mode').value

        # Convert string to RenderMode enum
        render_mode = RenderMode.HUMAN if render_mode_str == 'human' else RenderMode.HEADLESS

        module_path, class_name = env_class_path.rsplit('.', 1)
        mod = importlib.import_module(module_path)
        EnvClass = getattr(mod, class_name)

        self.get_logger().info(
            f'Creating env: {env_class_path} render_mode={render_mode}')
        self._env = EnvClass(env_kwargs={"render_mode": render_mode})
        obs, _ = self._env.reset()
        self._obs = obs
        self.get_logger().info('SOFA env reset -- bridge ready')
    
    # -- Callbacks ------------------------------------------------------------

    def _cb_joint_target(self, msg: Vector3):
        self._pending_delta += np.array([msg.x, msg.y, msg.z])

    def _cb_emergency_stop(self, msg: Bool):
        if msg.data:
            self.get_logger().error('EMERGENCY STOP received')
        self._emergency = msg.data

    # -- Main loop ------------------------------------------------------------

    def _step(self):
        if self._emergency:
            return
        now = self.get_clock().now().to_msg()
        if self._env is not None:
            self._step_with_env(now)
        else:
            self._step_stub(now)

    def _step_with_env(self, now: Time):
        action = self._pending_delta.copy()
        self._pending_delta[:] = 0.0

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._obs = obs
        self._step_count += 1

        tool_xyz = obs[:3].tolist()
        phase = float(obs[6]) if len(obs) > 6 else 0.0

        # /joint_states
        js = JointState()
        js.header.stamp = now
        js.name = ['instrument_x', 'instrument_y', 'instrument_z', 'phase']
        js.position = tool_xyz + [phase]
        self._pub_joint.publish(js)

        # /tissue_force_proxy
        force_val = self._compute_force_proxy(info)
        fm = Float32()
        fm.data = float(force_val)
        self._pub_force.publish(fm)

        if force_val >= self.FORCE_STOP_THRESHOLD:
            self.get_logger().error(
                f'Force {force_val:.3f} >= safety stop {self.FORCE_STOP_THRESHOLD}'
                ' -- triggering emergency stop')
            self._emergency = True

        # /camera/image_raw
        rgb = info.get('rgb_frame')
        if rgb is not None:
            self._pub_image.publish(self._rgb_to_image_msg(rgb, now))

        if terminated:
            self.get_logger().info(
                f'Task terminated at step {self._step_count} -- resetting')
            obs, _ = self._env.reset()
            self._obs = obs
            self._step_count = 0
        elif truncated:
            self.get_logger().info(
                f'Step limit reached at {self._step_count} -- continuing teleop')
            self._step_count = 0  # reset counter only, environment keeps running

    def _step_stub(self, now: Time):
        js = JointState()
        js.header.stamp = now
        js.name = ['instrument_x', 'instrument_y', 'instrument_z', 'phase']
        js.position = [0.0, 0.0, 0.0, 0.0]
        self._pub_joint.publish(js)
        fm = Float32()
        fm.data = 0.0
        self._pub_force.publish(fm)

    # -- Helpers --------------------------------------------------------------

    def _compute_force_proxy(self, info: dict) -> float:
        import cv2
        rgb = info.get('rgb_frame')
        if rgb is None:
            return 0.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self._last_rgb is None:
            self._last_rgb = gray.copy()
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._last_rgb, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        self._last_rgb = gray.copy()   # .copy() prevents aliasing bug
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.mean(mag))

    @staticmethod
    def _rgb_to_image_msg(rgb: np.ndarray, stamp: Time) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'sofa_camera'
        msg.height = rgb.shape[0]
        msg.width = rgb.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = rgb.shape[1] * 3
        msg.data = rgb.tobytes()
        return msg

    # -- Cleanup --------------------------------------------------------------

    def destroy_node(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SofaBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)