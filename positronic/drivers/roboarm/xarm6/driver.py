import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.xarm6.calibration import get_xarm_params_from_arm
from positronic.drivers.roboarm.xarm6.kinematics import DEFAULT_XARM6_URDF_PATH, XArm6IKSolver

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    from xarm import XArmAPI

_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
_log = logging.getLogger('xarm6')


class XArm6State(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(6 + 6 + 7 + 1,), dtype=np.float32)

    def instantiation_params(self):
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[:6].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[6:12].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        return geom.Transform3D(
            self.array[12 : 12 + 3].copy(), geom.Rotation.from_quat(self.array[12 + 3 : 12 + 7].copy())
        )

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[19]))

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D, status: RobotStatus):
        self.array[:6] = q
        self.array[6:12] = dq
        self.array[12 : 12 + 3] = ee_pose.translation
        self.array[12 + 3 : 12 + 7] = ee_pose.rotation.as_quat
        self.array[19] = status.value


class Robot(pimm.ControlSystem):
    def __init__(
        self,
        ip_address: str = '192.168.1.233',
        *,
        urdf_path: str | Path = DEFAULT_XARM6_URDF_PATH,
        use_robot_calibration: bool = True,
        speed: float = 50.0,
        acceleration: float = 1100.0,
        command_hz: float = 100.0,
        max_joint_step_rad: float = 0.05,
        home_joints: list[float] | None = None,
        has_gripper: bool = True,
        gripper_model: str = 'hand-e',
        gripper_open_width: int = 190,
        gripper_closed_width: int = 150,
        gripper_force: int = 255,
        gripper_speed: int = 100,
    ):
        self.ip_address = ip_address
        self.urdf_path = Path(urdf_path).expanduser()
        self.use_robot_calibration = use_robot_calibration
        self.speed = speed
        self.acceleration = acceleration
        self.command_hz = command_hz
        self.max_joint_step_rad = max_joint_step_rad
        self.home_joints = np.asarray(home_joints if home_joints is not None else [0.0, -0.6, -0.6, 0.0, 1.2, 0.0])
        self.has_gripper = has_gripper
        self.gripper_model = gripper_model.lower()
        self.gripper_open_width = gripper_open_width
        self.gripper_closed_width = gripper_closed_width
        self.gripper_force = gripper_force
        self.gripper_speed = gripper_speed
        self._last_q: np.ndarray | None = None
        self._last_grip = 0.0

        self.commands: pimm.SignalReceiver[roboarm_command.CommandType] = pimm.ControlSystemReceiver(self, default=None)
        self.target_grip: pimm.SignalReceiver[float] = pimm.ControlSystemReceiver(self, default=0.0)

        self.state: pimm.SignalEmitter[XArm6State] = pimm.ControlSystemEmitter(self)
        self.grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        _log.info('Connecting to xArm6 at %s', self.ip_address)
        arm = XArmAPI(self.ip_address)
        arm.connect()
        joint_params = get_xarm_params_from_arm(self.ip_address) if self.use_robot_calibration else None
        solver = XArm6IKSolver(self.urdf_path, joint_params=joint_params)
        self._enable_joint_servo_mode(arm)
        self._init_gripper(arm)
        self.robot_meta.emit({
            'urdf': self.urdf_path.read_text(),
            'joint_names': _JOINT_NAMES,
            'control_frame': 'link6',
            'uses_robot_calibration': joint_params is not None,
            'has_gripper': self.has_gripper,
            'gripper_model': self.gripper_model,
        })

        rate_limiter = pimm.RateLimiter(clock, hz=self.command_hz)
        state = XArm6State()
        last_q = self._read_joints(arm)
        last_ts = clock.now_ns()
        self._last_q = last_q

        try:
            while not should_stop.value:
                cmd_msg = self.commands.read()
                if cmd_msg.updated:
                    match cmd_msg.data:
                        case roboarm_command.Reset():
                            self._movej(arm, self.home_joints)
                            self._last_q = self.home_joints.copy()
                        case roboarm_command.Recover():
                            self._enable_joint_servo_mode(arm)
                        case roboarm_command.CartesianPosition(pose):
                            self._send_cartesian(arm, solver, pose)
                        case roboarm_command.JointPosition(positions):
                            self._send_joints(arm, np.asarray(positions, dtype=np.float32))
                        case roboarm_command.JointDelta(velocities=delta):
                            self._send_joints(arm, self._read_joints(arm) + delta)
                        case _:
                            raise ValueError(f'Unknown command: {cmd_msg.data}')

                grip_msg = self.target_grip.read()
                if grip_msg.updated:
                    self._send_gripper(arm, grip_msg.data)

                now = clock.now_ns()
                q = self._read_joints(arm)
                dq = self._estimate_velocity(q, last_q, now, last_ts)
                ee_pose = solver.fk_to_transform(q)
                status = self._classify_status(arm)

                state.encode(q, dq, ee_pose, status)
                self.state.emit(state)
                self.grip.emit(self._last_grip)
                last_q = q
                last_ts = now
                self._last_q = q
                yield pimm.Sleep(rate_limiter.wait_time())
        finally:
            arm.disconnect()

    def _enable_joint_servo_mode(self, arm: XArmAPI):
        arm.motion_enable(True)
        arm.clean_error()
        arm.set_mode(1)
        time.sleep(0.2)
        arm.set_state(0)

    def _init_gripper(self, arm: XArmAPI):
        if not self.has_gripper:
            return
        if self.gripper_model == 'pgc':
            frames = [[0x01, 0x06, 0x01, 0x00, 0x00, 0xA5]]
        else:
            frames = [
                [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00],
            ]
        for frame in frames:
            arm.core.tgpio_set_modbus(frame, len(frame))
            time.sleep(1.0)
        self._send_gripper(arm, 0.0)

    def _movej(self, arm: XArmAPI, joints: np.ndarray):
        arm.set_mode(0)
        time.sleep(0.2)
        arm.set_state(0)
        arm.set_servo_angle(
            angle=np.degrees(joints).tolist(), speed=self.speed, mvacc=self.acceleration, wait=True, is_radian=False
        )
        self._enable_joint_servo_mode(arm)

    def _send_cartesian(self, arm: XArmAPI, solver: XArm6IKSolver, pose: geom.Transform3D):
        initial_guess = self._last_q if self._last_q is not None else self._read_joints(arm)
        target_joints = solver.ik_from_transform(pose, initial_guess=initial_guess)
        if target_joints is None:
            _log.warning(
                'IK failed for target xyz=%s quat_xyzw=%s', pose.translation.tolist(), pose.rotation.as_quat_xyzw
            )
            return
        self._send_joints(arm, self._limit_joint_step(initial_guess, target_joints))

    def _send_joints(self, arm: XArmAPI, joints: np.ndarray):
        arm.set_servo_angle_j(np.degrees(joints).tolist(), is_radian=False)
        self._last_q = joints.copy()

    def _send_gripper(self, arm: XArmAPI, target_grip: float):
        if not self.has_gripper:
            return
        grip = min(1.0, max(0.0, float(target_grip)))
        width = round(self.gripper_open_width + grip * (self.gripper_closed_width - self.gripper_open_width))
        force = min(255, max(0, int(self.gripper_force)))
        speed = min(100, max(1, int(self.gripper_speed)))

        if self.gripper_model == 'pgc':
            pgc_force = max(20, round(force / 255.0 * 100))
            frames = [
                [0x01, 0x06, 0x01, 0x01, 0x00, pgc_force],
                [0x01, 0x06, 0x01, 0x04, 0x00, speed],
                [0x01, 0x06, 0x01, 0x03, width >> 8, width & 0xFF],
            ]
        else:
            frames = [[0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, 0x00, width, 0xFF, force]]

        for frame in frames:
            arm.core.tgpio_set_modbus(frame, len(frame))
            time.sleep(0.01)
        self._last_grip = grip

    def _limit_joint_step(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.max_joint_step_rad <= 0:
            return target
        return current + np.clip(target - current, -self.max_joint_step_rad, self.max_joint_step_rad)

    def _read_joints(self, arm: XArmAPI) -> np.ndarray:
        code, angles = arm.get_servo_angle()
        if code != 0:
            raise RuntimeError(f'xArm get_servo_angle failed with code {code}')
        return np.radians(np.asarray(angles[:6], dtype=np.float32))

    def _estimate_velocity(self, q: np.ndarray, last_q: np.ndarray, now_ns: int, last_ts_ns: int) -> np.ndarray:
        dt = (now_ns - last_ts_ns) / 1e9
        if dt <= 0:
            return np.zeros_like(q)
        return (q - last_q) / dt

    def _classify_status(self, arm: XArmAPI) -> RobotStatus:
        code, errors = arm.get_err_warn_code()
        if code != 0 or errors[0] != 0:
            return RobotStatus.ERROR
        if arm.get_is_moving():
            return RobotStatus.MOVING
        return RobotStatus.AVAILABLE
