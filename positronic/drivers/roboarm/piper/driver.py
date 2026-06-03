import logging
import time
from collections.abc import Iterator

import numpy as np
from piper_sdk import C_PiperInterface_V2

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command

_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
_METERS_TO_PIPER_POSITION = 1_000_000.0
_RADIANS_TO_PIPER_ANGLE = 180_000.0 / np.pi
_PIPER_POSITION_TO_METERS = 1.0 / _METERS_TO_PIPER_POSITION
_PIPER_ANGLE_TO_RADIANS = np.pi / 180_000.0

_log = logging.getLogger('piper')


def _err_status_to_dict(err_status) -> dict:
    return {k: v for k, v in vars(err_status).items() if v}


def _arm_status_snapshot(arm_status) -> dict:
    return {
        'ctrl_mode': str(arm_status.ctrl_mode),
        'arm_status': str(arm_status.arm_status),
        'mode_feed': str(arm_status.mode_feed),
        'motion_status': str(arm_status.motion_status),
        'err_code': f'0x{int(arm_status.err_code):04X}',
        'err_flags': _err_status_to_dict(arm_status.err_status),
    }


class PiperState(State, pimm.shared_memory.NumpySMAdapter):
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
        can_name: str = 'can0',
        *,
        judge_flag: bool = True,
        can_auto_init: bool = True,
        dh_is_offset: int = 0x01,
        start_sdk_joint_limit: bool = True,
        start_sdk_gripper_limit: bool = True,
        speed: int = 100,
        gripper_effort: int = 1000,
        gripper_range_m: float = 0.08,
        cartesian_rotation_mode: str = 'command',
        fixed_cartesian_rpy_deg: list[float] | None = None,
        max_cartesian_step_m: float = 0.01,
        home_joints: list[float] | None = None,
    ):
        self.can_name = can_name
        self.judge_flag = judge_flag
        self.can_auto_init = can_auto_init
        self.dh_is_offset = dh_is_offset
        self.start_sdk_joint_limit = start_sdk_joint_limit
        self.start_sdk_gripper_limit = start_sdk_gripper_limit
        self.speed = speed
        self.gripper_effort = gripper_effort
        self.gripper_range_m = gripper_range_m
        self.cartesian_rotation_mode = cartesian_rotation_mode
        self.fixed_cartesian_rpy = np.deg2rad(fixed_cartesian_rpy_deg or [0.0, 85.0, 0.0])
        self.max_cartesian_step_m = max_cartesian_step_m
        self.home_joints = home_joints if home_joints is not None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._last_cartesian_target: geom.Transform3D | None = None

        self.commands: pimm.SignalReceiver[roboarm_command.CommandType] = pimm.ControlSystemReceiver(self, default=None)
        self.target_grip: pimm.SignalReceiver[float] = pimm.ControlSystemReceiver(self, default=0.0)

        self.state: pimm.SignalEmitter[PiperState] = pimm.ControlSystemEmitter(self)
        self.grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        _log.info('Connecting to CAN %s (speed=%d%%)', self.can_name, self.speed)
        piper = C_PiperInterface_V2(
            self.can_name,
            self.judge_flag,
            self.can_auto_init,
            self.dh_is_offset,
            self.start_sdk_joint_limit,
            self.start_sdk_gripper_limit,
        )
        piper.ConnectPort()
        _log.info('Enabling arm...')
        while not should_stop.value and not piper.EnablePiper():
            yield pimm.Sleep(0.01)
        _log.info('Arm enabled. enable_status=%s', piper.GetArmEnableStatus())
        piper.GripperCtrl(0, self.gripper_effort, 0x01, 0)
        self.robot_meta.emit({'joint_names': _JOINT_NAMES, 'control_frame': 'end_effector'})

        rate_limiter = pimm.RateLimiter(clock, hz=200)
        state = PiperState()
        last_q = self._read_joints(piper)
        last_ts = clock.now_ns()
        last_status_snapshot: dict | None = None

        try:
            while not should_stop.value:
                cmd_msg = self.commands.read()
                if cmd_msg.updated:
                    match cmd_msg.data:
                        case roboarm_command.Reset():
                            _log.info('CMD Reset -> home joints=%s', self.home_joints)
                            self._send_joints(piper, np.asarray(self.home_joints, dtype=np.float32))
                        case roboarm_command.Recover():
                            _log.warning('CMD Recover -> EnablePiper')
                            while not should_stop.value and not piper.EnablePiper():
                                yield pimm.Sleep(0.01)
                            _log.info('Recover: arm re-enabled. enable_status=%s', piper.GetArmEnableStatus())
                        case roboarm_command.CartesianPosition(pose):
                            self._send_cartesian(piper, pose)
                        case roboarm_command.JointPosition(positions):
                            self._send_joints(piper, np.asarray(positions, dtype=np.float32))
                        case _:
                            raise ValueError(f'Unknown command: {cmd_msg.data}')

                grip_msg = self.target_grip.read()
                if grip_msg.updated:
                    self._send_gripper(piper, grip_msg.data)

                now = clock.now_ns()
                q = self._read_joints(piper)
                dq = self._estimate_velocity(q, last_q, now, last_ts)
                ee_pose = self._read_ee_pose(piper)
                raw_arm_status = piper.GetArmStatus().arm_status
                status = self._classify_status(raw_arm_status)
                snapshot = _arm_status_snapshot(raw_arm_status)
                if snapshot != last_status_snapshot:
                    _log.info('arm status change: %s -> %s (-> %s)', last_status_snapshot, snapshot, status.name)
                    last_status_snapshot = snapshot
                grip = self._read_gripper(piper)

                state.encode(q, dq, ee_pose, status)
                self.state.emit(state)
                self.grip.emit(grip)
                last_q = q
                last_ts = now

                yield pimm.Sleep(rate_limiter.wait_time())
        finally:
            piper.DisconnectPort()

    def _send_cartesian(self, piper: C_PiperInterface_V2, pose: geom.Transform3D):
        pose = self._prepare_cartesian_pose(piper, pose)
        xyz = np.rint(pose.translation * _METERS_TO_PIPER_POSITION).astype(int)
        rpy = np.rint(pose.rotation.as_euler * _RADIANS_TO_PIPER_ANGLE).astype(int)
        _log.debug(
            'CMD CartesianPosition xyz_m=%s rpy_rad=%s -> piper xyz=%s rpy=%s',
            pose.translation.tolist(),
            pose.rotation.as_euler.tolist(),
            xyz.tolist(),
            rpy.tolist(),
        )
        piper.MotionCtrl_2(0x01, 0x00, self.speed, 0x00)
        piper.EndPoseCtrl(int(xyz[0]), int(xyz[1]), int(xyz[2]), int(rpy[0]), int(rpy[1]), int(rpy[2]))
        self._last_cartesian_target = pose

    def _prepare_cartesian_pose(self, piper: C_PiperInterface_V2, pose: geom.Transform3D) -> geom.Transform3D:
        if self._last_cartesian_target is None:
            self._last_cartesian_target = self._read_ee_pose(piper)

        translation = self._limit_translation_step(self._last_cartesian_target.translation, pose.translation)
        match self.cartesian_rotation_mode:
            case 'command':
                rotation = pose.rotation
            case 'current':
                rotation = self._read_ee_pose(piper).rotation
            case 'fixed':
                rotation = geom.Rotation.from_euler(self.fixed_cartesian_rpy)
            case _:
                raise ValueError(f'Unknown cartesian_rotation_mode={self.cartesian_rotation_mode!r}')

        return geom.Transform3D(translation, rotation)

    def _limit_translation_step(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.max_cartesian_step_m <= 0:
            return target

        delta = target - start
        distance = np.linalg.norm(delta)
        if distance <= self.max_cartesian_step_m:
            return target
        return start + delta / distance * self.max_cartesian_step_m

    def _send_joints(self, piper: C_PiperInterface_V2, q: np.ndarray):
        joints = np.rint(q * _RADIANS_TO_PIPER_ANGLE).astype(int)
        _log.info('CMD JointPosition rad=%s -> piper=%s', q.tolist(), joints.tolist())
        piper.MotionCtrl_2(0x01, 0x01, self.speed, 0x00)
        piper.JointCtrl(int(joints[0]), int(joints[1]), int(joints[2]), int(joints[3]), int(joints[4]), int(joints[5]))

    def _send_gripper(self, piper: C_PiperInterface_V2, target_grip: float):
        grip = min(1.0, max(0.0, float(target_grip)))
        gripper_angle = round(grip * self.gripper_range_m * _METERS_TO_PIPER_POSITION)
        piper.GripperCtrl(abs(gripper_angle), self.gripper_effort, 0x01, 0)

    def _read_ee_pose(self, piper: C_PiperInterface_V2) -> geom.Transform3D:
        pose = piper.GetArmEndPoseMsgs().end_pose
        translation = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float32) * _PIPER_POSITION_TO_METERS
        rotation = geom.Rotation.from_euler(
            np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float32) * _PIPER_ANGLE_TO_RADIANS
        )
        return geom.Transform3D(translation, rotation)

    def _read_joints(self, piper: C_PiperInterface_V2) -> np.ndarray:
        joints = piper.GetArmJointMsgs().joint_state
        return (
            np.array(
                [joints.joint_1, joints.joint_2, joints.joint_3, joints.joint_4, joints.joint_5, joints.joint_6],
                dtype=np.float32,
            )
            * _PIPER_ANGLE_TO_RADIANS
        )

    def _read_gripper(self, piper: C_PiperInterface_V2) -> float:
        gripper = piper.GetArmGripperMsgs().gripper_state
        return min(1.0, max(0.0, gripper.grippers_angle * _PIPER_POSITION_TO_METERS / self.gripper_range_m))

    def _classify_status(self, arm_status) -> RobotStatus:
        # `arm_status.err_status` is an ErrStatus object — never compare to 0 directly.
        if int(arm_status.err_code) != 0 or any(vars(arm_status.err_status).values()):
            return RobotStatus.ERROR
        if int(arm_status.motion_status) != 0:
            return RobotStatus.MOVING
        return RobotStatus.AVAILABLE

    def _estimate_velocity(self, q: np.ndarray, last_q: np.ndarray, ts: int, last_ts: int) -> np.ndarray:
        dt = (ts - last_ts) / 1e9
        if dt <= 0:
            return np.zeros(6, dtype=np.float32)
        return (q - last_q) / dt


if __name__ == '__main__':
    with pimm.World() as world:
        robot = Robot()
        commands = world.pair(robot.commands)
        state = world.pair(robot.state)
        grip = world.pair(robot.target_grip)
        world.start([], background=robot)

        while state.read() is None:
            time.sleep(0.01)

        origin = state.value.ee_pose
        commands.emit(roboarm_command.CartesianPosition(origin))
        grip.emit(0.0)
