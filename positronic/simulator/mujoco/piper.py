import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import mujoco as mj
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.simulator.mujoco.sim import MujocoSim

PIPER_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
PIPER_GRIPPER_JOINT_NAMES = ['joint7', 'joint8']
PIPER_CONTROL_FRAME = 'end_effector'


def materialize_piper_mujoco_model(
    mujoco_model_path: str | Path, urdf_path: str | Path, initial_ctrl: list[float] | None = None
) -> str:
    source_path = Path(mujoco_model_path).expanduser().resolve()
    root = ET.parse(source_path).getroot()

    for mesh in root.findall('./asset/mesh'):
        mesh_file = mesh.get('file')
        if mesh_file is None or Path(mesh_file).is_absolute():
            continue
        mesh.set('file', str((source_path.parent / mesh_file).resolve()))

    _ensure_custom_text(root, 'initial_ctrl', ','.join(str(v) for v in (initial_ctrl or [0.0] * 8)))
    _ensure_end_effector_site(root)

    with tempfile.NamedTemporaryFile('w', suffix='.xml', prefix='positronic_piper_', delete=False) as f:
        ET.ElementTree(root).write(f, encoding='unicode')
        temp_path = f.name

    return temp_path


def _ensure_custom_text(root: ET.Element, name: str, data: str):
    custom = root.find('custom')
    if custom is None:
        custom = ET.SubElement(root, 'custom')

    text = custom.find(f"text[@name='{name}']")
    if text is None:
        ET.SubElement(custom, 'text', name=name, data=data)
    else:
        text.set('data', data)


def _ensure_end_effector_site(root: ET.Element):
    for body in root.iter('body'):
        if body.get('name') != 'link6':
            continue
        if body.find(f"site[@name='{PIPER_CONTROL_FRAME}']") is None:
            ET.SubElement(body, 'site', name=PIPER_CONTROL_FRAME, pos='0 0 0.1358', size='0.01', rgba='1 0 0 1')
        return

    raise ValueError('Could not find Piper link6 body to attach end-effector site')


class MujocoPiperState(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(6 + 6 + 7 + 1,), dtype=np.float32)
        self.array.fill(0.0)
        self.array[19] = RobotStatus.AVAILABLE.value

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

    def set_error(self):
        self.array[19] = RobotStatus.ERROR.value

    def clear_error(self):
        self.array[19] = RobotStatus.AVAILABLE.value

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D):
        status = self.status
        self.array[:6] = q
        self.array[6:12] = dq
        self.array[12 : 12 + 3] = ee_pose.translation
        self.array[12 + 3 : 12 + 7] = ee_pose.rotation.as_quat
        self.array[19] = status.value


class MujocoPiper(pimm.ControlSystem):
    def __init__(
        self,
        sim: MujocoSim,
        urdf_path: str | Path,
        *,
        cartesian_rotation_mode: str = 'current',
        rot_weight: float = 0.0,
        max_joint_step: float = 0.05,
    ):
        self.sim = sim
        self.physics = dm_mujoco.Physics.from_model(sim.data)
        self.urdf_path = Path(urdf_path).expanduser()
        self.cartesian_rotation_mode = cartesian_rotation_mode
        self.rot_weight = rot_weight
        self.max_joint_step = max_joint_step
        self.ee_name = PIPER_CONTROL_FRAME
        self.joint_names = PIPER_JOINT_NAMES
        self.actuator_names = PIPER_JOINT_NAMES
        self.gripper_joint_names = PIPER_GRIPPER_JOINT_NAMES
        self.gripper_actuator_names = PIPER_GRIPPER_JOINT_NAMES
        self.joint_qpos_ids = [self.sim.model.joint(joint).qposadr.item() for joint in self.joint_names]
        self.gripper_qpos_ids = [self.sim.model.joint(joint).qposadr.item() for joint in self.gripper_joint_names]

        self.commands: pimm.SignalReceiver[roboarm_command.CommandType] = pimm.ControlSystemReceiver(self, default=None)
        self.target_grip: pimm.SignalReceiver[float] = pimm.ControlSystemReceiver(self, default=0.0)

        self.state: pimm.SignalEmitter[MujocoPiperState] = pimm.ControlSystemEmitter(self)
        self.grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.robot_meta.emit({
            'urdf': self.urdf_path.read_text(),
            'joint_names': self.joint_names,
            'control_frame': self.ee_name,
        })
        state = MujocoPiperState()

        while not should_stop.value:
            state.encode(self.q, self.dq, self.ee_pose)

            cmd_msg = self.commands.read()
            if cmd_msg.updated:
                match cmd_msg.data:
                    case roboarm_command.CartesianPosition(pose=pose):
                        q = self._recalculate_ik(self._prepare_cartesian_pose(pose))
                        if q is None:
                            state.set_error()
                        else:
                            self.set_arm_actuator_values(q)
                            state.clear_error()
                    case roboarm_command.JointPosition(positions=positions):
                        self.set_arm_actuator_values(positions)
                        state.clear_error()
                    case roboarm_command.JointDelta(velocities=delta):
                        self.set_arm_actuator_values(self.q + delta)
                        state.clear_error()
                    case roboarm_command.Reset():
                        self.sim.reset()
                        state.clear_error()
                    case roboarm_command.Recover():
                        state.clear_error()
                    case _:
                        raise ValueError(f'Unknown command type: {type(cmd_msg.data)}')

            grip_msg = self.target_grip.read()
            if grip_msg.updated:
                self.set_target_grip(grip_msg.data)

            self.state.emit(state)
            self.grip.emit(self.current_grip)
            yield pimm.Pass()

    def _recalculate_ik(self, target_robot_position: geom.Transform3D) -> np.ndarray | None:
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name=self.ee_name,
            target_pos=target_robot_position.translation,
            target_quat=target_robot_position.rotation.as_quat,
            joint_names=self.joint_names,
            rot_weight=self.rot_weight,
        )

        if result.success:
            return self._limit_joint_step(result.qpos[self.joint_qpos_ids])

        return None

    def _prepare_cartesian_pose(self, pose: geom.Transform3D) -> geom.Transform3D:
        match self.cartesian_rotation_mode:
            case 'command':
                rotation = pose.rotation
            case 'current':
                rotation = self.ee_pose.rotation
            case _:
                raise ValueError(f'Unknown cartesian_rotation_mode={self.cartesian_rotation_mode!r}')

        return geom.Transform3D(pose.translation, rotation)

    def _limit_joint_step(self, target: np.ndarray) -> np.ndarray:
        if self.max_joint_step <= 0:
            return target

        delta = np.clip(target - self.q, -self.max_joint_step, self.max_joint_step)
        return self.q + delta

    @property
    def q(self) -> np.ndarray:
        return np.array([self.sim.data.qpos[i] for i in self.joint_qpos_ids])

    @property
    def dq(self) -> np.ndarray:
        return np.array([self.sim.data.qvel[i] for i in self.joint_qpos_ids])

    @property
    def ee_pose(self) -> geom.Transform3D:
        translation = self.sim.data.site(self.ee_name).xpos.copy()
        rotmat = self.sim.data.site(self.ee_name).xmat.copy()
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, rotmat)
        return geom.Transform3D(translation=translation, rotation=geom.Rotation.from_quat(quat))

    @property
    def current_grip(self) -> float:
        min_grip, max_grip = self.sim.model.actuator(self.gripper_actuator_names[0]).ctrlrange
        current = self.sim.data.qpos[self.gripper_qpos_ids[0]].item()
        return min(1.0, max(0.0, (current - min_grip) / (max_grip - min_grip)))

    def set_arm_actuator_values(self, actuator_values: np.ndarray):
        for actuator_name, value in zip(self.actuator_names, actuator_values, strict=True):
            self._set_actuator_value(actuator_name, value)

    def set_target_grip(self, target_grip: float):
        grip = min(1.0, max(0.0, float(target_grip)))
        joint7_range = self.sim.model.actuator(self.gripper_actuator_names[0]).ctrlrange
        joint8_range = self.sim.model.actuator(self.gripper_actuator_names[1]).ctrlrange
        self._set_actuator_value(
            self.gripper_actuator_names[0], joint7_range[0] + grip * (joint7_range[1] - joint7_range[0])
        )
        self._set_actuator_value(
            self.gripper_actuator_names[1], joint8_range[1] + grip * (joint8_range[0] - joint8_range[1])
        )

    def _set_actuator_value(self, actuator_name: str, value: float):
        ctrlrange = self.sim.model.actuator(actuator_name).ctrlrange
        self.sim.data.actuator(actuator_name).ctrl = min(ctrlrange[1], max(ctrlrange[0], value))
