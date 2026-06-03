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
from positronic.drivers.roboarm.xarm6.kinematics import XArm6IKSolver
from positronic.simulator.mujoco.sim import MujocoSim

XARM6_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
XARM6_CONTROL_FRAME = 'end_effector'
XARM6_DEFAULT_HOME = [0.0, -0.6, -0.6, 0.0, 1.2, 0.0]


def materialize_xarm6_mujoco_model(urdf_path: str | Path, initial_ctrl: list[float] | None = None) -> str:
    source_path = Path(urdf_path).expanduser().resolve()
    temp_urdf_path = _rewrite_mesh_paths(source_path)
    spec = mj.MjSpec.from_file(temp_urdf_path)
    spec.compiler.balanceinertia = True
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    _add_end_effector_site(spec)
    _add_position_actuators(spec)
    spec.add_text(name='initial_ctrl', data=','.join(str(v) for v in (initial_ctrl or XARM6_DEFAULT_HOME)))

    with tempfile.NamedTemporaryFile('w', suffix='.xml', prefix='positronic_xarm6_', delete=False) as f:
        f.write(spec.to_xml())
        return f.name


def _rewrite_mesh_paths(urdf_path: Path) -> str:
    description_dir = urdf_path.parent
    root = ET.parse(urdf_path).getroot()

    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename')
        if filename is None:
            continue
        mesh.set('filename', str(_local_mesh_path(description_dir, filename)))

    with tempfile.NamedTemporaryFile('w', suffix='.urdf', prefix='positronic_xarm6_', delete=False) as f:
        ET.ElementTree(root).write(f, encoding='unicode')
        return f.name


def _local_mesh_path(description_dir: Path, filename: str) -> Path:
    mesh_name = Path(filename).name
    stem = Path(mesh_name).stem.removesuffix('_vhacd')
    if stem == 'base':
        stem = 'link_base'

    if '/visual/' in filename:
        return (description_dir / 'visual' / f'{stem}.stl').resolve()
    if '/collision/' in filename:
        return (description_dir / 'collision' / f'{stem}.obj').resolve()

    return (description_dir / mesh_name).resolve()


def _add_end_effector_site(spec: mj.MjSpec):
    link6 = spec.body('link6')
    link6.add_site(name=XARM6_CONTROL_FRAME, pos=[0.0, 0.0, 0.0], size=[0.01], rgba=[1.0, 0.0, 0.0, 1.0])


def _add_position_actuators(spec: mj.MjSpec):
    for joint_name in XARM6_JOINT_NAMES:
        joint = spec.joint(joint_name)
        actuator = spec.add_actuator(name=joint_name, target=joint_name)
        actuator.trntype = mj.mjtTrn.mjTRN_JOINT
        actuator.set_to_position(kp=600.0, kv=80.0)
        actuator.ctrllimited = True
        actuator.ctrlrange = joint.range
        actuator.forcelimited = True
        actuator.forcerange = [-250.0, 250.0]


class MujocoXArm6State(State, pimm.shared_memory.NumpySMAdapter):
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


class MujocoXArm6(pimm.ControlSystem):
    def __init__(
        self,
        sim: MujocoSim,
        urdf_path: str | Path,
        *,
        cartesian_rotation_mode: str = 'command',
        rot_weight: float = 0.2,
        max_joint_step: float = 0.05,
        fixed_cartesian_rpy_deg: list[float] | None = None,
        ik_solver: str = 'fresenius',
    ):
        self.sim = sim
        self.physics = dm_mujoco.Physics.from_model(sim.data)
        self.urdf_path = Path(urdf_path).expanduser()
        self.solver = XArm6IKSolver(self.urdf_path)
        self.ik_solver = ik_solver
        self.cartesian_rotation_mode = cartesian_rotation_mode
        self.rot_weight = rot_weight
        self.max_joint_step = max_joint_step
        self.ee_name = XARM6_CONTROL_FRAME
        self.joint_names = XARM6_JOINT_NAMES
        self.actuator_names = XARM6_JOINT_NAMES
        self.joint_qpos_ids = [self.sim.model.joint(joint).qposadr.item() for joint in self.joint_names]
        self.fixed_cartesian_rotation = (
            geom.Rotation.from_euler(np.deg2rad(fixed_cartesian_rpy_deg))
            if fixed_cartesian_rpy_deg is not None
            else self.ee_pose.rotation
        )

        self.commands: pimm.SignalReceiver[roboarm_command.CommandType] = pimm.ControlSystemReceiver(self, default=None)

        self.state: pimm.SignalEmitter[MujocoXArm6State] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.robot_meta.emit({
            'urdf': self.urdf_path.read_text(),
            'joint_names': self.joint_names,
            'control_frame': self.ee_name,
        })
        state = MujocoXArm6State()

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

            self.state.emit(state)
            yield pimm.Pass()

    def _recalculate_ik(self, target_robot_position: geom.Transform3D) -> np.ndarray | None:
        match self.ik_solver:
            case 'fresenius':
                return self._recalculate_fresenius_ik(target_robot_position)
            case 'mujoco':
                return self._recalculate_mujoco_ik(target_robot_position)
            case _:
                raise ValueError(f'Unknown ik_solver={self.ik_solver!r}')

    def _recalculate_fresenius_ik(self, target_robot_position: geom.Transform3D) -> np.ndarray | None:
        result = self.solver.ik_from_transform(target_robot_position, initial_guess=self.q)
        if result is None:
            return None
        return self._limit_joint_step(result)

    def _recalculate_mujoco_ik(self, target_robot_position: geom.Transform3D) -> np.ndarray | None:
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
            case 'fixed':
                rotation = self.fixed_cartesian_rotation
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

    def set_arm_actuator_values(self, actuator_values: np.ndarray):
        for actuator_name, value in zip(self.actuator_names, actuator_values, strict=True):
            ctrlrange = self.sim.model.actuator(actuator_name).ctrlrange
            self.sim.data.actuator(actuator_name).ctrl = min(ctrlrange[1], max(ctrlrange[0], value))
