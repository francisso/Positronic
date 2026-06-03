import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from enum import Enum
from pathlib import Path

import configuronic as cfn
import numpy as np
import pos3

import pimm
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.simulator
import positronic.cfg.sound
import positronic.cfg.webxr
from positronic import geom, wire
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, Serializers, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.drivers.roboarm import State as RoboarmState
from positronic.drivers.webxr import WebXR
from positronic.gui.dpg import DearpyguiUi
from positronic.simulator.mujoco.piper import MujocoPiper, materialize_piper_mujoco_model
from positronic.simulator.mujoco.sim import MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.simulator.mujoco.xarm import XARM6_DEFAULT_HOME, MujocoXArm6, materialize_xarm6_mujoco_model
from positronic.utils import package_assets_path
from positronic.utils.buttons import ButtonHandler
from positronic.utils.logging import init_logging


def _parse_buttons(buttons: dict, button_handler: ButtonHandler):
    for side in ['left', 'right']:
        if buttons[side] is None:
            continue

        mapping = {
            f'{side}_A': buttons[side][4],
            f'{side}_B': buttons[side][5],
            f'{side}_trigger': buttons[side][0],
            f'{side}_thumb': buttons[side][1],
            f'{side}_stick': buttons[side][3],
        }
        button_handler.update_buttons(mapping)


def _check_error(is_error, was_error):
    return is_error, is_error and not was_error


class _Tracker:
    on = False
    _offset = geom.Transform3D()
    _teleop_t = geom.Transform3D()

    def __init__(self, operator_position: geom.Transform3D | None):
        self._operator_position = operator_position
        self.on = self.umi_mode

    @property
    def umi_mode(self):
        return self._operator_position is None

    def turn_on(self, robot_pos: geom.Transform3D):
        if self.umi_mode:
            logging.info('Ignoring tracking on/off in UMI mode')
            return

        self.on = True
        logging.info('Starting tracking')
        self._offset = geom.Transform3D(
            -self._teleop_t.translation + robot_pos.translation, self._teleop_t.rotation.inv * robot_pos.rotation
        )

    def turn_off(self):
        if self.umi_mode:
            logging.info('Ignoring tracking on/off in UMI mode')
            return
        self.on = False
        logging.info('Stopped tracking')

    def update(self, tracker_pos: geom.Transform3D):
        if self.umi_mode:
            return tracker_pos

        self._teleop_t = self._operator_position * tracker_pos * self._operator_position.inv
        return geom.Transform3D(
            self._teleop_t.translation + self._offset.translation, self._teleop_t.rotation * self._offset.rotation
        )


class OperatorPosition(Enum):
    # map xyz -> zxy
    FRONT = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
    # map xyz -> zxy + flip x and y
    BACK = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))


class DataCollectionController(pimm.ControlSystem):
    def __init__(
        self,
        operator_position: geom.Transform3D | None,
        *,
        static_meta: dict | None = None,
        metadata_getter: Callable[[], dict] | None = None,
    ):
        self.operator_position = operator_position
        self._static_meta = static_meta or {}
        self.metadata_getter = metadata_getter or (lambda: {})
        self.controller_positions = pimm.ControlSystemReceiver(self, default=None)
        self.buttons_receiver = pimm.ControlSystemReceiver(self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.FakeReceiver(self)  # To make compatible with other "policy" control systems
        self.frames = pimm.ReceiverDict(self, fake=True)
        self.robot_meta_in = pimm.ControlSystemReceiver(self, default={})

        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.ds_agent_commands = pimm.ControlSystemEmitter(self)
        self.sound = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        start_wav_path = 'positronic/assets/sounds/recording-has-started.wav'
        end_wav_path = 'positronic/assets/sounds/recording-has-stopped.wav'
        abort_wav_path = 'positronic/assets/sounds/recording-has-been-aborted.wav'
        error_wav_path = 'positronic/assets/sounds/error-occurred.wav'

        tracker = _Tracker(self.operator_position)
        button_handler = ButtonHandler()

        recording = False
        in_error = False

        while not should_stop.value:
            try:
                _parse_buttons(self.buttons_receiver.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    if not recording:
                        meta = dict(self._static_meta)
                        meta.update(self.robot_meta_in.value)
                        meta.update(self.metadata_getter())
                        self.ds_agent_commands.emit(DsWriterCommand.START(meta))
                        self.sound.emit(start_wav_path)
                    else:
                        self.ds_agent_commands.emit(DsWriterCommand.STOP())
                        self.sound.emit(end_wav_path)
                    recording = not recording
                elif button_handler.just_pressed('right_A'):
                    if tracker.on:
                        tracker.turn_off()
                    else:
                        tracker.turn_on(self.robot_state.value.ee_pose)
                elif button_handler.just_pressed('right_stick') and not tracker.umi_mode:
                    logging.info('Resetting robot')
                    if recording:
                        self.ds_agent_commands.emit(DsWriterCommand.ABORT())
                        self.sound.emit(abort_wav_path)
                    tracker.turn_off()
                    recording = False
                    self.robot_commands.emit(roboarm.command.Reset())

                self.target_grip.emit(button_handler.get_value('right_trigger'))
                cp_msg = self.controller_positions.read()
                if cp_msg.updated:
                    target_robot_pos = tracker.update(cp_msg.data['right'])

                if tracker.on:
                    in_error, entered_error = _check_error(
                        self.robot_state.value.status == roboarm.RobotStatus.ERROR, in_error
                    )
                    if entered_error:
                        self.sound.emit(error_wav_path)
                        self.robot_commands.emit(roboarm.command.Recover())
                    elif not in_error and cp_msg.updated:
                        self.robot_commands.emit(roboarm.command.CartesianPosition(target_robot_pos))

                yield pimm.Sleep(0.001)

            except pimm.NoValueException:
                yield pimm.Sleep(0.001)
                continue


def controller_positions_serializer(controller_positions: dict[str, geom.Transform3D]) -> dict[str, np.ndarray]:
    res = {}
    for side, pos in controller_positions.items():
        if pos is not None:
            res[f'.{side}'] = Serializers.transform_3d(pos)
    return res


def _wrench_to_level(state: RoboarmState) -> float | None:
    if state.ee_wrench is None:
        return None
    return np.linalg.norm(state.ee_wrench)


def _camera_adapter_array(adapter):
    return adapter.array


def _wire(
    world: pimm.World,
    ds_agent: DsWriterAgent | None,
    data_collection: DataCollectionController,
    webxr: WebXR,
    robot_arm: pimm.ControlSystem | None,
    sound: pimm.ControlSystem | None,
):
    world.connect(webxr.controller_positions, data_collection.controller_positions)
    world.connect(webxr.buttons, data_collection.buttons_receiver)

    if sound is not None:
        world.connect(data_collection.sound, sound.wav_path)
        if robot_arm is not None:
            world.connect(robot_arm.state, sound.level, receiver_wrapper=pimm.map(_wrench_to_level))

    if ds_agent is not None:
        if robot_arm is not None:
            ds_agent.add_signal('controller_positions', controller_positions_serializer)
            world.connect(webxr.controller_positions, ds_agent.inputs['controller_positions'])
        world.connect(data_collection.ds_agent_commands, ds_agent.command)

    return ds_agent


def main(
    robot_arm: pimm.ControlSystem | None,
    gripper: pimm.ControlSystem | None,
    webxr: WebXR,
    sound: pimm.ControlSystem | None,
    cameras: dict[str, pimm.ControlSystem] | None,
    output_dir: str | None = None,
    stream_video_to_webxr: str | None = None,
    operator_position: OperatorPosition = OperatorPosition.FRONT,
    task: str | None = None,
):
    """Runs data collection in real hardware."""
    camera_instances = cameras or {}
    camera_emitters = {name: cam.frame for name, cam in camera_instances.items()}
    static_meta = {}
    if task is not None:
        static_meta['task'] = task
    if robot_arm is not None:
        static_meta.update(wire.ROBOT_STATIC_META)
    data_collection = DataCollectionController(operator_position.value, static_meta=static_meta)

    writer_cm = (
        LocalDatasetWriter(pos3.sync(output_dir, sync_on_error=True)) if output_dir is not None else nullcontext()
    )
    with writer_cm as dataset_writer, pimm.World() as world:
        ds_agent = wire.wire(world, data_collection, dataset_writer, camera_emitters, robot_arm, gripper, None)
        _wire(world, ds_agent, data_collection, webxr, robot_arm, sound)

        bg_cs = [webxr, *camera_instances.values(), ds_agent, robot_arm, gripper, sound]

        if stream_video_to_webxr is not None:
            world.connect(
                camera_emitters[stream_video_to_webxr], webxr.frame, receiver_wrapper=pimm.map(_camera_adapter_array)
            )

        dc_steps = iter(world.start(data_collection, bg_cs))
        while not world.should_stop:
            try:
                time.sleep(next(dc_steps).seconds)
            except StopIteration:
                break


@cfn.config(
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    webxr=positronic.cfg.webxr.oculus,
    cameras={
        'image.wrist': 'handcam_left_ph',
        'image.exterior': 'back_view_ph',
        'image.handcam_right': 'handcam_right_ph',
        'image.wrist_2': 'wrist_cam_ph',
    },
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)
def main_sim(
    mujoco_model_path: str,
    webxr: WebXR,
    cameras: dict[str, str],
    sound: pimm.ControlSystem | None = None,
    loaders: Sequence[MujocoSceneTransform] = (),
    output_dir: str | None = None,
    fps: int = 30,
    operator_position: OperatorPosition = OperatorPosition.FRONT,
    task: str | None = None,
):
    """Runs data collection in simulator."""

    sim = MujocoSim(mujoco_model_path, loaders)
    robot_arm = MujocoFranka(sim, suffix='_ph')

    mujoco_cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=fps)
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in cameras.items()}
    gui = DearpyguiUi()
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')

    static_meta = dict(wire.ROBOT_STATIC_META)
    if task is not None:
        static_meta['task'] = task

    data_collection = DataCollectionController(
        operator_position.value,
        static_meta=static_meta,
        metadata_getter=lambda: {k: v.tolist() for k, v in sim.save_state().items()},
    )

    writer_cm = (
        LocalDatasetWriter(pos3.sync(output_dir, sync_on_error=True)) if output_dir is not None else nullcontext()
    )
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(world, data_collection, dataset_writer, cameras, robot_arm, gripper, gui, TimeMode.MESSAGE)
        _wire(world, ds_agent, data_collection, webxr, robot_arm, sound)

        sim_iter = world.start(
            [sim, mujoco_cameras, robot_arm, gripper, data_collection], [webxr, gui, ds_agent, sound]
        )
        sim_iter = iter(sim_iter)

        start_time = pimm.world.SystemClock().now_ns()
        sim_start_time = sim.now_ns()

        while not world.should_stop:
            try:
                time_since_start = pimm.world.SystemClock().now_ns() - start_time
                if sim.now_ns() < sim_start_time + time_since_start:
                    next(sim_iter)
                else:
                    time.sleep(0.001)
            except StopIteration:
                break


main_cfg = cfn.Config(
    main,
    robot_arm=None,
    gripper=positronic.cfg.hardware.gripper.dh_gripper,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={
        'image.left': positronic.cfg.hardware.camera.arducam_left,
        'image.right': positronic.cfg.hardware.camera.arducam_right,
    },
    operator_position=OperatorPosition.FRONT,
)


@cfn.config()
def piper_stack_cubes_loaders():
    from positronic.simulator.mujoco.transforms import AddBox, AddCameras, SetBodyPosition

    return [
        AddCameras(
            additional_cameras={
                'side_view': {'pos': [0.80, -0.80, 0.55], 'xyaxes': [0.707, 0.707, 0.000, -0.408, 0.408, 0.816]},
                'front_view': {'pos': [0.75, 0.00, 0.48], 'xyaxes': [0.000, 1.000, 0.000, -0.625, 0.000, 0.781]},
            }
        ),
        AddBox(name='table', size=[0.30, 0.30, 0.02], pos=[0.35, 0.0, 0.10], rgba=[0.65, 0.65, 0.65, 1]),
        AddBox(name='box_0', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[1, 0, 0, 1], freejoint=True),
        SetBodyPosition(body_name='box_0_body', random_position=[[0.20, -0.16, 0.13], [0.46, 0.16, 0.13]]),
        AddBox(name='box_1', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[0, 1, 0, 1], freejoint=True),
        SetBodyPosition(body_name='box_1_body', random_position=[[0.20, -0.16, 0.13], [0.46, 0.16, 0.13]]),
    ]


@cfn.config(
    urdf_path=str(Path(__file__).resolve().parents[1] / 'piper_description/urdf/piper_description.urdf'),
    mujoco_model_path=None,
    webxr=positronic.cfg.webxr.oculus,
    cameras={'image.exterior': 'side_view', 'image.agent_view': 'front_view'},
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    loaders=piper_stack_cubes_loaders,
    cartesian_rotation_mode='command',
    rot_weight=0.2,
    max_joint_step=0.05,
    fixed_cartesian_rpy_deg=None,
)
def main_piper_sim(
    urdf_path: str,
    mujoco_model_path: str | None,
    webxr: WebXR,
    cameras: dict[str, str],
    sound: pimm.ControlSystem | None = None,
    loaders: Sequence[MujocoSceneTransform] = (),
    output_dir: str | None = None,
    fps: int = 30,
    operator_position: OperatorPosition = OperatorPosition.FRONT,
    task: str | None = None,
    cartesian_rotation_mode: str = 'command',
    rot_weight: float = 0.2,
    max_joint_step: float = 0.05,
    fixed_cartesian_rpy_deg: list[float] | None = None,
):
    if mujoco_model_path is None:
        mujoco_model_path = str(Path(urdf_path).expanduser().parents[1] / 'mujoco_model/piper_description.xml')

    model_path = materialize_piper_mujoco_model(mujoco_model_path, urdf_path)
    sim = MujocoSim(model_path, loaders)
    robot_arm = MujocoPiper(
        sim,
        urdf_path,
        cartesian_rotation_mode=cartesian_rotation_mode,
        rot_weight=rot_weight,
        max_joint_step=max_joint_step,
        fixed_cartesian_rpy_deg=fixed_cartesian_rpy_deg,
    )
    mujoco_cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=fps)
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in cameras.items()}
    gui = DearpyguiUi()

    static_meta = dict(wire.ROBOT_STATIC_META)
    if task is not None:
        static_meta['task'] = task

    data_collection = DataCollectionController(
        operator_position.value,
        static_meta=static_meta,
        metadata_getter=lambda: {k: v.tolist() for k, v in sim.save_state().items()},
    )

    writer_cm = (
        LocalDatasetWriter(pos3.sync(output_dir, sync_on_error=True)) if output_dir is not None else nullcontext()
    )
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(
            world, data_collection, dataset_writer, cameras, robot_arm, robot_arm, gui, TimeMode.MESSAGE
        )
        _wire(world, ds_agent, data_collection, webxr, robot_arm, sound)

        sim_iter = world.start([sim, mujoco_cameras, robot_arm, data_collection], [webxr, gui, ds_agent, sound])
        sim_iter = iter(sim_iter)

        start_time = pimm.world.SystemClock().now_ns()
        sim_start_time = sim.now_ns()

        while not world.should_stop:
            try:
                time_since_start = pimm.world.SystemClock().now_ns() - start_time
                if sim.now_ns() < sim_start_time + time_since_start:
                    next(sim_iter)
                else:
                    time.sleep(0.001)
            except StopIteration:
                break


@cfn.config()
def xarm6_stack_cubes_loaders():
    from positronic.simulator.mujoco.transforms import AddBodyCameras, AddBox, AddCameras, SetBodyPosition

    return [
        AddCameras(
            additional_cameras={
                'side_view': {
                    'pos': [1.55, -1.55, 1.05],
                    'xyaxes': [0.707, 0.707, 0.000, -0.408, 0.408, 0.816],
                    'fovy': 70,
                },
                'front_view': {
                    'pos': [1.65, 0.00, 0.95],
                    'xyaxes': [0.000, 1.000, 0.000, -0.500, 0.000, 0.866],
                    'fovy': 70,
                },
                'side_view_2': {
                    'pos': [1.55, 1.55, 1.05],
                    'xyaxes': [-0.707, 0.707, 0.000, -0.408, -0.408, 0.816],
                    'fovy': 70,
                },
            }
        ),
        AddBodyCameras(
            body_name='link6',
            cameras={
                'wrist_view': {
                    'pos': [-0.05, 0.03, 0.015],
                    'xyaxes': [0.000, 1.000, 0.000, 1.000, 0.000, -0.200],
                    'fovy': 90,
                }
            },
        ),
        AddBox(name='table', size=[0.35, 0.35, 0.02], pos=[0.42, 0.0, 0.10], rgba=[0.65, 0.65, 0.65, 1]),
        AddBox(name='box_0', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[1, 0, 0, 1], freejoint=True),
        SetBodyPosition(body_name='box_0_body', random_position=[[0.25, -0.18, 0.13], [0.54, 0.18, 0.13]]),
        AddBox(name='box_1', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[0, 1, 0, 1], freejoint=True),
        SetBodyPosition(body_name='box_1_body', random_position=[[0.25, -0.18, 0.13], [0.54, 0.18, 0.13]]),
    ]


@cfn.config(
    urdf_path=str(Path(__file__).resolve().parents[1] / 'xarm6_description/xarm6.urdf'),
    home_joints=XARM6_DEFAULT_HOME,
    webxr=positronic.cfg.webxr.oculus,
    cameras={
        'image.exterior': 'side_view',
        'image.agent_view': 'front_view',
        'image.side': 'side_view_2',
        'image.wrist': 'wrist_view',
    },
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    loaders=xarm6_stack_cubes_loaders,
    cartesian_rotation_mode='command',
    rot_weight=0.2,
    max_joint_step=0.12,
    fixed_cartesian_rpy_deg=None,
    ik_solver='fresenius',
    stream_video_to_webxr=None,
)
def main_xarm6_sim(
    urdf_path: str,
    home_joints: list[float],
    webxr: WebXR,
    cameras: dict[str, str],
    sound: pimm.ControlSystem | None = None,
    loaders: Sequence[MujocoSceneTransform] = (),
    output_dir: str | None = None,
    fps: int = 30,
    operator_position: OperatorPosition = OperatorPosition.FRONT,
    task: str | None = None,
    cartesian_rotation_mode: str = 'command',
    rot_weight: float = 0.2,
    max_joint_step: float = 0.12,
    fixed_cartesian_rpy_deg: list[float] | None = None,
    ik_solver: str = 'fresenius',
    stream_video_to_webxr: str | None = None,
):
    model_path = materialize_xarm6_mujoco_model(urdf_path, initial_ctrl=home_joints)
    sim = MujocoSim(model_path, loaders)
    robot_arm = MujocoXArm6(
        sim,
        urdf_path,
        cartesian_rotation_mode=cartesian_rotation_mode,
        rot_weight=rot_weight,
        max_joint_step=max_joint_step,
        fixed_cartesian_rpy_deg=fixed_cartesian_rpy_deg,
        ik_solver=ik_solver,
    )
    mujoco_cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=fps)
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in cameras.items()}
    gui = DearpyguiUi()

    static_meta = dict(wire.ROBOT_STATIC_META)
    if task is not None:
        static_meta['task'] = task

    data_collection = DataCollectionController(
        operator_position.value,
        static_meta=static_meta,
        metadata_getter=lambda: {k: v.tolist() for k, v in sim.save_state().items()},
    )

    writer_cm = (
        LocalDatasetWriter(pos3.sync(output_dir, sync_on_error=True)) if output_dir is not None else nullcontext()
    )
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(world, data_collection, dataset_writer, cameras, robot_arm, None, gui, TimeMode.MESSAGE)
        _wire(world, ds_agent, data_collection, webxr, robot_arm, sound)

        if stream_video_to_webxr is not None:
            if stream_video_to_webxr not in cameras:
                raise ValueError(
                    f'Unknown stream_video_to_webxr={stream_video_to_webxr!r}. Available cameras: {list(cameras)}'
                )
            world.connect(cameras[stream_video_to_webxr], webxr.frame, receiver_wrapper=pimm.map(_camera_adapter_array))

        sim_iter = world.start([sim, mujoco_cameras, robot_arm, data_collection], [webxr, gui, ds_agent, sound])
        sim_iter = iter(sim_iter)

        start_time = pimm.world.SystemClock().now_ns()
        sim_start_time = sim.now_ns()

        while not world.should_stop:
            try:
                time_since_start = pimm.world.SystemClock().now_ns() - start_time
                if sim.now_ns() < sim_start_time + time_since_start:
                    next(sim_iter)
                else:
                    time.sleep(0.001)
            except StopIteration:
                break


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.so101,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    cameras={'image.right': positronic.cfg.hardware.camera.arducam_right},
)
def so101cfg(robot_arm, **kwargs):
    """Runs data collection on SO101 robot"""
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.piper,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    cameras={
        'image.usb': positronic.cfg.hardware.camera.opencv.override(
            camera_id=0, width=640, height=480, fps=30, buffer_size=1, auto_wb=0, wb_temperature=6000
        ),
        'image.zed': positronic.cfg.hardware.camera.zed.override(
            view='left', resolution='hd720', fps=30, output_width=640, output_height=480
        ),
    },
)
def pipercfg(robot_arm, **kwargs):
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.xarm6,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    cameras={
        'image.usb': positronic.cfg.hardware.camera.opencv.override(
            camera_id=0, width=640, height=480, fps=30, buffer_size=1, auto_wb=0, wb_temperature=6000
        ),
        'image.zed': positronic.cfg.hardware.camera.zed.override(
            view='left', resolution='hd720', fps=30, output_width=640, output_height=480
        ),
    },
)
def xarm6cfg(robot_arm, **kwargs):
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


droid = cfn.Config(
    main,
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={
        'image.wrist': positronic.cfg.hardware.camera.zed_m.override(view='left', resolution='hd720', fps=30),
        'image.exterior': positronic.cfg.hardware.camera.zed_2i.override(view='left', resolution='hd720', fps=30),
    },
    operator_position=OperatorPosition.BACK,
)


human = cfn.Config(
    main,
    robot_arm=None,
    gripper=None,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={'image.exterior': positronic.cfg.hardware.camera.zed_2i.override(view='left', resolution='hd720', fps=30)},
    operator_position=OperatorPosition.BACK,
)


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'real': main_cfg,
        'so101': so101cfg,
        'piper': pipercfg,
        'xarm6': xarm6cfg,
        'sim': main_sim,
        'piper_sim': main_piper_sim,
        'xarm6_sim': main_xarm6_sim,
        'sim_pnp': main_sim.override(loaders=positronic.cfg.simulator.multi_tote_loaders),
        'droid': droid,
        'human': human,
    })


if __name__ == '__main__':
    _internal_main()
