import configuronic as cfn

import positronic.cfg.hardware.motors


@cfn.config(
    ip='172.168.0.2',
    relative_dynamics_factor=0.2,
    home_joints=[0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0],
    load=None,
    collision_coeff=2.0,
)
def franka(
    ip: str,
    relative_dynamics_factor: float,
    home_joints: list[float],
    load: tuple | None = None,
    collision_coeff: float = 2.0,
):
    from positronic.drivers.roboarm import franka  # noqa: F401

    return franka.Robot(
        ip=ip,
        relative_dynamics_factor=relative_dynamics_factor,
        home_joints=home_joints,
        load=load,
        collision_coeff=collision_coeff,
    )


franka_droid = franka.override(load=(0.9, [0.0, 0.0, 0.057], [0.002768, 0, 0, 0, 0.003149, 0, 0, 0, 0.000564]))


@cfn.config(ip='192.168.1.10', relative_dynamics_factor=0.5)
def kinova(ip, relative_dynamics_factor):
    from positronic.drivers.roboarm.kinova.driver import Robot

    return Robot(ip=ip, relative_dynamics_factor=relative_dynamics_factor)


@cfn.config(motor_bus=positronic.cfg.hardware.motors.so101_follower)
def so101(motor_bus):
    from positronic.drivers.roboarm.so101.driver import Robot

    return Robot(motor_bus=motor_bus)


@cfn.config(
    can_name='can0',
    judge_flag=True,
    can_auto_init=True,
    dh_is_offset=0x01,
    start_sdk_joint_limit=True,
    start_sdk_gripper_limit=True,
    speed=30,
    gripper_effort=1000,
    gripper_range_m=0.08,
    cartesian_rotation_mode='command',
    fixed_cartesian_rpy_deg=[0.0, 85.0, 0.0],
    max_cartesian_step_m=0.01,
    home_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
)
def piper(
    can_name: str,
    judge_flag: bool,
    can_auto_init: bool,
    dh_is_offset: int,
    start_sdk_joint_limit: bool,
    start_sdk_gripper_limit: bool,
    speed: int,
    gripper_effort: int,
    gripper_range_m: float,
    cartesian_rotation_mode: str,
    fixed_cartesian_rpy_deg: list[float],
    max_cartesian_step_m: float,
    home_joints: list[float],
):
    from positronic.drivers.roboarm.piper.driver import Robot

    return Robot(
        can_name=can_name,
        judge_flag=judge_flag,
        can_auto_init=can_auto_init,
        dh_is_offset=dh_is_offset,
        start_sdk_joint_limit=start_sdk_joint_limit,
        start_sdk_gripper_limit=start_sdk_gripper_limit,
        speed=speed,
        gripper_effort=gripper_effort,
        gripper_range_m=gripper_range_m,
        cartesian_rotation_mode=cartesian_rotation_mode,
        fixed_cartesian_rpy_deg=fixed_cartesian_rpy_deg,
        max_cartesian_step_m=max_cartesian_step_m,
        home_joints=home_joints,
    )


@cfn.config(
    ip_address='192.168.1.233',
    use_robot_calibration=True,
    speed=50.0,
    acceleration=1100.0,
    command_hz=100.0,
    max_joint_step_rad=0.05,
    home_joints=[0.0, -0.6, -0.6, 0.0, 1.2, 0.0],
    has_gripper=True,
    gripper_model='hand-e',
    gripper_open_width=190,
    gripper_closed_width=150,
    gripper_force=255,
    gripper_speed=100,
)
def xarm6(
    ip_address: str,
    use_robot_calibration: bool,
    speed: float,
    acceleration: float,
    command_hz: float,
    max_joint_step_rad: float,
    home_joints: list[float],
    has_gripper: bool,
    gripper_model: str,
    gripper_open_width: int,
    gripper_closed_width: int,
    gripper_force: int,
    gripper_speed: int,
):
    from positronic.drivers.roboarm.xarm6.driver import Robot

    return Robot(
        ip_address=ip_address,
        use_robot_calibration=use_robot_calibration,
        speed=speed,
        acceleration=acceleration,
        command_hz=command_hz,
        max_joint_step_rad=max_joint_step_rad,
        home_joints=home_joints,
        has_gripper=has_gripper,
        gripper_model=gripper_model,
        gripper_open_width=gripper_open_width,
        gripper_closed_width=gripper_closed_width,
        gripper_force=gripper_force,
        gripper_speed=gripper_speed,
    )
