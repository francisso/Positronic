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
    speed=100,
    gripper_effort=1000,
    gripper_range_m=0.08,
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
        home_joints=home_joints,
    )
