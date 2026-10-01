"""PIE observation, reward and reset terms, implemented from the paper equations."""

import torch


def canonical_joint_ids(env, joint_names):
    robot = env.scene["robot"]
    return torch.tensor([robot.joint_names.index(name) for name in joint_names], device=env.device)


def proprioception(env, joint_names, angular_scale=0.25,
                   joint_velocity_scale=0.05, relative_joint_positions=True):
    data = env.scene["robot"].data
    ids = canonical_joint_ids(env, joint_names)
    joints = data.joint_pos[:, ids]
    if relative_joint_positions:
        joints = joints - data.default_joint_pos[:, ids]
    native_names = env.action_manager.get_term("joint_pos").target_names
    action_order = [native_names.index(name) for name in joint_names]
    command = env.command_manager.get_command("twist")
    return torch.cat((
        data.root_link_ang_vel_b * angular_scale, data.projected_gravity_b, command,
        joints, data.joint_vel[:, ids] * joint_velocity_scale,
        env.action_manager.action[:, action_order],
    ), dim=-1)


def track_linear_velocity(env):
    command = env.command_manager.get_command("twist")
    error = (command[:, :2] - env.scene["robot"].data.root_link_lin_vel_b[:, :2]).square().sum(-1)
    return torch.exp(-4.0 * error)


def track_angular_velocity(env):
    command = env.command_manager.get_command("twist")
    error = command[:, 2] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-4.0 * error.square())


def vertical_velocity(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()


def horizontal_angular_velocity(env):
    return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(-1)


def orientation(env):
    # ||g||^2 would be constant for a unit gravity vector. Penalize its XY component.
    return env.scene["robot"].data.projected_gravity_b[:, :2].square().sum(-1)


def joint_acceleration(env):
    return env.scene["robot"].data.joint_acc.square().sum(-1)


def mechanical_power(env):
    robot = env.scene["robot"]
    names = env.action_manager.get_term("joint_pos").target_names
    order = [robot.joint_names.index(name) for name in names]
    return (robot.data.actuator_force.abs() * robot.data.joint_vel[:, order].abs()).sum(-1)


def nonfoot_collision(env):
    found = env.scene["nonfoot_ground_contact"].data.found
    return found.flatten(1).float().sum(-1)


def action_rate(env):
    manager = env.action_manager
    return (manager.action - manager.prev_action).square().sum(-1)


def action_smoothness(env):
    manager = env.action_manager
    return (manager.action - 2 * manager.prev_action + manager.prev_prev_action).square().sum(-1)


def fallen(env):
    data = env.scene["robot"].data
    tilt = data.projected_gravity_b[:, 2] > -0.35
    below_track = data.root_link_pos_w[:, 2] < env.scene.env_origins[:, 2] - 0.30
    base_contact = env.scene["base_ground_contact"].data.found.flatten(1).any(-1)
    finite = torch.isfinite(data.root_link_pos_w).all(-1)
    for state in (data.root_link_lin_vel_b, data.root_link_ang_vel_b,
                  data.projected_gravity_b, data.joint_pos, data.joint_vel,
                  data.joint_acc, data.actuator_force):
        finite &= torch.isfinite(state).all(-1)
    return tilt | below_track | base_contact | ~finite


def track_bounds(env, width: float):
    relative = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    return (relative[:, 0] < -0.6) | (relative[:, 1].abs() > width / 2 - 0.15)


def course_complete(env, length: float):
    relative = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    return relative[:, 0] > length - 1.8


def terrain_curriculum(env, env_ids, length: float):
    terrain = env.scene.terrain
    displacement = env.scene["robot"].data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    up = displacement > length - 2.0
    down = (displacement < 0.5) & (env.episode_length_buf[env_ids] > 0)
    if env.common_step_counter == 0:
        up = torch.zeros_like(up)
        down = torch.zeros_like(down)
    terrain.update_env_origins(env_ids, up, down)
    return terrain.terrain_levels.float().mean()


def reset_joints_scaled(env, env_ids, low=0.5, high=1.5):
    # Paper's [0.5,1.5] initial-joint range is interpreted as a multiplier of stand pose.
    # Absolute positive radians would violate Lite3 HipY joint limits.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    default = robot.data.default_joint_pos[env_ids]
    positions = default * torch.empty_like(default).uniform_(low, high)
    limits = robot.data.soft_joint_pos_limits[env_ids]
    positions = positions.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=env_ids)
