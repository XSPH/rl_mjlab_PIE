"""Compose real mjlab v1.6.0 APIs into a PIE Lite3 environment."""

import math
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (CurriculumTermCfg, EventTermCfg, ObservationGroupCfg,
                            ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg)
from mjlab.scene import SceneCfg
from mjlab.sensor import (CameraSensorCfg, ContactMatch, ContactSensorCfg, GridPatternCfg,
                          ObjRef, RayCastSensorCfg, TerrainHeightSensorCfg)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg

from . import mdp
from .reproduction import EnvironmentConfig
from .robots import make_robot_config
from .terrains import ParkourTerrainCfg


@dataclass
class RaisedGridPatternCfg(GridPatternCfg):
    elevation: float = 1.0

    def generate_rays(self, mj_model, device):
        offsets, directions = super().generate_rays(mj_model, device)
        offsets[:, 2] = self.elevation
        return offsets, directions


def make_env_config(cfg: EnvironmentConfig) -> ManagerBasedRlEnvCfg:
    cfg.validate()
    entity, robot = make_robot_config(cfg)
    robot_spec = entity.spec_fn()
    action_joint_limits = {name: tuple(float(value) for value in robot_spec.joint(name).range)
                           for name in robot.joints}
    scan_shape = cfg.heightmap_shape
    height_scan = RayCastSensorCfg(
        name="height_scan", frame=ObjRef(type="body", name=robot.base, entity="robot"),
        pattern=RaisedGridPatternCfg(
            size=((scan_shape[0] - 1) * cfg.heightmap_spacing,
                  (scan_shape[1] - 1) * cfg.heightmap_spacing),
            resolution=cfg.heightmap_spacing),
        ray_alignment="yaw", max_distance=4.0, include_geom_groups=(0,),
    )
    feet_frames = tuple(ObjRef(type="site" if cfg.robot == "lite3" else "geom",
                              name=f"{foot}_scan" if cfg.robot == "lite3" else geom,
                              entity="robot") for foot, geom in zip(robot.feet, robot.feet_geoms))
    feet_scan = TerrainHeightSensorCfg(
        name="foot_clearance", frame=feet_frames,
        pattern=RaisedGridPatternCfg(size=(0.0, 0.0), elevation=1.0),
        ray_alignment="world", max_distance=4.0, include_geom_groups=(0,),
    )
    depth_camera = CameraSensorCfg(
        name="depth", camera_name="robot/depth_camera",
        width=cfg.camera.width, height=cfg.camera.height, data_types=("depth",),
        use_textures=False, use_shadows=False, enabled_geom_groups=(0,),
    )
    nonfoot = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=robot.nonfoot_geoms, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1,
    )
    feet_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=robot.feet_geoms, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
    )
    base_contact = ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(mode="body", pattern=robot.base, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1,
    )
    terrain = TerrainEntityCfg(
        terrain_type="generator", max_init_terrain_level=min(cfg.max_initial_level, cfg.terrain_rows - 1),
        textures=(), materials=(),
        terrain_generator=TerrainGeneratorCfg(
            seed=cfg.seed, size=(cfg.terrain_length, cfg.terrain_width),
            num_rows=cfg.terrain_rows, num_cols=5, curriculum=True,
            border_width=2.0, border_height=0.05, color_scheme="none",
            sub_terrains={kind: ParkourTerrainCfg(kind=kind)
                          for kind in ("flat", "gaps", "steps", "hurdles", "stairs")},
        ),
    )
    events = {
        "reset_robot": EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset"),
        "reset_pose": EventTermCfg(func=base_mdp.reset_root_state_uniform, mode="reset",
                                   params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1),
                                                          "yaw": (-0.1, 0.1)}, "velocity_range": {}}),
    }
    if cfg.randomization:
        events["reset_joints"] = EventTermCfg(func=mdp.reset_joints_scaled, mode="reset")
        base = SceneEntityCfg("robot", body_names=(robot.base,))
        camera = SceneEntityCfg("robot", camera_names=("depth_camera",))
        events.update({
            "payload": EventTermCfg(func=dr.body_mass, mode="startup",
                                    params={"asset_cfg": base, "ranges": (-1.0, 2.0), "operation": "add"}),
            "base_com": EventTermCfg(func=dr.body_com_offset, mode="startup",
                                     params={"asset_cfg": base, "ranges": (-0.05, 0.05)}),
            "pd_gains": EventTermCfg(func=dr.pd_gains, mode="startup",
                                     params={"kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1)}),
            "motor_strength": EventTermCfg(func=dr.effort_limits, mode="startup",
                                           params={"effort_limit_range": (0.9, 1.1)}),
            "foot_friction": EventTermCfg(func=dr.geom_friction, mode="startup",
                                          params={"asset_cfg": SceneEntityCfg("robot", geom_names=robot.feet_geoms),
                                                  "ranges": (0.2, 1.2), "operation": "abs", "axes": [0]}),
            "camera_position": EventTermCfg(func=dr.cam_pos, mode="reset",
                                            params={"asset_cfg": camera,
                                                    "ranges": (-cfg.camera.position_jitter_m, cfg.camera.position_jitter_m),
                                                    "operation": "add"}),
            "camera_pitch": EventTermCfg(func=dr.cam_quat, mode="reset",
                                         params={"asset_cfg": camera,
                                                 "pitch_range": (-math.radians(cfg.camera.pitch_jitter_degrees),
                                                                 math.radians(cfg.camera.pitch_jitter_degrees))}),
            "camera_fov": EventTermCfg(func=dr.cam_fovy, mode="reset",
                                       params={"asset_cfg": camera, "operation": "abs",
                                               "ranges": tuple(math.degrees(2 * math.atan(
                                                   math.tan(math.radians(hfov) / 2)
                                                   * cfg.camera.height / cfg.camera.width))
                                                   for hfov in (cfg.camera.hfov_degrees - cfg.camera.hfov_jitter_degrees,
                                                                cfg.camera.hfov_degrees + cfg.camera.hfov_jitter_degrees))}),
        })
    weights = {
        "track_linear_velocity": 1.5, "track_angular_velocity": 0.5,
        "vertical_velocity": -1.0, "horizontal_angular_velocity": -0.05,
        "orientation": -1.0, "joint_acceleration": -2.5e-7,
        "mechanical_power": -2e-5, "nonfoot_collision": -10.0,
        "action_rate": -0.01, "action_smoothness": -0.01,
    }
    pie = ManagerBasedRlEnvCfg(
        seed=cfg.seed, decimation=round(cfg.control_dt / cfg.physics_dt), auto_reset=False,
        episode_length_s=cfg.episode_seconds,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=cfg.physics_dt, cone="elliptic",
                                         iterations=50, ls_iterations=20),
                          nconmax=64, njmax=256, contact_sensor_maxmatch=64),
        scene=SceneCfg(num_envs=cfg.num_envs, entities={"robot": entity}, terrain=terrain,
                       sensors=(depth_camera, height_scan, feet_scan, feet_contact, nonfoot, base_contact)),
        observations={"proprio": ObservationGroupCfg(
            terms={"proprio": ObservationTermCfg(func=mdp.proprioception,
                    params={"joint_names": robot.joints, "angular_scale": cfg.angular_velocity_scale,
                            "joint_velocity_scale": cfg.joint_velocity_scale,
                            "relative_joint_positions": cfg.relative_joint_positions})},
            concatenate_terms=True, enable_corruption=False)},
        actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",),
                                                      scale=cfg.action_scale, use_default_offset=True,
                                                      clip=action_joint_limits)},
        commands={"twist": UniformVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.1, rel_forward_envs=0.3, heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 1.5),
                                                    lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.2, 1.2)))},
        events=events,
        rewards={name: RewardTermCfg(func=getattr(mdp, name), weight=weight)
                 for name, weight in weights.items()},
        terminations={
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "fallen": TerminationTermCfg(func=mdp.fallen),
            "track_bounds": TerminationTermCfg(func=mdp.track_bounds,
                                               params={"width": cfg.terrain_width}),
            "course_complete": TerminationTermCfg(func=mdp.course_complete,
                                                  params={"length": cfg.terrain_length}),
        },
        curriculum={"terrain": CurriculumTermCfg(func=mdp.terrain_curriculum,
                     params={"length": cfg.terrain_length})} if cfg.curriculum else {},
    )
    # Extend Unitree's real velocity task factory with PIE-specific terms.
    from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
    cfg_base = make_velocity_env_cfg()
    cfg_base.viewer.body_name = robot.base
    for name in ("seed", "decimation", "auto_reset", "episode_length_s", "sim", "scene",
                 "observations", "actions", "commands", "events", "rewards",
                 "terminations", "curriculum"):
        setattr(cfg_base, name, getattr(pie, name))
    return cfg_base
