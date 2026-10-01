"""Native mjlab articulation configurations and canonical FL/FR/RL/RR ordering."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .reproduction import EnvironmentConfig


@dataclass(frozen=True)
class RobotProfile:
    base: str
    joints: tuple[str, ...]
    feet: tuple[str, ...]
    feet_geoms: tuple[str, ...]
    nonfoot_geoms: tuple[str, ...]
    standing_height: float


def profile(name: str) -> RobotProfile:
    if name == "lite3":
        legs = ("FL", "FR", "HL", "HR")
        return RobotProfile(
            base="TORSO",
            joints=tuple(f"{leg}_{joint}_joint" for leg in legs
                         for joint in ("HipX", "HipY", "Knee")),
            feet=tuple(f"{leg}_FOOT" for leg in legs),
            feet_geoms=tuple(f"{leg}_FOOT_collision" for leg in legs),
            nonfoot_geoms=("TORSO_collision",) + tuple(
                f"{leg}_{segment}_collision" for leg in legs
                for segment in ("HIP", "THIGH", "SHANK")),
            standing_height=0.30,
        )
    if name == "go1":
        legs = ("FL", "FR", "RL", "RR")
        return RobotProfile(
            base="trunk",
            joints=tuple(f"{leg}_{joint}_joint" for leg in legs
                         for joint in ("hip", "thigh", "calf")),
            feet=legs,
            feet_geoms=tuple(f"{leg}_foot_collision" for leg in legs),
            nonfoot_geoms=("trunk_collision", "head_collision") + tuple(
                f"{leg}_hip_collision" for leg in legs) + tuple(
                f"{leg}_{segment}_collision{index}" for leg in legs
                for segment, count in (("thigh", 3), ("calf", 2))
                for index in range(1, count + 1)),
            standing_height=0.278,
        )
    raise ValueError(f"Unknown robot: {name}")


def _camera_quaternion(pitch_degrees: float) -> tuple[float, float, float, float]:
    # MuJoCo cameras look along local -Z; image right is local +X.
    pitch = np.deg2rad(pitch_degrees)
    right = np.array([0.0, -1.0, 0.0])
    up = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    back = np.array([-np.cos(pitch), 0.0, np.sin(pitch)])
    x, y, z, w = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
    return float(w), float(x), float(y), float(z)


def make_robot_config(cfg: EnvironmentConfig) -> tuple[EntityCfg, RobotProfile]:
    robot = profile(cfg.robot)
    if cfg.robot == "lite3":
        xml_path = Path(__file__).parents[2] / "assets" / "robots" / "lite3" / "Lite3.xml"

        def original_spec() -> mujoco.MjSpec:
            spec = mujoco.MjSpec.from_file(str(xml_path))
            # The official asset is a complete demo scene. Keep only the robot.
            for geom in list(spec.worldbody.geoms):
                spec.delete(geom)
            for light in list(spec.lights):
                spec.delete(light)
            for camera in list(spec.cameras):
                spec.delete(camera)
            for actuator in list(spec.actuators):
                spec.delete(actuator)
            for sensor in list(spec.sensors):
                spec.delete(sensor)
            # Zero foot inertia in the vendor demo fails current compiler checks.
            # A 22 mm solid sphere has I = (2/5)*m*r^2.
            for foot in robot.feet:
                body = spec.body(foot)
                body.inertia = np.full(3, 0.4 * 0.02 * 0.022**2)
                body.add_site(name=f"{foot}_scan", pos=(0, 0, 0), size=(0.005,))
            # Static terrain and the robot can collide; body/leg collision penalties
            # must see more than only the four feet.
            for geom in spec.geoms:
                if geom.name.endswith("_collision"):
                    geom.contype = 1
                    geom.conaffinity = 1
                    if geom.name in robot.feet_geoms:
                        geom.priority = 1
            return spec

        init = EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, robot.standing_height),
            joint_pos={".*HipX_joint": 0.0, ".*HipY_joint": -0.8, ".*Knee_joint": 1.6},
            joint_vel={".*": 0.0},
        )
    else:
        from mjlab.asset_zoo.robots.unitree_go1.go1_constants import get_spec, INIT_STATE
        original_spec = get_spec
        init = INIT_STATE

    def mounted_camera_spec() -> mujoco.MjSpec:
        spec = original_spec()
        for geom in spec.geoms:
            if geom.name in robot.feet_geoms:
                geom.priority = 1
        body = spec.body(robot.base)
        camera = cfg.camera
        fovy = np.rad2deg(2 * np.arctan(
            np.tan(np.deg2rad(camera.hfov_degrees) / 2) * camera.height / camera.width))
        body.add_camera(
            name="depth_camera", pos=camera.position,
            quat=_camera_quaternion(camera.pitch_degrees), fovy=float(fovy),
        )
        return spec

    hip_names = (".*Hip[XY]_joint",) if cfg.robot == "lite3" else (".*_(hip|thigh)_joint",)
    knee_names = (".*Knee_joint",) if cfg.robot == "lite3" else (".*_calf_joint",)
    motors = tuple(BuiltinPositionActuatorCfg(
        target_names_expr=names, stiffness=cfg.kp, damping=cfg.kd,
        effort_limit=min(limit, cfg.torque_limit), armature=0.01,
        delay_min_lag=0,
        delay_max_lag=int(0.015 / cfg.physics_dt + 1e-9) if cfg.randomization else 0,
    ) for names, limit in ((hip_names, 24.0), (knee_names, 30.5)))
    return EntityCfg(
        spec_fn=mounted_camera_spec, init_state=init,
        sort_actuators=True,
        articulation=EntityArticulationInfoCfg(actuators=motors, soft_joint_pos_limit_factor=0.9),
    ), robot
