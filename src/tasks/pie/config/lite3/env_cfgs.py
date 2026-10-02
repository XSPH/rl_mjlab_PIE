from copy import deepcopy
from dataclasses import dataclass, field, fields
from mjlab.envs import ManagerBasedRlEnvCfg
from ...reproduction import EnvironmentConfig
from ...env_config import make_env_config


@dataclass
class Lite3PIEEnvCfg(ManagerBasedRlEnvCfg):
    pie: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def lite3_pie_env_cfg(play=False):
    pie = EnvironmentConfig(num_envs=2) if play else EnvironmentConfig()
    if play:
        pie.randomization = False
        pie.curriculum = False
        pie.terrain_rows = 1
    base = make_env_config(pie)
    return Lite3PIEEnvCfg(**{f.name: getattr(base, f.name) for f in fields(ManagerBasedRlEnvCfg)}, pie=pie)


def apply_pie_settings(cfg: Lite3PIEEnvCfg, device: str, seed: int) -> EnvironmentConfig:
    """Apply parsed PIE settings before compiling native cameras/actuators.

    The ordinary Unitree num-envs and timing overrides remain authoritative when
    changed; camera, robot, terrain and DR settings come from ``env.pie``.
    """
    pie = deepcopy(cfg.pie)
    defaults = EnvironmentConfig()
    pie.num_envs, pie.device, pie.seed = cfg.scene.num_envs, device, seed
    native_dt = cfg.sim.mujoco.timestep
    if native_dt != defaults.physics_dt:
        pie.physics_dt = native_dt
    default_decimation = round(defaults.control_dt / defaults.physics_dt)
    if cfg.decimation != default_decimation or native_dt != defaults.physics_dt:
        pie.control_dt = cfg.decimation * pie.physics_dt
    if cfg.episode_length_s != defaults.episode_seconds:
        pie.episode_seconds = cfg.episode_length_s
    native = make_env_config(pie)
    for name in ("scene", "observations", "actions", "commands", "events",
                 "rewards", "terminations", "curriculum", "auto_reset",
                 "decimation", "episode_length_s"):
        setattr(cfg, name, getattr(native, name))
    cfg.sim.mujoco.timestep = pie.physics_dt
    cfg.seed, cfg.pie = seed, pie
    cfg.viewer.body_name = native.viewer.body_name
    return pie
