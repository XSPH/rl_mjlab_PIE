"""Serializable reproduction settings; unreported paper choices are explicit defaults."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class CameraConfig:
    width: int = 80
    height: int = 60
    history: int = 2
    frequency_hz: float = 10.0
    near: float = 0.1
    far: float = 3.0
    hfov_degrees: float = 87.0
    hfov_jitter_degrees: float = 1.0
    pitch_degrees: float = 30.0
    position: tuple[float, float, float] = (0.25, 0.0, 0.06)
    position_jitter_m: float = 0.01
    pitch_jitter_degrees: float = 1.0
    # Control ticks: five 20ms ticks = one 10Hz camera frame. Paper did not give camera latency.
    latency_steps: tuple[int, int] = (5, 5)


@dataclass
class EnvironmentConfig:
    robot: str = "lite3"
    num_envs: int = 2
    device: str = "cuda:0"
    seed: int = 42
    physics_dt: float = 0.005
    control_dt: float = 0.02
    episode_seconds: float = 20.0
    proprio_history: int = 10
    angular_velocity_scale: float = 0.25
    joint_velocity_scale: float = 0.05
    relative_joint_positions: bool = True
    action_scale: float = 0.25
    kp: float = 30.0
    kd: float = 0.8
    torque_limit: float = 30.5
    heightmap_shape: tuple[int, int] = (17, 11)
    heightmap_spacing: float = 0.1
    terrain_rows: int = 10
    terrain_length: float = 8.0
    terrain_width: float = 3.0
    curriculum: bool = True
    max_initial_level: int = 1
    randomization: bool = True
    camera: CameraConfig = field(default_factory=CameraConfig)

    def validate(self) -> None:
        if self.robot not in {"lite3", "go1"}:
            raise ValueError("robot must be lite3 or go1")
        if self.num_envs < 1 or self.terrain_rows < 1:
            raise ValueError("num_envs and terrain_rows must be positive")
        if self.proprio_history < 1 or self.camera.history < 1:
            raise ValueError("history lengths must be positive")
        if not 0 < self.camera.near < self.camera.far:
            raise ValueError("camera near/far clipping distances are invalid")
        ratio = self.control_dt / self.physics_dt
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        camera_ratio = 1.0 / self.camera.frequency_hz / self.control_dt
        if camera_ratio < 1 or abs(camera_ratio - round(camera_ratio)) > 1e-6:
            raise ValueError("camera period must be an integer multiple of control_dt")
        low, high = self.camera.latency_steps
        if low < 0 or high < low:
            raise ValueError("camera latency range must be nonnegative and ordered")
        if high > round(camera_ratio):
            raise ValueError("camera latency_steps cannot exceed one camera period")


def load_config(path: str | Path | None = None) -> EnvironmentConfig:
    data = {} if path is None else yaml.safe_load(Path(path).read_text()) or {}
    # Training/model settings are consumed by the project-local learner.
    data = data.get("environment", data)
    camera = CameraConfig(**data.pop("camera", {}))
    cfg = EnvironmentConfig(**data, camera=camera)
    cfg.validate()
    return cfg


def save_config(cfg: EnvironmentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
