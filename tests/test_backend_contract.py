"""Small CPU checks of episode isolation and camera geometry; no training here."""

import torch

from src.tasks.pie.reproduction import EnvironmentConfig, load_config
from src.tasks.pie.history import HistoryBuffer


def test_partial_reset_and_masked_camera_delivery():
    history = HistoryBuffer(2, 2, (1,), "cpu")
    history.append(torch.tensor([[1.0], [3.0]]))
    history.append(torch.tensor([[2.0], [4.0]]))
    history.reset(torch.tensor([[10.0], [99.0]]), torch.tensor([0]))
    assert history.data.tolist() == [[[10.0], [10.0]], [[3.0], [4.0]]]
    history.append(torch.tensor([[11.0], [5.0]]), torch.tensor([False, True]))
    assert history.data.tolist() == [[[10.0], [10.0]], [[4.0], [5.0]]]


def test_paper_sensor_timing_and_dimensions():
    cfg = EnvironmentConfig()
    cfg.validate()
    assert cfg.proprio_history == 10
    assert cfg.camera.history == 2
    assert round(1 / cfg.camera.frequency_hz / cfg.control_dt) == 5
    assert cfg.heightmap_shape[0] * cfg.heightmap_shape[1] == 187


def test_config_load_does_not_mutate_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("environment:\n  num_envs: 2\n  camera:\n    width: 80\n")
    cfg = load_config(path)
    assert cfg.num_envs == 2
    assert cfg.camera.height == 60
    assert load_config(path) == cfg


def test_cli_pie_settings_reach_native_camera_and_actuators():
    from src.tasks.pie.config.lite3.env_cfgs import apply_pie_settings, lite3_pie_env_cfg
    cfg = lite3_pie_env_cfg()
    cfg.scene.num_envs = 3
    cfg.pie.camera.width = 64
    cfg.pie.camera.height = 48
    cfg.pie.kp = 35.0
    cfg.pie.randomization = False
    pie = apply_pie_settings(cfg, "cuda:0", 7)
    camera = next(sensor for sensor in cfg.scene.sensors if sensor.name == "depth")
    assert (camera.width, camera.height) == (64, 48)
    assert cfg.scene.num_envs == pie.num_envs == 3
    assert cfg.scene.entities["robot"].articulation.actuators[0].stiffness == 35.0
    assert "camera_position" not in cfg.events
    assert cfg.seed == pie.seed == 7
