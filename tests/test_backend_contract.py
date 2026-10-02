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


def test_formal_defaults_restore_native_noise_events_and_solver():
    from src.tasks.pie.config.lite3.env_cfgs import lite3_pie_env_cfg
    from src.tasks.pie.config.lite3.rl_cfg import lite3_pie_runner_cfg
    cfg, runner = lite3_pie_env_cfg(), lite3_pie_runner_cfg()
    assert cfg.scene.num_envs == cfg.pie.num_envs == 4096
    assert (runner.max_iterations, runner.num_steps_per_env, runner.save_interval, runner.seed) == (15000, 24, 500, 42)
    assert (runner.ppo.learning_rate, runner.ppo.epochs, runner.ppo.minibatches) == (.001, 5, 4)
    assert (runner.ppo.schedule, runner.ppo.desired_kl) == ("adaptive", .01)
    assert runner.model.initial_std == 1.0
    assert runner.model.actor_obs_normalization and runner.model.critic_obs_normalization
    assert (cfg.pie.angular_velocity_scale, cfg.pie.joint_velocity_scale) == (1., 1.)
    assert cfg.scene.terrain.max_init_terrain_level == 5
    assert (cfg.sim.mujoco.iterations, cfg.sim.nconmax, cfg.sim.njmax, cfg.sim.mujoco.cone) == (10, 35, 1500, "pyramidal")
    assert cfg.actions["joint_pos"].clip is None and runner.clip_actions is None
    command = cfg.commands["twist"]
    assert (command.resampling_time_range, command.rel_standing_envs, command.rel_forward_envs) == ((3., 8.), .05, 0.)
    assert command.ranges.lin_vel_x == (0., 1.5)
    assert command.ranges.lin_vel_y == (0., 0.) and command.ranges.ang_vel_z == (-1.2, 1.2)
    assert cfg.events["push_robot"].interval_range_s == (5., 6.)
    assert cfg.events["encoder_bias"].params["bias_range"] == (-.015, .015)
    proprio = cfg.observations["proprio"]
    assert proprio.enable_corruption and proprio.history_length == 1
    assert not cfg.observations["critic_proprio"].enable_corruption
    noise = proprio.terms["proprio"].noise
    magnitudes = noise.n_max
    assert magnitudes == (.2,) * 3 + (.05,) * 3 + (0.,) * 3 + (.01,) * 12 + (1.5,) * 12 + (0.,) * 12
    assert noise.n_min == tuple(-v for v in magnitudes)
    assert cfg.events["foot_friction"].params["ranges"] == (.2, 1.2)
    assert cfg.events["payload"].params["ranges"] == (-1., 2.)
    assert len(cfg.rewards) == 10
    camera = next(sensor for sensor in cfg.scene.sensors if sensor.name == "depth")
    assert (camera.width, camera.height, camera.data_types) == (80, 60, ("depth",))


def test_snapshot_keeps_actor_noise_out_of_critic_and_physical_labels():
    from types import SimpleNamespace
    from src.tasks.pie.backend import PieMjlabEnv
    adapter = PieMjlabEnv.__new__(PieMjlabEnv)
    adapter.cfg = EnvironmentConfig(num_envs=2)
    data = SimpleNamespace(root_link_pos_w=torch.ones(2, 3),
                           root_link_lin_vel_b=torch.full((2, 3), 3.0))
    adapter.env = SimpleNamespace(
        obs_buf={"proprio": torch.full((2, 45), 10.), "critic_proprio": torch.zeros(2, 45)},
        scene={"robot": SimpleNamespace(data=data),
               "height_scan": SimpleNamespace(data=SimpleNamespace(
                   hit_pos_w=torch.zeros(2, 187, 3), distances=torch.ones(2, 187))),
               "foot_clearance": SimpleNamespace(data=SimpleNamespace(heights=torch.ones(2, 4)))})
    adapter._proprio_history = HistoryBuffer(2, 10, (45,), "cpu")
    adapter._depth_history = HistoryBuffer(2, 2, (60, 80), "cpu")
    obs = adapter._snapshot()
    assert torch.all(obs["proprio"] == 10)
    assert torch.all(obs["critic"][:, :45] == 0)
    assert torch.all(obs["targets"]["velocity"] == 3)
    assert torch.all(obs["targets"]["heightmap"] == .5)
    torch.testing.assert_close(obs["targets"]["foot_clearance"], torch.full((2, 4), .978))
    obs["critic"].fill_(-100)
    assert torch.all(adapter.env.obs_buf["critic_proprio"] == 0)
    assert torch.all(obs["targets"]["velocity"] == 3)
