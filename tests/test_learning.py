"""CPU checks for observation privacy, recurrent replay, and time limits."""
import torch
import pytest
from dataclasses import asdict, replace
from types import SimpleNamespace
from src.tasks.pie.rl.models import ModelConfig, PIEActorCritic
from src.tasks.pie.rl.learner import (
    PPO, PPOConfig, PIERunnerCfg, PIEOnPolicyRunner, gae,
    checkpoint_model_config, checkpoint_model_state, evaluate,
)

def make_obs(batch=2):
    torch.manual_seed(5)
    return {"proprio": torch.randn(batch, 45),
            "proprio_history": torch.randn(batch, 10, 45),
            "depth": torch.rand(batch, 2, 24, 32),
            "critic": torch.randn(batch, 59),
            "targets": {"velocity": torch.randn(batch, 3),
                        "foot_clearance": torch.rand(batch, 4),
                        "heightmap": torch.randn(batch, 11)}}

def make_model():
    return PIEActorCritic(ModelConfig(heightmap_dim=11, critic_dim=59,
                                     token_dim=16, gru_dim=16,
                                     latent_dim=4, map_latent_dim=8))

def test_privileged_targets_cannot_change_actor():
    model = make_model().eval()
    obs = make_obs(); hidden = model.initial_state(2)
    action1 = model.act(obs, hidden, deterministic=True)[0]
    obs["critic"].add_(1000)
    for target in obs["targets"].values():
        target.add_(1000)
    action2 = model.act(obs, hidden, deterministic=True)[0]
    torch.testing.assert_close(action1, action2, rtol=0, atol=0)

def test_recurrent_reset_and_probability_replay():
    model = make_model().eval(); obs = make_obs()
    hidden = model.initial_state(2)
    action, old_logp, _, _ = model.act(obs, hidden)
    new_logp = model.evaluate(obs, hidden, action)[0]
    torch.testing.assert_close(old_logp, new_logp, rtol=0, atol=0)
    dirty = torch.randn_like(hidden)
    reset = torch.ones(2, dtype=torch.bool)
    with_reset = model.act(obs, dirty, reset, deterministic=True)[0]
    fresh = model.act(obs, hidden, deterministic=True)[0]
    torch.testing.assert_close(with_reset, fresh, rtol=0, atol=0)

def test_time_limits_bootstrap_without_cross_episode_gae():
    rewards = torch.tensor([[1., 1.], [100., 100.]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10., 10.], [0., 0.]])
    terminated = torch.tensor([[True, False], [False, False]])
    truncated = torch.tensor([[False, True], [False, False]])
    advantage, _ = gae(rewards, values, next_values, terminated, truncated, .9, .95)
    torch.testing.assert_close(advantage[0], torch.tensor([1., 10.]))

class ResetEnvironment:
    def __init__(self):
        self.obs = make_obs(); self.steps = 0
    def reset(self):
        return self.obs
    def step(self, actions):
        self.steps += 1
        final = make_obs()
        final["proprio"].fill_(7)
        final["critic"].fill_(7)
        reset_obs = make_obs(); reset_obs["proprio"].fill_(-99)
        reset_obs["critic"].fill_(-99)
        done = torch.ones(2, dtype=torch.bool)
        return reset_obs, torch.ones(2), ~done, done, {
            "terminal_proprio": final["proprio"],
            "terminal_observation": final}

def test_rollout_uses_pre_reset_successor_and_joint_loss_updates():
    torch.set_num_threads(1)
    model = make_model(); env = ResetEnvironment()
    algorithm = PPO(model, PPOConfig(epochs=1, minibatches=1))
    batch, *_ = algorithm.collect(env, env.reset(), model.initial_state(2),
                                 torch.ones(2, dtype=torch.bool), 3)
    assert all(torch.all(frame["successor"] == 7) for frame in batch["frames"])
    with torch.no_grad():
        final_obs = make_obs(); final_obs["critic"].fill_(7)
        expected_value = model.value(final_obs)
        reset_obs = make_obs(); reset_obs["critic"].fill_(-99)
        reset_value = model.value(reset_obs)
    assert not torch.equal(expected_value, reset_value)
    for frame in batch["frames"]:
        torch.testing.assert_close(frame["next_values"], expected_value)
    before_actor = model.actor[0].weight.detach().clone()
    before_encoder = model.depth_encoder[0].weight.detach().clone()
    report = algorithm.update(batch)
    assert all(torch.isfinite(torch.tensor(x)) for x in report.values())
    assert not torch.equal(before_actor, model.actor[0].weight)
    assert not torch.equal(before_encoder, model.depth_encoder[0].weight)

def test_invalid_successor_is_masked_without_nan():
    model = make_model(); obs = make_obs()
    _, _, est = model.distribution(obs, model.initial_state(2))
    successor = obs["proprio"].clone(); successor[0].fill_(float("nan"))
    losses = model.auxiliary_losses(est, obs["targets"], successor, torch.tensor([False, True]))
    assert all(torch.isfinite(loss) for loss in losses.values())

def test_invalid_successor_keeps_current_labels_and_gradients():
    model = make_model(); obs = make_obs()
    _, _, estimates = model.distribution(obs, model.initial_state(2))
    successor = torch.full_like(obs["proprio"], float("nan"))
    losses = model.auxiliary_losses(estimates, obs["targets"], successor,
                                   successor_valid=torch.zeros(2, dtype=torch.bool))
    assert losses["successor"].item() == 0
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert model.clearance_head.weight.grad.abs().sum() > 0
    assert model.height_decoder[0].weight.grad.abs().sum() > 0

def test_policy_gradients_train_estimator_without_privileged_supervision():
    model = make_model(); obs = make_obs()
    distribution, _, _ = model.distribution(obs, model.initial_state(2))
    actions = distribution.mean.detach() + 0.1
    distribution.log_prob(actions).sum().backward()
    assert model.depth_encoder[0].weight.grad.abs().sum() > 0
    assert model.gru.weight_ih.grad.abs().sum() > 0
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert model.map_head.weight.grad.abs().sum() > 0
    assert model.mu_head.weight.grad.abs().sum() > 0
    # Posterior variance is trained by reconstruction/KL, not policy sampling.
    assert model.logvar_head.weight.grad is None
    assert model.critic[0].weight.grad is None

class MixedResetEnvironment:
    def __init__(self):
        self.steps = 0
    def reset(self):
        return make_obs(3)
    def step(self, actions):
        self.steps += 1
        obs = make_obs(3)
        terminated = torch.zeros(3, dtype=torch.bool)
        truncated = torch.zeros(3, dtype=torch.bool)
        if self.steps == 2:
            terminated[0] = True
        if self.steps == 3:
            truncated[1] = True
        return obs, torch.ones(3), terminated, truncated, {
            "terminal_observation": obs, "terminal_proprio": obs["proprio"]}

def test_updated_weight_replay_preserves_continuing_episode_memory():
    torch.set_num_threads(1)
    model = make_model(); env = MixedResetEnvironment()
    algorithm = PPO(model, PPOConfig(epochs=1, minibatches=1))
    batch, _, old_hidden, _ = algorithm.collect(
        env, env.reset(), model.initial_state(3),
        torch.ones(3, dtype=torch.bool), 3)
    algorithm.update(batch)
    refreshed, reset_mask = algorithm.refresh_hidden(batch)
    with torch.no_grad():
        expected = batch["hidden"].clone()
        for frame in batch["frames"]:
            _, expected = model.encode(frame["obs"], expected, frame["reset"])
        expected[1] = 0
    torch.testing.assert_close(refreshed, expected)
    assert reset_mask.tolist() == [False, True, False]
    assert torch.all(refreshed[1] == 0)
    assert refreshed[0].norm() > 0 and refreshed[2].norm() > 0
    assert not torch.equal(refreshed[2], old_hidden[2])
    assert not refreshed.requires_grad


def test_native_scalar_std_and_adaptive_policy_kl():
    model = make_model()
    assert model.output_distribution.std_type == "scalar"
    assert model.output_distribution.std_range == [1e-6, 1e6]
    torch.testing.assert_close(model.output_distribution.std_param, torch.ones(12))
    algorithm = PPO(model)
    mean, std = torch.zeros(2, 12), torch.ones(2, 12)
    high_kl = algorithm.adapt_learning_rate(mean, std, mean + 1, std)
    assert high_kl > 0.02
    assert algorithm.learning_rate == pytest.approx(1e-3 / 1.5)
    low_kl = algorithm.adapt_learning_rate(mean, std, mean + .01, std)
    assert 0 < low_kl < .005
    assert algorithm.learning_rate == pytest.approx(1e-3)
    assert algorithm.optimizer.param_groups[0]["lr"] == algorithm.learning_rate
    algorithm.cfg.schedule = "fixed"
    algorithm.adapt_learning_rate(mean, std, mean + 1, std)
    assert algorithm.learning_rate == pytest.approx(1e-3)


def test_normalization_keeps_raw_labels_and_likelihood_replay():
    model = make_model()
    obs = make_obs()
    labels = {k: v.clone() for k, v in obs["targets"].items()}
    successor = obs["proprio"].clone()
    hidden = model.initial_state(2)
    estimates, _ = model.encode(obs, hidden)
    actor_inputs = torch.cat((obs["proprio"], model.features(estimates)), -1)
    model.update_normalization(actor_inputs, obs["critic"])
    assert model.actor_normalizer.count.item() == model.critic_normalizer.count.item() == 2
    actions, logp, _, _ = model.act(obs, hidden)
    torch.testing.assert_close(logp, model.evaluate(obs, hidden, actions)[0], rtol=0, atol=0)
    for key, label in labels.items():
        torch.testing.assert_close(obs["targets"][key], label)
    torch.testing.assert_close(obs["proprio"], successor)
    assert model.actor_normalizer.count.item() == 2
    model.eval()
    model.update_normalization(actor_inputs + 100, obs["critic"] + 100)
    assert model.actor_normalizer.count.item() == 2


class CpuRunnerEnvironment:
    """Only a two-world tensor contract; never constructs a simulator."""
    def __init__(self):
        from src.tasks.pie.reproduction import EnvironmentConfig
        self.device = torch.device("cpu")
        self.num_envs, self.num_actions = 2, 12
        self.cfg = EnvironmentConfig(num_envs=2, device="cpu")
        self.env = SimpleNamespace(episode_length_buf=torch.zeros(2, dtype=torch.long),
                                   max_episode_length=1000)
        self.reset_calls = 0
        self.last_actions = None

    def reset(self):
        self.reset_calls += 1
        self.env.episode_length_buf.zero_()
        return make_obs()

    def step(self, actions):
        self.last_actions = actions.clone()
        self.env.episode_length_buf.add_(1)
        obs = make_obs()
        done = torch.zeros(2, dtype=torch.bool)
        return obs, torch.tensor([1., 2.]), done, done, {
            "terminal_proprio": obs["proprio"],
            "log": {"Episode_Reward/track_linear_velocity": torch.tensor(0.75)}}


def cpu_runner_cfg():
    model = ModelConfig(heightmap_dim=11, critic_dim=59, token_dim=16, gru_dim=16,
                        latent_dim=4, map_latent_dim=8,
                        proprio_hidden_dims=(16,), actor_hidden_dims=(16,),
                        critic_hidden_dims=(16,), cnn_hidden_channels=(4, 8),
                        successor_hidden_dims=(16,), height_decoder_hidden_dims=(16,))
    return PIERunnerCfg(model=model, ppo=PPOConfig(epochs=1, minibatches=1),
                        num_steps_per_env=2, max_iterations=3, save_interval=2)


def test_periodic_atomic_checkpoint_tensorboard_and_resume(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(1)
    cfg = cpu_runner_cfg()
    env = CpuRunnerEnvironment()
    runner = PIEOnPolicyRunner(env, asdict(cfg), str(tmp_path), "cpu")
    result = runner.learn()
    assert runner.current_learning_iteration == 3
    assert env.env.episode_length_buf.min() >= 6
    assert result["total_transitions"] == 12
    assert (tmp_path / "model_2.pt").is_file()
    assert (tmp_path / "model_3.pt").is_file()
    assert not (tmp_path / "model_1.pt").exists()
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert saved["iter"] == saved["iterations"] == 3
    assert saved["optimizer_state_dict"]["state"]
    assert all(t.device.type == "cpu" for t in saved["model"].values())
    assert saved["model"]["actor_normalizer.count"].item() == 12
    assert saved["model"]["critic_normalizer.count"].item() == 12
    events = EventAccumulator(str(tmp_path)).Reload()
    assert {"Train/mean_reward", "Train/policy_kl", "Train/kl", "Train/velocity",
            "Train/foot_clearance", "Train/heightmap", "Train/successor",
            "Episode_Reward/track_linear_velocity"} <= set(events.Tags()["scalars"])
    assert events.Scalars("Train/mean_reward")[-1].value == 1.5

    restored_env = CpuRunnerEnvironment()
    resumed_cfg = replace(cfg, model=replace(cfg.model, initial_std=.9),
                          ppo=replace(cfg.ppo, epochs=2))
    resumed = PIEOnPolicyRunner(restored_env, resumed_cfg, str(tmp_path / "resume"), "cpu")
    resumed.load(result["checkpoint"])
    assert resumed.current_learning_iteration == 3
    assert resumed.cfg.model.initial_std == .9
    assert resumed._training_state["algorithm"].cfg.epochs == 2
    assert torch.all(resumed._training_state["hidden"] == 0)
    assert resumed._training_state["reset_mask"].all()
    for key, expected in saved["model"].items():
        torch.testing.assert_close(resumed._training_state["model"].state_dict()[key], expected)
    restored_optimizer = resumed._training_state["algorithm"].optimizer.state_dict()
    assert restored_optimizer["param_groups"] == saved["optimizer_state_dict"]["param_groups"]
    for key, expected in saved["optimizer_state_dict"]["state"].items():
        for moment, value in expected.items():
            torch.testing.assert_close(restored_optimizer["state"][key][moment], value)
    resumed.learn(num_learning_iterations=1)
    assert resumed.current_learning_iteration == 4
    assert resumed._training_state["total_transitions"] == 16
    assert resumed._training_state["model"].actor_normalizer.count.item() == 16

    # A failed save must preserve the last complete checkpoint and clean its temp file.
    original_bytes = (tmp_path / "checkpoint.pt").read_bytes()
    def failed_save(*_args, **_kwargs):
        raise OSError("simulated disk write failure")
    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="simulated"):
        runner.save(tmp_path / "checkpoint.pt")
    assert (tmp_path / "checkpoint.pt").read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_minimal_weights_remain_evaluable(tmp_path):
    cfg = replace(cpu_runner_cfg().model, initial_std=.5,
                  actor_obs_normalization=False, critic_obs_normalization=False)
    old_model = PIEActorCritic(cfg).eval()
    weights = dict(old_model.state_dict())
    weights["log_std"] = weights.pop("output_distribution.std_param").log()
    model_config = asdict(cfg)
    model_config.pop("actor_obs_normalization")
    model_config.pop("critic_obs_normalization")
    checkpoint = tmp_path / "legacy.pt"
    saved = {"model_config": model_config, "model": weights, "iterations": 1}
    torch.save(saved, checkpoint)
    loaded_cfg = checkpoint_model_config(saved)
    assert not loaded_cfg.actor_obs_normalization and not loaded_cfg.critic_obs_normalization
    loaded_model = PIEActorCritic(loaded_cfg).eval()
    loaded_model.load_state_dict(checkpoint_model_state(saved))
    obs = make_obs()
    expected = old_model.act(obs, old_model.initial_state(2), deterministic=True)[0]
    actual = loaded_model.act(obs, loaded_model.initial_state(2), deterministic=True)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert evaluate(CpuRunnerEnvironment(), checkpoint, steps=1)["steps"] == 1


@pytest.mark.parametrize("restored_iteration, expected_remaining", [(14900, 100), (15000, 0)])
def test_training_entry_uses_total_target_after_resume(tmp_path, monkeypatch, restored_iteration, expected_remaining):
    import scripts.train as entry
    import src.tasks.pie.backend as backend
    from src.tasks.pie.config.lite3.env_cfgs import lite3_pie_env_cfg
    calls = {}
    class NativeStub:
        def __init__(self, **_kwargs):
            pass
        def close(self):
            calls["closed"] = True
    class RunnerStub:
        def __init__(self, *_args, **_kwargs):
            self.current_learning_iteration = 0
        def add_git_repo_to_log(self, *_args):
            pass
        def load(self, *_args):
            self.current_learning_iteration = restored_iteration
        def learn(self, **kwargs):
            calls["learn"] = kwargs
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(entry, "configure_torch_backends", lambda: None)
    monkeypatch.setattr(entry, "ManagerBasedRlEnv", NativeStub)
    monkeypatch.setattr(entry, "load_runner_cls", lambda _task: RunnerStub)
    monkeypatch.setattr(entry, "get_checkpoint_path", lambda *_args: tmp_path / "model.pt")
    monkeypatch.setattr(entry, "dump_yaml", lambda *_args: None)
    monkeypatch.setattr(backend, "PieMjlabEnv", lambda _cfg, native_env, **_kwargs: native_env)
    cfg = entry.TrainConfig(env=lite3_pie_env_cfg(), agent=PIERunnerCfg(resume=True))
    entry.run_train("Unitree-Lite3-PIE", cfg, tmp_path / "new-run")
    assert calls["closed"]
    if expected_remaining:
        assert calls["learn"] == {"num_learning_iterations": expected_remaining, "init_at_random_ep_len": True}
    else:
        assert "learn" not in calls
