"""Recurrent clipped PPO with concurrent PIE estimator losses."""
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import json
import os
import random
import tempfile
import torch
from torch.distributions import Normal, kl_divergence
from .models import ModelConfig, PIEActorCritic
from mjlab.rl import RslRlBaseRunnerCfg

@dataclass
class PPOConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    schedule: str = "adaptive"
    desired_kl: float | None = 0.01
    entropy_weight: float = 0.01
    value_weight: float = 1.0
    estimation_weight: float = 1.0
    kl_weight: float = 1.0
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if self.schedule not in ("fixed", "adaptive"):
            raise ValueError("PIE schedule must be fixed or adaptive")
        if self.learning_rate <= 0 or self.epochs < 1 or self.minibatches < 1:
            raise ValueError("learning_rate, epochs and minibatches must be positive")
        if self.desired_kl is not None and self.desired_kl <= 0:
            raise ValueError("desired_kl must be positive or None")

@dataclass
class PIERunnerCfg(RslRlBaseRunnerCfg):
    """Native mjlab task registration with PIE-specific estimator/PPO settings."""
    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 15000
    save_interval: int = 500
    experiment_name: str = "lite3_pie"
    logger: str = "tensorboard"
    upload_model: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

class PIEOnPolicyRunner:
    """Custom native-registry runner: RSL MLPs plus PIE's recurrent joint loss.

    This runner consumes PieMjlabEnv's synchronized image/history/target dict.
    Stock RSL VecEnvWrapper/OnPolicyRunner would discard these auxiliary labels.
    """
    def __init__(self, env, train_cfg, log_dir="runs/lite3_pie", device=None):
        self.env = env
        device = torch.device(device or env.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != torch.device(env.device):
            raise ValueError("PIE requires the RL device to match the simulation device.")
        if isinstance(train_cfg, dict):
            data = dict(train_cfg)
            data["model"] = ModelConfig(**data["model"])
            data["ppo"] = PPOConfig(**data["ppo"])
            train_cfg = PIERunnerCfg(**data)
        self.cfg = train_cfg
        if self.cfg.logger != "tensorboard":
            raise ValueError("The PIE runner currently implements TensorBoard logging only.")
        if self.cfg.upload_model:
            raise ValueError("PIE model upload/export is not implemented; use upload_model=False.")
        if self.cfg.save_interval < 1:
            raise ValueError("save_interval must be positive")
        self.log_dir = log_dir
        self._training_state = {}
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations=None, init_at_random_ep_len=True):
        result = train(
            self.env,
            iterations=self.cfg.max_iterations if num_learning_iterations is None else num_learning_iterations,
            rollout_steps=self.cfg.num_steps_per_env, output_dir=self.log_dir,
            seed=self.cfg.seed, model_config=self.cfg.model, ppo_config=self.cfg.ppo,
            init_at_random_ep_len=init_at_random_ep_len,
            save_interval=self.cfg.save_interval,
            _state=self._training_state)
        self.current_learning_iteration = self._training_state["iterations"]
        return result

    def add_git_repo_to_log(self, source_file):
        if self.log_dir is None:
            return
        import subprocess
        root = Path(source_file).resolve().parent
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        output = Path(self.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "upstream_commit.txt").write_text(result.stdout)

    def save(self, path, infos=None):
        if not self._training_state:
            raise RuntimeError("Initialize the PIE model with learn() or load() before saving.")
        save_checkpoint(path, self.env, self._training_state, self.cfg.seed, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        """Restore weights/Adam/iteration; simulation and GRU start new episodes.

        Read tensors on CPU before transferring model/optimizer state to the RL
        device. This follows the native runner load contract without claiming to
        serialize a MuJoCo simulation or partially completed episodes.
        """
        del map_location
        saved = torch.load(path, map_location="cpu", weights_only=True)
        load_cfg = ({"actor": True, "critic": True, "optimizer": True, "iteration": True}
                    if load_cfg is None else load_cfg)
        unknown = set(load_cfg) - {"actor", "critic", "optimizer", "iteration"}
        if unknown:
            raise ValueError("Unsupported PIE load options: " + ", ".join(sorted(unknown)))
        if load_cfg.get("optimizer") and "log_std" in saved.get("model", {}):
            raise ValueError("Legacy log-std Adam state cannot resume the scalar-std model; load weights only.")
        saved_cfg = checkpoint_model_config(saved)
        saved_architecture, configured_architecture = asdict(saved_cfg), asdict(self.cfg.model)
        saved_architecture.pop("initial_std")
        configured_architecture.pop("initial_std")
        if saved_architecture != configured_architecture:
            raise ValueError("Checkpoint model_config differs from the configured PIE model.")
        if not self._training_state:
            initialize_state(self.env, self._training_state, self.cfg.seed, self.cfg.model, self.cfg.ppo)
        state = self._training_state
        weights = checkpoint_model_state(saved)
        if load_cfg.get("actor") and load_cfg.get("critic"):
            state["model"].load_state_dict(weights, strict=strict)
        else:
            selected = {k: v for k, v in weights.items() if
                        load_cfg.get("critic") and k.startswith(("critic.", "critic_normalizer."))}
            if load_cfg.get("actor"):
                selected.update({k: v for k, v in weights.items()
                                 if not k.startswith(("critic.", "critic_normalizer."))})
            current = state["model"].state_dict()
            if strict:
                expected = {k for k in current if
                            (bool(load_cfg.get("critic")) if k.startswith(("critic.", "critic_normalizer."))
                             else bool(load_cfg.get("actor")))}
                if set(selected) != expected:
                    raise RuntimeError("Checkpoint is missing or has unexpected selected PIE model keys.")
            current.update(selected)
            state["model"].load_state_dict(current, strict=strict)
        if load_cfg.get("optimizer"):
            state["algorithm"].optimizer.load_state_dict(
                saved.get("optimizer_state_dict", saved.get("optimizer")))
            state["algorithm"].learning_rate = state["algorithm"].optimizer.param_groups[0]["lr"]
            # Native resume restores Adam/LR; epochs, schedule and loss weights
            # remain governed by the current task, not old smoke-run defaults.
        if load_cfg.get("iteration"):
            state["iterations"] = int(saved.get("iter", saved.get("iterations", 0)))
            state["total_transitions"] = int(saved.get("total_transitions", 0))
            self.current_learning_iteration = state["iterations"]
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"].cpu())
        if "python_rng" in saved:
            random.setstate(saved["python_rng"])
        if "cuda_rng" in saved and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        # Existing observations and GRU are deliberately restarted on resume.
        state["obs"] = self.env.reset()
        state["hidden"] = state["model"].initial_state(self.env.num_envs)
        state["reset_mask"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=state["hidden"].device)
        state["randomize_episode_length"] = True
        return saved.get("infos")

def clone_observation(obs):
    return {key: value.detach().clone() for key, value in obs.items()
            if key in ("proprio", "proprio_history", "depth", "critic")}

def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Bootstrap time limits, but never propagate advantage across a reset."""
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        bootstrap = torch.where(terminated[t], torch.zeros_like(next_values[t]), next_values[t])
        trace = torch.where(terminated[t] | truncated[t], torch.zeros_like(following), following)
        delta = rewards[t] + gamma * bootstrap - values[t]
        following = delta + gamma * lam * trace
        advantages[t] = following
    return advantages, advantages + values

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PPO:
    """One optimizer jointly trains actor, critic, encoder, and estimators.

    Actor gradients pass through the estimator as well as its supervised losses.
    This is an explicit reproduction choice; the paper specifies concurrent
    optimization without defining the optimizer/gradient boundary.
    """
    def __init__(self, model, cfg=None):
        self.model = model
        self.cfg = cfg or PPOConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        self.learning_rate = self.cfg.learning_rate

    def adapt_learning_rate(self, old_mean, old_std, mean, std):
        """Native RSL-RL Gaussian KL(old || new) and adaptive schedule."""
        with torch.no_grad():
            policy_kl = kl_divergence(Normal(old_mean, old_std), Normal(mean, std)).sum(-1).mean()
            if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
                if policy_kl > self.cfg.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0.0 < policy_kl < self.cfg.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
        return policy_kl

    @torch.no_grad()
    def collect(self, env, obs, hidden, reset_mask, steps):
        if steps < 1:
            raise ValueError("rollout_steps must be positive")
        frames = []
        self.model.train()
        start_hidden = hidden.detach().clone()
        for _ in range(steps):
            stored = clone_observation(obs)
            targets = {k: v.detach().clone() for k, v in obs["targets"].items()}
            actions, logp, values, next_hidden = self.model.act(obs, hidden, reset_mask)
            old_mean, old_std = self.model.output_distribution.params
            # The actor's raw features are recovered from the same recurrent
            # forward pass; no privileged labels enter its statistics.
            actor_inputs = torch.cat((obs["proprio"], self.model.features_from_hidden(next_hidden)), -1)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            terminated, truncated = terminated.bool().clone(), truncated.bool().clone()
            next_values = self.model.value(next_obs)
            if truncated.any():
                final_obs = info.get("terminal_observation")
                if final_obs is None:
                    raise RuntimeError("time limits require terminal_observation BEFORE reset")
                final_value = self.model.value(final_obs)
                next_values = torch.where(truncated, final_value, next_values)
            successor = info.get("terminal_proprio")
            if successor is None:
                if (terminated | truncated).any():
                    raise RuntimeError("autoreset transitions require pre-reset terminal_proprio")
                successor = next_obs["proprio"]
            valid = torch.isfinite(stored["proprio"]).all(-1)
            successor_valid = torch.isfinite(successor).all(-1)
            frames.append({
                "obs": stored, "targets": targets, "actions": actions.detach().clone(),
                "old_logp": logp.detach().clone(), "values": values.detach().clone(),
                "old_mean": old_mean.detach().clone(), "old_std": old_std.detach().clone(),
                "actor_inputs": actor_inputs.detach().clone(),
                "rewards": rewards.detach().clone(), "next_values": next_values.detach().clone(),
                "terminated": terminated, "truncated": truncated,
                "successor": successor.detach().clone(), "valid": valid,
                "successor_valid": successor_valid,
                "reset": reset_mask.detach().clone(),
                "log": {k: float(v.detach().float().mean()) if isinstance(v, torch.Tensor) else float(v)
                        for k, v in info.get("log", {}).items()}})
            obs, hidden = next_obs, next_hidden.detach()
            reset_mask = terminated | truncated
        def stack(key):
            return torch.stack([frame[key] for frame in frames])
        advantages, returns = gae(
            stack("rewards"), stack("values"), stack("next_values"),
            stack("terminated"), stack("truncated"), self.cfg.gamma, self.cfg.gae_lambda)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # Only the explicit 1-transition smoke helper can reach this case.
            advantages = advantages - advantages.mean()
        batch = {"frames": frames, "hidden": start_hidden,
                 "advantages": advantages, "returns": returns}
        return batch, obs, hidden, reset_mask

    def update(self, batch):
        """Minibatches contain whole environment trajectories, not shuffled steps."""
        frames = batch["frames"]
        batch_size = batch["hidden"].shape[0]
        records = []
        if self.cfg.epochs < 1 or self.cfg.minibatches < 1:
            raise ValueError("epochs/minibatches must be positive")
        self.model.train()
        for _ in range(self.cfg.epochs):
            indices = torch.randperm(batch_size, device=batch["hidden"].device)
            for ids in torch.tensor_split(indices, min(batch_size, self.cfg.minibatches)):
                hidden = batch["hidden"][ids]
                logps, entropies, values, aux, means, stds = [], [], [], [], [], []
                for frame in frames:
                    obs = {k: v[ids] for k, v in frame["obs"].items()}
                    logp, entropy, value, hidden, estimates = self.model.evaluate(
                        obs, hidden, frame["actions"][ids], frame["reset"][ids])
                    logps.append(logp); entropies.append(entropy); values.append(value)
                    mean, std = self.model.output_distribution.params
                    means.append(mean); stds.append(std)
                    targets = {k: v[ids] for k, v in frame["targets"].items()}
                    aux.append(self.model.auxiliary_losses(
                        estimates, targets, frame["successor"][ids], frame["valid"][ids],
                        frame["successor_valid"][ids]))
                logps, entropies, values = torch.stack(logps), torch.stack(entropies), torch.stack(values)
                old_logps = torch.stack([f["old_logp"][ids] for f in frames])
                policy_kl = self.adapt_learning_rate(
                    torch.stack([f["old_mean"][ids] for f in frames]),
                    torch.stack([f["old_std"][ids] for f in frames]),
                    torch.stack(means), torch.stack(stds))
                ratio = (logps - old_logps).exp()
                advantage = batch["advantages"][:, ids]
                policy = -torch.minimum(ratio * advantage,
                    ratio.clamp(1-self.cfg.clip, 1+self.cfg.clip) * advantage).mean()
                old_values = torch.stack([f["values"][ids] for f in frames])
                clipped_values = old_values + (values-old_values).clamp(-self.cfg.clip, self.cfg.clip)
                returns = batch["returns"][:, ids]
                value_loss = torch.maximum((values-returns).square(),
                                           (clipped_values-returns).square()).mean()
                losses = {k: torch.stack([a[k] for a in aux]).mean() for k in aux[0]}
                estimation = sum(losses[k] for k in ("velocity", "foot_clearance", "heightmap", "successor"))
                total = (policy + self.cfg.value_weight * value_loss
                         - self.cfg.entropy_weight * entropies.mean()
                         + self.cfg.estimation_weight * (estimation + self.cfg.kl_weight * losses["kl"]))
                if not torch.isfinite(total):
                    raise FloatingPointError("non-finite PPO/estimator loss")
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                records.append({**{k: float(v.detach()) for k, v in losses.items()},
                    "loss": float(total.detach()), "policy": float(policy.detach()),
                    "value": float(value_loss.detach()), "grad_norm": float(grad_norm.detach()),
                    "policy_kl": float(policy_kl), "entropy": float(entropies.detach().mean())})
        metrics = {key: sum(r[key] for r in records)/len(records) for key in records[0]}
        metrics["learning_rate"] = self.learning_rate
        metrics["action_std"] = float(self.model.output_distribution.std_param.detach().mean())
        return metrics

    @torch.no_grad()
    def refresh_hidden(self, batch):
        """Replay this rollout under updated weights, retaining episode memory.

        The detached rollout-start state is the truncated recurrent boundary.
        Done masks reset only the corresponding episode. The successor
        observation has not been encoded yet and belongs to the next rollout.
        """
        hidden = batch["hidden"].detach().clone()
        for frame in batch["frames"]:
            _, hidden = self.model.encode(frame["obs"], hidden, frame["reset"])
        final = batch["frames"][-1]
        reset_mask = final["terminated"] | final["truncated"]
        hidden = hidden * (~reset_mask).unsqueeze(-1)
        return hidden.detach(), reset_mask.detach().clone()

def initialize_state(env, state, seed, model_config, ppo_config):
    seed_everything(seed)
    obs = env.reset()
    cfg = model_config or ModelConfig(
        proprio_dim=obs["proprio"].shape[-1],
        proprio_history=obs["proprio_history"].shape[1],
        depth_history=obs["depth"].shape[1],
        heightmap_dim=obs["targets"]["heightmap"].shape[-1],
        critic_dim=obs["critic"].shape[-1])
    cfg.validate_observation(obs, env.num_actions)
    model = PIEActorCritic(cfg).to(obs["proprio"].device)
    hidden = model.initial_state(obs["proprio"].shape[0])
    state.update(model_config=cfg, model=model, algorithm=PPO(model, ppo_config),
                 obs=obs, hidden=hidden, iterations=0, total_transitions=0,
                 reset_mask=torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device),
                 randomize_episode_length=True)


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(v) for v in value)
    return value


def save_checkpoint(path, env, state, seed, infos=None):
    """Atomically write native runner keys plus the existing PIE evaluation keys.

    Actor weights include PIE's estimator, Gaussian and actor normalizer; these
    are consumed by PIEOnPolicyRunner, not the stock non-PIE MLP runner.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = cpu_state(state["model"].state_dict())
    optimizer = cpu_state(state["algorithm"].optimizer.state_dict())
    environment_cfg = getattr(env, "config", getattr(env, "cfg", None))
    environment_cfg = asdict(environment_cfg) if is_dataclass(environment_cfg) else {}
    critic_prefix = ("critic.", "critic_normalizer.")
    saved = {
        "format_version": 2, "model": model, "model_config": asdict(state["model_config"]),
        "actor_state_dict": {k: v for k, v in model.items() if not k.startswith(critic_prefix)},
        "critic_state_dict": {k: v for k, v in model.items() if k.startswith(critic_prefix)},
        "ppo_config": asdict(state["algorithm"].cfg), "optimizer": optimizer,
        "optimizer_state_dict": optimizer, "learning_rate": state["algorithm"].learning_rate,
        "iterations": state["iterations"], "iter": state["iterations"], "infos": infos,
        "seed": seed, "environment_config": environment_cfg,
        "total_transitions": state.get("total_transitions", 0),
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
    }
    if torch.cuda.is_initialized():
        saved["cuda_rng"] = [rng.cpu() for rng in torch.cuda.get_rng_state_all()]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = stream.name
            torch.save(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_model_state(saved):
    weights = dict(saved["model"] if "model" in saved else
                   {**saved["actor_state_dict"], **saved["critic_state_dict"]})
    # Preserve evaluation of the earlier minimal checkpoints. Adam moments in
    # log space cannot be converted into scalar-space moments for training.
    if "log_std" in weights:
        weights["output_distribution.std_param"] = weights.pop("log_std").clamp(-5.0, 2.0).exp()
    return weights


def checkpoint_model_config(saved):
    data = dict(saved["model_config"])
    weights = checkpoint_model_state(saved)
    data.setdefault("actor_obs_normalization", "actor_normalizer.count" in weights)
    data.setdefault("critic_obs_normalization", "critic_normalizer.count" in weights)
    return ModelConfig(**data)


def train(env, iterations=15000, rollout_steps=24, output_dir="runs/lite3_pie",
          seed=42, model_config=None, ppo_config=None, init_at_random_ep_len=True,
          save_interval=500, _state=None):
    """Train recurrent PIE PPO; repeated runner calls retain optimizer/episodes."""
    if iterations < 1 or rollout_steps < 1 or save_interval < 1:
        raise ValueError("iterations, rollout_steps and save_interval must be positive")
    state = {} if _state is None else _state
    if not state:
        initialize_state(env, state, seed, model_config, ppo_config)
    if state.pop("randomize_episode_length", False) and init_at_random_ep_len:
        if hasattr(env, "episode_steps"):
            counter = env.episode_steps
            limit = round(env.config.episode_seconds / env.config.policy_dt)
        elif hasattr(env, "env"):
            counter = env.env.episode_length_buf
            limit = env.env.max_episode_length
        else:
            raise ValueError("Random episode initialization requires an episode-length buffer.")
        counter.copy_(torch.randint(max(1, limit), counter.shape, device=counter.device))
    model, algorithm = state["model"], state["algorithm"]
    obs, hidden, reset_mask = state["obs"], state["hidden"], state["reset_mask"]
    output = None if output_dir is None else Path(output_dir)
    writer = None
    if output is not None:
        from torch.utils.tensorboard import SummaryWriter
        output.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(output))
    last, checkpoint = {}, None
    try:
        for _ in range(iterations):
            batch, obs, hidden, reset_mask = algorithm.collect(env, obs, hidden, reset_mask, rollout_steps)
            last = algorithm.update(batch)
            hidden, reset_mask = algorithm.refresh_hidden(batch)
            # All likelihoods above used the same statistics. Update them once
            # for the next rollout without changing raw observations or targets.
            model.update_normalization(
                torch.cat([f["actor_inputs"] for f in batch["frames"]]),
                torch.cat([f["obs"]["critic"] for f in batch["frames"]]))
            transitions = len(batch["frames"]) * hidden.shape[0]
            state.update(obs=obs, hidden=hidden, reset_mask=reset_mask,
                         iterations=state["iterations"] + 1,
                         total_transitions=state.get("total_transitions", 0) + transitions)
            last.update(iteration=state["iterations"],
                        mean_reward=float(torch.stack([f["rewards"] for f in batch["frames"]]).mean()),
                        transitions=transitions, total_transitions=state["total_transitions"])
            if output is not None:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(last, allow_nan=False) + "\n")
                for key, value in last.items():
                    writer.add_scalar("Train/" + key, value, state["iterations"])
                logs = {}
                for frame in batch["frames"]:
                    for key, value in frame["log"].items():
                        logs.setdefault(key, []).append(value)
                for key, values in logs.items():
                    writer.add_scalar(key, sum(values)/len(values), state["iterations"])
                writer.flush()
                if state["iterations"] % save_interval == 0:
                    save_checkpoint(output / f"model_{state['iterations']}.pt", env, state, seed)
            print(json.dumps(last, allow_nan=False), flush=True)
            # Release image trajectories before collecting the next rollout.
            del batch
        if output is not None:
            save_checkpoint(output / f"model_{state['iterations']}.pt", env, state, seed)
            checkpoint = output / "checkpoint.pt"
            save_checkpoint(checkpoint, env, state, seed)
    finally:
        if writer is not None:
            writer.close()
    return {"checkpoint": None if checkpoint is None else str(checkpoint), **last}

@torch.no_grad()
def evaluate(env, checkpoint, steps=20):
    if steps < 1:
        raise ValueError("steps must be positive")
    obs = env.reset()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint_model_config(saved).validate_observation(obs, env.num_actions)
    model = PIEActorCritic(cfg).to(obs["proprio"].device)
    model.load_state_dict(checkpoint_model_state(saved)); model.eval()
    hidden = model.initial_state(obs["proprio"].shape[0])
    reset_mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
    rewards, terminated_count, timeout_count = [], 0, 0
    for _ in range(steps):
        actions, _, _, hidden = model.act(obs, hidden, reset_mask, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(actions)
        rewards.append(float(reward.mean()))
        terminated_count += int(terminated.sum()); timeout_count += int(truncated.sum())
        reset_mask = terminated | truncated
    return {"steps": steps, "mean_reward": sum(rewards)/len(rewards),
            "terminations": terminated_count, "timeouts": timeout_count,
            "claim": "bounded checkpoint evaluation; not a paper-performance benchmark"}
