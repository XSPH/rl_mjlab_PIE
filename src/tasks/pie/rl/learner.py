"""Recurrent clipped PPO with concurrent PIE estimator losses."""
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import json
import random
import torch
from .models import ModelConfig, PIEActorCritic
from mjlab.rl import RslRlBaseRunnerCfg

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 2
    minibatches: int = 2
    entropy_weight: float = 0.01
    value_weight: float = 1.0
    estimation_weight: float = 1.0
    kl_weight: float = 1.0
    max_grad_norm: float = 1.0

@dataclass
class PIERunnerCfg(RslRlBaseRunnerCfg):
    """Native mjlab task registration with PIE-specific estimator/PPO settings."""
    seed: int = 0
    num_steps_per_env: int = 8
    max_iterations: int = 1
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
    def __init__(self, env, train_cfg, log_dir="runs/minimal", device=None):
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
        self.log_dir = log_dir
        self._training_state = {}
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations=None, init_at_random_ep_len=False):
        result = train(
            self.env,
            iterations=self.cfg.max_iterations if num_learning_iterations is None else num_learning_iterations,
            rollout_steps=self.cfg.num_steps_per_env, output_dir=self.log_dir,
            seed=self.cfg.seed, model_config=self.cfg.model, ppo_config=self.cfg.ppo,
            init_at_random_ep_len=init_at_random_ep_len,
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

    def load(self, path):
        raise NotImplementedError("Use bounded play for checkpoint evaluation; PIE optimizer resume is not implemented.")

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

    @torch.no_grad()
    def collect(self, env, obs, hidden, reset_mask, steps):
        if steps < 1:
            raise ValueError("rollout_steps must be positive")
        frames = []
        start_hidden = hidden.detach().clone()
        for _ in range(steps):
            stored = clone_observation(obs)
            targets = {k: v.detach().clone() for k, v in obs["targets"].items()}
            actions, logp, values, next_hidden = self.model.act(obs, hidden, reset_mask)
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
                "rewards": rewards.detach().clone(), "next_values": next_values.detach().clone(),
                "terminated": terminated, "truncated": truncated,
                "successor": successor.detach().clone(), "valid": valid,
                "successor_valid": successor_valid,
                "reset": reset_mask.detach().clone()})
            obs, hidden = next_obs, next_hidden.detach()
            reset_mask = terminated | truncated
        def stack(key):
            return torch.stack([frame[key] for frame in frames])
        advantages, returns = gae(
            stack("rewards"), stack("values"), stack("next_values"),
            stack("terminated"), stack("truncated"), self.cfg.gamma, self.cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
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
                logps, entropies, values, aux = [], [], [], []
                for frame in frames:
                    obs = {k: v[ids] for k, v in frame["obs"].items()}
                    logp, entropy, value, hidden, estimates = self.model.evaluate(
                        obs, hidden, frame["actions"][ids], frame["reset"][ids])
                    logps.append(logp); entropies.append(entropy); values.append(value)
                    targets = {k: v[ids] for k, v in frame["targets"].items()}
                    aux.append(self.model.auxiliary_losses(
                        estimates, targets, frame["successor"][ids], frame["valid"][ids],
                        frame["successor_valid"][ids]))
                logps, entropies, values = torch.stack(logps), torch.stack(entropies), torch.stack(values)
                old_logps = torch.stack([f["old_logp"][ids] for f in frames])
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
                # Keep the parameter inside the distribution's clamp bounds,
                # so a large update cannot leave its gradient permanently zero.
                with torch.no_grad():
                    self.model.log_std.clamp_(-5.0, 2.0)
                records.append({**{k: float(v.detach()) for k, v in losses.items()},
                    "loss": float(total.detach()), "policy": float(policy.detach()),
                    "value": float(value_loss.detach()), "grad_norm": float(grad_norm.detach())})
        return {key: sum(r[key] for r in records)/len(records) for key in records[0]}

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

def train(env, iterations=1, rollout_steps=8, output_dir="runs/minimal",
          seed=0, model_config=None, ppo_config=None, init_at_random_ep_len=False,
          _state=None):
    """Bounded training; a runner retains its optimizer and recurrent state.

    Standalone calls start a new run. The runner supplies a private state dict
    so repeated learn() calls continue the same model, optimizer and episodes.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    state = {} if _state is None else _state
    if not state:
        seed_everything(seed)
        obs = env.reset()
        if init_at_random_ep_len:
            if hasattr(env, "episode_steps"):
                counter = env.episode_steps
                limit = round(env.config.episode_seconds / env.config.policy_dt)
            else:
                counter = env.env.episode_length_buf
                limit = round(env.cfg.episode_seconds / env.cfg.control_dt)
            counter.copy_(torch.randint(max(1, limit), counter.shape, device=counter.device))
        cfg = model_config or ModelConfig(
            proprio_dim=obs["proprio"].shape[-1],
            proprio_history=obs["proprio_history"].shape[1],
            depth_history=obs["depth"].shape[1],
            heightmap_dim=obs["targets"]["heightmap"].shape[-1],
            critic_dim=obs["critic"].shape[-1])
        cfg.validate_observation(obs, env.num_actions)
        model = PIEActorCritic(cfg).to(obs["proprio"].device)
        algorithm = PPO(model, ppo_config)
        hidden = model.initial_state(obs["proprio"].shape[0])
        reset_mask = torch.ones(obs["proprio"].shape[0], dtype=torch.bool, device=hidden.device)
        state.update(model_config=cfg, model=model, algorithm=algorithm,
                     obs=obs, hidden=hidden, reset_mask=reset_mask, iterations=0)
    cfg, model, algorithm = state["model_config"], state["model"], state["algorithm"]
    obs, hidden, reset_mask = state["obs"], state["hidden"], state["reset_mask"]
    output = None if output_dir is None else Path(output_dir)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    last = {}
    for _ in range(iterations):
        batch, obs, hidden, reset_mask = algorithm.collect(env, obs, hidden, reset_mask, rollout_steps)
        last = algorithm.update(batch)
        # Refresh the same recurrent rollout under the new weights. PPO updates
        # do not imply environment resets or erase continuing episode memories.
        hidden, reset_mask = algorithm.refresh_hidden(batch)
        state.update(obs=obs, hidden=hidden, reset_mask=reset_mask,
                     iterations=state["iterations"] + 1)
        last.update(iteration=state["iterations"],
                    mean_reward=float(torch.stack([f["rewards"] for f in batch["frames"]]).mean()),
                    transitions=len(batch["frames"])*hidden.shape[0])
        if output is not None:
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(last, allow_nan=False) + "\n")
        print(json.dumps(last, allow_nan=False), flush=True)
    checkpoint = None
    if output is not None:
        environment_cfg = getattr(env, "config", getattr(env, "cfg", None))
        environment_cfg = asdict(environment_cfg) if is_dataclass(environment_cfg) else {}
        checkpoint = output / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "model_config": asdict(cfg),
                    "ppo_config": asdict(algorithm.cfg), "optimizer": algorithm.optimizer.state_dict(),
                    "iterations": state["iterations"], "seed": seed, "environment_config": environment_cfg,
                    "torch_rng": torch.get_rng_state()}, checkpoint)
    return {"checkpoint": None if checkpoint is None else str(checkpoint), **last}

@torch.no_grad()
def evaluate(env, checkpoint, steps=20):
    if steps < 1:
        raise ValueError("steps must be positive")
    obs = env.reset()
    saved = torch.load(checkpoint, map_location=obs["proprio"].device, weights_only=True)
    cfg = ModelConfig(**saved["model_config"]).validate_observation(obs, env.num_actions)
    model = PIEActorCritic(cfg).to(obs["proprio"].device)
    model.load_state_dict(saved["model"]); model.eval()
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
            "claim": "untrained/minimally trained plumbing check; not paper performance"}
