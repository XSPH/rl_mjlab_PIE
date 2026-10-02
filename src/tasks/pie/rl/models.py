"""PIE estimator and asymmetric actor-critic. No simulator imports."""
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.modules.normalization import EmpiricalNormalization

@dataclass
class ModelConfig:
    proprio_dim: int = 45
    proprio_history: int = 10
    depth_history: int = 2
    action_dim: int = 12
    heightmap_dim: int = 187
    critic_dim: int = 235
    token_dim: int = 128
    gru_dim: int = 128
    latent_dim: int = 16
    map_latent_dim: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 1
    initial_std: float = 1.0
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    # Reproduction choices, with evidence/rationale in PIE_NETWORK.md.
    proprio_hidden_dims: Tuple[int, ...] = (512, 256)
    cnn_hidden_channels: Tuple[int, int] = (32, 64)
    cnn_kernel_sizes: Tuple[int, int, int] = (5, 3, 3)
    cnn_strides: Tuple[int, int, int] = (2, 2, 2)
    cnn_paddings: Tuple[int, int, int] = (2, 1, 1)
    visual_grid: Tuple[int, int] = (4, 4)
    transformer_ffn_multiplier: int = 2
    transformer_dropout: float = 0.0
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    successor_hidden_dims: Tuple[int, ...] = (128, 128)
    height_decoder_hidden_dims: Tuple[int, ...] = (128, 128)

    def __post_init__(self):
        # YAML/JSON may deserialize tuples as lists; keep checkpoint comparison stable.
        sequence_fields = ("proprio_hidden_dims", "cnn_hidden_channels", "cnn_kernel_sizes",
                           "cnn_strides", "cnn_paddings", "visual_grid", "actor_hidden_dims",
                           "critic_hidden_dims", "successor_hidden_dims", "height_decoder_hidden_dims")
        for name in sequence_fields:
            setattr(self, name, tuple(getattr(self, name)))
        positive = (self.proprio_dim, self.proprio_history, self.depth_history, self.action_dim,
                    self.heightmap_dim, self.critic_dim, self.token_dim, self.gru_dim,
                    self.latent_dim, self.map_latent_dim, self.transformer_heads,
                    self.transformer_layers, self.transformer_ffn_multiplier)
        if any(value <= 0 for value in positive):
            raise ValueError("PIE dimensions and layer counts must be positive")
        if self.token_dim % self.transformer_heads:
            raise ValueError("token_dim must be divisible by transformer_heads")
        for name in sequence_fields:
            values = getattr(self, name)
            if not values or any(v < (0 if name == "cnn_paddings" else 1) for v in values):
                raise ValueError("Invalid network dimensions in " + name)
        expected_lengths = {"cnn_hidden_channels": 2, "cnn_kernel_sizes": 3,
                            "cnn_strides": 3, "cnn_paddings": 3, "visual_grid": 2}
        if any(len(getattr(self, name)) != length for name, length in expected_lengths.items()):
            raise ValueError("PIE uses three CNN layers and a two-dimensional token grid")
        if self.transformer_dropout != 0.0:
            raise ValueError("PIE PPO replay requires transformer_dropout=0")
        if self.initial_std <= 0:
            raise ValueError("initial_std must be positive")

    def validate_observation(self, obs, num_actions):
        """Check the environment schema once before training or playback."""
        observed = {
            "proprio_dim": obs["proprio"].shape[-1],
            "proprio_history": obs["proprio_history"].shape[1],
            "depth_history": obs["depth"].shape[1],
            "heightmap_dim": obs["targets"]["heightmap"].shape[-1],
            "critic_dim": obs["critic"].shape[-1],
            "action_dim": num_actions,
        }
        for name, actual in observed.items():
            if getattr(self, name) != actual:
                raise ValueError("PIE model/environment mismatch for {}: configured {}, observed {}".format(
                    name, getattr(self, name), actual))
        return self

    @property
    def token_count(self):
        return 1 + self.visual_grid[0] * self.visual_grid[1]

def mlp(input_dim, widths, output_dim):
    # mjlab 1.6 uses RSL-RL 5.4, whose network API exposes reusable MLPs.
    return MLP(input_dim=input_dim, output_dim=output_dim,
               hidden_dims=list(widths), activation="elu")

class PIEActorCritic(nn.Module):
    """The actor uses estimates, never privileged targets.
    
    PPO uses posterior means for repeatable action probabilities. Only the
    successor decoder samples the VAE. This is an explicit reproduction choice,
    because the paper does not specify how latent samples are replayed in PPO.
    Transformer dropout is zero for the same reason. GRU state is passed in and
    out, so it cannot leak between environments or survive episode boundaries.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proprio_encoder = mlp(cfg.proprio_dim * cfg.proprio_history,
                                  cfg.proprio_hidden_dims, cfg.token_dim)
        channels = (cfg.depth_history,) + cfg.cnn_hidden_channels + (cfg.token_dim,)
        convolution = []
        for index in range(3):
            convolution.extend((
                nn.Conv2d(channels[index], channels[index + 1], cfg.cnn_kernel_sizes[index],
                          stride=cfg.cnn_strides[index], padding=cfg.cnn_paddings[index]),
                nn.ELU()))
        self.depth_encoder = nn.Sequential(*convolution, nn.AdaptiveAvgPool2d(cfg.visual_grid))
        self.position = nn.Parameter(torch.zeros(1, cfg.token_count, cfg.token_dim))
        layer = nn.TransformerEncoderLayer(
            cfg.token_dim, cfg.transformer_heads, cfg.token_dim * cfg.transformer_ffn_multiplier,
            dropout=cfg.transformer_dropout, batch_first=True, activation="gelu",
            layer_norm_eps=1e-5, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, cfg.transformer_layers)
        self.gru = nn.GRUCell(cfg.token_count * cfg.token_dim, cfg.gru_dim)
        self.velocity_head = nn.Linear(cfg.gru_dim, 3)
        self.clearance_head = nn.Linear(cfg.gru_dim, 4)
        self.map_head = nn.Linear(cfg.gru_dim, cfg.map_latent_dim)
        self.mu_head = nn.Linear(cfg.gru_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.gru_dim, cfg.latent_dim)
        estimate_dim = 3 + 4 + cfg.map_latent_dim + cfg.latent_dim
        self.successor_decoder = mlp(estimate_dim, cfg.successor_hidden_dims, cfg.proprio_dim)
        self.height_decoder = mlp(cfg.map_latent_dim, cfg.height_decoder_hidden_dims, cfg.heightmap_dim)
        self.actor = mlp(cfg.proprio_dim + estimate_dim, cfg.actor_hidden_dims, cfg.action_dim)
        self.critic = mlp(cfg.critic_dim, cfg.critic_hidden_dims, 1)
        self.actor_normalizer = (EmpiricalNormalization(cfg.proprio_dim + estimate_dim)
                                 if cfg.actor_obs_normalization else nn.Identity())
        self.critic_normalizer = (EmpiricalNormalization(cfg.critic_dim)
                                  if cfg.critic_obs_normalization else nn.Identity())
        self.output_distribution = GaussianDistribution(
            cfg.action_dim, init_std=cfg.initial_std, std_type="scalar")

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.cfg.gru_dim,
                           device=device or self.output_distribution.std_param.device)

    def encode(self, obs, hidden, reset_mask=None):
        if reset_mask is not None:
            hidden = hidden * (~reset_mask.bool()).unsqueeze(-1)
        prop = self.proprio_encoder(obs["proprio_history"].flatten(1)).unsqueeze(1)
        depth = self.depth_encoder(obs["depth"]).flatten(2).transpose(1, 2)
        tokens = self.transformer(torch.cat((prop, depth), dim=1) + self.position)
        hidden = self.gru(tokens.flatten(1), hidden)
        estimates = {
            "velocity": self.velocity_head(hidden),
            "foot_clearance": self.clearance_head(hidden),
            "map_latent": self.map_head(hidden),
            "mu": self.mu_head(hidden),
            "logvar": self.logvar_head(hidden).clamp(-10.0, 5.0)}
        return estimates, hidden

    @staticmethod
    def features(estimates, z=None):
        return torch.cat((estimates["velocity"], estimates["foot_clearance"],
                          estimates["map_latent"], estimates["mu"] if z is None else z), dim=-1)

    def features_from_hidden(self, hidden):
        return torch.cat((self.velocity_head(hidden), self.clearance_head(hidden),
                          self.map_head(hidden), self.mu_head(hidden)), dim=-1)

    def distribution(self, obs, hidden, reset_mask=None):
        estimates, hidden = self.encode(obs, hidden, reset_mask)
        inputs = torch.cat((obs["proprio"], self.features(estimates)), dim=-1)
        mean = self.actor(self.actor_normalizer(inputs))
        self.output_distribution.update(mean)
        return self.output_distribution._distribution, hidden, estimates

    def value(self, obs):
        return self.critic(self.critic_normalizer(obs["critic"])).squeeze(-1)

    @torch.no_grad()
    def update_normalization(self, actor_inputs, critic_inputs):
        """Update only network-input statistics; estimator labels remain in physical units.

        The learner calls this at rollout boundaries, keeping statistics fixed
        throughout collection, likelihood replay and optimization.
        """
        if self.cfg.actor_obs_normalization:
            self.actor_normalizer.update(actor_inputs)
        if self.cfg.critic_obs_normalization:
            self.critic_normalizer.update(critic_inputs)

    def act(self, obs, hidden, reset_mask=None, deterministic=False):
        dist, hidden, estimates = self.distribution(obs, hidden, reset_mask)
        actions = dist.mean if deterministic else dist.sample()
        return actions, dist.log_prob(actions).sum(-1), self.value(obs), hidden

    def evaluate(self, obs, hidden, actions, reset_mask=None):
        dist, hidden, estimates = self.distribution(obs, hidden, reset_mask)
        return (dist.log_prob(actions).sum(-1), dist.entropy().sum(-1),
                self.value(obs), hidden, estimates)

    def auxiliary_losses(self, estimates, targets, successor, valid=None,
                         successor_valid=None):
        z = estimates["mu"] + torch.randn_like(estimates["mu"]) * (
            0.5 * estimates["logvar"]).exp()
        next_prediction = self.successor_decoder(self.features(estimates, z))
        map_prediction = self.height_decoder(estimates["map_latent"])
        if valid is None:
            valid = torch.ones(successor.shape[0], dtype=torch.bool, device=successor.device)
        if successor_valid is None:
            successor_valid = valid
        def masked_mean(loss, mask):
            weights = mask.to(successor.dtype)
            return (loss * weights).sum() / weights.sum().clamp_min(1.0)
        def mse(prediction, truth, mask=valid):
            # Each label has its own validity. An invalid successor must not
            # discard valid current-state velocity/map/clearance supervision.
            mask = mask & torch.isfinite(truth).all(-1)
            # torch.where prevents NaN labels in invalid entries poisoning a batch.
            safe_truth = torch.where(mask[:, None], truth, prediction.detach())
            return masked_mean((prediction - safe_truth).square().mean(-1), mask)
        kl = 0.5 * (estimates["mu"].square() + estimates["logvar"].exp()
                    - estimates["logvar"] - 1).sum(-1)
        return {
            "velocity": mse(estimates["velocity"], targets["velocity"]),
            "foot_clearance": mse(estimates["foot_clearance"], targets["foot_clearance"]),
            "heightmap": mse(map_prediction, targets["heightmap"]),
            "successor": mse(next_prediction, successor, valid & successor_valid),
            "kl": masked_mean(kl, valid)}
