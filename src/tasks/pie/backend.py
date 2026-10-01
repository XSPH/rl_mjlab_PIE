"""GPU PIE environment adapter for pinned mjlab 1.6.0.

The native environment runs with auto_reset=False. True terminal observations are
copied before selected environments reset, preserving successor and timeout labels.
"""

from importlib.metadata import version
from types import MethodType

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from .reproduction import EnvironmentConfig
from .env_config import make_env_config
from .history import HistoryBuffer
from .robots import profile


def clone_observation(observation):
    return {key: clone_observation(value) if isinstance(value, dict) else value.clone()
            for key, value in observation.items()}


class PieMjlabEnv:
    """Learner-facing vector environment with clean depth and privileged targets."""

    def __init__(self, cfg: EnvironmentConfig, native_cfg=None, native_env=None):
        cfg.validate()
        if version("mjlab") != "1.6.0":
            raise RuntimeError("This backend requires mjlab==1.6.0; use this project's separate environment.")
        if not cfg.device.startswith("cuda"):
            raise ValueError("PIE training and Warp camera verification require device cuda:0 or another CUDA GPU.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the dedicated PIE Conda environment and a CUDA GPU.")
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_actions = 12
        self.device = torch.device(cfg.device)
        self.heightmap_dim = cfg.heightmap_shape[0] * cfg.heightmap_shape[1]
        self.critic_dim = 45 + 3 + self.heightmap_dim
        self.env = native_env if native_env is not None else ManagerBasedRlEnv(
            native_cfg if native_cfg is not None else make_env_config(cfg), device=cfg.device)
        try:
            native_names = self.env.action_manager.get_term("joint_pos").target_names
            canonical = profile(cfg.robot).joints
            self._to_native = torch.tensor([canonical.index(name) for name in native_names], device=self.device)
            self._proprio_history = HistoryBuffer(self.num_envs, cfg.proprio_history, (45,), self.device)
            self._depth_history = HistoryBuffer(self.num_envs, cfg.camera.history,
                                               (cfg.camera.height, cfg.camera.width), self.device)
            self._depth_pending = torch.zeros((self.num_envs, cfg.camera.height, cfg.camera.width), device=self.device)
            self._depth_due_at = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._depth_pending_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self._depth_pending_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._depth_frame_ids = HistoryBuffer(self.num_envs, cfg.camera.history, (), self.device,
                                                  dtype=torch.long)
            self._camera_period = round(1 / cfg.camera.frequency_hz / cfg.control_dt)
            self._step = 0
            self._render_due = True
            self.depth_render_count = 0
            self._native_sense = self.env.sim.sense
            # Pinned upstream integration: preserve native full sensing on camera frames;
            # between camera frames, keep terrain supervision raycasts at the policy rate.
            self.env.sim.sense = MethodType(lambda _sim: self._sense(), self.env.sim)
            self._observation = None
        except Exception:
            # A caller owns a supplied native environment; the standalone
            # constructor must release the native environment it created.
            if hasattr(self, "_native_sense"):
                self.env.sim.sense = self._native_sense
            if native_env is None:
                self.env.close()
            raise

    def _sense(self):
        if self._render_due:
            self._native_sense()
            self.depth_render_count += 1
            return
        sim = self.env.sim
        context = sim._sensor_context  # verified private API in pinned mjlab 1.6.0
        if context is None:
            return
        context.prepare()
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.refit_bvh(sim.wp_model, sim.wp_data, context.render_context)
            for sensor in context.raycast_sensors:
                sensor.raycast_kernel(rc=context.render_context)
        context.finalize()

    def _proprio(self):
        return self.env.obs_buf["proprio"]

    def _depth(self):
        raw = self.env.scene["depth"].data.depth[..., 0]
        # MuJoCo Warp uses depth from camera plane. No image noise or filtering.
        camera = self.cfg.camera
        valid = torch.isfinite(raw) & (raw > 0)
        metres = torch.where(valid, raw, torch.full_like(raw, camera.far)).clamp(camera.near, camera.far)
        return (metres - camera.near) / (camera.far - camera.near) - 0.5

    def _targets(self):
        robot = self.env.scene["robot"].data
        scan = self.env.scene["height_scan"].data
        # Same map convention as the independent Isaac Gym reproduction: clearance
        # below the base minus 0.5m, clipped to [-1,1]. Misses denote a deep drop.
        terrain_z = scan.hit_pos_w[..., 2]
        heightmap = robot.root_link_pos_w[:, 2:3] - terrain_z - 0.5
        heightmap = torch.where(scan.distances >= 0, heightmap, torch.ones_like(heightmap))
        foot_height = self.env.scene["foot_clearance"].data.heights
        # Sphere bottom is center world-Z minus radius, independent of leg rotation.
        foot_height = foot_height - (0.022 if self.cfg.robot == "lite3" else 0.02)
        return {"velocity": robot.root_link_lin_vel_b.clone(),
                "foot_clearance": foot_height.clamp(0.0, 2.0),
                "heightmap": heightmap.clamp(-1.0, 1.0)}

    def _snapshot(self):
        proprio = self._proprio().clone()
        targets = self._targets()
        return {"proprio": proprio,
                "proprio_history": self._proprio_history.data.clone(),
                "depth": self._depth_history.data.clone(),
                "critic": torch.cat((proprio, targets["velocity"], targets["heightmap"]), -1),
                "targets": targets}

    def _reset_histories(self, ids):
        self._proprio_history.reset(self._proprio(), ids)
        depth = self._depth()
        self._depth_history.reset(depth, ids)
        self._depth_frame_ids.reset(torch.full((self.num_envs,), self._step, device=self.device), ids)
        self._depth_pending_valid[ids] = False

    def reset(self, *, seed=None, env_ids=None):
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        if env_ids is None:
            self._step = 0
        self._render_due = True
        self.env.reset(seed=seed, env_ids=ids)
        self._reset_histories(ids)
        self._observation = self._snapshot()
        return self._observation

    def get_observations(self):
        if self._observation is None:
            return self.reset()
        return self._observation

    def step(self, actions):
        if self._observation is None:
            self.reset()
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected actions[{self.num_envs},12], got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("Actions contain nonfinite values.")
        self._step += 1
        self._render_due = self._step % self._camera_period == 0
        _, rewards, terminated, truncated, extras = self.env.step(
            actions.to(self.device).clamp(-4.0, 4.0)[:, self._to_native])
        rewards, terminated, truncated = rewards.clone(), terminated.clone(), truncated.clone()
        invalid_reward = ~torch.isfinite(rewards)
        terminated |= invalid_reward
        truncated &= ~terminated
        rewards = torch.where(invalid_reward, torch.zeros_like(rewards), rewards)
        self._proprio_history.append(self._proprio())
        # Deliver the preceding frame BEFORE overwriting the pending buffer. This
        # order is essential when latency equals the camera period (default 100ms).
        delivered = self._depth_pending_valid & (self._depth_due_at <= self._step)
        self._depth_history.append(self._depth_pending, delivered)
        self._depth_frame_ids.append(self._depth_pending_step, delivered)
        self._depth_pending_valid[delivered] = False
        if self._render_due:
            self._depth_pending.copy_(self._depth())
            low, high = self.cfg.camera.latency_steps if self.cfg.randomization else (0, 0)
            delays = torch.randint(low, high + 1, (self.num_envs,), device=self.device)
            self._depth_due_at.copy_(self._step + delays)
            self._depth_pending_step.fill_(self._step)
            self._depth_pending_valid.fill_(True)
        delivered = self._depth_pending_valid & (self._depth_due_at <= self._step)
        self._depth_history.append(self._depth_pending, delivered)
        self._depth_frame_ids.append(self._depth_pending_step, delivered)
        self._depth_pending_valid[delivered] = False
        terminal = self._snapshot()
        done = terminated | truncated
        relative = self.env.scene["robot"].data.root_link_pos_w - self.env.scene.env_origins
        success = relative[:, 0] > self.cfg.terrain_length - 1.8
        ids = done.nonzero(as_tuple=False).flatten()
        info = {"terminal_observation": terminal,
                "terminal_proprio": terminal["proprio"].clone(),
                "time_outs": truncated.clone(), "reset_mask": done.clone(),
                "success": success.clone(),
                "log": dict(extras.get("log", {})),
                "depth_render_count": self.depth_render_count}
        info["depth_frame_ids"] = self._depth_frame_ids.data.clone().long()
        if len(ids):
            self._render_due = True
            self.env.reset(env_ids=ids)
            self._reset_histories(ids)
            observation = self._snapshot()
            # A partial reset recomputes raycasts for all worlds, but proprio histories
            # and depth timelines in continuing worlds retain their previous entries.
        else:
            observation = clone_observation(terminal)
        self._observation = observation
        return observation, rewards, terminated, truncated, info

    def close(self):
        # Restore the borrowed simulation method and break its reference to this adapter.
        self.env.sim.sense = self._native_sense
        self.env.close()
