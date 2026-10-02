"""Two-world camera timing and autoreset checks; no policy training."""
import json
import torch
from mjlab.envs import ManagerBasedRlEnv
from src.tasks.pie.backend import PieMjlabEnv
from src.tasks.pie.config.lite3.env_cfgs import apply_pie_settings, lite3_pie_env_cfg


def main():
    cfg = lite3_pie_env_cfg()
    cfg.scene.num_envs = 2
    cfg.pie.terrain_rows = 1
    cfg.pie.curriculum = False
    pie = apply_pie_settings(cfg, "cuda:0", 7)
    # Isolate sensing and time-limit reset semantics from untrained-policy falls.
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    native = ManagerBasedRlEnv(cfg, device=pie.device)
    env = native
    try:
        env = PieMjlabEnv(pie, native_env=native)
        obs = env.reset()
        assert tuple(obs["critic"].shape) == (2, 235)
        assert tuple(obs["depth"].shape) == (2, 2, 60, 80)
        actions = torch.full((2, 12), 0.1, device=env.device)
        frame_at_step_10 = None
        for step in range(1, 17):
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert not (terminated | truncated).any()
            assert torch.isfinite(obs["depth"]).all() and torch.isfinite(rewards).all()
            if step == 10:
                frame_at_step_10 = info["depth_frame_ids"][:, -1].clone()
        assert torch.equal(frame_at_step_10, torch.full_like(frame_at_step_10, 5))
        assert info["depth_frame_ids"].tolist() == [[5, 10], [5, 10]]
        render_count = env.depth_render_count
        assert render_count == 4  # Reset, then capture at 5/10/15; no per-action rendering.
        depth_range = [obs["depth"].min().item(), obs["depth"].max().item()]
        env.env.episode_length_buf.fill_(env.env.max_episode_length - 2)
        env.step(actions)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert truncated.all() and not terminated.any()
        assert torch.allclose(info["terminal_proprio"][:, -12:], actions)
        assert torch.equal(info["terminal_proprio"], info["terminal_observation"]["proprio"])
        assert torch.equal(obs["proprio"][:, -12:], torch.zeros_like(actions))
        assert torch.equal(obs["proprio_history"][:, 0], obs["proprio_history"][:, -1])
        assert not env._depth_pending_valid.any()
    finally:
        env.close()
    print(json.dumps({"checks": 2, "passed": True, "num_envs": 2, "steps": 18,
                      "depth_renders_in_first_16_steps": render_count, "depth_range": depth_range,
                      "step_10_delivers_step_5_frame": True, "pre_reset_labels": True,
                      "history_reset": True}))


if __name__ == "__main__":
    main()
