# Lite3 PIE 扩展

网络超参数已按 PIE、DreamWaQ、LocoTransformer 与 Extreme Parkour 的公开资料确定，具体数值、选择理由及工程假设见 [PIE_NETWORK.md](PIE_NETWORK.md)。本项目独立的 `src/tasks/pie/rl/models.py::ModelConfig` 保存全部层宽与 CNN/token 参数，随 checkpoint 保存；可通过 `--agent.model.*` 配置。

直接基于 `unitreerobotics/unitree_rl_mjlab` 修改，基线 `1425b15f73bd4095f0df53709d7c389c3eb9e790`。新增 `Unitree-Lite3-PIE`，由原 `scripts/train.py`、`scripts/play.py` 和 mjlab registry 调用。

环境扩展宇树原 `make_velocity_env_cfg()`。原入口创建 `ManagerBasedRlEnv`，PIE adapter 包装同一实例。与 Isaac Gym 项目无相互运行依赖。

## 独立环境与最小运行

```bash
conda activate /home/asuka/Legged/parkour/.conda-envs/pie-mjlab
cd /home/asuka/Legged/parkour/unitree_rl_mjlab
python -s -m pytest -q tests
python -s scripts/check_pie_backend.py
python -s scripts/train.py Unitree-Lite3-PIE --env.scene.num-envs=2 --agent.max-iterations=1 --agent.num-steps-per-env=16 --agent.run-name=minimal
python -s scripts/play.py Unitree-Lite3-PIE --num-envs 2 --steps 4 --viewer none --checkpoint-file logs/rsl_rl/lite3_pie/2026-09-30_21-53-53_minimal/checkpoint.pt
```

play 使用本次已保存 checkpoint；新训练目录带新时间戳。默认 2 环境、8 步 rollout、1 iteration。check 仅做 2 环境、18 个控制步，无 PPO 更新。

环境为 Python 3.11、Torch 2.14.0、mjlab 1.6.0、MuJoCo/MuJoCo Warp 3.11.0、Warp 1.17.0、rsl_rl 5.4.2。只更新新环境，原 mjlab 1.2 环境保留。`environment.pie.yml` 是配方；依赖齐全后在新环境运行 `python -s -m pip install -e . --no-deps --no-build-isolation` 安装本仓库。

## 修改位置

| 路径 | 内容 |
| --- | --- |
| `src/tasks/pie/config/lite3` | 原生注册、Lite3 环境/runner 配置、CLI 参数传递 |
| `src/tasks/pie/env_config.py` | 扩展宇树原环境工厂，组装 managers/sensors |
| `src/tasks/pie/reproduction.py` | 控制、相机、随机化与地形默认值 |
| `src/tasks/pie/backend.py` | 原生环境包装、图像时延与 pre-reset 标签 |
| `src/tasks/pie/rl/models.py`, `learner.py` | rsl_rl MLP、PIE 估计器和循环 PPO |
| `src/tasks/pie/terrains.py`, `mdp.py` | 地形、奖励、观测、监督与课程 |
| `src/assets/robots/lite3` | 独立官方 Lite3 MJCF、meshes、许可证 |
| `src/assets/robots/compat.py` | 上游资产在 mjlab 1.6 下的必要兼容 |

原生 `CameraSensorCfg` 使用 MuJoCo Warp GPU 深度。`RayCastSensorCfg` / `TerrainHeightSensorCfg` 提供高程图/足高监督。10 Hz 图像、50 Hz 控制，默认 100 ms 图像延迟；无图像噪声或滤波，reset 时随机化相机姿态与 FOV。

网络为本体 MLP + 深度 CNN → Transformer → GRU，显式估计速度/足高，隐式编码地图，另有动力学 VAE 与两个解码器。策略使用 posterior mean，单优化器联合循环 PPO 和五项辅助损失。actor 无特权状态，critic 使用真实速度/地图。

相机、PD、地形和随机化用 `--env.pie.*`；环境数用 `--env.scene.num-envs`。常规 `--env.sim.mujoco.timestep`、`--env.decimation`、`--env.episode-length-s` 的非默认覆盖优先于 PIE timing。创建环境前重建相机/执行器。低层 sensor/action 定义请编辑任务工厂，避免被 PIE 配方重建覆盖。模型/PPO 参数用 `--agent.model.*` / `--agent.ppo.*`。

修改本体历史长度或高程图网格时，需要同步 `--agent.model.*` 中的 history、heightmap/critic 维数；训练与回放入口会检查模型配置是否匹配实际观测。

## 当前边界

此前只验证 CPU、GPU 相机/延迟/reset、一次 2×16 条 transition 联合更新和 checkpoint 4 步 headless 回放。2026-10-01 网络配置调整与随后的代码修复仅做静态审查，历史记录不代表当前源码已重新运行。[完整审查记录](docs/code_review_2026-10-01.md)。

同一 PIE runner 多次调用 `learn()` 会保留模型、Adam 和 GRU/观测状态并累计 iteration。未验证跑酷成功率或真机。PIE 仅支持单 GPU、无视频的最小训练；optimizer resume、ONNX 导出和交互 viewer 未实现。

相机降频依赖固定版本 `Simulation._sensor_context` 内部接口。mjlab 电机强度随机化用 effort cap 倍率，与 Isaac Gym 完整力矩倍率不同。上游任务保留并通过注册导入；为兼容 1.6 调整资产加载和显式 collision mask，本轮未物理验证所有其他机器人任务。
