# rl_mjlab_PIE 发布范围

GitHub：<https://github.com/XSPH/rl_mjlab_PIE>。

发布内容是当前宇树项目的源码快照，保留上游许可证、原任务源码和机器人模型资产。上游基线为 `unitreerobotics/unitree_rl_mjlab@1425b15f73bd4095f0df53709d7c389c3eb9e790`，此前本地 Git 历史仍保留在原工作区。

包含 `src/tasks/pie` 的配置、环境、深度观测、PIE 网络、循环 PPO 和训练/回放入口，以及 Lite3 XML、mesh、许可证和资产来源。rsl_rl 使用独立环境中的 `rsl-rl-lib==5.4.2`，依赖配方在 `environment.pie.yml`。

`.gitignore` 排除 Conda/venv、编辑器配置、缓存、日志、checkpoint、导出策略、数据集、压缩包和预编译库。本次不携带上游演示 ONNX 策略/NPZ 运动数据，以及 ONNX Runtime 的 `.so` 二进制；原部署示例若使用它们，需要按上游说明另外获取。相关 C++ 源码、头文件、配置和许可证，以及机器人 XML/mesh 均保留。

`PUBLICATION_MANIFEST.json` 列出上传文件的字节数、权限模式和 SHA-256，以及有意排除的文件。单文件大小门限为 25 MiB；超过门限会中止准备，不能静默漏传。发布后重新克隆远端，逐文件检查清单、内容哈希、权限及 Python 源码语法，同时核对 GitHub `main` 的 commit。

本次仅检查发布完整性，没有运行强化学习测试、仿真、训练或回放。`docs/validation*` 保存的是此前历史记录，相关日志与 checkpoint 已排除。仓库里的原工作区命令含本地路径；在新机器上需要按实际克隆目录和独立环境路径调整。
