from mjlab.tasks.registry import register_mjlab_task
from ...rl.learner import PIEOnPolicyRunner
from .env_cfgs import lite3_pie_env_cfg
from .rl_cfg import lite3_pie_runner_cfg

register_mjlab_task(
    task_id='Unitree-Lite3-PIE', env_cfg=lite3_pie_env_cfg(),
    play_env_cfg=lite3_pie_env_cfg(play=True), rl_cfg=lite3_pie_runner_cfg(),
    runner_cls=PIEOnPolicyRunner)
