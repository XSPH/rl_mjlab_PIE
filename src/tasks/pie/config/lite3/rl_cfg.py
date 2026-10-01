from ...rl.learner import PIERunnerCfg


def lite3_pie_runner_cfg():
    return PIERunnerCfg(max_iterations=1, num_steps_per_env=8,
                        experiment_name='lite3_pie', upload_model=False)
