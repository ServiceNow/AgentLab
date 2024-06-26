from agentlab.experiments import exp_configs


def test_all_configs():
    exp_name_list = [
        "generic_agent_test",
        # "tgi_toolkit_test",
        "generic_agent_eval_llm",
        "progression_study",
        "ablation_study",
        "demo_maker",
        "final_run",
        "ablation_study_GPT_3_5",
        "ablation_study_GPT_4",
    ]

    for exp_name in exp_name_list:
        exp_group_name, exp_args_list = exp_configs.get_exp_args_list(exp_name)

        assert exp_group_name == exp_name
        assert isinstance(exp_args_list, list)
        assert len(exp_args_list) > 0
        assert all(isinstance(exp_args, exp_configs.ExpArgs) for exp_args in exp_args_list)
