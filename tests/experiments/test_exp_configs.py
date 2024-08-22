from agentlab.experiments import study_generators


def test_all_configs():
    generators = [
        study_generators.ablation_study,
        study_generators.demo_maker,
        study_generators.run_agents_on_benchmark,
    ]

    for generator in generators:
        study_name, exp_args_list = generator()
        assert isinstance(study_name, str)
        assert isinstance(exp_args_list, list)
        assert len(exp_args_list) > 0
        assert isinstance(exp_args_list[0], study_generators.ExpArgs)


if __name__ == "__main__":
    test_all_configs()
