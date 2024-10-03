from agentlab.experiments import study_generators


def test_all_configs():
    generators = [
        # study_generators.ablation_study,
        study_generators.run_agents_on_benchmark,
        study_generators.random_search,
    ]

    for generator in generators:
        study = generator()
        assert isinstance(study, study_generators.Study)
        assert isinstance(study.exp_args_list, list)
        assert len(study.exp_args_list) > 0
        assert isinstance(study.exp_args_list[0], study_generators.ExpArgs)


if __name__ == "__main__":
    test_all_configs()
