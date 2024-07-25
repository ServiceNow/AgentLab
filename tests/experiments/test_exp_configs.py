from agentlab.agents.generic_agent.agent_configs import AGENT_3_5
from agentlab.agents.generic_agent.exp_configs import ExpArgs
from agentlab.experiments.launch_exp import import_object


def test_all_configs():
    exp_name_list = [
        "agentlab.agents.generic_agent.generic_agent_test",
        "agentlab.agents.generic_agent.progression_study",
        "agentlab.agents.generic_agent.ablation_study",
        "agentlab.agents.generic_agent.demo_maker",
        "agentlab.agents.generic_agent.final_run",
    ]

    for exp_name in exp_name_list:
        exp_args_list = import_object(exp_name)(AGENT_3_5, "miniwob")

        assert isinstance(exp_args_list, list)
        assert len(exp_args_list) > 0
        assert all(isinstance(exp_args, ExpArgs) for exp_args in exp_args_list)


if __name__ == "__main__":
    test_all_configs()
