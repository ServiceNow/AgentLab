import pytest
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs
from agentlab.experiments.study import ParallelStudies, make_study, Study
from agentlab.experiments.multi_server import WebArenaInstanceVars
import logging


logging.getLogger().setLevel(logging.INFO)


def _make_agent_args_list():
    # CheatMiniWoB agents won't succeed on WebArena, this is just for testing parallelization
    agent_args_list = []
    for i in range(2):
        agent_args = GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_4o,
        )

        agent_args.agent_name = agent_args.agent_name + f"_{i}"
        agent_args_list.append(agent_args)
    return agent_args_list


@pytest.mark.skip(reason="This test requires WebArena instances to be running")
def manual_test_launch_parallel_study_webarena():
    agent_args_list = _make_agent_args_list()

    server_instance_1 = WebArenaInstanceVars.from_env_vars()
    server_instance_2 = server_instance_1.clone()
    server_instance_2.base_url = "http://webarena-slow.eastus.cloudapp.azure.com"
    parallel_servers = [server_instance_1, server_instance_2]
    # parallel_servers = [server_instance_2]

    for server in parallel_servers:
        print(server)

    study = make_study(
        agent_args_list,
        benchmark="webarena_tiny",
        parallel_servers=parallel_servers,
        ignore_dependencies=True,
    )
    study.override_max_steps(2)
    assert isinstance(study, ParallelStudies)

    study.run(n_jobs=4, parallel_backend="ray", n_relaunch=1)


@pytest.mark.skip(reason="This usecase isnt relevant atm")
def test_launch_parallel_study():
    agent_args_list = _make_agent_args_list()

    study = make_study(agent_args_list, benchmark="miniwob_tiny_test", parallel_servers=2)
    assert isinstance(study, ParallelStudies)

    study.run(n_jobs=4, parallel_backend="ray", n_relaunch=1)
    _, summary_df, _ = study.get_results()
    assert len(summary_df) == 2
    for n_completed in summary_df["n_completed"]:
        assert n_completed == "4/4"


if __name__ == "__main__":
    # test_launch_parallel_study()
    manual_test_launch_parallel_study_webarena()
