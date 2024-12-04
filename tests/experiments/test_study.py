from copy import deepcopy
from pathlib import Path
import tempfile
import time
import multiprocessing
import bgym

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs
from agentlab.experiments.study import Study, SequentialStudies


def _make_click_test_benchmark():
    click_test_benchmark = bgym.DEFAULT_BENCHMARKS["miniwob"]().subset_from_glob("task_name", "*click-test")
    click_test_benchmark.env_args_list = click_test_benchmark.env_args_list[:3]

    for env_args in click_test_benchmark.env_args_list:
        env_args.max_steps = 3

    return click_test_benchmark


def _run_with_timeout(func, kwargs, timeout):
    """Run a function in a separate process and kill it after timeout seconds"""
    process = multiprocessing.Process(target=func, kwargs=kwargs)
    process.start()
    time.sleep(timeout)
    process.terminate()
    process.join()


def test_sequential_study_timeout():
    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_dir = Path("/tmp/agentlab_test")

        agent_args_1 = GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(wait_time=10),
            flags=FLAGS_GPT_4o,
        )

        agent_args_1.agent_name = "agent1"
        agent_args_2 = deepcopy(agent_args_1)
        agent_args_2.agent_name = "agent2"

        benchmark = _make_click_test_benchmark()

        studies = [Study([agent_args], benchmark) for agent_args in [agent_args_1, agent_args_2]]
        study = SequentialStudies(studies)
        
        for s in study.studies:
            print(s.dir)
            print(s.agent_args[0].agent_name)
            print(len(s.exp_args_list))

        kwargs = dict(n_jobs=1, parallel_backend="ray", exp_root=tmp_dir)
        t0 = time.time()
        # kill it after 4 seconds to test the restart system
        _run_with_timeout(study.run, kwargs=kwargs, timeout=4)
        dt = time.time() - t0
        assert dt < 6, f"Process terminated in {dt} seconds"
        assert dt > 4, f"Process terminated in {dt} seconds"

        study_ = Study.load_most_recent(tmp_dir)
        assert isinstance(study_, SequentialStudies)
        assert len(study_.studies) == 2 # there should be two agents

        n_incomplete, n_errors = study_.find_incomplete()

        print(n_incomplete, n_errors)   
        for sub_study in study_.studies:
            assert len(sub_study.exp_args_list) == 3 

        study_.run(n_jobs=3, parallel_backend="ray")


if __name__ == "__main__":
    test_sequential_study_timeout()