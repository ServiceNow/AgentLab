import pytest
from agentlab.experiments.launch_exp import relaunch_study, run_experiments, make_study_dir
from agentlab.experiments.study_generators import run_agents_on_benchmark
from browsergym.experiments.loop import EnvArgs, ExpArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5, AGENT_4o_MINI
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs
from agentlab.analyze import inspect_results
import tempfile

from pathlib import Path


def test_relaunch_study():
    study_dir = Path(__file__).parent.parent / "data" / "test_study"
    exp_args_list, study_dir_ = relaunch_study(study_dir, relaunch_mode="incomplete_only")

    assert study_dir_ == study_dir
    assert len(exp_args_list) == 1
    assert exp_args_list[0].env_args.task_name == "miniwob.ascending-numbers"

    exp_args_list, study_dir_ = relaunch_study(study_dir, relaunch_mode="incomplete_or_error")

    assert study_dir_ == study_dir
    assert len(exp_args_list) == 2


@pytest.mark.repeat(3)  # there was stochastic bug caused by asyncio loop not started
def test_launch_system(backend="dask"):
    exp_args_list = []
    for seed in range(3):
        exp_args_list.append(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=CheatMiniWoBLLMArgs(),
                    flags=FLAGS_GPT_3_5,
                ),
                env_args=EnvArgs(task_name="miniwob.click-test", task_seed=seed),
            )
        )

    with tempfile.TemporaryDirectory() as tmp_dir:

        study_dir = make_study_dir(tmp_dir, "generic_agent_test")
        run_experiments(
            n_jobs=2, exp_args_list=exp_args_list, exp_dir=study_dir, parallel_backend=backend
        )

        results_df = inspect_results.load_result_df(study_dir, progress_fn=None)
        assert len(results_df) == len(exp_args_list)

        for _, row in results_df.iterrows():
            if row.stack_trace is not None:
                print(row.stack_trace)
            assert row.err_msg is None
            assert row.cum_reward == 1.0

        global_report = inspect_results.global_report(results_df)
        assert len(global_report) == 2
        assert global_report.std_err.iloc[0] == 0
        assert global_report.n_completed.iloc[0] == "3/3"
        assert global_report.avg_reward.iloc[0] == 1.0


def test_launch_system_joblib():
    test_launch_system(backend="joblib")


def test_launch_system_sequntial():
    test_launch_system(backend="sequential")


@pytest.mark.pricy
def test_4o_mini_on_miniwob_tiny_test():
    """Run with `pytest -m pricy`."""
    with tempfile.TemporaryDirectory() as tmp_dir:

        study_name, exp_args_list = run_agents_on_benchmark(
            agents=AGENT_4o_MINI, benchmark="miniwob_tiny_test"
        )
        study_dir = make_study_dir(tmp_dir, study_name)

        run_experiments(n_jobs=4, exp_args_list=exp_args_list, exp_dir=study_dir)

        results_df = inspect_results.load_result_df(study_dir, progress_fn=None)
        for row in results_df.iterrows():
            if row[1].err_msg:
                print(row[1].err_msg)
                print(row[1].stack_trace)

        assert len(results_df) == len(exp_args_list)
        global_report = inspect_results.global_report(results_df)
        print(global_report)
        assert global_report.avg_reward["[ALL TASKS]"] == 1.0


if __name__ == "__main__":
    # test_4o_mini_on_miniwob_tiny_test()
    # test_launch_system()
    test_launch_system_joblib()
