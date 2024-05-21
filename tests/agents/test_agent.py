import tempfile
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import BASIC_FLAGS
from agentlab.llm.chat_api import ChatModelArgs
from browsergym.experiments.loop import EnvArgs, ExpArgs
from agentlab.experiments import launch_exp
from agentlab.analyze import inspect_results


def test_generic_agent():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=ChatModelArgs(model_name="test/CheatMiniWoBLLM"),
            flags=BASIC_FLAGS,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        launch_exp.main(
            exp_root=tmp_dir,
            exp_group_name="generic_agent_test",
            exp_args_list=[exp_args],
            n_jobs=1,
            relaunch_mode=None,
            shuffle_jobs=True,
            auto_accept=True,
        )

        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        # for key, val in result_record.items():
        #     print(key, val[0])
        # pass
        # # exp_args.prepare(tmp_dir)
        # # exp_args.run()
        # # exp_result = get_exp_result(exp_args.exp_dir)
        # # exp_record = exp_result.get_exp_record()

        target = {
            "n_steps": 1,
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
            "stack_trace": None,
            "agent_args.flags.obs.use_ax_tree": True,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key][0] == target_val

        # for key, target_val in target.items():
        #     assert key in exp_record
        #     assert exp_record[key] == target_val

        # # TODO investigate why it's taking almost 5 seconds to solve
        # assert exp_record["stats.cum_step_elapsed"] < 5
        # if exp_record["stats.cum_step_elapsed"] > 3:
        #     t = exp_record["stats.cum_step_elapsed"]
        #     logging.warning(
        #         f"miniwob.click-test is taking {t:.2f}s (> 3s) to solve with an oracle."
        #     )


if __name__ == "__main__":
    test_generic_agent()
