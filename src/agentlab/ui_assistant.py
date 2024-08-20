import argparse

from browsergym.experiments.loop import AbstractAgentArgs, EnvArgs, ExpArgs

from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.launch_exp import import_object


def make_exp_args(agent_args: AbstractAgentArgs, start_url="https://www.google.com"):

    try:
        agent_args.flags.action.demo_mode = "default"
    except AttributeError:
        pass

    exp_args = ExpArgs(
        agent_args=agent_args,
        env_args=EnvArgs(
            max_steps=1000,
            task_seed=None,
            task_name="openended",
            task_kwargs={
                "start_url": start_url,
            },
            headless=False,
            record_video=True,
            wait_for_user_message=True,
            viewport={"width": 1500, "height": 1280},
            slow_mo=1000,
        ),
    )

    return exp_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent_config",
        type=str,
        default="agentlab.agents.generic_agent.AGENT_4o",
        help="Python path to the agent config",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="The start page of the agent",
    )

    args, unknown = parser.parse_known_args()
    agent_args = import_object(args.agent_config)
    exp_args = make_exp_args(agent_args, args.start_url)
    exp_args.prepare(RESULTS_DIR / "ui_assistant_logs")
    exp_args.run()
