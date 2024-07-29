import logging

from browsergym.experiments.loop import EnvArgs, ExpArgs

from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.experiments import task_collections as tasks
from agentlab.experiments import args
from agentlab.agents.step_agent.browsergym_step_agent import BrowserGymStepAgentArgs


STEP_AGENT_ARGS = BrowserGymStepAgentArgs(
    model=CHAT_MODEL_ARGS_DICT["azure/gpt-35-turbo/gpt-35-turbo"],
    low_level_action_list=["click", "type", "stop"],
    use_dom=True,
    benchmark="miniwob",
    logging=True,
)


def step_agent_test(agent: BrowserGymStepAgentArgs = STEP_AGENT_ARGS, benchmark="miniwob"):
    """Minimalistic experiment to test the system."""
    return args.expand_cross_product(
        ExpArgs(
            agent_args=agent,
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd([None] * 2),
                task_name=args.CrossProd(tasks.step_tasks),
            ),
            enable_debug=True,
        )
    )
