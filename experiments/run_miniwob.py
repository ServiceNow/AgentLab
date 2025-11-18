import logging
import os

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

from agentlab.agents.generic_agent.agent_configs import GPT5_MINI_FLAGS
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.tapeagent.agent import TapeAgentArgs, load_config
from agentlab.backends.browser.mcp_playwright import MCPPlaywright
from agentlab.backends.browser.playwright import AsyncPlaywright
from agentlab.benchmarks.miniwob import MiniWobBenchmark
from agentlab.experiments.study import make_study
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
load_dotenv()

if __name__ == "__main__":
    config = load_config("miniwob")

    # benchmark = DEFAULT_BENCHMARKS["miniwob"](n_repeats=1)
    # benchmark = MiniWobBenchmark(backend=MCPPlaywright())
    benchmark = MiniWobBenchmark(backend=AsyncPlaywright())

    # agent_args = GenericAgentArgs(
    #     chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-mini-2025-08-07"],
    #     flags=GPT5_MINI_FLAGS,
    # )
    # agent_args.flags.obs.use_ax_tree = False
    # agent_args.flags.obs.use_html = True
    # agent_args.flags.obs.use_focused_element = False
    agent_args = TapeAgentArgs(agent_name=config.name, config=config)


    study = make_study(
        benchmark=benchmark,
        agent_args=agent_args,
        comment=config.comment,
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )
    if os.environ.get("AGENTLAB_DEBUG"):
        study.exp_args_list = study.exp_args_list[23:27]
        study.run(n_jobs=1, n_relaunch=1, parallel_backend="sequential")
    else:
        study.run(n_jobs=config.n_jobs, n_relaunch=1, parallel_backend=config.parallel_backend)
