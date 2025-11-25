import argparse
import logging
import os
import sys

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



def parse_args():
    parser = argparse.ArgumentParser(description="Run MiniWob benchmark experiments")
    parser.add_argument(
        "--backend",
        choices=["playwright", "mcp", "bgym"],
        default="playwright",
        help="Browser backend to use (default: playwright)",
    )
    parser.add_argument(
        "--agent",
        choices=["tape", "generic"],
        default="tape",
        help="Agent type to use (default: tape)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="miniwob",
        help="Hydra config name to load (default: miniwob)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    if args.backend == "bgym":
        benchmark = DEFAULT_BENCHMARKS["miniwob"](n_repeats=1)
    elif args.backend == "playwright":
        benchmark = MiniWobBenchmark(backend_cls=AsyncPlaywright)
    elif args.backend == "mcp":
        benchmark = MiniWobBenchmark(backend_cls=MCPPlaywright)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if args.agent == "generic":
        agent_args = GenericAgentArgs(
            chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-mini-2025-08-07"],
            flags=GPT5_MINI_FLAGS,
        )
    else:
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
