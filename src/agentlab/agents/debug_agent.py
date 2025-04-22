from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial

import bgym
from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, overlay_som, prune_html

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import ParseError, image_to_png_base64_url, parse_html_tags_raise, retry
from agentlab.llm.tracking import cost_tracker_decorator


@dataclass
class DebugAgentArgs(AgentArgs):

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"debug".replace("/", "_")
        except AttributeError:
            pass
        self.action_set_args = bgym.DEFAULT_BENCHMARKS[
            "miniwob_tiny_test"
        ]().high_level_action_set_args
        self.use_html = False

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        if benchmark.name.startswith("miniwob"):
            self.use_html = True
        self.action_set_args = benchmark.high_level_action_set_args

    def make_agent(self):
        return DebugAgent(self.action_set_args, use_html=self.use_html)


class DebugAgent(Agent):
    def __init__(
        self,
        action_set_args,
        use_html=False,
    ):
        self.action_set = action_set_args.make_action_set()
        self.use_html = use_html

    def obs_preprocessor(self, obs):
        obs = deepcopy(obs)
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            extra_properties=obs["extra_element_properties"],
            with_visible=True,
            with_clickable=True,
            with_center_coords=True,
            with_bounding_box_coords=True,
            filter_visible_only=False,
            filter_with_bid_only=False,
            filter_som_only=False,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=obs["extra_element_properties"],
            with_visible=True,
            with_clickable=True,
            with_center_coords=True,
            with_bounding_box_coords=True,
            filter_visible_only=False,
            filter_with_bid_only=False,
            filter_som_only=False,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])
        obs["screenshot_som"] = overlay_som(
            obs["screenshot"], extra_properties=obs["extra_element_properties"]
        )
        return obs

    def get_action(self, obs):

        # print(obs["pruned_html"])
        print("\n")
        observation = obs["pruned_html"] if self.use_html else obs["axtree_txt"]
        action = input(observation + "\n")
        agent_info = AgentInfo(
            think="nope",
            chat_messages=[],
            stats={},
        )
        return action, agent_info


DEBUG_AGENT = DebugAgentArgs()
