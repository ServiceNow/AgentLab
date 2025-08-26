import os
from copy import deepcopy
from dataclasses import dataclass

import bgym

os.environ["LITELLM_LOG"] = "WARNING"

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.litellm_api import APIPayload, LiteLLMModelArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    overlay_som,
    prune_html,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "send_action_candidates",
            "description": (
                "Submit up to 3 plausible NEXT ACTION CANDIDATES for the web agent. "
                "Each entry must include both the THOUGHT PROCESS and the ACTION itself. "
                "The action must be atomic, executable in the current UI context, and "
                "use the environment's high-level primitives (click, type, select, submit, scroll, wait, navigate)."
                "\n\nGUIDELINES:\n"
                "- Always return a JSON array assigned to the 'candidates' field.\n"
                "- Each item is an object with keys: 'thought' (string rationale) and 'action' (string command).\n"
                "- Provide at most 3 candidates, ordered from most to least promising.\n"
                "- Ground actions in visible/clickable DOM/AXTree elements, preferring stable locators "
                "(bid/test-id, role+name, aria-label, or unique text).\n"
                "- Each action string must be atomic, e.g., 'click(text=\"Sign in\")'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "description": (
                            "Ranked list of candidate actions. Each item must contain both "
                            "a 'thought' (the reasoning) and an 'action' (the executable step)."
                        ),
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "thought": {
                                    "type": "string",
                                    "description": "The reasoning behind proposing this action.",
                                },
                                "action": {
                                    "type": "string",
                                    "description": "The atomic, executable action string.",
                                },
                            },
                            "required": ["thought", "action"],
                        },
                    }
                },
                "required": ["candidates"],
            },
        },
    }
]


SYS_MSG = "You are a web navigation agent that proposes the NEXT ACTIONS an agent should take to achieve the goal"


@dataclass
class HumanGuidedAgentArgs(AgentArgs):

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"HumanGuided".replace("/", "_")
        except AttributeError:
            pass
        self.action_set_args = bgym.DEFAULT_BENCHMARKS[
            "miniwob_tiny_test"
        ]().high_level_action_set_args
        self.use_html = False
        self.model_name = "openai/gpt-5-mini-2025-08-07"

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        if benchmark.name.startswith("miniwob"):
            self.use_html = True
        self.action_set_args = benchmark.high_level_action_set_args

    def make_agent(self):
        return HumanGuidedAgent(
            self.action_set_args, use_html=self.use_html, model_name=self.model_name
        )


class HumanGuidedAgent(Agent):
    def __init__(
        self,
        action_set_args,
        model_name,
        use_html=False,
    ):
        self.action_set = action_set_args.make_action_set()
        self.use_html = use_html
        self.model_name = model_name
        self.model_args = LiteLLMModelArgs(
            model_name=self.model_name,
            max_new_tokens=2000,
            temperature=None,
            use_only_first_toolcall=True,
        )
        self.llm, self.msg = self.model_args.make_model(), self.model_args.get_message_builder()
        self._step = 0
        self.hints = []

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

    def get_candidates(self, messages):
        """
        Returns a list of dict with keys 'action' and 'thought'
        """
        response = self.llm(
                    APIPayload(messages=messages, tools=tools, force_call_tool="send_action_candidates")
                )

        candidates = response.tool_calls.tool_calls[0].arguments["candidates"]

        return candidates

    def get_action(self, obs):
        action_description = self.action_set.describe(
            with_long_description=True, with_examples=True
        )
        messages = [
            self.msg.system().add_text(SYS_MSG),
            self.msg.user().add_text(f"""##ACTION DESCRIPTION\n {action_description}"""),
            self.msg.user().add_text(f"##Goal:\n {obs['goal_object'][0]['text']}"),
            self.msg.user().add_text(f"""##Pruned HTML\n {obs["pruned_html"]}"""),
            self.msg.user().add_text(f"""##AXTREE HTML\n {obs["axtree_txt"]}"""),
            # self.msg.user().add_text("## SCREENSHOT"),
            # self.msg.user().add_image(image_to_png_base64_url(obs["screenshot"])),
        ]
        # Init Action Candidates
        candidates = self.get_candidates(messages)
        candidates_str = "\n".join([f"{i}. {c['action']} (Reasoning: {c['thought']})" for i, c in enumerate(candidates, 1)])
        while True:

            prompt = (
                f"\n\033[1m{'':=^60}\033[0m\n"                           
                f"\033[92m{'STEP ' + str(self._step+1):^60}\033[0m\n"   
                f"\033[96m{'SELECT ACTION:':^60}\033[0m\n\n"             
                f"{candidates_str}\n\n"                                  
                f"\033[93m{'OR Type a hint:':^60}\033[0m\n"              
                f"\033[1m{'':=^60}\033[0m\n> "                            
            )

            choice = input(prompt)
            if not choice.strip(): 
                continue
            hint = not (len(choice) == 1 and choice.isdigit() and 1 <= int(choice) <= len(candidates))
            if not hint:
                action = candidates[int(choice) - 1]['action']
                think = candidates[int(choice) - 1]['thought']
                break
            else:
                messages += [self.msg.user().add_text(f"## HINT\n {choice}")]
                self.hints.append(choice)
                candidates = self.get_candidates(messages)
                candidates_str = "\n".join(
                    [
                        f"{i}. {c['action']} (Reasoning: {c['thought']})"
                        for i, c in enumerate(candidates, 1)
                    ]
                )

        self._step += 1
        agent_info = AgentInfo(
            think=think,
            chat_messages=messages,
            stats={},
            extra_info={"hints": self.hints},
        )
        return action, agent_info


HUMAN_GUIDED_AGENT = HumanGuidedAgentArgs()

if __name__ == "__main__":
    import logging

    from agentlab.agents.human_trace_recorder.human_guidance_agent import (
        HUMAN_GUIDED_AGENT,
    )
    from agentlab.experiments.study import Study
    import logging


    agent_configs = [HUMAN_GUIDED_AGENT]
    benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l1"]()
    benchmark = benchmark.subset_from_glob("task_name", "*filter*")
    benchmark.env_args_list = benchmark.env_args_list[2:3]

    for env_args in benchmark.env_args_list:
        env_args.max_steps = 100  # max human steps
        env_args.headless = False

    Study(agent_configs, benchmark, logging_level=logging.WARNING).run(
        n_jobs=1,
        parallel_backend="sequential",
        n_relaunch=1,
    )
