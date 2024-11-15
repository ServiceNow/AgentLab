from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import Agent, AgentInfo
from browsergym.experiments.benchmark import Benchmark, HighLevelActionSetArgs

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import make_system_message, make_user_message
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_utils import ParseError, extract_code_blocks, retry
from agentlab.llm.tracking import cost_tracker_decorator

if TYPE_CHECKING:
    from agentlab.llm.chat_api import BaseModelArgs


@dataclass
class WebArenaBasicAgentArgs(AgentArgs):
    agent_name: str = "WebArenaBasicAgent"
    temperature: float
    chat_model_args: "BaseModelArgs"
    action_set_args: HighLevelActionSetArgs

    def __post_init__(self):
        self.agent_name = f"WebArenaBasicAgent-{self.chat_model_args.model_name}".replace("/", "_")

    def make_agent(self) -> Agent:
        return WebArenaBasicAgent(
            temperature=self.temperature,
            chat_model_args=self.chat_model_args,
            action_set=HighLevelActionSetArgs.make_action_set(),
        )

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        self.action_set_args = deepcopy(benchmark.high_level_action_set_args)

    def set_reproducibility_mode(self):
        self.temperature = 0.0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()


# Best-attempt reproduction of original prompts from
# https://github.com/web-arena-x/webarena/blob/e31c190c9b43f63e5724322b847e00249300df40/agent/prompts/raw/p_cot_id_actree_2s_no_na.py#L1
webarena_intro_template = """\
You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

{action_space_description}

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the next action I will perform is ```click("1234")```".
"""

webarena_examples = [
    """\
OBSERVATION:
[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
		[1749] StaticText '$279.49'
		[1757] button 'Add to Cart'
		[1760] button 'Add to Wish List'
		[1761] button 'Add to Compare'
URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the price of HP Inkjet Fax Machine
PREVIOUS ACTION: None
ACTION: Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```
""",
    """\
OBSERVATION:
[164] textbox 'Search' focused: True required: False
[171] button 'Go'
[174] link 'Find directions between two points'
[212] heading 'Search Results'
[216] button 'Close'
URL: http://openstreetmap.org
OBJECTIVE: Show me the restaurants near CMU
PREVIOUS ACTION: None
ACTION: Let's think step-by-step. This page has a search box whose ID is [164]. According to the nominatim rule of openstreetmap, I can search for the restaurants near a location by \"restaurants near\". I can submit my typing by pressing the Enter afterwards. In summary, the next action I will perform is ```type [164] [restaurants near CMU] [1]```
""",
]

webarena_template = """\
OBSERVATION:
{observation}
URL: {url}
OBJECTIVE: {objective}
PREVIOUS ACTION: {previous_action}"""


def parser(response: str) -> tuple[dict, bool, str]:
    blocks = extract_code_blocks(response)
    if len(blocks) == 0:
        raise ParseError("No code block found in the response")
    action = blocks[0][1]
    thought = response
    return {"action": action, "think": thought}


class WebArenaBasicAgent(Agent):
    def __init__(self, temperature: float, chat_model_args: "BaseModelArgs"):
        self.temperature = temperature
        self.chat = chat_model_args.make_model()
        self.chat_model_args = chat_model_args

        self.action_set = HighLevelActionSet(["chat", "bid"], strict=False, multiaction=False)
        self.action_history = ["None"]

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str, dict]:
        """
        Replica of WebArena agent
        https://github.com/web-arena-x/webarena/blob/e31c190c9b43f63e5724322b847e00249300df40/agent/prompts/prompt_constructor.py#L38
        """
        messages = []

        # intro
        messages.append(
            make_system_message(
                webarena_intro_template.format(
                    action_space_description=self.action_set.describe(
                        with_long_description=False, with_examples=False
                    )
                )
            )
        )

        # examples
        for i, example in enumerate(webarena_examples):
            messages.append(
                make_system_message(
                    f"""\
Example {i + 1}/{len(webarena_examples)}:

{example}
"""
                )
            )

        # current observation
        cur_tabs_txt = " | ".join(
            f"Tab {i}{' (current)' if i == obs['active_page_index'][0] else ''}: {title}"
            for i, title in enumerate(obs["open_pages_titles"])
        )
        cur_axtree_txt = obs["axtree_txt"]

        messages.append(
            make_user_message(
                webarena_template.format(
                    observation=f"""\
{cur_tabs_txt}

{cur_axtree_txt}
""",
                    url=obs["url"],
                    objective=obs["goal"],  # TODO convert goal_object to text (images?)
                    previous_action=self.action_history[-1],
                )
            )
        )

        ans_dict = retry(self.chat, messages, n_retry=3, parser=parser)

        action = ans_dict.get("action", None)
        thought = ans_dict.get("think", None)

        self.action_history.append(action)

        return (
            action,
            AgentInfo(
                think=thought,
                chat_messages=messages,
                # put any stats that you care about as long as it is a number or a dict of numbers
                stats={
                    "prompt_length": sum(len(message.content) for message in messages),
                    "response_length": len(thought),
                },
                extra_info={"chat_model_args": asdict(self.chat_model_args)},
            ),
        )


AGENT_4o_MINI = WebArenaBasicAgentArgs(
    temperature=0.1,
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
)

AGENT_4o = WebArenaBasicAgentArgs(
    temperature=0.1,
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4o-2024-08-06"],
)

AGENT_SONNET = WebArenaBasicAgentArgs(
    temperature=0.1,
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/anthropic/claude-3.5-sonnet:beta"],
)
