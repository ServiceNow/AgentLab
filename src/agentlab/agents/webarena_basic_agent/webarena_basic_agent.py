import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.experiments.loop import EnvArgs, ExpArgs
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_utils import ParseError, extract_code_blocks, retry_raise

if TYPE_CHECKING:
    from agentlab.llm.chat_api import BaseModelArgs


@dataclass
class WebArenaBasicAgentArgs(AgentArgs):
    agent_name: str = "WebArenaBasicAgent"
    temperature: float = 0.0
    chat_model_args: "BaseModelArgs" = None

    def make_agent(self) -> Agent:
        return WebArenaBasicAgent(
            temperature=self.temperature,
            chat_model_args=self.chat_model_args,
        )

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()


class WebArenaBasicAgent(Agent):
    def __init__(self, temperature: float, chat_model_args: "BaseModelArgs"):
        self.temperature = temperature
        self.chat = chat_model_args.make_model()
        self.chat_model_args = chat_model_args

        self.action_set = HighLevelActionSet(["chat", "bid"], strict=False, multiaction=False)
        self.prev_actions = []

    def get_action(self, obs: Any) -> tuple[str, dict]:
        system_prompt = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

# Goal:
{obs["goal"]}

# Action Space
{self.action_set.describe(with_long_description=False, with_examples=True)}
"""
        cur_axtree_txt = obs["axtree_txt"]
        prev_action_str = "\n".join(self.prev_actions)

        prompt = f"""\
# Current Accessibility Tree:
{cur_axtree_txt}

# Previous Actions
{prev_action_str}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```
click("12")
```
"

Here is another example with chain of thought of a valid action when providing a concise answer to user:
"
In order to accomplish my goal I need to send the information asked back to the user. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I will send a message back to user with the answer.
```
send_msg_to_user("$279.49")
```
"
""".strip()

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        def parser(response: str) -> tuple[dict, bool, str]:
            blocks = extract_code_blocks(response)
            if len(blocks) == 0:
                raise ParseError("No code block found in the response")
            action = blocks[0][1]
            thought = response
            return {"action": action, "think": thought}

        ans_dict = retry_raise(self.chat, messages, n_retry=3, parser=parser)

        action = ans_dict.get("action", None)
        thought = ans_dict.get("think", None)

        self.prev_actions.append(action)

        return (
            action,
            AgentInfo(
                think=thought,
                chat_messages=messages,
                # put any stats that you care about as long as it is a number or a dict of numbers
                stats={"prompt_length": len(prompt), "response_length": len(thought)},
                markup_page="Add any txt information here, including base 64 images, to display in xray",
                extra_info={"chat_model_args": asdict(self.chat_model_args)},
            ),
        )


env_args = EnvArgs(
    task_name="miniwob.click-button",
    task_seed=0,
    max_steps=10,
    headless=True,
)

chat_model_args = CHAT_MODEL_ARGS_DICT["azure/gpt-35-turbo/gpt-35-turbo"]

exp_args = [
    ExpArgs(
        agent_args=WebArenaBasicAgentArgs(
            temperature=0.1,
            chat_model_args=chat_model_args,
        ),
        env_args=env_args,
        logging_level=logging.INFO,
    ),
    ExpArgs(
        agent_args=WebArenaBasicAgentArgs(
            temperature=0.0,
            chat_model_args=chat_model_args,
        ),
        env_args=env_args,
        logging_level=logging.INFO,
    ),
]


def experiment_config():
    return exp_args


if __name__ == "__main__":
    from agentlab.experiments.exp_utils import RESULTS_DIR

    exp_dir = os.path.join(RESULTS_DIR, "webarena_basic_agent")

    for exp_arg in exp_args:
        exp_arg.agent_args.prepare()
        exp_arg.prepare(exp_dir)

    for exp_arg in exp_args:
        exp_arg.run()
        exp_arg.agent_args.close()
