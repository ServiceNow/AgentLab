import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments.agent import Agent
from browsergym.experiments.loop import AbstractAgentArgs
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from agentlab.llm.llm_utils import ParseError, extract_code_blocks, retry_raise

if TYPE_CHECKING:
    from agentlab.llm.chat_api import BaseModelArgs


@dataclass
class BasicAgentArgs(AbstractAgentArgs):
    agent_name: str = "BasicAgent"
    temperature: float = 0.1
    use_chain_of_thought: bool = False
    chat_model_args: "BaseModelArgs" = None

    def make_agent(self) -> Agent:
        return BasicAgent(
            temperature=self.temperature,
            use_chain_of_thought=self.use_chain_of_thought,
            chat_model_args=self.chat_model_args,
        )


class BasicAgent(Agent):
    def __init__(
        self, temperature: float, use_chain_of_thought: bool, chat_model_args: "BaseModelArgs"
    ):
        self.temperature = temperature
        self.use_chain_of_thought = use_chain_of_thought
        self.chat = chat_model_args.make_model()

        self.action_set = HighLevelActionSet(["bid"], multiaction=False)

    def get_action(self, obs: Any) -> tuple[str, dict]:
        system_prompt = f"""
You are a web assistant.
"""
        prompt = f"""
You are helping a user to accomplish the following goal on a website:

{obs["goal"]}

Here is the current state of the website, in the form of an html:

{obs["dom_txt"]}

To do so, you can interact with the environment using the following actions:

{self.action_set.describe(with_long_description=False)}

The inputs to those functions are the bids given in the html.

The action you provide must be in between triple ticks.
Here is an example of how to use the bid action:

```
click('a314')
```

Please provide a single action at a time and wait for the next observation. Provide only a single action per step. 
Focus on the bid that are given in the html, and use them to perform the actions.
"""
        if self.use_chain_of_thought:
            prompt += f"""
Provide a chain of thoughts reasoning to decompose the task into smaller steps. And execute only the next step.
"""

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

        messages.append(AIMessage(content=thought))

        return (
            action,
            AgentInfo(
                think=thought,
                chat_messages=messages,  # TODO put lanchgchain objects
                # put any stats that you care about as long as it is a number or a dict of numbers
                stats={"prompt_length": len(prompt), "response_length": len(thought)},
                markup_page="Add any txt information here, including base 64 images, to display in xray",
            ),
        )


@dataclass
class AgentInfo:
    think: str = None
    chat_messages: list[str] = None
    stats: dict = None
    markup_page: str = None
    extra_info: dict = None

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        return {
            "think": self.think,
            "chat_messages": self.chat_messages,
            "stats": self.stats,
            "markup_page": self.markup_page,
        }


if __name__ == "__main__":
    from browsergym.experiments.loop import EnvArgs, ExpArgs

    env_args = EnvArgs(
        task_name="miniwob.click-menu",
        task_seed=0,
        max_steps=10,
        headless=True,
    )

    exp_args = ExpArgs(
        agent_args=BasicAgentArgs(temperature=0.1, use_chain_of_thought=True),
        env_args=env_args,
        logging_level=logging.INFO,
    )

    exp_args.prepare("exp/basic_agent")
    exp_args.run()
