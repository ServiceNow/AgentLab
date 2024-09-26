from dataclasses import dataclass
from functools import partial
from typing import Any

from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs, make_system_message, make_user_message
from agentlab.llm.llm_utils import ParseError, RetryError, parse_html_tags_raise, retry_raise


@dataclass
class VerifierWrapperArgs(AgentArgs):
    """Arguments for the WrapperAgent."""

    agent_args: AgentArgs
    chat_model_args: BaseModelArgs
    agent_name: str = "VerifierWrapperAgent"

    def __post_init__(self):
        self.agent_name = f"VerifierWrapped-{self.agent_args.agent_name}"

    def set_benchmark(self, benchmark: str):
        self.agent_args.set_benchmark(benchmark)

    def make_agent(self):
        return VerifierAgent(self.agent_args, self.chat_model_args)


class VerifierAgent(Agent):
    """An agent that wraps another agent."""

    def __init__(self, agent_args: AgentArgs, chat_model_args: BaseModelArgs):
        self.agent = agent_args.make_agent()
        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args

        self.reset()

    def get_action(self, obs: Any) -> tuple[str, AgentInfo]:
        self.obs_history.append(obs)
        preprocessed_obs, preprocess_info = self.preprocess_agent(obs)

        action, agent_info = self.agent.get_action(preprocessed_obs)

        _, postprocess_info = self.postprocess_agent(action, obs)

        return action, self.aggregate_info(agent_info, preprocess_info, postprocess_info)

    def reset(self):
        self.obs_history = []
        self.intention_history = []
        self.action_history = []

    def preprocess_agent(self, obs: Any) -> tuple[dict, dict]:

        if len(self.obs_history) == 1:
            return obs, {}

        old_goal = obs["goal"]

        previous_html = self.obs_history[-2]["pruned_html"]
        new_html = obs["pruned_html"]
        action = self.action_history[-1]

        intention = self.intention_history[-1]

        system_prompt = """
You are a controller for a web agent. Your goal is to verify if an action was successful based on:
- The previous observation
- The current observation
- The action executed between the two observations by the agent
- The intention behind the action

You need to assess wether the action was successful in achieving the intention.

Use <success>True</success> or <success>False</success> to encapsulate the success of the action.

Add <extra_info> and </extra_info> to add any additional information about the success or failure of the action.
In case of success, give ideas for future steps.
In case of failure, give ideas to fix the issue.

Here are the previous and current observations, the action and the intention you need to verify:
"""

        obs_prompt = f"""
### Previous Observation
{previous_html}

### Current Observation
{new_html}

### Action
{action}

### Intention
{intention}
"""

        def parser(text: str) -> dict:
            success = partial(parse_html_tags_raise, keys=["success"])(text)
            if success["success"] == "False":
                try:
                    extra_info = parse_html_tags_raise(text, keys=["extra_info"])["extra_info"]
                except ParseError:
                    extra_info = "Failed to identify reason for failure"

                return {"success": success["success"], "extra_info": extra_info}

            return {"success": success["success"]}

        try:
            prompt = [
                make_system_message(system_prompt),
                make_user_message(obs_prompt),
            ]
            verification = retry_raise(
                self.chat_llm,
                prompt,
                parser=parser,
                n_retry=3,
            )
        except RetryError as e:
            verification = {"success": "Failed to verify success"}

        if verification["success"] == "False":
            obs["goal"] += f"\n\nLast attempt failed because: {verification['extra_info']}"

        return obs, {"verification": verification, "old_goal": old_goal}

    def postprocess_agent(self, action: Any, obs: Any) -> tuple[str, dict]:
        self.action_history.append(action)

        axtree = obs["pruned_html"]
        goal = obs["goal"]

        system_prompt = """
You are a controller of a web agent. Your goal is to identify the intentions of the agent based on:
- The current goal
- The current state of the web page, represented by the html.
- Its action

First describe the webpage with respect to the goal. Then describe the action of the agent.

Finally, identify the intention of the agent and the impact the action should have on the webpage.

Use <intention> and </intention> to encapsulate the intention of the agent.

Example answer:

The webpage is a store page with a list of products. The goal is to buy a red car.
The agent is typing "red car" in the search bar.

<intention>
The agent is searching for a red car.
The search bar should have the text "red car".
</intention>

Here are the goal, observation and action you need to identify:
"""

        obs_prompt = f"""
### Goal
{goal}

### HTML
{axtree}

### Action
{action}
"""
        parser = partial(parse_html_tags_raise, keys=["intention"])

        try:
            prompt = [
                make_system_message(system_prompt),
                make_user_message(obs_prompt),
            ]
            intention = retry_raise(
                self.chat_llm,
                prompt,
                parser=parser,
                n_retry=3,
            )
        except RetryError as e:
            intention = "Failed to identify intention"

        self.intention_history.append(intention)
        return intention, {"intention_messages": prompt}

    def aggregate_info(
        self, agent_info: AgentInfo, preprocess_info: dict, postprocess_info: dict
    ) -> AgentInfo:
        agent_info.extra_info["preprocess_info"] = preprocess_info
        agent_info.extra_info["postprocess_info"] = postprocess_info

        markup = ""

        if "verification" in preprocess_info:
            verification = preprocess_info["verification"]
            if verification["success"] == "False":
                markup = f"Verification failed: {verification['extra_info']}"
            else:
                markup = "Verification successful"

        if "intention_messages" in postprocess_info:
            for message in postprocess_info["intention_messages"]:
                markup += f"\n\n{message.get('content', 'No content')}"
        # add old and new goal
        markup += f"\n\nOld goal: {preprocess_info.get('old_goal', 'No old goal')}"

        agent_info.markup_page = markup
        return agent_info
