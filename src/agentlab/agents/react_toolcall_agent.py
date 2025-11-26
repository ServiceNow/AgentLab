import json
import logging
import pprint
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
from litellm import completion
from litellm.types.utils import Message, ModelResponse
from litellm.utils import token_counter
from PIL import Image
from termcolor import colored

from agentlab.actions import ToolCall, ToolsActionSet, ToolSpec
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import image_to_png_base64_url

logger = logging.getLogger(__name__)


class LLMArgs(BaseModelArgs):
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low"
    num_retries: int = 3

    def make_model(self) -> Callable:
        return partial(
            completion,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_total_tokens,
            max_completion_tokens=self.max_new_tokens,
            reasoning_effort=self.reasoning_effort,
            num_retries=self.num_retries,
            tool_choice="auto",
            parallel_tool_calls=False,
        )


@dataclass
class AgentConfig:
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    max_actions: int = 10
    max_history_tokens: int = 120000
    system_prompt: str = """
You are an expert AI Agent trained to assist users with complex web tasks.
Your role is to understand the goal, perform actions until the goal is accomplished and respond in a helpful and accurate manner.
Keep your replies brief, concise, direct and on topic. Prioritize clarity and avoid over-elaboration.
Do not express emotions or opinions."""
    guidance: str = """
Think along the following lines:
1. Summarize the last observation and describe the visible changes in the state.
2. Evaluate action success, explain impact on task and next steps.
3. If you see any errors in the last observation, think about it. If there is no error, just move on.
4. List next steps to move towards the goal and propose next immediate action.
Then produce the single function call that performs the proposed action. If the task is complete, produce the final step."""
    summarize_system_prompt: str = """
You are a helpful assistant that summarizes agent interaction history. Following messages is the history to summarize:"""
    summarize_prompt: str = """
Summarize the presented agent interaction history concisely.
Focus on:
- The original goal
- Key actions taken and their outcomes
- Important errors or obstacles encountered
- Current progress toward the goal
Provide a concise summary that preserves all information needed to continue the task."""


def user_message(content: str | list[dict]) -> dict:
    return {"role": "user", "content": content}


class ReactToolCallAgent:
    def __init__(
        self,
        action_set: ToolsActionSet,
        llm: Callable[..., ModelResponse],
        token_counter: Callable[..., int],
        config: AgentConfig,
    ):
        self.action_set = action_set
        self.tools = self.action_set.tools()
        self.history: list[dict | Message] = [{"role": "system", "content": config.system_prompt}]
        self.llm = llm
        self.token_counter = token_counter
        self.config = config
        self.last_tool_call_id: str = ""

    def obs_preprocessor(self, obs: dict) -> dict:
        return obs

    def obs_to_messages(self, obs: dict) -> list[dict]:
        """
        Convert the observation dictionary into a list of chat messages for Lite LLM
        """
        images = {k: v for k, v in obs.items() if isinstance(v, (Image.Image, np.ndarray))}
        texts = {k: v for k, v in obs.items() if k not in images and v is not None and v != ""}
        messages = []

        if not self.last_tool_call_id and (goal_obj := texts.pop("goal_object", None)):
            # its a first observation when there are no tool_call_id, so include goal
            goal = goal_obj[0]["text"]
            messages.append(user_message(f"Goal: {goal}"))

        text = "\n\n".join([f"## {k}\n{v}" for k, v in texts.items()])
        if self.last_tool_call_id:
            message = {
                "role": "tool",
                "tool_call_id": self.last_tool_call_id,
                "content": text,
            }
        else:
            message = user_message(text)
        messages.append(message)

        if self.config.use_screenshot:
            for caption, image in images.items():
                image_content = [
                    {"type": "text", "text": caption},
                    {"type": "image_url", "image_url": {"url": image_to_png_base64_url(image)}},
                ]
                messages.append(user_message(image_content))

        return messages

    def get_action(self, obs: dict) -> tuple[ToolCall, dict]:
        if self.max_actions_reached():
            logger.warning("Max actions reached, stopping agent.")
            return ToolCall(name="final_step"), {}

        self.history += self.obs_to_messages(obs)
        self.maybe_compact_history()
        messages = self.history + [{"role": "user", "content": self.config.guidance}]

        try:
            logger.info(colored(f"Prompt:\n{pprint.pformat(messages, width=120)}", "blue"))
            response = self.llm(tools=self.tools, messages=messages)
            message = response.choices[0].message  # type: ignore
        except Exception as e:
            logger.exception(f"Error getting LLM response: {e}. Prompt: {messages}")
            raise e
        logger.info(colored(f"LLM response:\n{pprint.pformat(message, width=120)}", "green"))

        self.history.append(message)
        thoughts = self.thoughts_from_message(message)
        action = self.action_from_message(message)
        return action, {"think": thoughts, "chat_messages": self.history}

    def max_actions_reached(self) -> bool:
        prev_actions = [msg for msg in self.history if isinstance(msg, Message) and msg.tool_calls]
        return len(prev_actions) >= self.config.max_actions

    def thoughts_from_message(self, message: Message) -> str:
        """Extract the agent's thoughts from the LLM message."""
        thoughts = []
        if reasoning := message.get("reasoning_content"):
            thoughts.append(reasoning)
        if blocks := message.get("thinking_blocks"):
            for block in blocks:
                if thinking := getattr(block, "content", None) or getattr(block, "thinking", None):
                    thoughts.append(thinking)
        if message.content:
            thoughts.append(message.content)
        logger.info(colored(f"LLM thoughts: {thoughts}", "cyan"))
        return "\n\n".join(thoughts)

    def action_from_message(self, message: Message) -> ToolCall:
        """Parse the ToolCall from the LLM message."""
        if message.tool_calls:
            if len(message.tool_calls) > 1:
                logger.warning("Multiple tool calls found in LLM response, using the first one.")
            tool_call = message.tool_calls[0]
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            action = ToolCall(id=tool_call.id, name=name, arguments=args)
            self.last_tool_call_id = action.id
            logger.info(colored(f"Parsed tool call: {action}", "magenta"))
        else:
            raise ValueError(f"No tool call found in LLM response: {message}")
        return action

    def maybe_compact_history(self):
        tokens = self.token_counter(messages=self.history)
        if tokens > self.config.max_history_tokens:
            logger.info("Compacting history due to length.")
            self.compact_history()
            short_tokens = self.token_counter(messages=self.history)
            logger.info(f"Compacted history from {tokens} to {short_tokens} tokens.")

    def compact_history(self):
        """
        Compact the history by summarizing the first half of messages with the LLM.
        Updates self.history in place by replacing the first half with the summary message.
        """
        system_msg = self.history[0]
        rest = self.history[1:]
        midpoint = len(rest) // 2
        messages = [
            {"role": "system", "content": self.config.summarize_system_prompt},
            *rest[:midpoint],
            {"role": "user", "content": self.config.summarize_prompt},
        ]

        try:
            response = self.llm(messages=messages, tool_choice="none")
            summary = response.choices[0].message.content  # type: ignore
        except Exception as e:
            logger.exception(f"Error compacting history: {e}")
            raise

        logger.info(colored(f"Compacted {midpoint} messages into summary:\n{summary}", "cyan"))
        # Rebuild history: system + summary + remaining messages
        summary_message = {"role": "user", "content": f"## Previous Interaction :\n{summary}"}
        self.history = [system_msg, summary_message, *rest[midpoint:]]


@dataclass
class ReactToolCallAgentArgs(AgentArgs):
    llm_args: LLMArgs | None = None
    config: AgentConfig | None = None

    def make_agent(self, actions: list[ToolSpec]) -> ReactToolCallAgent:
        llm = self.llm_args.make_model()
        counter = partial(token_counter, model=self.llm_args.model_name)
        action_set = ToolsActionSet(actions=actions)
        return ReactToolCallAgent(action_set, llm, counter, self.config)
