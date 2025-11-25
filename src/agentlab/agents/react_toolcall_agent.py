import json
import logging
import pprint
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

from litellm import completion
from litellm.types.utils import ChatCompletionMessageToolCall, Message, ModelResponse
from PIL import Image
from termcolor import colored

from agentlab.actions import FunctionCall, ToolCallAction, ToolsActionSet, ToolSpec
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
    system_prompt: str = """
You are an expert AI Agent trained to assist users with complex web tasks.
Your role is to understand the goal, perform actions until the goal is accomplished and respond in a helpful and accurate manner.
Keep your replies brief, concise, direct and on topic. Prioritize clarity and avoid over-elaboration.
Do not express emotions or opinions.
"""
    guidance: str = """
Think along the following lines:
1. Summarize the last observation and describe the visible changes in the state.
2. Evaluate action success, explain impact on task and next steps.
3. If you see any errors in the last observation, think about it. If there is no error, just move on.
4. List next steps to move towards the goal and propose next immediate action.
Then produce the single function call that performs the proposed action. If the task is complete, produce the final step.
"""


class ReactToolCallAgent:
    def __init__(
        self, action_set: ToolsActionSet, llm: Callable[..., ModelResponse], config: AgentConfig
    ):
        self.action_set = action_set
        self.history: list[dict | Message] = [{"role": "system", "content": config.system_prompt}]
        self.llm = llm
        self.config = config
        self.last_tool_call_id: str = ""

    def obs_preprocessor(self, obs: dict) -> dict:
        return obs

    def obs_to_messages(self, obs: dict) -> list[dict]:
        """
        Convert the observation dictionary into a list of chat messages for Lite LLM
        """
        messages = []
        if obs.get("goal_object") and not self.last_tool_call_id:
            # its a first observation when there are no tool_call_id, so include goal
            goal = obs["goal_object"][0]["text"]
            messages.append({"role": "user", "content": f"## Goal:\n{goal}"})
        text_obs = []
        if result := obs.get("action_result"):
            text_obs.append(f"## Action Result:\n{result}")
        if error := obs.get("last_action_error"):
            text_obs.append(f"## Action Error:\n{error}")
        if self.config.use_html and (html := obs.get("pruned_html")):
            text_obs.append(f"## HTML:\n{html}")
        if self.config.use_axtree and (axtree := obs.get("axtree_txt")):
            text_obs.append(f"## Accessibility Tree:\n{axtree}")
        content = "\n\n".join(text_obs)
        if content:
            if self.last_tool_call_id:
                message = {
                    "role": "tool",
                    "tool_call_id": self.last_tool_call_id,
                    "content": content,
                }
            else:
                message = {"role": "user", "content": content}
            messages.append(message)
        if self.config.use_screenshot and (scr := obs.get("screenshot")):
            if isinstance(scr, Image.Image):
                image_content = [
                    {"type": "image_url", "image_url": {"url": image_to_png_base64_url(scr)}}
                ]
                messages.append({"role": "user", "content": image_content})
            else:
                raise ValueError(
                    f"Expected Image.Image in screenshot obs, got {type(obs['screenshot'])}"
                )
        return messages

    def get_action(self, obs: dict) -> tuple[ToolCallAction, dict]:
        actions_count = len(
            [msg for msg in self.history if isinstance(msg, Message) and msg.tool_calls]
        )
        if actions_count >= self.config.max_actions:
            logger.warning("Max actions reached, stopping agent.")
            stop_action = ToolCallAction(
                id="stop", function=FunctionCall(name="final_step", arguments={})
            )
            return stop_action, {}
        self.history += self.obs_to_messages(self.obs_preprocessor(obs))
        tools = [tool.model_dump() for tool in self.action_set.actions]
        messages = self.history + [{"role": "user", "content": self.config.guidance}]

        try:
            logger.info(colored(f"Prompt:\n{pprint.pformat(messages, width=120)}", "blue"))
            response = self.llm(tools=tools, messages=messages)
            message = response.choices[0].message  # type: ignore
        except Exception as e:
            logger.exception(f"Error getting LLM response: {e}. Prompt: {messages}")
            raise e
        logger.info(colored(f"LLM response:\n{pprint.pformat(message, width=120)}", "green"))

        self.history.append(message)
        thoughts = self.thoughts_from_message(message)
        action = self.action_from_message(message)
        return action, {"think": thoughts}

    def thoughts_from_message(self, message) -> str:
        thoughts = []
        if reasoning := message.get("reasoning_content"):
            logger.info(colored(f"LLM reasoning:\n{reasoning}", "yellow"))
            thoughts.append(reasoning)
        if blocks := message.get("thinking_blocks"):
            for block in blocks:
                if thinking := getattr(block, "content", None) or getattr(block, "thinking", None):
                    logger.info(colored(f"LLM thinking block:\n{thinking}", "yellow"))
                    thoughts.append(thinking)
        if message.content:
            logger.info(colored(f"LLM text output:\n{message.content}", "cyan"))
            thoughts.append(message.content)
        return "\n\n".join(thoughts)

    def action_from_message(self, message) -> ToolCallAction:
        if message.tool_calls:
            if len(message.tool_calls) > 1:
                logger.warning("Multiple tool calls found in LLM response, using the first one.")
            tool_call: ChatCompletionMessageToolCall = message.tool_calls[0]
            assert isinstance(tool_call.function.name, str)
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.exception(
                    f"Error in json parsing of tool call arguments, {e}: {tool_call.function.arguments}"
                )
                raise e
            action = ToolCallAction(
                id=tool_call.id, function=FunctionCall(name=tool_call.function.name, arguments=args)
            )
            self.last_tool_call_id = action.id
            logger.info(f"Parsed tool call action: {action}")
        else:
            raise ValueError(f"No tool call found in LLM response: {message}")
        return action


@dataclass
class ReactToolCallAgentArgs(AgentArgs):
    llm_args: LLMArgs | None = None
    config: AgentConfig | None = None

    def make_agent(self, actions: list[ToolSpec]) -> ReactToolCallAgent:
        llm = self.llm_args.make_model()
        action_set = ToolsActionSet(actions=actions)
        return ReactToolCallAgent(action_set=action_set, llm=llm, config=self.config)
