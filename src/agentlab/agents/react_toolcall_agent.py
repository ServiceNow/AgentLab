import json
import logging
import pprint
from dataclasses import dataclass
from functools import partial
from typing import Callable

from litellm import completion_with_retries
from litellm.types.utils import ChatCompletionMessageToolCall, Message, ModelResponse
from PIL import Image
from termcolor import colored

from agentlab.actions import FunctionCall, ToolCallAction, ToolsActionSet, ToolSpec
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import image_to_png_base64_url

logger = logging.getLogger(__name__)

@dataclass
class Observation:
    data: dict # expected keys: goal_object, pruned_html, axtree_txt, screenshot, last_action_error, action_result

    def to_messages(self) -> list[dict]:
        """
        Convert the observation dictionary into a list of chat messages for Lite LLM
        """
        messages = []
        tool_call_id = self.data.get("tool_call_id")
        if self.data.get("goal_object") and not tool_call_id: # its a first observation when there are no tool_call_id, so include goal
            goal=self.data["goal_object"][0]["text"]
            messages.append({
                "role": "user",
                "content": f"## Goal:\n{goal}"
            })
        text_obs = []
        if self.data.get("action_result"):
            result=self.data["action_result"]
            text_obs.append(f"Action Result:\n{result}")
        if self.data.get("pruned_html"):
            html=self.data["pruned_html"]
            text_obs.append(f"Pruned HTML:\n{html}")
        if self.data.get("axtree_txt"):
            axtree=self.data["axtree_txt"]
            text_obs.append(f"Accessibility Tree:\n{axtree}")
        if self.data.get("last_action_error"):
            error = self.data['last_action_error']
            text_obs.append(f"Action Error:\n{error}")
        if text_obs:
            if tool_call_id:
                message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "\n\n".join(text_obs),
                }
            else:
                message = {
                    "role": "user",
                    "content": "\n\n".join(text_obs),
                }
            messages.append(message)
        if self.data.get("screenshot"):
            if isinstance(self.data["screenshot"], Image.Image):
                image_content_url = image_to_png_base64_url(self.data["screenshot"])
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image_content_url}}],
                })
            else:
                raise ValueError(f"Expected Image.Image, got {type(self.data['screenshot'])}")
        return messages

@dataclass
class LLMOutput:
    """
    LiteLLM output message containing all the llm response details, suitable for putting back into prompt to reuse KV cache
    """
    message: Message
    def to_messages(self) -> list[Message]:
        return [self.message]

@dataclass
class SystemMessage:
    message: str
    def to_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.message}]

@dataclass
class UserMessage:
    message: str
    def to_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.message}]

Step = LLMOutput | Observation | SystemMessage | UserMessage

@dataclass
class AgentConfig:
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    max_actions: int = 10
    max_retry: int = 4
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
Then produce the function call that performs the proposed action. If the task is complete, produce the final step.
"""

class LLMArgs(BaseModelArgs):
    reasoning_effort: str = "low"

    def make_model(self) -> Callable:
        return partial(
            completion_with_retries, 
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_total_tokens,
            max_completion_tokens=self.max_new_tokens,
            reasoning_effort=self.reasoning_effort,
        )

class ReactToolCallAgent:
    def __init__(self, action_set: ToolsActionSet, llm: Callable, config: AgentConfig):
        self.action_set = action_set
        self.history: list[Step] = [SystemMessage(message=config.system_prompt)]
        self.llm = llm
        self.config = config
        self.last_tool_call_id: str = ""

    def obs_preprocessor(self, obs: dict) -> dict:
        if not self.config.use_html:
            obs.pop("pruned_html", None)
        if not self.config.use_axtree:
            obs.pop("axtree_txt", None)
        if not self.config.use_screenshot:
            obs.pop("screenshot", None)
        if self.last_tool_call_id:
            # add tool_call_id to obs for linking observation to the last executed action
            obs["tool_call_id"] = self.last_tool_call_id
        return obs

    def get_action(self, obs: dict) -> tuple[ToolCallAction, dict]:
        prev_actions = [step for step in self.history if isinstance(step, LLMOutput)]
        if len(prev_actions) >= self.config.max_actions:
            logger.warning("Max actions reached, stopping agent.")
            stop_action = ToolCallAction(id="stop", function=FunctionCall(name="final_step", arguments={}))
            return stop_action, {}
        self.history.append(Observation(data=obs))
        steps = self.history + [UserMessage(message=self.config.guidance)]
        messages = [m for step in steps for m in step.to_messages()]
        tools = [tool.model_dump() for tool in self.action_set.actions]
        try:
            logger.info(colored(f"Prompt:\n{pprint.pformat(messages, width=120)}", "blue"))
            response: ModelResponse = self.llm(
                tools=tools,
                messages=messages,
                num_retries=self.config.max_retry,
            )
            message = response.choices[0].message # type: ignore
        except Exception as e:
            logger.exception(f"Error getting LLM response: {e}. Prompt: {messages}")
            raise e
        logger.info(colored(f"LLM response:\n{pprint.pformat(message, width=120)}", "green"))
        self.history.append(LLMOutput(message=message))
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
            logger.info(colored(f"LLM output:\n{message.content}", "cyan"))
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
                action = ToolCallAction(
                    id=tool_call.id,
                    function=FunctionCall(name=tool_call.function.name, arguments=args)
                )
            except json.JSONDecodeError as e:
                logger.exception(f"Error in json parsing of tool call arguments, {e}: {tool_call.function.arguments}")
                raise e
            
            self.last_tool_call_id = action.id
        else:
            raise ValueError(f"No tool call found in LLM response: {message}")
        return action
    

@dataclass
class ReactToolCallAgentArgs(AgentArgs):
    llm_args: LLMArgs = None # type: ignore
    config: AgentConfig = None # type: ignore

    def make_agent(self, actions: list[ToolSpec]) -> ReactToolCallAgent:
        llm = self.llm_args.make_model()
        action_set = ToolsActionSet(actions=actions)
        return ReactToolCallAgent(action_set=action_set, llm=llm, config=self.config)
    
