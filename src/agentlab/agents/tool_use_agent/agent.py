import json
import logging
from copy import deepcopy as copy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import bgym
from browsergym.core.observation import extract_screenshot

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from agentlab.llm.response_api import OpenAIResponseModelArgs
from agentlab.llm.tracking import cost_tracker_decorator

if TYPE_CHECKING:
    from openai.types.responses import Response


@dataclass
class ToolUseAgentArgs(AgentArgs):
    temperature: float = 0.1
    model_args: OpenAIResponseModelArgs = None

    def __post_init__(self):
        try:
            self.agent_name = f"ToolUse-{self.model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self) -> bgym.Agent:
        return ToolUseAgent(
            temperature=self.temperature,
            model_args=self.model_args,
        )

    def set_reproducibility_mode(self):
        self.temperature = 0

    def prepare(self):
        return self.model_args.prepare_server()

    def close(self):
        return self.model_args.close_server()


class ToolUseAgent(bgym.Agent):
    def __init__(
        self,
        temperature: float,
        model_args: OpenAIResponseModelArgs,
    ):
        self.temperature = temperature
        self.chat = model_args.make_model()
        self.model_args = model_args

        self.action_set = bgym.HighLevelActionSet(["coord"], multiaction=False)

        self.tools = self.action_set.to_tool_description()

        # self.tools.append(
        #     {
        #         "type": "function",
        #         "name": "chain_of_thought",
        #         "description": "A tool that allows the agent to think step by step. Every other action must ALWAYS be preceeded by a call to this tool.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "thoughts": {
        #                     "type": "string",
        #                     "description": "The agent's reasoning process.",
        #                 },
        #             },
        #             "required": ["thoughts"],
        #         },
        #     }
        # )

        self.llm = model_args.make_model(extra_kwargs={"tools": self.tools})

        self.messages = []

    def obs_preprocessor(self, obs):
        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
        else:
            raise ValueError("No page found in the observation.")

        return obs

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str, dict]:

        if len(self.messages) == 0:
            system_message = {
                "role": "system",
                "content": "You are an agent. Based on the observation, you will decide which action to take to accomplish your goal.",
            }
            goal_object = [el for el in obs["goal_object"]]
            for content in goal_object:
                if content["type"] == "text":
                    content["type"] = "input_text"
                elif content["type"] == "image_url":
                    content["type"] = "input_image"
            goal_message = {"role": "user", "content": goal_object}
            goal_message["content"].append(
                {
                    "type": "input_image",
                    "image_url": image_to_png_base64_url(obs["screenshot"]),
                }
            )
            self.messages.append(system_message)
            self.messages.append(goal_message)
        else:
            if obs["last_action_error"] == "":
                self.messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": self.previous_call_id,
                        "output": "Function call executed, see next observation.",
                    }
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": image_to_png_base64_url(obs["screenshot"]),
                            }
                        ],
                    }
                )
            else:
                self.messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": self.previous_call_id,
                        "output": f"Function call failed: {obs['last_action_error']}",
                    }
                )

        response: "Response" = self.llm(
            messages=self.messages,
            temperature=self.temperature,
        )

        action = "noop()"
        think = ""
        for output in response.output:
            if output.type == "function_call":
                arguments = json.loads(output.arguments)
                action = f"{output.name}({", ".join([f"{k}={v}" for k, v in arguments.items()])})"
                self.previous_call_id = output.call_id
                self.messages.append(output)
                break
            elif output.type == "reasoning":
                if len(output.summary) > 0:
                    think += output.summary[0].text + "\n"
                self.messages.append(output)

        return (
            action,
            bgym.AgentInfo(
                think=think,
                chat_messages=[],
                stats={},
            ),
        )


MODEL_CONFIG = OpenAIResponseModelArgs(
    model_name="o4-mini-2025-04-16",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=100_000,
    temperature=0.1,
    vision_support=True,
)

AGENT_CONFIG = ToolUseAgentArgs(
    temperature=0.1,
    model_args=MODEL_CONFIG,
)
