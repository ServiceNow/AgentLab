import json
from dataclasses import dataclass
from typing import Any, Dict, List

from agentlab.llm.llm_utils import call_openai_api_with_retries
from agentlab.llm.response_api import (
    MessageBuilder,
    OpenAIResponseAPIMessageBuilder,
    OpenAIResponseModel,
    OpenAIResponseModelArgs,
    ToolCalls,
)

from .tool_use_agent import (
    GeneralHints,
    Goal,
    Obs,
    PromptConfig,
    Summarizer,
    TaskHint,
    ToolUseAgentArgs,
)


class OpenAICUAModel(OpenAIResponseModel):

    def _call_api(self, messages: list[ToolCalls | MessageBuilder], tool_choice="auto", **kwargs) -> dict:
        input = self.convert_messages_to_api_format(messages)

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "truncation": "auto",  # truncation is required for OpenAI CUA
            "tool_choice": "auto",  # Tool choice can only be auto
            **self.extra_kwargs,  
        }

        if "tools" in api_params:
            cua_tool_present = any(
                tool.get("type") == "computer_use_preview" for tool in api_params["tools"]
            )
            # CUA requires this tool 
            if not cua_tool_present:
                api_params["tools"].extend(
                    [
                        {
                            "type": "computer_use_preview",
                            "display_width": 1024,   
                            "display_height": 768,
                            "environment": "browser",  # TODO: Parametrize this 
                        }
                    ]
                )

        response = call_openai_api_with_retries(
            self.client.responses.create,
            api_params,
        )

        return response

    def cua_action_to_env_tool_name_and_args(self, action) -> str:
        """
        Given a computer action (e.g., click, double_click, scroll, etc.),
        convert it to a text description.
        """
        #TODO: #Provide an alternate implementation for OS-World.

        action_type = action.type

        try:
            action_mapping = {
                "click": lambda: self._handle_click_action(action),
                "scroll": lambda: self._handle_scroll_action(action),
                "keypress": lambda: self._handle_keypress_action(action),
                "type": lambda: self._handle_type_action(action),
                "wait": lambda: self._handle_wait_action(action),
                "screenshot": lambda: self._handle_screenshot_action(action),
                "drag": lambda: self._handle_drag_action(action),
            }

            if action_type in action_mapping:
                return action_mapping[action_type]()
            else:
                raise ValueError(f"Unrecognized openAI CUA action type: {action_type}")

        except Exception as e:
            print(f"Error handling action {action}: {e}")

    def _handle_click_action(self, action):
        x, y = action.x, action.y
        button = action.button
        if button != "left" and button != "right":
            button = "left"
        return "mouse_click", {"x": x, "y": y, "button": button}

    def _handle_scroll_action(self, action):
        x, y = action.x, action.y
        scroll_x, scroll_y = action.scroll_x, action.scroll_y
        return "scroll_at", {"x": x, "y": y, "scroll_x": scroll_x, "scroll_y": scroll_y}

    def _handle_keypress_action(self, action):
        keys = action.keys
        #TODO: Check this if is suitable for BGYM env.
        for k in keys:
            print(f"Action: keypress '{k}'")
            if k.lower() == "enter":
                key = "Enter"
            elif k.lower() == "space":
                key = " "
            return "keyboard_press", {"key": key}

    def _handle_type_action(self, action):
        text = action.text
        return "keyboard_type", {"text": text}

    def _handle_wait_action(self, action):
        return "noop", {}

    def _handle_screenshot_action(self, action):
        return "noop", {}

    def _handle_drag_action(self, action):
        x1, y1 = action.path[0].x, action.path[0].y
        x2, y2 = action.path[1].x, action.path[1].y
        print(f"Action: drag from ({x1}, {y1}) to ({x2}, {y2})")
        return "mouse_drag_and_drop", {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

@dataclass
class OpenAICUAModelArgs(OpenAIResponseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "openai"

    def make_model(self, extra_kwargs=None, **kwargs):
        return OpenAICUAModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
            pricing_api="openai",
            **kwargs,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIResponseAPIMessageBuilder


# Default configuration for Computer Use Agent
DEFAULT_CUA_PROMPT_CONFIG = PromptConfig(
    tag_screenshot=True,
    goal=Goal(goal_as_system_msg=True),
    obs=Obs(
        use_last_error=True,
        use_screenshot=True,
        use_axtree=True,
        use_dom=False,
        use_som=False,
        use_tabs=False,
        openai_cua_mode=True,  # Enable CUA mode for OpenAI
    ),
    summarizer=Summarizer(do_summary=True),
    general_hints=GeneralHints(use_hints=False),
    task_hint=TaskHint(use_task_hint=False),
    keep_last_n_obs=1,  #NOTE: API error if more than 1 obs is used. There can be only one computer call output in the response.
    multiaction=True,  # whether to use multi-action or not
    # action_subsets=("bid",),
    action_subsets=("coord"),
)

OAI_CUA_TOOL_AGENT = ToolUseAgentArgs(
    model_args=OpenAICUAModelArgs(model_name="computer-use-preview"),
    config=DEFAULT_CUA_PROMPT_CONFIG,
)
