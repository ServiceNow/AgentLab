import json
from dataclasses import dataclass
from typing import Any, Dict, List

from agentlab.llm.llm_utils import call_openai_api_with_retries
from agentlab.llm.response_api import (
    ContentItem,
    LLMOutput,
    Message,
    MessageBuilder,
    OpenAIResponseAPIMessageBuilder,
    OpenAIResponseModel,
    OpenAIResponseModelArgs,
    ToolCall,
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

    def _call_api(self, messages: list[Any | MessageBuilder], tool_choice="auto", **kwargs) -> dict:
        input = []
        for msg in messages:
            if isinstance(msg, MessageBuilder):
                temp = msg.prepare_message()
            elif isinstance(msg, ToolCalls):
                temp = msg.raw_calls
            else:
                raise TypeError('Unsupported message type: {}'.format(type(msg)))
            input.extend(temp)

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
            if not cua_tool_present:
                api_params["tools"].extend(
                    [
                        {
                            "type": "computer_use_preview",
                            "display_width": 1024,
                            "display_height": 768,
                            "environment": "browser",  # other possible values: "mac", "windows", "ubuntu"
                        }
                    ]
                )

        response = call_openai_api_with_retries(
            self.client.responses.create,
            api_params,
        )

        return response

    def _parse_response(self, response: dict) -> dict:
        result = LLMOutput(
            raw_response=response,
            think="",
            action=None,
            tool_calls=ToolCalls(),
        )
        interesting_keys = ["output_text"]
        actions = []  # Collect all actions for multi-action support

        for output in response.output:
            if output.type in "computer_call":
                # Mapping CUA action space to bgym coord action space.
                bgym_fn, bgym_fn_args, action_str = (
                    self.cua_action_to_bgym_action(output.action)
                )
                tool_call = ToolCall(
                    name=bgym_fn,
                    arguments=bgym_fn_args,
                    raw_call=output,
                )
                result.tool_calls.add_tool_call(tool_call)
                actions.append(action_str)

            elif output.type == "function_call":
                arguments = json.loads(output.arguments)
                func_args_str = ", ".join(
                    [
                        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                        for k, v in arguments.items()
                    ]
                )
                action_str = f"{output.name}({func_args_str})"
                tool_call = ToolCall(
                    name=output.name,
                    arguments=arguments,
                    raw_call=output,
                )
                result.tool_calls.add_tool_call(tool_call)
                if tool_call.is_bgym_action():
                    actions.append(action_str)

            elif output.type == "reasoning":
                if len(output.summary) > 0:
                    result.think += output.summary[0].text + "\n"

            elif output.type == "message" and output.content:
                result.think += output.content[0].text + "\n"

        result.action = actions
        result.tool_calls.raw_calls = response.output

        for key in interesting_keys:
            if key_content := getattr(output, "output_text", None) is not None:
                result.think += f"<{key}>{key_content}</{key}>"
        return result

    @staticmethod
    def cua_action_to_bgym_action(action) -> str:
        """
        Given a computer action (e.g., click, double_click, scroll, etc.),
        convert it to a text description.
        """

        action_type = action.type

        try:
            match action_type:

                case "click":
                    x, y = action.x, action.y
                    button = action.button
                    print(f"Action: click at ({x}, {y}) with button '{button}'")
                    # Not handling things like middle click, etc.
                    if button != "left" and button != "right":
                        button = "left"
                    action_str = f"mouse_click({x}, {y}, button='{button}')"
                    (
                        bgym_fn,
                        bgym_fn_args,
                    ) = "mouse_click", {"x": x, "y": y, "button": button}

                case "scroll":
                    x, y = action.x, action.y
                    scroll_x, scroll_y = action.scroll_x, action.scroll_y
                    action_str = f"scroll_at({x}, {y}, {scroll_x},  {scroll_y})"
                    bgym_fn, bgym_fn_args = "scroll_at", {
                        "x": x,
                        "y": y,
                        "scroll_x": scroll_x,
                        "scroll_y": scroll_y,
                    }

                case "keypress":
                    keys = action.keys
                    for k in keys:
                        print(f"Action: keypress '{k}'")
                        # A simple mapping for common keys; expand as needed.
                        if k.lower() == "enter":
                            action_str = "keyboard_press('Enter')"
                        elif k.lower() == "space":
                            action_str = "keyboard_press(' ')"
                        else:
                            action_str = f"keyboard_press('{k}')"

                        bgym_fn, bgym_fn_args = "keyboard_press", {"key": k}

                case "type":
                    text = action.text
                    print(f"Action: type text: {text}")
                    action_str = f"keyboard_type('{text}')"
                    bgym_fn, bgym_fn_args = "keyboard_type", {"text": text}

                case "wait":
                    print("Action: wait")
                    action_str = "noop()"
                    bgym_fn, bgym_fn_args = "noop", {}

                case "screenshot":
                    # Not a valid bgym action
                    action_str = "noop()"
                    bgym_fn, bgym_fn_args = "noop", {}

                case "drag":
                    x1, y1 = action.path[0].x, action.path[0].y
                    x2, y2 = action.path[1].x, action.path[1].y
                    print(f"Action: drag from ({x1}, {y1}) to ({x2}, {y2})")
                    action_str = f"mouse_drag_and_drop({x1}, {y1}, {x2}, {y2})"
                    bgym_fn, bgym_fn_args = "mouse_drag_and_drop", {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }

                case _:
                    raise ValueError(f"Unrecognized action type: {action_type}")

            # Return the function name and arguments for bgym

            return bgym_fn, bgym_fn_args, action_str

        except Exception as e:
            print(f"Error handling action {action}: {e}")


class OpenaAICUAMessageBuilder(OpenAIResponseAPIMessageBuilder):

    def prepare_message(self) -> List[Message]:
        content = []
        for item in self.content:
            content.append(self.convert_content_to_expected_format(item))
        output = [{"role": self.role, "content": content}]

        if self.role != "tool":
            return output
        else:
            return self.handle_tool_call()

    def convert_content_to_expected_format(self, content: ContentItem) -> ContentItem:
        """Convert the content item to the expected format for OpenAI Responses."""
        if "text" in content:
            content_type = "input_text" if self.role != "assistant" else "output_text"
            return {"type": content_type, "text": content["text"]}
        elif "image" in content:
            return {"type": "input_image", "image_url": content["image"]}
        else:
            raise ValueError(f"Unsupported content type: {content}")

    def handle_tool_call(self):
        """Handle the tool call response from the last raw response."""
        if self.responsed_tool_calls is None:
            raise ValueError("No tool calls found in responsed_tool_calls")

        output = []
        for fn_call in self.responsed_tool_calls:
            call_type = fn_call.raw_call.type
            call_id = fn_call.raw_call.call_id
            call_response = fn_call.tool_response  # List[ContentItem]

            match call_type:
                case "function_call":
                    # image output is not supported in function calls response.
                    fn_call_response = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": [
                            self.convert_content_to_expected_format(item) for item in call_response
                        ],
                    }
                    output.append(fn_call_response)

                case "computer_call":
                    # For computer calls, use only images are expected.
                    computer_call_output = {
                        "type": "computer_call_output",
                        "call_id": call_id,
                        "output": self.convert_content_to_expected_format(call_response[0]), # list needs to be flattened
                    }
                    output.append(computer_call_output)  # this needs to be a screenshot

        return output

    def mark_all_previous_msg_for_caching(self):
        pass


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
        return OpenaAICUAMessageBuilder


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
