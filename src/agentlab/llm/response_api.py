import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import openai
from openai import OpenAI

from .base_api import AbstractChatModel, BaseModelArgs

type ContentItem = Dict[str, Any]
type Message = Dict[str, Union[str, List[ContentItem]]]


class MessageBuilder:
    def __init__(self, role: str):
        self.role = role
        self.content: List[ContentItem] = []
        self.tool_call_id = None

    @staticmethod
    def system() -> "MessageBuilder":
        return MessageBuilder(role="system")

    @staticmethod
    def user() -> "MessageBuilder":
        return MessageBuilder(role="user")

    @staticmethod
    def assistant() -> "MessageBuilder":
        return MessageBuilder(role="assistant")

    @staticmethod
    def tool() -> "MessageBuilder":
        return MessageBuilder(role="tool")

    def add_text(self, text: str) -> "MessageBuilder":
        self.content.append({"text": text})
        return self

    def add_image(self, image: str) -> "MessageBuilder":
        self.content.append({"image": image})
        return self

    def add_tool_id(self, tool_id: str) -> "MessageBuilder":
        self.tool_call_id = tool_id
        return self

    def to_openai(self) -> List[Message]:
        content = []
        for item in self.content:
            if "text" in item:
                content.append({"type": "input_text", "text": item["text"]})
            elif "image" in item:
                content.append({"type": "input_image", "image_url": item["image"]})
        res = [{"role": self.role, "content": content}]

        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            # tool messages can only take text with openai
            # we need to split the first content element if it's text and use it
            # then open a new (user) message with the rest
            res[0]["tool_call_id"] = self.tool_call_id
            text_content = (
                content.pop(0)["text"]
                if "text" in content[0]
                else "Tool call answer in next message"
            )
            res[0]["content"] = text_content
            res.append({"role": "user", "content": content})

        return res

    def to_anthropic(self) -> List[Message]:
        content = []

        if self.role == "system":
            logging.warning(
                "In the Anthropic API, system messages should be passed as a direct input to the client."
            )
            return []

        for item in self.content:
            if "text" in item:
                content.append({"type": "text", "text": item["text"]})
            elif "image" in item:
                img_str: str = item["image"]
                # make sure to get rid of the image type for anthropic
                # e.g. "data:image/png;base64"
                if img_str.startswith("data:image/png;base64,"):
                    img_str = img_str[len("data:image/png;base64,") :]
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",  # currently only base64 is supported
                            "media_type": "image/png",  # currently only png is supported
                            "data": img_str,
                        },
                    }
                )
        res = [{"role": self.role, "content": content}]

        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            res[0]["role"] = "user"
            res[0]["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": self.tool_call_id,
                    "content": res[0]["content"],
                }
            ]
        return res

    def to_markdown(self) -> str:
        content = []
        for item in self.content:
            if "text" in item:
                content.append(item["text"])
            elif "image" in item:
                content.append(f"![image]({item['image']})")
        res = f"{self.role}: " + "\n".join(content)
        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            res += f"\nTool call ID: {self.tool_call_id}"
        return res


class ResponseModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        extra_kwargs=None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = extra_kwargs or {}
        self.client = OpenAI(api_key=api_key)

    def __call__(self, content: dict, temperature: float = None) -> dict:
        temperature = temperature if temperature is not None else self.temperature
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=content,
                temperature=temperature,
                # previous_response_id=content.get("previous_response_id", None),
                max_output_tokens=self.max_tokens,
                **self.extra_kwargs,
                tool_choice="required",
                # reasoning={
                #     "effort": "low",
                #     "summary": "detailed",
                # },
            )
            return response
        except openai.OpenAIError as e:
            logging.error(f"Failed to get a response from the API: {e}")
            raise e


class OpenAIResponseModel(ResponseModel):
    def __init__(
        self, model_name, api_key=None, temperature=0.5, max_tokens=100, extra_kwargs=None
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
        )

    def __call__(self, messages: list[dict], temperature: float = None) -> dict:
        return super().__call__(messages, temperature)
        # outputs = response.output
        # last_computer_call_id = None
        # answer_type = "call"
        # reasoning = "No reasoning"
        # for output in outputs:
        #     if output.type == "reasoning":
        #         reasoning = output.summary[0].text
        #     elif output.type == "computer_call":
        #         action = output.action
        #         last_computer_call_id = output.call_id
        #         res = response_to_text(action)
        #     elif output.type == "message":
        #         res = "noop()"
        #         answer_type = "message"
        #     else:
        #         logging.warning(f"Unrecognized output type: {output.type}")
        #         continue
        # return {
        #     "think": reasoning,
        #     "action": res,
        #     "last_computer_call_id": last_computer_call_id,
        #     "last_response_id": response.id,
        #     "outputs": outputs,
        #     "answer_type": answer_type,
        # }


def response_to_text(action):
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
                return f"mouse_click({x}, {y}, button='{button}')"

            case "scroll":
                x, y = action.x, action.y
                scroll_x, scroll_y = action.scroll_x, action.scroll_y
                print(
                    f"Action: scroll at ({x}, {y}) with offsets (scroll_x={scroll_x}, scroll_y={scroll_y})"
                )
                return f"mouse_move({x}, {y})\nscroll({scroll_x}, {scroll_y})"

            case "keypress":
                keys = action.keys
                for k in keys:
                    print(f"Action: keypress '{k}'")
                    # A simple mapping for common keys; expand as needed.
                    if k.lower() == "enter":
                        return "keyboard_press('Enter')"
                    elif k.lower() == "space":
                        return "keyboard_press(' ')"
                    else:
                        return f"keyboard_press('{k}')"

            case "type":
                text = action.text
                print(f"Action: type text: {text}")
                return f"keyboard_type('{text}')"

            case "wait":
                print(f"Action: wait")
                return "noop()"

            case "screenshot":
                # Nothing to do as screenshot is taken at each turn
                print(f"Action: screenshot")

            # Handle other actions here

            case "drag":
                x1, y1 = action.path[0].x, action.path[0].y
                x2, y2 = action.path[1].x, action.path[1].y
                print(f"Action: drag from ({x1}, {y1}) to ({x2}, {y2})")
                return f"mouse_drag_and_drop({x1}, {y1}, {x2}, {y2})"

            case _:
                print(f"Unrecognized action: {action}")

    except Exception as e:
        print(f"Error handling action {action}: {e}")


@dataclass
class OpenAIResponseModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self, extra_kwargs=None):
        return OpenAIResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
        )


import anthropic


class ClaudeResponseModel(ResponseModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        extra_kwargs=None,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
        )
        self.client = anthropic.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = extra_kwargs or {}
        self.model_name = model_name
        self.api_key = api_key

    def __call__(self, messages: list[dict], temperature: float = None) -> dict:
        temperature = temperature if temperature is not None else self.temperature
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                **self.extra_kwargs,
            )
            return response
        except Exception as e:
            logging.error(f"Failed to get a response from the API: {e}")
            raise e


@dataclass
class ClaudeResponseModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self, extra_kwargs=None):
        return ClaudeResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
        )
