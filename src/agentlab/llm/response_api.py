import logging
from dataclasses import dataclass

import openai
from openai import OpenAI

from .base_api import AbstractChatModel, BaseModelArgs


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
                # temperature=temperature,
                # previous_response_id=content.get("previous_response_id", None),
                max_output_tokens=self.max_tokens,
                **self.extra_kwargs,
                tool_choice="required",
                reasoning={
                    "effort": "low",
                    "summary": "detailed",
                },
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
