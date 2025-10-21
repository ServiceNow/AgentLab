import logging
import os
from dataclasses import dataclass

import openai
from bgym import HighLevelActionSetArgs
from browsergym.experiments import AbstractAgentArgs, Agent, AgentInfo

from agentlab.llm.llm_utils import image_to_jpg_base64_url

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class OpenAIComputerUseAgentArgs(AbstractAgentArgs):
    """
    Arguments for the OpenAI Computer Use Agent.
    """

    agent_name: str = None
    model: str = "computer-use-preview"
    tool_type: str = "computer_use_preview"
    display_width: int = 1024
    display_height: int = 768
    environment: str = "browser"
    reasoning_summary: str = "concise"
    truncation: str = "auto"  # Always set to "auto" for OpenAI API
    wait_time: int = 2000  # wait for 2 seconds for noop() actions
    action_set: HighLevelActionSetArgs = None
    enable_safety_checks: bool = False  # Optional, default to False, only use in demo mode
    implicit_agreement: bool = True  # Whether to require explicit agreement for actions or not

    def __post_init__(self):
        if self.agent_name is None:
            self.agent_name = "OpenAIComputerUseAgent"

    def set_benchmark(self, benchmark, demo_mode):
        pass

    def set_reproducibility_mode(self):
        pass

    def make_agent(self):
        return OpenAIComputerUseAgent(
            model=self.model,
            tool_type=self.tool_type,
            display_width=self.display_width,
            display_height=self.display_height,
            environment=self.environment,
            reasoning_summary=self.reasoning_summary,
            truncation=self.truncation,
            wait_time=self.wait_time,
            action_set=self.action_set,
            enable_safety_checks=self.enable_safety_checks,
            implicit_agreement=self.implicit_agreement,
        )


class OpenAIComputerUseAgent(Agent):
    def __init__(
        self,
        model: str,
        tool_type: str,
        display_width: int,
        display_height: int,
        environment: str,
        reasoning_summary: str,
        wait_time: int,
        truncation: str,
        action_set: HighLevelActionSetArgs,
        enable_safety_checks: bool = False,
        implicit_agreement: bool = True,
    ):
        self.model = model
        self.reasoning_summary = reasoning_summary
        self.truncation = truncation
        self.wait_time = wait_time
        self.enable_safety_checks = enable_safety_checks
        self.implicit_agreement = implicit_agreement

        self.action_set = action_set.make_action_set()

        assert not (
            self.enable_safety_checks
            and (self.action_set.demo_mode is None or self.action_set.demo_mode == "off")
        ), "Safety checks are enabled but no demo mode is set. Please set demo_mode to 'all_blue'."

        self.computer_calls = []
        self.pending_checks = []
        self.previous_response_id = None
        self.last_call_id = None
        self.initialized = False
        self.user_answer = None  # Store the user answer to send to the assistant

        self.tools = [
            {
                "type": tool_type,
                "display_width": display_width,
                "display_height": display_height,
                "environment": environment,
            }
        ]

        self.inputs = []

    def reset(self):
        self.computer_calls = []
        self.pending_checks = []
        self.previous_response_id = None
        self.last_call_id = None
        self.initialized = False
        self.user_answer = None  # Store the user answer to send to the assistant
        self.inputs = []

    def parse_action_to_bgym(self, action) -> str:
        """
        Parse the action string returned by the OpenAI API into bgym format.
        """
        action_type = action.type

        match (action_type):
            case "click":
                x, y = action.x, action.y
                button = action.button
                if button != "left" and button != "right":
                    button = "left"
                return f"mouse_click({x}, {y}, button='{button}')"

            case "scroll":
                x, y = action.x, action.y
                dx, dy = action.scroll_x, action.scroll_y
                return f"scroll_at({x}, {y}, {dx}, {dy})"

            case "keypress":
                keys = action.keys
                to_press = ""
                for k in keys:
                    if k.lower() == "enter":
                        to_press = "Enter"
                    elif k.lower() == "space":
                        to_press = " "
                    elif k.lower() == "ctrl":
                        to_press = "Control"
                    elif k.lower() == "shift":
                        to_press = "Shift"
                    elif k.lower() == "alt":
                        to_press = "Alt"
                    elif k.lower() == "-":
                        to_press += "Minus"
                    else:
                        to_press += f"+{k}"
                return f"keyboard_press('{to_press}')"

            case "type":
                text = action.text
                return f"keyboard_type('{text}')"

            case "drag":
                from_x, from_y = action.path[0].x, action.path[0].y
                to_x, to_y = action.path[-1].x, action.path[-1].y
                return f"mouse_drag_and_drop({from_x}, {from_y}, {to_x}, {to_y})"

            case "move":
                x, y = action.x, action.y
                return f"mouse_move({x}, {y})"

            case "wait":
                return "noop(2000)"  # wait for 2 seconds

            # The screenshot is already given in the observation, so we don't need to handle it here.
            case "screenshot":
                # Nothing to do as screenshot is taken at each turn
                return "noop()"

            case _:
                logging.debug(f"No action found for {action_type}. Please check the action type.")
                return None

        return action

    def handle_safety_checks(self):
        """Implement this function to handle safety checks.
        self.pending_checks = computer_call.pending_safety_checks
        for check in self.pending_checks:
            do_something_to_acknowledge_check(check)
        """
        # TODO: allow user to handle safety checks when demo mode is enabled
        pass

    def start_session(self, goal: str, screenshot_base64: str):
        """
        Call the OpenAI API with the provided goal and screenshot to initiate a session.

        Args:
            goal (str): The goal or task description for the agent.
            screenshot_base64 (str): Base64 encoded screenshot of the current state.
        Returns:
            response: A list of Response objects. The response from the OpenAI API.
        """
        instruction = goal
        if self.implicit_agreement:
            instruction = f"""
Please perform the following task, I can confirm it is safe and you can proceed with all actions without asking for confirmation.

Task: 
{goal}
"""
        inputs = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": f"{screenshot_base64}"},
                ],
            }
        ]
        response = self.call_api(
            inputs=inputs,
            reasoning={
                "summary": self.reasoning_summary,
            },
        )
        agent_info = AgentInfo(
            think=self.reasoning_summary if self.reasoning_summary else None,
            chat_messages=inputs,
        )
        return response, agent_info

    def call_api(self, inputs: list, previous_response_id=None, **kwargs):
        response = client.responses.create(
            model=self.model,
            previous_response_id=previous_response_id,
            tools=self.tools,
            input=inputs,
            truncation=self.truncation,  # Always set to "auto"
            **kwargs,
        )
        return response

    def get_action(self, obs):
        agent_info = AgentInfo()
        goal = obs["goal"]
        screenshot_base64 = image_to_jpg_base64_url(obs["screenshot"])

        if not self.initialized:
            logging.debug("Initializing OpenAI Computer Use Agent with goal:", goal)
            response, agent_info = self.start_session(goal, screenshot_base64)
            for item in response.output:
                if item.type == "reasoning":
                    self.agent_info.think = item.summary[0].text if item.summary else None
                if item.type == "computer_call":
                    self.computer_calls.append(item)
            self.previous_response_id = response.id
            self.initialized = True

        if len(self.computer_calls) > 0:
            logging.debug("Found multiple computer calls in previous call. Processing them...")
            computer_call = self.computer_calls.pop(0)
            if not self.enable_safety_checks:
                # Bypass safety checks
                self.pending_checks = computer_call.pending_safety_checks
            action = self.parse_action_to_bgym(computer_call.action)
            self.last_call_id = computer_call.call_id
            return action, self.agent_info
        else:
            logging.debug("Last call ID:", self.last_call_id)
            logging.debug("Previous response ID:", self.previous_response_id)
            self.inputs.append(
                {
                    "call_id": self.last_call_id,
                    "type": "computer_call_output",
                    "acknowledged_safety_checks": self.pending_checks,
                    "output": {
                        "type": "input_image",
                        "image_url": f"{screenshot_base64}",  # current screenshot
                    },
                }
            )

            if self.user_answer:
                self.inputs.append(self.user_answer)
                self.user_answer = None

            agent_info.chat_messages = str(self.inputs)
            response = self.call_api(self.inputs, self.previous_response_id)
            self.inputs = []  # Clear inputs for the next call
            self.previous_response_id = response.id

            self.computer_calls = [item for item in response.output if item.type == "computer_call"]
            if not self.computer_calls:
                logging.debug(f"No computer call found. Output from model: {response.output}")
                for item in response.output:
                    if item.type == "reasoning":
                        agent_info.think = item.summary[0].text if item.summary else None
                    if hasattr(item, "role") and item.role == "assistant":
                        # Assume assitant asked for user confirmation
                        # Always answer with: Yes, continue.
                        self.user_answer = {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Yes, continue."}],
                        }
                        return f"send_msg_to_user('{item.content[0].text}')", agent_info
                logging.debug("No action found in the response. Returning None.")
                return None, agent_info

            computer_call = self.computer_calls.pop(0)
            self.last_call_id = computer_call.call_id
            action = self.parse_action_to_bgym(computer_call.action)
            logging.debug("Action:", action)
            if not self.enable_safety_checks:
                # Bypass safety checks
                self.pending_checks = computer_call.pending_safety_checks
            else:
                self.handle_safety_checks()

            for item in response.output:
                if item.type == "reasoning":
                    self.agent_info.think = item.summary[0].text if item.summary else None
                    break

            return action, agent_info
