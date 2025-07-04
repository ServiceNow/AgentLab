import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bgym import AbstractActionSet
from dataclasses_json import DataClassJsonMixin
from desktop_env.actions import ACTION_SPACE
from desktop_env.desktop_env import DesktopEnv

from agentlab.benchmarks.abstract_env import (
    AbstractBenchmark,
    AbstractEnv,
    AbstractEnvArgs,
    add_step_timing_to_env_info_decorator,
)

logger = logging.getLogger(__name__)


class OsworldGym(AbstractEnv):
    def __init__(
        self,
        task: dict,
        provider_name: str,
        region: str | None,
        path_to_vm: str | None,
        snapshot_name: str,
        action_space: str,
        cache_dir: str,
        screen_size: tuple[int, int],
        headless: bool,
        require_a11y_tree: bool,
        require_terminal: bool,
        os_type: str,
        enable_proxy: bool,
        max_steps: int = 50,
    ):
        self.task = task
        self.env_info = {
            "provider_name": provider_name,
            "region": region,
            "path_to_vm": path_to_vm,
            "snapshot_name": snapshot_name,
            "action_space": action_space,
            "cache_dir": cache_dir,
            "screen_size": screen_size,
            "headless": headless,
            "require_a11y_tree": require_a11y_tree,
            "require_terminal": require_terminal,
            "os_type": os_type,
            "enable_proxy": enable_proxy,
        }
        self.env = DesktopEnv(
            action_space=action_space,
            provider_name=provider_name,
            region=region,  # type: ignore
            path_to_vm=path_to_vm,  # type: ignore
            snapshot_name=snapshot_name,
            cache_dir=cache_dir,
            screen_size=screen_size,  # type: ignore
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            os_type=os_type,
        )
        self._step_count = 0
        self.max_steps = max_steps

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_obs = self.env.reset(task_config=self.task, seed=seed)
        obs = self.env_to_agentlab_observation(raw_obs)
        self._step_count = 0
        return obs, self.env_info

    @add_step_timing_to_env_info_decorator
    def step(self, action: str):
        """Execute the action in the OS-world environment."""
        env_action = self.agentlab_to_env_action(action)
        logger.info(f"AgentLab Action returned: {action}, converted to: {env_action}")
        raw_obs, reward, done, info = self.env.step(env_action)
        self._step_count += 1
        truncated = info.get('fail', False) or self._step_count >= self.max_steps
        if done or truncated:
            try:
                reward = self.env.evaluate()
            except Exception as e:
                logger.warning(f"Failed to evaluate {self.task} task: {e}")
        obs = self.env_to_agentlab_observation(raw_obs)
        return obs, reward, done, truncated, info

    def agentlab_to_env_action(self, action: str) -> str:
        """Convert AgentLab agents action format to OSWorld action format."""
        if self.env.action_space == "computer_13":
            # expects dictionary with 'keys' action_type and parameters
            return self.convert_agentlab_action_to_computer_13(action)
        elif self.env.action_space == "pyautogui":
            pattern = r"pyautogui_action\(action=['\"](.*)['\"]\)"
            match = re.search(pattern, action)
            if match:
                return match.group(1)
            return action

    def env_to_agentlab_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Convert OSWorld observation to AgentLab format."""
        converted_obs = {}

        # Core visual and interaction components
        self._add_screenshot(converted_obs, obs)
        # TODO: Check if the unprocessesed ax_tree is a suitable representation for agentlab agents or use the utility functions from os-world agents to convert them.
        # TODO: Check if there is something equivalent to bid in OSWorld Axtree. and how it is used in the action space. This can be used with GenericAgent.
        converted_obs["axtree_object"] = obs["accessibility_tree"]
        converted_obs["last_action_error"] = ""  # OSWorld doesn't provide this directly
        converted_obs["focused_element_bid"] = ""  # Extract from accessibility tree if available
        # Browser-like context (adapted for desktop environment)
        converted_obs = self._add_browser_context(converted_obs)
        # Task and instruction context
        converted_obs = self._add_task_context(converted_obs, obs)

        return converted_obs

    def _add_screenshot(self, converted_obs: dict[str, Any], obs: dict[str, Any]) -> None:
        """Convert screenshot to numpy array format expected by AgentLab"""
        if "screenshot" not in obs:
            return

        screenshot = obs["screenshot"]

        try:
            from io import BytesIO

            import numpy as np
            from PIL import Image

            if isinstance(screenshot, bytes):
                image = Image.open(BytesIO(screenshot))
            elif hasattr(screenshot, "convert"):  # PIL Image
                image = screenshot
            elif hasattr(screenshot, "__array__"):  # numpy array
                converted_obs["screenshot"] = np.array(screenshot)
                return
            else:
                raise ValueError(f"Unexpected screenshot type: {type(screenshot)}")

            # Convert PIL image to RGB numpy array
            if image.mode != "RGB":
                image = image.convert("RGB")
            converted_obs["screenshot"] = np.array(image)

        except Exception as e:
            logger.warning(f"Failed to process screenshot: {e}")
            converted_obs["screenshot"] = None

    def _add_browser_context(self, converted_obs: dict[str, Any]):
        """Add browser-like context fields adapted for desktop environment."""
        converted_obs["url"] = ""
        converted_obs["open_pages_urls"] = []
        converted_obs["open_pages_titles"] = []
        converted_obs["active_page_index"] = 0
        return converted_obs

    def _add_task_context(self, converted_obs: dict[str, Any], obs: dict[str, Any]):
        """Add task and instruction context fields."""
        instruction = obs.get("instruction", "")
        converted_obs["goal_object"] = [{"type": "text", "text": instruction}]
        # Terminal output (preserve if available)
        if obs.get("terminal"):
            converted_obs["terminal_output"] = obs["terminal"]
        return converted_obs

    def convert_agentlab_action_to_computer_13(self, action: str) -> dict[str, Any]:
        """Convert action string to dictionary format"""
        import ast

        pattern = r"computer_\d+_action\(action_type=\"(\w+)\",\s*parameters=({.*?})\)"
        match = re.match(pattern, action)

        if match:
            action_type = match.group(1)
            params_str = match.group(2)
            # Safely evaluate the parameters dictionary
            try:
                parameters = ast.literal_eval(params_str)
            except (ValueError, SyntaxError):
                # Handle malformed parameter strings
                parameters = {}

            return {"action_type": action_type, "parameters": parameters}
        else:
            raise ValueError("Invalid action string format")

    def close(self):
        return self.env.close()


class OSWorldActionSet(AbstractActionSet):
    # TODO: Define and use agentlab AbstractActionSet
    # TODO: AbstractActionSet should define some standard format to represent actions.(list of dict with keys that are MCP compatible)
    # (list of callables? that have extensive docstring with examples. We can then use inspect module to extract relevant info)
    # TODO: Should we have 'abstract function' here for action conversion for backend LLM with fixed action set like UI-Tars or Semi-fixed action set LLMs like OpenAI CUA?
    # TODO: We need to support both 'action space as tools' and 'action space as prompt' for agentlab agents and have conversion functions to convert them to format acceptable by environment.
    def __init__(self, action_space: Literal["computer_13", "pyautogui"]):
        self.action_space = action_space

    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """Describe the OSWorld action set for desktop interactions."""
        description = "OSWorld Desktop Action Set\n\n"

        if with_long_description:
            description += f"Action space: {self.action_space}\n\n"
            description += (
                "This action set provides comprehensive desktop interaction capabilities including:\n"
                "- Mouse operations: click, double-click, right-click, move, drag\n"
                "- Keyboard operations: type text, press keys, key combinations\n"
                "- Scrolling operations\n"
                "- Task control: wait, done, fail\n\n"
            )

            if self.action_space == "computer_13":
                description += "Actions are provided as structured dictionaries with action_type and parameters.\n"
                description += str(ACTION_SPACE)
            else:
                description += "Actions are provided as executable Python code using pyautogui.\n"

        if with_examples:
            description += "\nAvailable actions:\n"
            actions = [
                "MOVE_TO - Move cursor to position",
                "CLICK - Click at position or current cursor location",
                "RIGHT_CLICK - Right click at position",
                "DOUBLE_CLICK - Double click at position",
                "DRAG_TO - Drag to position with left button",
                "MOUSE_DOWN/MOUSE_UP - Press/release mouse button",
                "SCROLL - Scroll horizontally/vertically",
                "TYPING - Type text string",
                "PRESS - Press and release a key",
                "KEY_DOWN/KEY_UP - Press/release a key",
                "HOTKEY - Press key combination",
                "WAIT - Wait for action to complete",
                "DONE - Mark task as completed",
                "FAIL - Mark task as failed",
            ]
            description += "\n".join(f"- {action}" for action in actions)

        return description

    def example_action(self, abstract: bool) -> str:
        """Provide example actions for the action set."""
        if self.action_space == "computer_13":
            if abstract:
                return '{"action_type": "CLICK", "parameters": {"x": 100, "y": 200}}'
            else:
                examples = [
                    '{"action_type": "CLICK", "parameters": {"x": 500, "y": 300, "button": "left"}}',
                    '{"action_type": "TYPING", "parameters": {"text": "Hello World"}}',
                    '{"action_type": "PRESS", "parameters": {"key": "enter"}}',
                    '{"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "c"]}}',
                    '{"action_type": "SCROLL", "parameters": {"dx": 0, "dy": -3}}',
                ]
                return "\n".join(examples)
        else:  # pyautogui
            if abstract:
                return "pyautogui.click(x=100, y=200)"
            else:
                examples = [
                    "pyautogui.click(x=500, y=300, button='left')",
                    "pyautogui.typewrite('Hello World')",
                    "pyautogui.press('enter')",
                    "pyautogui.hotkey('ctrl', 'c')",
                    "pyautogui.vscroll(-3)",
                ]
                return "\n".join(examples)

    def to_python_code(self, action) -> str:
        """We use the OS-world/desktop_env environment controller"""
        pass

    def to_tool_descriptor(self):
        """Convert the action set to a tool descriptor for LLMs."""
        return ACTION_SPACE

    def to_tool_description(self, api="openai"):
        """Convert the action set to a tool description for Tool-Use LLMs."""
        if self.action_space == "computer_13":
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "computer_13_action",
                        "description": self.describe(
                            with_long_description=True, with_examples=True
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action_type": {
                                    "type": "string",
                                    "enum": [
                                        "CLICK",
                                        "DOUBLE_CLICK",
                                        "RIGHT_CLICK",
                                        "MOVE_TO",
                                        "DRAG_TO",
                                        "MOUSE_DOWN",
                                        "MOUSE_UP",
                                        "SCROLL",
                                        "TYPING",
                                        "PRESS",
                                        "KEY_DOWN",
                                        "KEY_UP",
                                        "HOTKEY",
                                        "WAIT",
                                        "DONE",
                                        "FAIL",
                                    ],
                                },
                                "parameters": {"type": "object"},
                            },
                            "required": ["action_type"],
                        },
                    },
                }
            ]
        else:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "pyautogui_action",
                        "description": self.describe(
                            with_long_description=True, with_examples=True
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                # Represent any action that pyautogui can perform in string format.
                                "action": {
                                    "type": "string",
                                    "description": "A pyautogui action in string format, e.g., 'pyautogui.click(x=100, y=200)'",
                                }
                            },
                        },
                    },
                }
            ]
        if api == "anthropic":
            return format_tools_from_openai_to_anthropic(tools)
        else:
            return tools


def format_tools_from_openai_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tool format to Anthropic tool format."""
    formatted_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            raise ValueError(f"Unsupported tool type: {tool.get('type')}")

        function_def = tool["function"]
        formatted_tool = {
            "name": function_def["name"],
            "description": function_def["description"],
            "input_schema": function_def["parameters"],
        }
        formatted_tools.append(formatted_tool)

    return formatted_tools


@dataclass
class OSWorldActionSetArgs(DataClassJsonMixin):
    action_space: Literal["computer_13", "pyautogui"] = "computer_13"

    def make_action_set(self):
        logger.info(f"Creating OSWorld Action Set with action space: {self.action_space}")
        return OSWorldActionSet(action_space=self.action_space)


@dataclass
class OsworldEnvArgs(AbstractEnvArgs):
    task: dict[str, Any]
    task_seed: int = 0
    task_name: str | None = None
    path_to_vm: str | None = None  # path to .vmx file
    provider_name: str = "docker"
    region: str = "us-east-1"  # AWS specific, does not apply to all providers
    snapshot_name: str = "init_state"  # snapshot name to revert to
    action_space: Literal["computer_13", "pyautogui"] = "computer_13"
    cache_dir: str = "cache"
    screen_size: tuple[int, int] = (1920, 1080)
    headless: bool = False
    require_a11y_tree: bool = True
    require_terminal: bool = False
    os_type: str = "Ubuntu"
    enable_proxy: bool = False
    max_steps: int = 100  

    def make_env(
        self, exp_dir: Path, action_mapping=None, use_raw_page_output: bool = False
    ) -> OsworldGym:
        logger.info(f"Creating OSWorld Gym with task: {self.task}")
        gym = OsworldGym(
            task=self.task,
            provider_name=self.provider_name,
            region=self.region,
            path_to_vm=self.path_to_vm,
            snapshot_name=self.snapshot_name,
            action_space=self.action_space,
            cache_dir=self.cache_dir,
            screen_size=self.screen_size,
            headless=self.headless,
            require_a11y_tree=self.require_a11y_tree,
            require_terminal=self.require_terminal,
            os_type=self.os_type,
            enable_proxy=self.enable_proxy,
            max_steps=self.max_steps,
        )
        return gym


class OsworldBenchmark(AbstractBenchmark):
    name: str = "osworld"
    is_multi_tab: bool = False
    high_level_action_set_args: OSWorldActionSetArgs = None  # type: ignore
    test_set_path: str = "OSWorld/evaluation_examples"
    test_set_name: str = "test_all.json"
    domain: str = "all"
    env_args: OsworldEnvArgs = None  # type: ignore # basic env configuration for all tasks
    env_args_list: list[OsworldEnvArgs] = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        if not self.env_args:
            self.env_args = OsworldEnvArgs(task={})
        self.high_level_action_set_args = OSWorldActionSetArgs(
            action_space=self.env_args.action_space
        )
        with open(os.path.join(self.test_set_path, self.test_set_name)) as f:
            tasks = json.load(f)
        if self.domain != "all":
            tasks = {self.domain: tasks[self.domain]}

        for domain in tasks:
            for task_id in tasks[domain]:
                task_file = os.path.join(self.test_set_path, f"examples/{domain}/{task_id}.json")
                with open(task_file) as f:
                    task = json.load(f)
                name = f"{self.name}.{task['id']}"
                task_env_args = deepcopy(self.env_args)
                task_env_args.task = task
                task_env_args.task_name = name
                self.env_args_list.append(task_env_args)
        logger.info(f"Loaded {len(self.env_args_list)} tasks from domain '{self.domain}'")
