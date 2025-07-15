import ast
import importlib.util
import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from bgym import AbstractActionSet
from dataclasses_json import DataClassJsonMixin
from PIL import Image

from agentlab.benchmarks.abstract_env import (
    AbstractBenchmark,
    AbstractEnv,
    AbstractEnvArgs,
    add_step_timing_to_env_info_decorator,
)
from agentlab.benchmarks.osworld_axtree_preprocessing import (
    linearize_accessibility_tree,
    tag_screenshot,
)

spec = importlib.util.find_spec("desktop_env")
if spec is not None:  # desktop_env is available
    from desktop_env.actions import KEYBOARD_KEYS, X_MAX, Y_MAX
    from desktop_env.desktop_env import DesktopEnv
else:
    # If desktop_env is not available, set to None or default values
    DesktopEnv = None
    KEYBOARD_KEYS = [
        "\t",
        "\n",
        "\r",
        " ",
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "{",
        "|",
        "}",
        "~",
        "accept",
        "add",
        "alt",
        "altleft",
        "altright",
        "apps",
        "backspace",
        "browserback",
        "browserfavorites",
        "browserforward",
        "browserhome",
        "browserrefresh",
        "browsersearch",
        "browserstop",
        "capslock",
        "clear",
        "convert",
        "ctrl",
        "ctrlleft",
        "ctrlright",
        "decimal",
        "del",
        "delete",
        "divide",
        "down",
        "end",
        "enter",
        "esc",
        "escape",
        "execute",
        "f1",
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f2",
        "f20",
        "f21",
        "f22",
        "f23",
        "f24",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "final",
        "fn",
        "hanguel",
        "hangul",
        "hanja",
        "help",
        "home",
        "insert",
        "junja",
        "kana",
        "kanji",
        "launchapp1",
        "launchapp2",
        "launchmail",
        "launchmediaselect",
        "left",
        "modechange",
        "multiply",
        "nexttrack",
        "nonconvert",
        "num0",
        "num1",
        "num2",
        "num3",
        "num4",
        "num5",
        "num6",
        "num7",
        "num8",
        "num9",
        "numlock",
        "pagedown",
        "pageup",
        "pause",
        "pgdn",
        "pgup",
        "playpause",
        "prevtrack",
        "print",
        "printscreen",
        "prntscrn",
        "prtsc",
        "prtscr",
        "return",
        "right",
        "scrolllock",
        "select",
        "separator",
        "shift",
        "shiftleft",
        "shiftright",
        "sleep",
        "stop",
        "subtract",
        "tab",
        "up",
        "volumedown",
        "volumemute",
        "volumeup",
        "win",
        "winleft",
        "winright",
        "yen",
        "command",
        "option",
        "optionleft",
        "optionright",
    ]
    X_MAX = 1920
    Y_MAX = 1080

logger = logging.getLogger(__name__)
COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move the cursor to the specified position",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "X coordinate",
                        "minimum": 0,
                        "maximum": X_MAX,
                    },
                    "y": {
                        "type": "number",
                        "description": "Y coordinate",
                        "minimum": 0,
                        "maximum": Y_MAX,
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click the left button if the button not specified, otherwise click the specified button; click at the current position if x and y are not specified, otherwise click at the specified position",
            "parameters": {
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to click",
                    },
                    "x": {
                        "type": "number",
                        "description": "X coordinate",
                        "minimum": 0,
                        "maximum": X_MAX,
                    },
                    "y": {
                        "type": "number",
                        "description": "Y coordinate",
                        "minimum": 0,
                        "maximum": Y_MAX,
                    },
                    "num_clicks": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "Number of clicks",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_down",
            "description": "Press the left button if the button not specified, otherwise press the specified button",
            "parameters": {
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to press",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_up",
            "description": "Release the left button if the button not specified, otherwise release the specified button",
            "parameters": {
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to release",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Right click at the current position if x and y are not specified, otherwise right click at the specified position",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "X coordinate",
                        "minimum": 0,
                        "maximum": X_MAX,
                    },
                    "y": {
                        "type": "number",
                        "description": "Y coordinate",
                        "minimum": 0,
                        "maximum": Y_MAX,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double click at the current position if x and y are not specified, otherwise double click at the specified position",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "X coordinate",
                        "minimum": 0,
                        "maximum": X_MAX,
                    },
                    "y": {
                        "type": "number",
                        "description": "Y coordinate",
                        "minimum": 0,
                        "maximum": Y_MAX,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag_to",
            "description": "Drag the cursor to the specified position with the left button pressed",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "X coordinate",
                        "minimum": 0,
                        "maximum": X_MAX,
                    },
                    "y": {
                        "type": "number",
                        "description": "Y coordinate",
                        "minimum": 0,
                        "maximum": Y_MAX,
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the mouse wheel up or down",
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {"type": "integer", "description": "Horizontal scroll amount"},
                    "dy": {"type": "integer", "description": "Vertical scroll amount"},
                },
                "required": ["dx", "dy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "typing",
            "description": "Type the specified text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to type"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press",
            "description": "Press the specified key and release it",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": KEYBOARD_KEYS, "description": "Key to press"}
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "key_down",
            "description": "Press the specified key",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": KEYBOARD_KEYS,
                        "description": "Key to press down",
                    }
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "key_up",
            "description": "Release the specified key",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": KEYBOARD_KEYS,
                        "description": "Key to release",
                    }
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hotkey",
            "description": "Press the specified key combination",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string", "enum": KEYBOARD_KEYS},
                        "description": "Array of keys to press simultaneously",
                    }
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait until the next action",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fail",
            "description": "Decide the task cannot be performed",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Decide the task is done",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


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
        max_steps: int,
        exp_dir: Path,
        record_video: bool = True,
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
        if DesktopEnv is None:
            raise ImportError(
                "desktop_env is not installed. Please install it (use `make osworld`) to use OSWorld Gym."
            )
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
        self.exp_dir = exp_dir
        self.record_video = record_video

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.env.reset(task_config=self.task, seed=seed)
        logging.info(f"Start solving task: {self.task['instruction']}")
        time.sleep(
            60
        )  # Wait for the environment to be ready, as in https://github.com/xlang-ai/OSWorld/blob/main/lib_run_single.py#L15
        raw_obs = self.env._get_obs()  # Get the initial observation
        if self.record_video:
            self.env.controller.start_recording()
            logging.info("Started recording the environment video")
        obs = self.to_agentlab_observation(raw_obs)
        self._step_count = 0
        return obs, self.env_info

    @add_step_timing_to_env_info_decorator
    def step(self, action: str):
        """Execute the action in the OS-world environment."""
        env_action = self.agentlab_to_env_action(action)
        logger.info(f"AgentLab Action returned: {action}, converted to: {env_action}")
        raw_obs, reward, done, info = self.env.step(env_action)
        logger.info(f"STEP {self.task['id']} {self._step_count + 1}/{self.max_steps}")
        self._step_count += 1
        truncated = info.get("fail", False) or self._step_count >= self.max_steps
        if done or truncated:
            if done:
                logger.info(f"Task {self.task['id']} completed successfully.")
            else:
                logger.warning(f"Task {self.task['id']} truncated after {self._step_count} steps.")
            try:
                reward = self.env.evaluate()
                logger.info(f"Evaluated reward: {reward}")
            except Exception as e:
                logger.error(f"Failed to evaluate {self.task} task: {e}")
        obs = self.to_agentlab_observation(raw_obs)
        return obs, reward, done, truncated, info

    def agentlab_to_env_action(self, action: str) -> Any:
        """Convert AgentLab agents action format to OSWorld action format."""
        if self.env.action_space == "computer_13":
            return self.convert_agentlab_action_to_computer_13(action)
        elif self.env.action_space == "pyautogui":
            raise NotImplementedError(
                "PyAutoGUI action space is not supported yet. Please use 'computer_13' action space."
            )

    def to_agentlab_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Convert OSWorld observation to AgentLab format."""
        converted_obs = {}

        self._add_screenshot(converted_obs, obs)
        # self._add_som_screenshot(converted_obs, obs)  #TODO: test this
        converted_obs["axtree_txt"] = linearize_accessibility_tree(
            accessibility_tree=obs["accessibility_tree"], platform="ubuntu"
        )
        converted_obs["last_action_error"] = ""  # OSWorld doesn't provide this directly
        converted_obs["focused_element_bid"] = ""  # Extract from accessibility tree if available
        converted_obs = self._add_browser_context(converted_obs)
        converted_obs = self._add_task_context(converted_obs, obs)

        return converted_obs

    def convert_screenshot_to_numpy(self, screenshot) -> np.ndarray:
        """Convert screenshot to numpy array format expected by AgentLab."""
        image = Image.open(BytesIO(screenshot))
        image = image.convert("RGB") if image.mode != "RGB" else image
        return np.array(image)

    def _add_screenshot(self, converted_obs: dict[str, Any], obs: dict[str, Any]) -> None:
        """Convert screenshot to numpy array format expected by AgentLab"""
        converted_obs["screenshot"] = self.convert_screenshot_to_numpy(obs["screenshot"])

    def _add_som_screenshot(self, converted_obs: dict[str, Any], obs: dict[str, Any]) -> None:
        """Convert SOM screenshot to numpy array format expected by AgentLab"""
        masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(
            obs["screenshot"], obs["accessibility_tree"], platform="ubuntu"
        )
        converted_obs["som_screenshot"] = self.convert_screenshot_to_numpy(tagged_screenshot)

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
        if obs.get("terminal"):
            converted_obs["terminal_output"] = obs["terminal"]
        return converted_obs

    def convert_agentlab_action_to_computer_13(self, action: str) -> dict[str, Any] | str:
        """Convert action string to dictionary format.

        Args:
            action (str): Action string in AgentLab format, e.g., "move_to(x=100, y=200)".

        Returns:
            dict[str, Any] | str: Action in OSWorld Computer 13 format as a dictionary,
            or a string for simple actions like "wait", "done", or "fail".

        Examples:
        >>> env = OsworldGym(task={}, provider_name="vmware", region=None, path_to_vm=None,
        ...                  snapshot_name="init_state", action_space="computer_13",
        ...                  cache_dir="cache", screen_size=(1920, 1080), headless=True,
        ...                  require_a11y_tree=True, require_terminal=False, os_type="Ubuntu",
        ...                  enable_proxy=False, max_steps=50, exp_dir=Path("."))
        >>> env.convert_agentlab_action_to_computer_13("move_to(x=100, y=200)")
        {'action_type': 'MOVE_TO', 'parameters': {'x': 100, 'y': 200}}
        >>> env.convert_agentlab_action_to_computer_13("wait()")
        'WAIT'
        """

        action_type, action_args, action_kwargs = self.parse_agentlab_action_str_to_func_args(
            action
        )

        if action_type in ["wait", "done", "fail"]:
            return str(action_type).upper()
        if action_args:
            logger.warning(
                f"""Action '{action_type}' has unexpected positional arguments: {action_args}.
                OSWorld Computer 13 actions are processed as dictionaries."""
            )
        action_kwargs = action_kwargs if action_kwargs is not None else {}

        return {"action_type": str(action_type).upper(), "parameters": action_kwargs}

    @staticmethod
    def parse_agentlab_action_str_to_func_args(action: str):
        """Parse the agentlab action string to extract function name, args, and kwargs.

        Args:
            action (str): Action string in AgentLab format, e.g., "move_to(x=100, y=200)".

        Returns:
            tuple: A tuple containing the function name, a list of positional arguments,
                   and a dictionary of keyword arguments.

        Examples:
        >>> parse_agentlab_action_str_to_func_args("move_to(x=100, y=200)")
        ('move_to', [], {'x': 100, 'y': 200})
        >>> parse_agentlab_action_str_to_func_args("hotkey(keys=['ctrl', 'alt', 't'])")
        ('hotkey', [], {'keys': ['ctrl', 'alt', 't']})
        """
        try:
            action = action.strip()
            parsed = ast.parse(action, mode="eval")
            if isinstance(parsed.body, ast.Call):
                func_name = ast.unparse(parsed.body.func)
                args = [ast.literal_eval(arg) for arg in parsed.body.args]
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in parsed.body.keywords}
                return func_name, args, kwargs
        except Exception as e:
            logger.warning(
                f"Failed to parse agentlab agent's str function call: {action}, error: {e}"
            )
        return None, None, None

    def close(self):
        if self.record_video:
            video_name = str(self.exp_dir / "recording.mp4")
            self.env.controller.end_recording(video_name)
            logger.info(f"Recorded video saved to {video_name}")
        return self.env.close()


@dataclass
class OSWorldActionSet(AbstractActionSet, DataClassJsonMixin):
    # TODO: Define and use agentlab AbstractActionSet
    # AbstractActionSet should define some standard format to represent actions.(list of dict with keys that are MCP compatible)
    # Should we have 'abstract function' here for action conversion for backend LLM with fixed action set like UI-Tars or Semi-fixed action set LLMs like OpenAI CUA?
    # TODO: We need to support both 'action space as tools' and 'action space as prompt' for agentlab agents
    # and have conversion functions to convert them to format acceptable by environment.
    action_space: Literal["computer_13", "pyautogui"] = "computer_13"
    multiaction: bool = False

    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """Describe the OSWorld action set for desktop interactions."""
        pass

    def example_action(self, abstract: bool) -> str:
        """Provide example actions for the action set."""
        pass

    def to_python_code(self, action) -> str:
        """We use the OS-world/desktop_env environment controller"""
        pass

    def to_tool_description(self, api="openai"):
        """Convert the action set to a tool description for Tool-Use LLMs.

        The default for openai is openai Response API tools format.

        Args:
            api (str): The API format to use. Defaults to "openai".

        Returns:
            list[dict]: List of tool descriptions in the specified API format.

        Raises:
            ValueError: If an unsupported action space is specified.
        """
        # TODO: Rename bgym AbstractActionSet 'to_tool_descriptor' method as 'to_tool_description' for consistency.
        if self.action_space == "computer_13":
            tools = COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS

        else:
            raise ValueError(
                "Only 'computer_13' action space is currently supported for tool description."
            )
        api_formatters = {
            "openai": lambda: format_chat_completion_tools_to_response_api(tools),
            "chatcompletion": lambda: tools,
            "anthropic": lambda: format_chat_completion_tools_to_anthropic(tools),
        }

        if api not in api_formatters:
            raise ValueError(f"Unsupported API type: {api}")

        return api_formatters[api]()


def format_chat_completion_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI Response API tool format to Anthropic tool format."""
    formatted_tools = []
    for tool in tools:
        function_def = tool["function"]
        formatted_tool = {
            "name": function_def["name"],
            "description": function_def["description"],
            "input_schema": function_def["parameters"],
        }
        formatted_tools.append(formatted_tool)

    return formatted_tools


def format_chat_completion_tools_to_response_api(tools: list[dict]) -> list[dict]:
    """Convert tools from OpenAI Chat Completion format to Responses API format.

    Args:
        tools: List of tools in Chat Completion format with nested function object

    Returns:
        List of tools in Responses API format with flattened structure
    """
    formatted_tools = []
    for tool in tools:
        function_def = tool["function"]
        formatted_tool = {
            "type": "function",
            "name": function_def["name"],
            "description": function_def["description"],
            "parameters": function_def["parameters"],
        }

        # Handle the strict field if present
        if "strict" in function_def:
            formatted_tool["strict"] = function_def["strict"]

        formatted_tools.append(formatted_tool)

    return formatted_tools


@dataclass
class OsworldEnvArgs(AbstractEnvArgs):
    task: dict[str, Any]
    task_seed: int = 0
    task_name: str | None = None
    path_to_vm: str | None = None  # path to .vmx file
    provider_name: str = "docker"  # path to .vmx file
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
    max_steps: int = 50

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
            exp_dir=exp_dir,
        )
        return gym


class OsworldBenchmark(AbstractBenchmark):
    name: str = "osworld"
    is_multi_tab: bool = False
    high_level_action_set_args: OSWorldActionSet = None  # type: ignore
    test_set_path: str = "OSWorld/evaluation_examples"
    test_set_name: str = "test_all.json"
    domain: str = "all"
    env_args: OsworldEnvArgs = None  # type: ignore # basic env configuration for all tasks
    env_args_list: list[OsworldEnvArgs] = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        if not self.env_args:
            self.env_args = OsworldEnvArgs(task={})
        self.high_level_action_set_args = OSWorldActionSet(action_space=self.env_args.action_space)
        with open(os.path.join(self.test_set_path, self.test_set_name)) as f:
            tasks = json.load(f)
        if self.domain != "all":
            tasks = {self.domain: tasks[self.domain]}

        for domain in tasks:
            for task_id in tasks[domain]:
                task_file = os.path.join(self.test_set_path, f"examples/{domain}/{task_id}.json")
                with open(task_file) as f:
                    task = json.load(f)
                    task = self.fix_settings_file_path_in_config(task)
                name = f"{self.name}.{task['id']}"
                task_env_args = deepcopy(self.env_args)
                task_env_args.task = task
                task_env_args.task_name = name
                self.env_args_list.append(task_env_args)
        logger.info(f"Loaded {len(self.env_args_list)} tasks from domain '{self.domain}'")

    def fix_settings_file_path_in_config(self, task: dict) -> dict:
        """Fix the settings file path in the task configuration.

        Args:
            task: Task configuration dictionary.

        Returns:
            Updated task configuration with fixed settings file paths.
        """
        osworld_repo = os.getenv("OSWORLD_REPO", "OSWorld")
        updated_task = deepcopy(task)  # Avoid modifying the original task
        for config in updated_task["config"]:
            if config.get("parameters", False) and config["parameters"].get("settings_file", False):
                config["parameters"]["settings_file"] = os.path.join(
                    osworld_repo, config["parameters"]["settings_file"]
                )
        return updated_task
