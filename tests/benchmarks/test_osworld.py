import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.find_spec("desktop_env")
if spec is None:
    DESKTOP_ENV_AVAILABLE = False
    OSWorldActionSet = None
    OsworldEnvArgs = None
    OsworldGym = None
else:
    # If desktop_env is available, import the necessary classes
    from agentlab.benchmarks.osworld import (
        OSWorldActionSet,
        OsworldEnvArgs,
        OsworldGym,
    )

    DESKTOP_ENV_AVAILABLE = True


# Skip the entire module if desktop_env is not available
pytestmark = pytest.mark.skipif(not DESKTOP_ENV_AVAILABLE, reason="desktop_env not installed")


def mock_task_config() -> dict:
    """Mock task configuration for testing."""
    return {
        "id": "bb5e4c0d-f964-439c-97b6-bdb9747de3f4",
        "snapshot": "chrome",
        "instruction": "Can you make Bing the main search thingy when I look stuff up on the internet?",
        "source": "https://support.google.com/chrome/answer/95426",
        "config": [
            {
                "type": "launch",
                "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]},
            }
        ],
        "trajectory": "trajectories/",
        "related_apps": ["chrome"],
        "evaluator": {
            "func": "match_in_list",
            "result": {"type": "default_search_engine"},
            "expected": {"type": "rule", "rules": {"expected": ["Microsoft Bing", "Bing"]}},
        },
        "proxy": False,
    }


class TestOSWorldActionSet:
    """Test cases for OSWorld action set functionality."""

    def test_action_set_creation(self):
        """Test basic action set creation."""
        action_set = OSWorldActionSet(action_space="computer_13")
        assert action_set.action_space == "computer_13"

    def test_to_tool_description_openai(self):
        """Test tool description conversion for OpenAI format."""
        action_set = OSWorldActionSet(action_space="computer_13")
        tools = action_set.to_tool_description(api="openai")

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check that tools have the expected structure
        tool = tools[0]
        assert "type" in tool
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["type"] == "function"

    def test_to_tool_description_anthropic(self):
        """Test tool description conversion for Anthropic format."""
        action_set = OSWorldActionSet(action_space="computer_13")
        tools = action_set.to_tool_description(api="anthropic")

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check that tools have the Anthropic format
        tool = tools[0]
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        # Anthropic format doesn't have "type" field

    def test_unsupported_action_space(self):
        """Test that unsupported action spaces raise ValueError."""
        action_set = OSWorldActionSet(action_space="pyautogui")
        with pytest.raises(
            ValueError, match="Only 'computer_13' action space is currently supported"
        ):
            action_set.to_tool_description()


class TestOsworldEnvArgs:
    """Test cases for OSWorld environment arguments."""

    def test_env_args_creation(self):
        """Test basic environment args creation."""
        task = mock_task_config()
        env_args = OsworldEnvArgs(task=task, task_name="test_task", max_steps=10)

        assert env_args.task == task
        assert env_args.task_name == "test_task"
        assert env_args.max_steps == 10
        assert env_args.action_space == "computer_13"  # default
        assert env_args.provider_name == "docker"  # default

    def test_env_args_custom_config(self):
        """Test environment args with custom configuration."""
        task = mock_task_config()
        env_args = OsworldEnvArgs(
            task=task,
            task_name="custom_task",
            action_space="computer_13",
            provider_name="vmware",
            headless=True,
            screen_size=(1280, 720),
            max_steps=25,
        )

        assert env_args.action_space == "computer_13"
        assert env_args.provider_name == "vmware"
        assert env_args.headless is True
        assert env_args.screen_size == (1280, 720)
        assert env_args.max_steps == 25

    @patch("agentlab.benchmarks.osworld.OsworldGym")
    def test_make_env(self, mock_gym_class):
        """Test environment creation from args."""
        task = mock_task_config()
        env_args = OsworldEnvArgs(task=task, task_name="test_task")

        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir)
            env_args.make_env(exp_dir)

            # Verify that OsworldGym was called with correct arguments
            mock_gym_class.assert_called_once()
            call_args = mock_gym_class.call_args[1]
            assert call_args["task"] == task
            assert call_args["exp_dir"] == exp_dir


class TestOsworldGym:
    """Test cases for OSWorld gym functionality."""

    def test_gym_action_parsing(self):
        """Test gym action parsing functionality."""

        from agentlab.benchmarks.osworld import OsworldGym

        # Test various action strings including edge cases
        test_cases = [
            # Basic actions
            ("wait()", ("wait", [], {})),
            ("done()", ("done", [], {})),
            ("move_to(x=100, y=200)", ("move_to", [], {"x": 100, "y": 200})),
            ('typing(text="hello world")', ("typing", [], {"text": "hello world"})),
            ("hotkey(keys=['ctrl', 'c'])", ("hotkey", [], {"keys": ["ctrl", "c"]})),
            # Edge cases with strings
            ('typing(text="")', ("typing", [], {"text": ""})),  # Empty string
            ('typing(text="line1\\nline2")', ("typing", [], {"text": "line1\nline2"})),  # Newlines
            ('typing(text="tab\\there")', ("typing", [], {"text": "tab\there"})),  # Tabs
            (
                'typing(text="quote\\"test")',
                ("typing", [], {"text": 'quote"test'}),
            ),  # Escaped quotes
            (
                'typing(text="single\'quote")',
                ("typing", [], {"text": "single'quote"}),
            ),  # Single quotes
            ('typing(text="unicode: café")', ("typing", [], {"text": "unicode: café"})),  # Unicode
            # Edge cases with coordinates
            ("move_to(x=0, y=0)", ("move_to", [], {"x": 0, "y": 0})),  # Zero coordinates
            (
                "move_to(x=-10, y=-20)",
                ("move_to", [], {"x": -10, "y": -20}),
            ),  # Negative coordinates
            (
                "move_to(x=9999, y=9999)",
                ("move_to", [], {"x": 9999, "y": 9999}),
            ),  # Large coordinates
            # Edge cases with lists
            ("hotkey(keys=[])", ("hotkey", [], {"keys": []})),  # Empty list
            ("hotkey(keys=['ctrl'])", ("hotkey", [], {"keys": ["ctrl"]})),  # Single key
            (
                "hotkey(keys=['ctrl', 'shift', 'alt', 'a'])",
                ("hotkey", [], {"keys": ["ctrl", "shift", "alt", "a"]}),
            ),  # Multiple keys
            # Edge cases with boolean values
            ("scroll(direction='up', clicks=3)", ("scroll", [], {"direction": "up", "clicks": 3})),
            (
                "click(x=100, y=200, button='left')",
                ("click", [], {"x": 100, "y": 200, "button": "left"}),
            ),
            # Edge cases with mixed parameter types
            (
                "complex_action(text='test', x=50, enabled=True, items=['a', 'b'])",
                (
                    "complex_action",
                    [],
                    {"text": "test", "x": 50, "enabled": True, "items": ["a", "b"]},
                ),
            ),
            # Edge cases with whitespace
            ("  wait()  ", ("wait", [], {})),  # Leading/trailing spaces
            (
                "move_to( x=100 , y=200 )",
                ("move_to", [], {"x": 100, "y": 200}),
            ),  # Spaces around params
            # Edge cases with special characters in strings
            (
                'typing(text="@#$%^&*()+={}[]|\\:;\'<>?,./")',
                ("typing", [], {"text": "@#$%^&*()+={}[]|\\:;'<>?,./"}),
            ),
        ]

        for action_str, expected in test_cases:
            result = OsworldGym.parse_agentlab_action_str_to_func_args(action_str)
            assert result == expected, f"Failed parsing: {action_str}"

    @patch("agentlab.benchmarks.osworld.DesktopEnv")
    def test_gym_creation(self, mock_desktop_env):
        """Test OSWorld gym creation."""
        task = mock_task_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir)
            gym = OsworldGym(
                task=task,
                provider_name="docker",
                region=None,
                path_to_vm=None,
                snapshot_name="init_state",
                action_space="computer_13",
                cache_dir="cache",
                screen_size=(1920, 1080),
                headless=True,
                require_a11y_tree=True,
                require_terminal=False,
                os_type="Ubuntu",
                enable_proxy=False,
                max_steps=50,
                exp_dir=exp_dir,
            )

            assert gym.task == task
            assert gym._step_count == 0
            assert gym.max_steps == 50
            assert gym.exp_dir == exp_dir

    def test_convert_agentlab_action_to_computer_13(self):
        """Test action conversion from AgentLab to Computer 13 format."""
        task = mock_task_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir)

            with patch("agentlab.benchmarks.osworld.DesktopEnv"):
                gym = OsworldGym(
                    task=task,
                    provider_name="docker",
                    region=None,
                    path_to_vm=None,
                    snapshot_name="init_state",
                    action_space="computer_13",
                    cache_dir="cache",
                    screen_size=(1920, 1080),
                    headless=True,
                    require_a11y_tree=True,
                    require_terminal=False,
                    os_type="Ubuntu",
                    enable_proxy=False,
                    max_steps=50,
                    exp_dir=exp_dir,
                )

                # Test simple action
                result = gym.convert_agentlab_action_to_computer_13("wait()")
                assert result == "WAIT"

                # Test action with parameters
                result = gym.convert_agentlab_action_to_computer_13("move_to(x=100, y=200)")
                expected = {"action_type": "MOVE_TO", "parameters": {"x": 100, "y": 200}}
                assert result == expected

                # Test typing action
                result = gym.convert_agentlab_action_to_computer_13('typing(text="hello")')
                expected = {"action_type": "TYPING", "parameters": {"text": "hello"}}
                assert result == expected
