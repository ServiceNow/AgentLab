from typing import Any, Literal, Union

from pydantic import Annotated, Field, TypeAdapter
from tapeagents.core import Action, Observation, Tape
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.base import Multitool, Tool

from agentlab.benchmarks.abstract_env import AbstractEnv

EnvTape = Tape[None, Action | Observation]


class FunctionCall(Action):
    kind: Literal["function_call_action"] = ["function_call_action"]
    function_name: str
    args: list[Any] | None
    kwargs: dict[str, Any] | None


class FunctionCallResult(Observation):
    kind: Literal["function_call_result"] = ["function_call_result"]
    result: Any


class SimpleFunctionCallTool(Tool):
    action = FunctionCall
    observation = FunctionCallResult
    function: callable
    function_name: str = ""

    def model_post_init(self, __context):
        function_name = getattr(self.function, "__name__", "")
        if not function_name and not self.function_name:
            raise ValueError("Function has no name, function_name must be provided")

    def execute_action(self, action: FunctionCall) -> FunctionCallResult:
        if not self.function_name == action.function_name:
            raise ValueError(
                f"Unexpected function action {action.function_name}, expected {self.function_name}"
            )
        result = self.function(*action.args, **action.kwargs)
        return FunctionCallResult(result=result)


class MultiToolGym(AbstractEnv):
    def __init__(self, tools: list[Tool | Multitool]):
        self._env = ToolCollectionEnvironment(tools)
        self._actions = self._env.actions()
        self._actions_parser: TypeAdapter = TypeAdapter(
            Annotated[Union[self._actions], Field(discriminator="kind")]
        )
        self.reset()

    def reset(self, seed=None):
        self._tape: EnvTape = EnvTape(steps=[])

    def step(self, action: str):
        try:
            action_step = self._actions_parser.validate_json(action)
        except Exception:
            raise ValueError("Action must be a valid JSON dict")
        assert isinstance(action_step, Action), "{action_step.kind} is not an Action"
        self._tape += [action_step]
        self._tape = self._env.react(self._tape)
        observation_step: Observation = self._tape.steps[-1]
        reward = self.calculate_reward()
        terminated = False
        truncated = False
        env_info = {"step_metadata": observation_step.metadata}
        return observation_step.llm_dict(), reward, terminated, truncated, env_info

    def calculate_reward(self) -> float:
        return 0.0

    def close(self):
        self._env.close()
