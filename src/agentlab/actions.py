import json
import logging
from typing import Callable, Literal
from uuid import uuid4

from bgym import AbstractActionSet
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from agentlab.llm.llm_utils import parse_html_tags_raise

logger = logging.getLogger(__name__)


class FunctionSpec(BaseModel):
    """
    A class representing the specification of a function.

    Attributes:
        name (str): The name of the function.
        description (str): A brief description of the function.
        parameters (dict): A dictionary containing the parameters of the function.
    """

    name: str
    description: str
    parameters: dict



class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    arguments: dict = Field(default_factory=dict)

    def llm_view(self, **kwargs) -> str:
        return self.model_dump_json(indent=2)


class ToolSpec(BaseModel):
    """
    ToolSpec is a model that represents a tool specification with a type and a function.

    Attributes:
        type (Literal["function"]): The type of the tool, which is always "function".
        function (FunctionSpec): The specification of the function.
    """

    type: Literal["function"] = "function"
    function: FunctionSpec

    def description(self) -> str:
        return f"{self.function.name} - {self.function.description}"

    @classmethod
    def from_function(cls, function: Callable):
        """
        Creates an instance of the class by validating the model from a given function.

        Args:
            function (Callable): The function to be converted and validated.

        Returns:
            (ToolSpec): An instance of the class with the validated model.
        """
        return cls.model_validate(convert_to_openai_tool(function))


class ToolsActionSet(AbstractActionSet):
    multiaction: bool = False
    strict: bool = False

    def __init__(self, actions: list[ToolSpec]):
        self.actions = actions

    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        descs = []
        for action in self.actions:
            desc = f"## {action.description()}.\n Schema: {action.model_dump_json(indent=2)}"
            descs.append(desc)
        tools_description = "\n".join(descs)
        return tools_description

    def example_action(self, abstract: bool) -> str:
        if abstract:
            return """{
    "name": "<action_name>",
    "arguments": {
        "<argument_name_1>": "<argument_value_1>",
        "<argument_name_2>": "<argument_value_2>",
        ...
    }
}"""
        else:
            return """{
    "name": "browser_click",
    "arguments": {
        "element": "buttom with year 2022",
        "ref": "e26"
    }
}"""

    @classmethod
    def parse_action(cls, llm_output: str) -> ToolCall:
        logger.info(f"Parsing action: {llm_output}")
        if "<action>" in llm_output:
            content_dict, valid, retry_message = parse_html_tags_raise(llm_output, keys=["action"])
            if not valid or "action" not in content_dict:
                raise ValueError(f"Invalid action: llm_output: {llm_output}, retry_message: {retry_message}")
            action_str = content_dict["action"]
        else:
            action_str = llm_output
        try:
            action_dict = json.loads(action_str)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse action: {action_str}")
        return ToolCall(name=action_dict["name"], arguments=action_dict["arguments"])

    def to_python_code(self, action) -> str:
        return action
