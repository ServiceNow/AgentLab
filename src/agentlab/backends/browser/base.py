import logging
from typing import Any, Callable, Literal

from langchain_core.utils.function_calling import convert_to_openai_tool
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FunctionCall(BaseModel):
    """
    A class representing a function call.

    Attributes:
        name (str): The name of the function being called.
        arguments (Any): The arguments to be passed to the function.
    """

    name: str
    arguments: Any


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


class ToolCallAction(BaseModel):
    id: str = ""
    function: FunctionCall


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


class BrowserBackend(BaseModel):
    def initialize(self) -> None:
        raise NotImplementedError

    def run_js(self, js: str):
        raise NotImplementedError

    def goto(self, url: str) -> str:
        raise NotImplementedError

    def page_snapshot(self) -> str:
        raise NotImplementedError

    def page_screenshot(self) -> Image:
        raise NotImplementedError

    def step(self, action: ToolCallAction) -> str:
        raise NotImplementedError

    def actions(self) -> tuple[ToolSpec]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
