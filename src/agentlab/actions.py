import json
import logging
from typing import Literal

from bgym import AbstractActionSet
from pydantic import BaseModel, Field

from agentlab.backends.browser.base import FunctionCall, ToolCallAction, ToolSpec
from agentlab.llm.llm_utils import parse_html_tags_raise

logger = logging.getLogger(__name__)


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
            return """<action>
{
    "name": "<action_name>",
    "arguments": {
        "<argument_name_1>": "<argument_value_1>",
        "<argument_name_2>": "<argument_value_2>",
        ...
    }
}
</action>
"""
        else:
            return """<action>
{
    "name": "browser_navigate",
    "arguments": {
        "url": "https://www.google.com"
    }
}
</action>
"""

    @classmethod
    def parse_action(cls, llm_output: str) -> ToolCallAction:
        logger.info(f"Parsing action: {llm_output}")
        if "<action>" in llm_output:
            content_dict, valid, retry_message = parse_html_tags_raise(llm_output, keys=["action"])
            if not valid or "action" not in content_dict:
                raise ValueError(
                    f"Invalid action: llm_output: {llm_output}, retry_message: {retry_message}"
                )
            action_str = content_dict["action"]
        else:
            action_str = llm_output
        try:
            action_dict = json.loads(action_str)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse action: {action_str}")
        return ToolCallAction(
            function=FunctionCall(name=action_dict["name"], arguments=action_dict["arguments"])
        )

    def to_python_code(self, action) -> str:
        return action
