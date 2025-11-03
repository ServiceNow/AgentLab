from bgym import AbstractActionSet
from tapeagents.tool_calling import FunctionCall, ToolCallAction, ToolSpec

from agentlab.llm.llm_utils import parse_html_tags_raise


class ToolsActionSet(AbstractActionSet):
    def __init__(self, actions:list[ToolSpec]):
        self.actions = actions

    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        tools_description = "\n".join([action.description() for action in self.actions])
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
        content_dict, valid, retry_message = parse_html_tags_raise(llm_output, keys=["action"])
        if not valid or "action" not in content_dict:
            raise ValueError(f"Invalid action: llm_output: {llm_output}, retry_message: {retry_message}")
        action_str = content_dict["action"]
        return ToolCallAction(function=FunctionCall(name=action_str["name"], arguments=action_str["arguments"]))

    def to_python_code(self, action) -> str:
        return action.model_dump_json(indent=2)