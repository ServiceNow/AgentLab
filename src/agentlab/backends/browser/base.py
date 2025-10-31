from tapeagents.environment import FunctionCall
from tapeagents.mcp import MCPEnvironment, ToolCallAction
from tapeagents.tool_calling import as_openai_tool


class BrowserBackend():
    def run_js(self, js: str):
        raise NotImplementedError

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        raise NotImplementedError

    def tools_description(self) -> str:
        raise NotImplementedError

    def tools(self) -> list[dict]:
        raise NotImplementedError


class MCPBrowserBackend(BrowserBackend):
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.mcp = MCPEnvironment(config_path=self.config_path)
        self.mcp.initialize()

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        action = ToolCallAction(
            function=FunctionCall(name=tool_name, arguments=arguments)
        )
        tool_result = self.mcp.step(action)
        return tool_result.content.content[0].text


    def tools_description(self) -> str:
        return self.mcp.tools_description()

    def tools(self) -> list[dict]:
        actions = self.mcp.actions()
        tools = [as_openai_tool(a).model_dump() for a in actions]
        return tools
