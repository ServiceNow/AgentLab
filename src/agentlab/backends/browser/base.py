from pydantic import BaseModel
from tapeagents.mcp import MCPEnvironment
from tapeagents.tool_calling import FunctionCall, ToolCallAction, ToolSpec


class BrowserBackend(BaseModel):
    def initialize(self) -> None:
        raise NotImplementedError

    def run_js(self, js: str):
        raise NotImplementedError

    def goto(self, url: str) -> str:
        raise NotImplementedError

    def page_snapshot(self) -> str:
        raise NotImplementedError

    def step(self, action: ToolCallAction) -> str:
        raise NotImplementedError

    def actions(self) -> tuple[ToolSpec]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MCPBrowserBackend(BrowserBackend):
    config_path: str
    _mcp = None

    def initialize(self) -> None:
        self._mcp = MCPEnvironment(config_path=self.config_path)
        self._mcp.initialize()

    def step(self, action: ToolCallAction) -> str:
        return self._call_mcp(action)

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        return self._call_mcp(
            ToolCallAction(function=FunctionCall(name=tool_name, arguments=arguments))
        )

    def _call_mcp(self, action: ToolCallAction) -> str:
        tool_result = self._mcp.step(action)
        texts = [c.text for c in tool_result.content.content]
        return "\n\n".join(texts)

    def actions(self) -> tuple[ToolSpec]:
        return self._mcp.actions()

    def close(self) -> None:
        self._mcp.close()
