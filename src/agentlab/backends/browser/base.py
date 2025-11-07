import logging

from mcp.types import ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel
from tapeagents.mcp import MCPEnvironment
from tapeagents.tool_calling import FunctionCall, ToolCallAction, ToolSpec

logger = logging.getLogger(__name__)


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


class MCPBrowserBackend(BrowserBackend):
    config_path: str
    _mcp = None

    def initialize(self) -> None:
        self._mcp = MCPEnvironment(config_path=self.config_path)
        self._mcp.initialize()

    def step(self, action: ToolCallAction) -> dict:
        contents = self._call_mcp(action)
        text = "\n".join([c.text for c in contents if c.type == "text"])
        return {"pruned_html": text, "axtree_txt": text}

    def call_tool(self, tool_name: str, arguments: dict) -> list[TextContent | ImageContent]:
        return self._call_mcp(
            ToolCallAction(function=FunctionCall(name=tool_name, arguments=arguments))
        )

    def _call_mcp(self, action: ToolCallAction) -> list[TextContent | ImageContent]:
        tool_result = self._mcp.step(action)
        return tool_result.content.content

    def actions(self) -> tuple[ToolSpec]:
        return self._mcp.actions()

    def close(self) -> None:
        self._mcp.close()
