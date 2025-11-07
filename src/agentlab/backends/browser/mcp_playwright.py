import logging

from tapeagents.tool_calling import ToolCallAction

from agentlab.backends.browser.base import MCPBrowserBackend

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "src/agentlab/backends/browser/mcp_playwright.json"


class MCPPlaywright(MCPBrowserBackend):
    config_path: str = DEFAULT_CONFIG_PATH

    def run_js(self, js: str):
        raw_response = self.call_tool("browser_evaluate", {"function": js})
        try:
            _, half_response = raw_response.split("### Result", maxsplit=1)
            result_str, _ = half_response.split("\n### Ran", maxsplit=1)
            result_str = result_str.strip()
        except Exception as e:
            logger.error(f"Error parsing JS result: {e}. Raw result: {raw_response}")
            raise e
        return result_str

    def step(self, action: ToolCallAction) -> str:
        tool_result = self._call_mcp(action)
        logger.info(f"Tool result: {tool_result}")
        return tool_result

    def page_snapshot(self) -> str:
        return self.call_tool("browser_snapshot", {})

    def goto(self, url: str) -> str:
        tool_result = self.call_tool("browser_navigate", {"url": url})
        return tool_result
