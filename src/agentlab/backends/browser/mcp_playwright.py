import base64
import logging
from io import BytesIO

from PIL import Image

from agentlab.actions import ToolCall
from agentlab.backends.browser.mcp import MCPBrowserBackend

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "src/agentlab/backends/browser/mcp_playwright.json"


class MCPPlaywright(MCPBrowserBackend):
    config_path: str = DEFAULT_CONFIG_PATH

    def run_js(self, js: str):
        contents = self.call_tool("browser_evaluate", {"function": js})
        raw_response = "\n".join([c.text for c in contents if c.type == "text"])
        try:
            _, half_response = raw_response.split("### Result", maxsplit=1)
            result_str, _ = half_response.split("\n### Ran", maxsplit=1)
            result_str = result_str.strip()
        except Exception as e:
            logger.error(f"Error parsing JS result: {e}. Raw result: {raw_response}")
            raise e
        return result_str

    def step(self, action: ToolCall) -> dict:
        contents = self.call_tool(action.name, action.arguments)
        logger.info(f"Step result has {len(contents)} contents")
        tool_result = "\n".join(
            [c.text for c in contents if c.type == "text" and "# Ran Playwright code" not in c.text]
        )
        html = self.page_html()
        screenshot = self.page_screenshot()
        axtree = self.page_axtree()
        return {
            "tool_result": tool_result,
            "pruned_html": html,
            "axtree_txt": axtree,
            "screenshot": screenshot,
        }

    def page_html(self) -> str:
        contents = self.call_tool(
            "browser_evaluate", {"function": "document.documentElement.outerHTML"}
        )
        raw_response = "\n".join([c.text for c in contents if c.type == "text"])
        try:
            _, half_response = raw_response.split("### Result", maxsplit=1)
            result_str, _ = half_response.split("\n### Ran", maxsplit=1)
            return result_str.strip()
        except Exception as e:
            logger.error(f"Error parsing page_html result: {e}. Raw result: {raw_response}")
            return ""

    def page_axtree(self) -> str:
        contents = self.call_tool("browser_snapshot", {})
        return "\n".join([c.text for c in contents if c.type == "text"])

    def page_screenshot(self) -> Image:
        contents = self.call_tool("browser_take_screenshot", {})
        content = [c for c in contents if c.type == "image"][0]
        image_base64 = content.data
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        return image

    def goto(self, url: str) -> str:
        contents = self.call_tool("browser_navigate", {"url": url})
        return "\n".join([c.text for c in contents if c.type == "text"])
