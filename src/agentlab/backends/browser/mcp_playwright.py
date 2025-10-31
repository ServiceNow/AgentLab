from agentlab.backends.browser.base import MCPBrowserBackend

DEFAULT_CONFIG_PATH = "src/agentlab/backends/browser/mcp_playwright.json"

class MCPPlaywright(MCPBrowserBackend):
    def __init__(self, config_path: str | None = None):
        super().__init__(config_path or DEFAULT_CONFIG_PATH)

    def run_js(self, js: str):
        raw_response = self.call_tool("browser_evaluate", {"function": js})
        _, half_response = raw_response.split("### Result", maxsplit=1)
        result_str, _ = half_response.split("\n### Ran", maxsplit=1)
        result_str = result_str.strip()
        return result_str
