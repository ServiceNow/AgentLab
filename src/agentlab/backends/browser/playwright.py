import logging
import time
from io import BytesIO
from typing import Any, Callable

from PIL import Image
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright
from playwright.sync_api import Page as SyncPage
from playwright.sync_api import sync_playwright

from agentlab.actions import ToolCall, ToolSpec
from agentlab.backends.browser.base import AsyncBrowserBackend, BrowserBackend

logger = logging.getLogger(__name__)


_pw = None  # Global Playwright instance for SyncPlaywright
_browser = None  # Global Browser instance for SyncPlaywright


class SyncPlaywright(BrowserBackend):
    """Fully synchronous Playwright backend using playwright.sync_api."""

    has_pw_page: bool = True
    _actions: dict[str, Callable]
    _page: SyncPage

    def model_post_init(self, __context: Any):
        self._actions = {
            "browser_press_key": self.browser_press_key,
            "browser_type": self.browser_type,
            "browser_click": self.browser_click,
            "browser_drag": self.browser_drag,
            "browser_hover": self.browser_hover,
            "browser_select_option": self.browser_select_option,
            "browser_mouse_click_xy": self.browser_mouse_click_xy,
        }

    def initialize(self):
        global _pw, _browser
        if _pw is None:
            _pw = sync_playwright().start()
        if _browser is None:
            _browser = _pw.chromium.launch(headless=True, chromium_sandbox=True)

        self._page = _browser.new_page()

    @property
    def page(self) -> SyncPage:
        return self._page

    def browser_press_key(self, key: str):
        """Press a key on the keyboard."""
        self._page.keyboard.press(key)

    def browser_type(self, selector: str, text: str):
        """Type text into the focused element."""
        self._page.type(selector, text)

    def browser_click(self, selector: str):
        """Click on a selector."""
        self._page.click(selector, timeout=3000, strict=True)

    def browser_drag(self, from_selector: str, to_selector: str):
        """Drag and drop from one selector to another."""
        from_elem = self._page.locator(from_selector)
        from_elem.hover(timeout=500)
        self._page.mouse.down()

        to_elem = self._page.locator(to_selector)
        to_elem.hover(timeout=500)
        self._page.mouse.up()

    def browser_hover(self, selector: str):
        """Hover over a given element."""
        self._page.hover(selector, timeout=3000, strict=True)

    def browser_select_option(self, selector: str, value: str):
        """Select an option from a given element."""
        self._page.select_option(selector, value)

    def browser_mouse_click_xy(self, x: int, y: int):
        """Click at a given x, y coordinate using the mouse."""
        self._page.mouse.click(x, y, delay=100)

    def browser_wait(self, seconds: int):
        """Wait for a given number of seconds, up to 10 seconds."""
        time.sleep(min(seconds, 10))

    def evaluate_js(self, js: str):
        js_result = self._page.evaluate(js)
        logger.info(f"JS result: {js_result}")
        return js_result

    def goto(self, url: str):
        """Navigate to a specified URL."""
        self._page.goto(url)

    def browser_back(self):
        """Navigate back in browser history."""
        self._page.go_back()

    def browser_forward(self):
        """Navigate forward in browser history."""
        self._page.go_forward()

    def page_html(self) -> str:
        return self._page.content()

    def page_screenshot(self) -> Image.Image:
        scr_bytes = self._page.screenshot()
        return Image.open(BytesIO(scr_bytes))

    def page_axtree(self) -> str:
        axtree = self._page.accessibility.snapshot()
        return flatten_axtree(axtree)

    def step(self, action: ToolCall) -> dict:
        fn = self._actions[action.name]
        try:
            action_result = fn(**action.arguments)
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.error(action_result)
        html = self.page_html()
        screenshot = self.page_screenshot()
        axtree = self.page_axtree()
        return {
            "action_result": action_result,
            "html": html,
            "axtree_txt": axtree,
            "screenshot": screenshot,
        }

    def actions(self) -> list[ToolSpec]:
        return [ToolSpec.from_function(fn) for fn in self._actions.values()]

    def close(self):
        self._page.close()


_apw = None  # Global Playwright instance for AsyncPlaywright
_abrowser = None  # Global Browser instance for AsyncPlaywright


class AsyncPlaywright(AsyncBrowserBackend):
    """Fully asynchronous Playwright backend using playwright.async_api."""

    has_pw_page: bool = False
    _actions: dict[str, Callable]
    _page: AsyncPage

    def model_post_init(self, __context: Any):
        self._actions = {
            "browser_press_key": self.browser_press_key,
            "browser_type": self.browser_type,
            "browser_click": self.browser_click,
            "browser_drag": self.browser_drag,
            "browser_hover": self.browser_hover,
            "browser_select_option": self.browser_select_option,
            "browser_mouse_click_xy": self.browser_mouse_click_xy,
        }

    async def initialize(self):
        global _apw, _abrowser
        if _apw is None:
            _apw = await async_playwright().start()
        if _abrowser is None:
            _abrowser = await _apw.chromium.launch(headless=False, chromium_sandbox=True)
        self._page = await _abrowser.new_page()

    async def browser_press_key(self, key: str):
        """Press a key on the keyboard."""
        await self._page.keyboard.press(key)

    async def browser_type(self, selector: str, text: str):
        """Type text into the focused element."""
        await self._page.type(selector, text)

    async def browser_click(self, selector: str):
        """Click on a selector."""
        await self._page.click(selector, timeout=3000, strict=True)

    async def browser_drag(self, from_selector: str, to_selector: str):
        """Drag and drop from one selector to another."""
        from_elem = self._page.locator(from_selector)
        await from_elem.hover(timeout=500)
        await self._page.mouse.down()

        to_elem = self._page.locator(to_selector)
        await to_elem.hover(timeout=500)
        await self._page.mouse.up()

    async def browser_hover(self, selector: str):
        """Hover over a given element."""
        await self._page.hover(selector, timeout=3000, strict=True)

    async def browser_select_option(self, selector: str, value: str):
        """Select an option from a given element."""
        await self._page.select_option(selector, value)

    async def browser_mouse_click_xy(self, x: int, y: int):
        """Click at a given x, y coordinate using the mouse."""
        await self._page.mouse.click(x, y, delay=100)

    async def evaluate_js(self, js: str):
        js_result = await self._page.evaluate(js)
        logger.info(f"JS result: {js_result}")
        return js_result

    async def goto(self, url: str):
        await self._page.goto(url)

    async def page_html(self) -> str:
        return await self._page.content()

    async def page_screenshot(self) -> Image.Image:
        scr_bytes = await self._page.screenshot()
        return Image.open(BytesIO(scr_bytes))

    async def page_axtree(self) -> str:
        axtree = await self._page.accessibility.snapshot()
        return flatten_axtree(axtree)

    async def step(self, action: ToolCall) -> dict:
        fn = self._actions[action.name]
        try:
            action_result = await fn(**action.arguments)
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.error(action_result)
        html = await self.page_html()
        screenshot = await self.page_screenshot()
        axtree = await self.page_axtree()
        return {
            "action_result": action_result,
            "html": html,
            "axtree_txt": axtree,
            "screenshot": screenshot,
        }

    def actions(self) -> list[ToolSpec]:
        return [ToolSpec.from_function(fn) for fn in self._actions.values()]

    async def close(self):
        await self._page.close()


def flatten_axtree(axtree_dict: dict | None) -> str:
    """
    Traverses accessibility tree dictionary and returns its markdown view.

    Args:
        axtree_dict: Accessibility tree from playwright page.accessibility.snapshot()
                     Structure: dict with 'role', 'name', 'value', 'children' keys

    Returns:
        String representation of the accessibility tree in markdown format
    """
    if axtree_dict is None:
        return ""

    def traverse_node(node: dict, depth: int = 0) -> list[str]:
        """Recursively traverse the accessibility tree and build markdown lines."""
        lines = []
        indent = "  " * depth  # 2 spaces per indent level

        # Extract node information
        role = node.get("role", "")
        name = node.get("name", "")
        value = node.get("value", "")

        # Build the node representation
        parts = []
        if role:
            parts.append(f"{role}:")
        if name.strip():
            parts.append(f"{name}")
        if value:
            parts.append(f"[value: {value}]")

        # Only add line if there's meaningful content
        if parts:
            line = f"{indent}{' '.join(parts)}"
            lines.append(line)

        # Recursively process children
        children = node.get("children", [])
        for child in children:
            child_lines = traverse_node(child, depth + 1)
            lines.extend(child_lines)

        return lines

    # Start traversal from root
    all_lines = traverse_node(axtree_dict, depth=0)
    return "\n".join(all_lines)
