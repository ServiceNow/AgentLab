import asyncio
import logging
from io import BytesIO
from typing import Any, Callable

from PIL import Image
from playwright.async_api import Browser, Page, async_playwright

from agentlab.actions import ToolCallAction, ToolSpec
from agentlab.backends.browser.base import BrowserBackend

logger = logging.getLogger(__name__)


class AsyncPlaywright(BrowserBackend):
    _actions: dict[str, Callable]
    _loop: asyncio.AbstractEventLoop
    _browser: Browser
    _page: Page

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

    def initialize(self, loop: asyncio.AbstractEventLoop | None = None):
        self._loop = loop or asyncio.get_event_loop()
        self._loop.run_until_complete(self.ainitialize())

    async def ainitialize(self):
        pw = await async_playwright().start()
        self._browser = await pw.chromium.launch(headless=True, chromium_sandbox=True)
        self._page = await self._browser.new_page()

    async def browser_press_key(self, key: str):
        """
        Press a key on the keyboard.
        """
        await self._page.keyboard.press(key)

    async def browser_type(self, selector: str, text: str):
        """
        Type text into the focused element.
        """
        await self._page.type(selector, text)

    async def browser_click(self, selector: str):
        """
        Click on a selector.
        """
        await self._page.click(selector, timeout=3000, strict=True)

    async def browser_drag(self, from_selector: str, to_selector: str):
        """
        Drag and drop from one selector to another.
        """
        from_elem = self._page.locator(from_selector)
        await from_elem.hover(timeout=500)
        await self._page.mouse.down()

        to_elem = self._page.locator(to_selector)
        await to_elem.hover(timeout=500)
        await self._page.mouse.up()

    async def browser_hover(self, selector: str):
        """
        Hover over a given element.
        """
        await self._page.hover(selector, timeout=3000, strict=True)

    async def browser_select_option(self, selector: str, value: str):
        """
        Select an option from a given element.
        """
        await self._page.select_option(selector, value)

    async def browser_mouse_click_xy(self, x: int, y: int):
        """
        Click at a given x, y coordinate using the mouse.
        """
        await self._page.mouse.click(x, y, delay=100)

    def run_js(self, js: str):
        js_result = self._loop.run_until_complete(self._page.evaluate(js))
        logger.info(f"JS result: {js_result}")
        return js_result

    def goto(self, url: str):
        self._loop.run_until_complete(self._page.goto(url))

    def page_html(self):
        return self._loop.run_until_complete(self._page.content())

    def page_screenshot(self):
        scr_bytes = self._loop.run_until_complete(self._page.screenshot())
        return Image.open(BytesIO(scr_bytes))

    def page_axtree(self):
        axtree = self._loop.run_until_complete(self._page.accessibility.snapshot())
        flat_axtree = flatten_axtree(axtree)
        return flat_axtree

    def step(self, action: ToolCallAction):
        fn = self._actions[action.function.name]
        try:
            action_result = self._loop.run_until_complete(fn(**action.function.arguments))
        except Exception as e:
            logger.error(f"Error executing action {action.function.name}: {e}")
            action_result = f"Error executing action {action.function.name}: {e}"
        html = self.page_html()
        screenshot = self.page_screenshot()
        axtree = self.page_axtree()
        return {
            "tool_result": action_result,
            "pruned_html": html,
            "axtree_txt": axtree,
            "screenshot": screenshot,
        }

    def actions(self) -> tuple[ToolSpec]:
        specs = [ToolSpec.from_function(fn) for fn in self._actions.values()]
        return tuple(specs)

    def close(self):
        self._loop.run_until_complete(self._browser.close())


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
