import logging
from io import BytesIO
from typing import Any, Callable

from PIL import Image
from playwright.sync_api import Page, sync_playwright

from agentlab.backends.browser.base import BrowserBackend, ToolCallAction, ToolSpec

logger = logging.getLogger(__name__)


class PlaywrightSyncBackend(BrowserBackend):
    _actions: dict[str, Callable]
    _browser: Any
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

    def browser_press_key(self, key: str):
        """
        Press a key on the keyboard.
        """
        self._page.keyboard.press(key)

    def browser_type(self, text: str):
        """
        Type text into the focused element.
        """
        self._page.type(text)

    def browser_click(self, selector: str):
        """
        Click on a selector.
        """
        self._page.click(selector)

    def browser_drag(self, from_selector: str, to_selector: str):
        """
        Drag and drop from one selector to another.
        """
        from_elem = self._page.locator(from_selector)
        from_elem.hover(timeout=500)
        self._page.mouse.down()

        to_elem = self._page.locator(to_selector)
        to_elem.hover(timeout=500)
        self._page.mouse.up()

    def browser_hover(self, selector: str):
        """
        Hover over a given element.
        """
        self._page.hover(selector)

    def browser_select_option(self, selector: str):
        """
        Select an option from a given element.
        """
        self._page.select_option(selector)

    def browser_mouse_click_xy(self, x: int, y: int):
        """
        Click at a given x, y coordinate using the mouse.
        """
        self._page.mouse.click(x, y)

    def initialize(self):
        self._browser = sync_playwright().start().chromium.launch(headless=True, chromium_sandbox=True)
        self._page = self._browser.new_page()

    def run_js(self, js: str):
        js_result = self._page.evaluate(js)
        logger.info(f"JS result: {js_result}")
        return js_result

    def goto(self, url: str):
        self._page.goto(url)

    def page_snapshot(self):
        return self._page.content()

    def page_screenshot(self):
        scr_bytes = self._page.screenshot()
        return Image.open(BytesIO(scr_bytes))

    def step(self, action: ToolCallAction):
        fn = self._actions[action.function.name]
        action_result = fn(**action.function.arguments)
        snapshot = self.page_snapshot()
        screenshot = self.page_screenshot()
        return {
            "pruned_html": f"{action_result or ''}\n{snapshot}",
            "axtree_txt": snapshot,
            "screenshot": screenshot,
        }
    def actions(self) -> tuple[ToolSpec]:
        specs = [ToolSpec.from_function(fn) for fn in self._actions.values()]
        return tuple(specs)

    def close(self):
        self._browser.close()
