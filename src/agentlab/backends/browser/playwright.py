from playwright.sync_api import sync_playwright

from agentlab.backends.browser.base import BrowserBackend, ToolCallAction


class PlaywrightSyncBackend(BrowserBackend):
    def __init__(self):
        self.actions = {
            "browser_press_key": lambda key: self.page.keyboard.press(key),
            "browser_type": lambda text: self.page.type(text),
            "browser_click": lambda selector: self.page.click(selector),
            "browser_drag": lambda from_selector, to_selector: self.drag_and_drop(
                from_selector, to_selector
            ),
            "browser_hover": lambda selector: self.page.hover(selector),
            "browser_select_option": lambda selector: self.page.select_option(selector),
            "browser_mouse_click_xy": lambda x, y: self.page.mouse.click(x, y),
        }

    def drag_and_drop(self, from_selector: str, to_selector: str):
        from_elem = self.page.locator(from_selector)
        from_elem.hover(timeout=500)
        self.page.mouse.down()

        to_elem = self.page.locator(to_selector)
        to_elem.hover(timeout=500)
        self.page.mouse.up()

    def initialize(self):
        self.browser = sync_playwright().start().chromium.launch(headless=True)
        self.page = self.browser.new_page()

    def run_js(self, js: str):
        return self.page.evaluate(js)

    def goto(self, url: str):
        self.page.goto(url)

    def page_snapshot(self):
        return self.page.content()

    def page_screenshot(self):
        return self.page.screenshot()

    def step(self, action: ToolCallAction):
        fn = self.actions[action.function.name]
        return fn(**action.function.arguments)

    def actions(self):
        return self.page.actions()

    def close(self):
        self.browser.close()
