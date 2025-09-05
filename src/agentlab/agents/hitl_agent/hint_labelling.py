import json
import logging
from importlib import resources
from queue import Queue
from typing import Dict, List, Optional

import playwright.sync_api
from browsergym.core import _get_global_playwright
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HINT_LABELING_DIR = resources.files("agentlab.agents.hitl_agent.hint_labelling_ui_files")


class HintLabelingInputs(BaseModel):
    goal: str
    error_feedback: str = ""
    screenshot: str  # base64 screenshot (original/current)
    screenshots: List[str] = Field(default_factory=list)  # list of base64 screenshots for hover
    axtree: str
    hints: List[str] = Field(default_factory=list)
    suggestions: List[Dict[str, str]] = Field(default_factory=list)


class HintLabeling:
    def __init__(self, headless: bool, *args, **kwargs):
        pw_opt = _get_global_playwright()
        pw: playwright.sync_api.Playwright = pw_opt  # type: ignore[assignment]
        self.browser = pw.chromium.launch(headless=headless)
        self.context = self.browser.new_context(
            no_viewport=True,
        )
        self.page = self.context.new_page()
        self._resp_queue = Queue()

        self.page.route("**/api/reprompt", self._route_reprompt)
        self.page.route("**/api/submit", self._route_submit)
        self.page.set_content(get_hint_labeling_ui(HINT_LABELING_DIR))

        # internal state
        self._context = None
        self._running = False

    def _route_reprompt(
        self, route: playwright.sync_api.Route, request: playwright.sync_api.Request
    ):
        logger.info("Route hit: %s %s", request.method, request.url)
        try:
            body = json.loads(request.post_data or "{}")
        except Exception:
            body = {}
        # enqueue output 1 (reprompt)
        hints = body.get("hints")
        if not isinstance(hints, list):
            # Back-compat: accept single 'hint' string
            h = body.get("hint")
            hints = [h] if isinstance(h, str) and h.strip() else []
        msg = {"type": "reprompt", "payload": {"hints": hints}}
        self._resp_queue.put(msg)
        # Respond something minimal so UI doesn’t break; it will be refreshed by a later update_context()
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"suggestions": []}),
        )

    def _route_submit(self, route: playwright.sync_api.Route, request: playwright.sync_api.Request):
        logger.info("Route hit: %s %s", request.method, request.url)
        try:
            body = json.loads(request.post_data or "{}")
        except Exception:
            body = {}
        # Map UI payload -> your step shape
        msg = {
            "type": "step",
            "payload": {
                "think": body.get("think", ""),
                "action": body.get("action", ""),
            },
        }
        self._resp_queue.put(msg)
        # UI expects 200 JSON; we can optionally send new suggestions here too.
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"suggestions": []}),
        )

    def _to_ui_bootstrap(self, ctx: HintLabelingInputs) -> dict:
        return {
            "goal": ctx.goal,
            "error_feedback": ctx.error_feedback,
            "screenshot": ctx.screenshot,
            "screenshots": ctx.screenshots,  # list of screenshots for hover
            "axtree": ctx.axtree,
            "hints": ctx.hints,
            "suggestions": ctx.suggestions,
        }

    def update_context(self, context: HintLabelingInputs):
        self._context = context
        ui_payload = self._to_ui_bootstrap(context)
        # call JS function with arg (no string concat)
        self.page.evaluate("(d) => updateContext(d)", ui_payload)

    def wait_for_response(self, timeout: Optional[float] = 600) -> dict:
        """
        Wait until the page makes a request to /api/reprompt or /api/submit,
        then parse the request body and return it in your schema.

        Args:
            timeout (Optional[float]): Maximum time to wait for the request in seconds. If None or 0,
                waits indefinitely. Defaults to 600 seconds.

        Returns:
            dict: A dictionary containing the parsed response with 'type' and 'payload' keys.
                For /api/reprompt: {'type': 'reprompt', 'payload': {'hints': list[str]}}
                For /api/submit: {'type': 'step', 'payload': {'think': str, 'action': str}}

        """
        logger.info("Waiting for response from Hint Labeling UI...")

        def is_api(req: playwright.sync_api.Request) -> bool:
            u = req.url
            return (
                u.endswith("/api/reprompt") or u.endswith("/api/submit")
            ) and req.method == "POST"

        # This pumps Playwright internally; no busy waiting.
        with self.page.expect_request(
            is_api, timeout=(timeout * 1000 if timeout else 0)
        ) as req_info:
            req = req_info.value

        body_text = req.post_data or "{}"
        try:
            body = json.loads(body_text)
        except Exception as e:
            print("JSON parse error:", e)
            body = {}

        if req.url.endswith("/api/reprompt"):
            hints = body.get("hints")
            if not isinstance(hints, list):
                h = body.get("hint")
                hints = [h] if isinstance(h, str) and h.strip() else []
            msg = {"type": "reprompt", "payload": {"hints": hints}}
        else:
            msg = {
                "type": "step",
                "payload": {"think": body.get("think", ""), "action": body.get("action", "")},
            }

        logger.info("Response received: %s", msg)
        return msg

    def close(self):
        self.context.close()
        self.browser.close()


def get_hint_labeling_ui(hint_labeling_dir) -> str:
    with open(hint_labeling_dir / "hint_labeling_ui.html", "r") as file:
        hint_labeling_html = file.read()
    return hint_labeling_html
