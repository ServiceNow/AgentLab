import logging
from typing import ClassVar

from browsergym.utils.obs import prune_html
from browsergym.workarena.instance import SNowInstance
from browsergym.workarena.tasks.base import AbstractServiceNowTask
from pydantic import ConfigDict

from agentlab.backends.browser import BrowserBackend
from agentlab.benchmarks.web_task import AbstractWebTask

logger = logging.getLogger(__name__)


class WorkarenaTask(AbstractWebTask):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: str = "workarena"
    level: str
    task_cls: type[AbstractServiceNowTask]
    seed: int
    instance: SNowInstance
    _task_obj: AbstractServiceNowTask = None  # type: ignore
    actions_whitelist: ClassVar[list[str]] = [
        "browser_press_key",
        "browser_type",
        "browser_click",
        "browser_drag",
        "browser_hover",
        "browser_select_option",
        "browser_mouse_click_xy",
        "browser_wait",
    ]

    def setup(self, backend: BrowserBackend) -> tuple[str, dict]:
        if not backend.has_pw_page:
            raise ValueError("Workarena task requires a backend with playwright page access.")
        self._backend = backend
        self._task_obj = self.task_cls(instance=self.instance, seed=self.seed) # type: ignore
        self.url = self._task_obj.start_url
        goal, info = self._task_obj.setup(backend.page)
        logger.info(f"Current backend page URL: {backend.page.url}")
        # backend.goto(self.url)
        return goal, info

    def teardown(self) -> None:
        self._task_obj.teardown()

    def validate(self) -> tuple[float, dict]:
        reward, done, _, info = self._task_obj.validate(page=self._backend.page, chat_messages=[])
        info["done"] = done
        return reward, info

    def obs_postprocess(self, obs: dict) -> dict:
        html = obs.pop("html", "")
        obs["pruned_html"] = prune_html(html)
        return obs