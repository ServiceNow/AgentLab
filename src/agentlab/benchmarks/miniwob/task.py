
import os
from typing import Any

from browsergym.miniwob import ALL_MINIWOB_TASKS

from agentlab.benchmarks.web_task import AbstractWebTask


class MiniWobTask(AbstractWebTask):
    dataset: str = "miniwob"
    task_id: str
    desc: str
    subdomain: str
    base_url: str = None
    url: str = None
    remove_human_display: bool = True
    episode_max_time: int = 1000000

    def model_post_init(self, __context: Any):
        self.url = f"{self.base_url}/{self.subdomain}.html"
        
    def get_setup_js(self) -> str:
        if self.remove_human_display:
            js = r"""
let __display_ids = ['reward-display', 'click-canvas', 'sync-task-cover'];
let __display_divs = {};
let __query_div_hidden_copy = null;

removeDisplay = function() {
  core.clearTimer();
  document.body.removeEventListener('click', core.canvasDrawClick);

  __query_div_hidden_copy = document.getElementById('query').cloneNode(true);
  document.getElementById('query').innerHTML = '';

  for (i in __display_ids) {
    elem_id = __display_ids[i];
    elem = document.getElementById(elem_id);
    // remove elem from the document
    elem.remove();
    // but keep it stored somewhere to bring back later
    __display_divs[elem_id] = elem;
  }
};

bringBackDisplay = function() {
  document.getElementById('query').innerHTML = __query_div_hidden_copy.innerHTML;
  for (var elem_id in __display_divs){
    document.body.appendChild(__display_divs[elem_id]);
  }
  core.createDisplay();
};

core.endEpisode_legacy = core.endEpisode;
core.startEpisodeReal_legacy = core.startEpisodeReal;
core.getUtterance_legacy = core.getUtterance;

core.getUtterance = function () {
  bringBackDisplay();
  utterance = core.getUtterance_legacy();
  removeDisplay();
  return utterance;
};

core.endEpisode = function(reward, time_proportional, reason){
  bringBackDisplay();
  core.endEpisode_legacy(reward, time_proportional, reason);
  removeDisplay();
};

core.startEpisodeReal = function() {
  bringBackDisplay();
  core.startEpisodeReal_legacy();
  removeDisplay();
};

removeDisplay();
"""
        else:
            js = ""
        js += f"""
Math.seedrandom(42);
core.EPISODE_MAX_TIME = {self.episode_max_time};
core.startEpisodeReal();
while (!WOB_TASK_READY) {{
  await new Promise(resolve => setTimeout(resolve, 100));
}}
return core.getUtterance();
    """
        return f"async () => {{{js}}}"

    def get_teardown_js(self) -> str:
        return ""

    def get_step_validate_js(self) -> str:
        return """() => {
return [WOB_REWARD_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON, WOB_DONE_GLOBAL, WOB_EPISODE_ID, WOB_TASK_READY];
}"""

    def get_task_validate_js(self) -> str:
        return """() => {
return [WOB_REWARD_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON, WOB_DONE_GLOBAL, WOB_EPISODE_ID, WOB_TASK_READY];
}"""


    def parse_validation_result(self, validation_result: str) -> tuple[float, dict]:
        chunks = [c.strip() for c in validation_result.split(",")]
        raw_reward = float(chunks[1])
        done = bool(chunks[3])
        reward = float(raw_reward > 0)
        return reward, {
          "raw_reward": raw_reward,
          "reward_reason": chunks[2],
          "done": done,
        }

def get_miniwob_tasks(base_url: str | None = None, remove_human_display: bool = True, episode_max_time: int = 1000000) -> list[MiniWobTask]:
    if base_url is None:
        base_url = os.environ.get("MINIWOB_URL")
        if base_url is None:
            raise ValueError("MINIWOB_URL environment variable is not set")
    return [
        MiniWobTask(
            task_id=task.subdomain, 
            desc=task.desc, 
            subdomain=task.subdomain, 
            base_url=base_url, 
            remove_human_display=remove_human_display, 
            episode_max_time=episode_max_time) for task in ALL_MINIWOB_TASKS
    ]