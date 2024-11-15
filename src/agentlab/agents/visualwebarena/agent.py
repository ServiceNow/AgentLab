import base64
import dataclasses
import io
import re
from io import BytesIO

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import Agent, AgentInfo
from browsergym.utils.obs import flatten_axtree_to_str, overlay_som
from PIL import Image

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_utils import Discussion, HumanMessage, ParseError, SystemMessage, retry


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def b64_to_pil(img_b64: str) -> str:
    if not img_b64.startswith("data:image/png;base64,"):
        raise ValueError(f"Unexpected base64 encoding: {img_b64}")
    img_b64 = img_b64.removeprefix("data:image/png;base64,")
    img_data = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_data))
    return img


class VWAAgent(Agent):
    """
    Re-implementation of the web agent from VisualWebArena.
    Credits to Lawrence Jang (@ljang0)
    https://github.com/web-arena-x/visualwebarena/blob/main/agent/agent.py
    """

    action_set = HighLevelActionSet(
        subsets=["chat", "bid", "infeas", "nav", "tab"],
        strict=False,
        multiaction=False,
        demo_mode="off",
    )

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "axtree_txt": flatten_axtree_to_str(
                obs["axtree_object"], obs["extra_element_properties"]
            ),
            "extra_properties": obs["extra_element_properties"],
            "url": obs["url"],
            "screenshot": obs["screenshot"],
        }

    def __init__(self, chat_model_args: BaseModelArgs, n_retry: int) -> None:
        super().__init__()
        self.model_name = chat_model_args.model_name
        self.chat_llm = chat_model_args.make_model()
        self.n_retry = n_retry

        self.goal_images = None

    def get_action(self, obs: dict) -> tuple[str, dict]:

        system_prompt = f"""\
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions."""

        prompt = Discussion(SystemMessage(system_prompt))

        prompt.append(
            HumanMessage(
                f"""\
# Goal:
"""
            )
        )
        for goal in obs["goal_object"]:
            prompt.add_content(goal["type"], goal[goal["type"]])

        prompt.add_text(
            f"""
# Current Accessibility Tree:
{obs["axtree_txt"]}

# Action Space
{self.action_set.describe(with_long_description=False, with_examples=True)}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"

If you have completed the task, use the chat to return an answer. For example, if you are asked what is the color of the sky, return
"
```send_msg_to_user("blue")```
"
"""
        )

        prompt.add_text("IMAGES: current page screenshot")
        prompt.add_image(
            pil_to_b64(Image.fromarray(overlay_som(obs["screenshot"], obs["extra_properties"])))
        )

        def parser(response: str) -> tuple[dict, bool, str]:
            pattern = r"```((.|\\n)*?)```"
            match = re.search(pattern, response)
            if not match:
                raise ParseError("No code block found in the response")
            action = match.group(1).strip()
            thought = response
            return {"action": action, "think": thought}

        response = retry(self.chat_llm, prompt, n_retry=self.n_retry, parser=parser)

        action = response.get("action", None)
        stats = self.chat_llm.get_stats()
        return action, AgentInfo(
            chat_messages=prompt.to_markdown(),
            think=response.get("think", None),
            stats=stats,
        )


@dataclasses.dataclass
class VWAAgentArgs(AgentArgs):
    """
    This class is meant to store the arguments that define the agent.

    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    agent_name: str = "vwa"
    temperature: float = 0.1
    chat_model_args: BaseModelArgs = None

    def make_agent(self):
        return VWAAgent(chat_model_args=self.chat_model_args, n_retry=3)


CONFIG = VWAAgentArgs(CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"])


def main():
    from pathlib import Path

    from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result

    exp_args = ExpArgs(
        agent_args=VWAAgentArgs(model_name="gpt-4-1106-preview"),
        env_args=EnvArgs(
            task_name="visualwebarena.423",
            task_seed=42,
            headless=False,  # shows the browser
        ),
    )
    exp_args.prepare(exp_root=Path("./results"))
    exp_args.run()
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
