"""
Prompt builder for GenericAgent

It is based on the dynamic_prompting module from the agentlab package.
"""

import logging
from dataclasses import dataclass

from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import BaseMessage, HumanMessage, image_to_jpg_base64_url


@dataclass
class PromptFlags(dp.Flags):
    """
    A class to represent various flags used to control features in an application.

    Attributes:
        use_criticise (bool): Ask the LLM to first draft and criticise the action before producing it.
        use_thinking (bool): Enable a chain of thoughts.
        use_concrete_example (bool): Use a concrete example of the answer in the prompt for a generic task.
        use_abstract_example (bool): Use an abstract example of the answer in the prompt.
        use_hints (bool): Add some human-engineered hints to the prompt.
        enable_chat (bool): Enable chat mode, where the agent can interact with the user.
        max_prompt_tokens (int): Maximum number of tokens allowed in the prompt.
        be_cautious (bool): Instruct the agent to be cautious about its actions.
        extra_instructions (Optional[str]): Extra instructions to provide to the agent.
        add_missparsed_messages (bool): When retrying, add the missparsed messages to the prompt.
        flag_group (Optional[str]): Group of flags used.
    """

    obs: dp.ObsFlags
    action: dp.ActionFlags
    use_criticise: bool = False  #
    use_thinking: bool = False
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    max_trunc_itr: int = 20
    flag_group: str = None


class SystemPrompt(dp.PromptElement):
    _prompt = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You can interact with the page and explore, and send messages to the user. Each time you
submit an action it will be sent to the browser and you will receive a new page."""


def make_instructions(obs: dict, from_chat: bool, extra_instructions: str | None):
    """Convenient wrapper to extract instructions from either goal or chat"""
    if from_chat:
        instructions = dp.ChatInstructions(
            obs["chat_messages"], extra_instructions=extra_instructions
        )
    else:
        if sum([msg["role"] == "user" for msg in obs.get("chat_messages", [])]) > 1:
            logging.warning(
                "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
            )
        instructions = dp.GoalInstructions(
            obs["goal_object"], extra_instructions=extra_instructions
        )
    return instructions


class History(dp.PromptElement):
    """
    Format the actions and thoughts of previous steps."""

    def __init__(self, actions, thoughts) -> None:

        prompt_elements = []
        for i, (action, thought) in enumerate(zip(actions, thoughts)):
            prompt_elements.append(
                f"""
## Step {i}
### Thoughts:
{thought}
### Action:
{action}
"""
            )
        self._prompt = "\n".join(prompt_elements) + "\n"


class Observation(dp.PromptElement):
    """Observation of the current step.

    Contains the html, the accessibility tree and the error logs.
    """

    def __init__(self, obs, flags: dp.ObsFlags) -> None:
        super().__init__()
        self.flags = flags
        self.obs = obs

        # for a multi-tab browser, we need to show the current tab
        self.tabs = dp.Tabs(
            obs,
            visible=lambda: flags.use_tabs,
            prefix="## ",
        )

        # if an error is present, we need to show it
        self.error = dp.Error(
            obs["last_action_error"],
            visible=lambda: flags.use_error_logs and obs["last_action_error"],
            prefix="## ",
        )

    @property
    def _prompt(self) -> str:
        return f"""
# Observation of current step:
{self.tabs.prompt}{self.focused_element.prompt}{self.error.prompt}

"""

    def add_screenshot(self, prompt: BaseMessage) -> BaseMessage:
        if self.flags.use_screenshot:
            if self.flags.use_som:
                screenshot = self.obs["screenshot_som"]
                prompt.add_text(
                    "\n## Screenshot:\nHere is a screenshot of the page, it is annotated with bounding boxes and corresponding bids:"
                )
            else:
                screenshot = self.obs["screenshot"]
                prompt.add_text("\n## Screenshot:\nHere is a screenshot of the page:")
            img_url = image_to_jpg_base64_url(screenshot)
            prompt.add_image(img_url, detail=self.flags.openai_vision_detail)
        return prompt


class MainPrompt(dp.PromptElement):

    def __init__(
        self,
        action_set: AbstractActionSet,
        obs: dict,
        actions: list[str],
        thoughts: list[str],
        flags: PromptFlags,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = History(obs, actions, thoughts)
        self.instructions = make_instructions(obs, flags.enable_chat, flags.extra_instructions)
        self.obs = dp.Observation(
            obs,
            self.flags.obs,
        )

        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)
        self.think = dp.Think(visible=lambda: flags.use_thinking)

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.think.prompt}\
"""
        )

        if self.flags.use_abstract_example:
            prompt.add_text(
                f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.think.abstract_ex}\
{self.action_prompt.abstract_ex}\
"""
            )

        if self.flags.use_concrete_example:
            prompt.add_text(
                f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.think.concrete_ex}\
{self.action_prompt.concrete_ex}\
"""
            )
        return self.obs.add_screenshot(prompt)

    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.think.parse_answer(text_answer))
        ans_dict.update(self.action_prompt.parse_answer(text_answer))
        return ans_dict
