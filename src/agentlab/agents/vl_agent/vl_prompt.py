from abc import ABC, abstractmethod
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage
from dataclasses import dataclass
from PIL import Image
from typing import Optional, Union


class VLPrompt(ABC):
    @abstractmethod
    def get_message(self) -> HumanMessage:
        raise NotImplementedError

    @abstractmethod
    def answer_parser(self, answer_text: str) -> dict:
        raise NotImplementedError


@dataclass
class VLPromptArgs(ABC):
    prompt_name: str

    @abstractmethod
    def make_prompt(
        self, obs_history: list[dict], actions: list[str], thoughts: list[str]
    ) -> VLPrompt:
        raise NotImplementedError


@dataclass
class UIPrompt(VLPrompt):
    instructions: Union[dp.ChatInstructions, dp.GoalInstructions]
    screenshot: Optional[Image.Image]
    observation: dp.Observation
    history: dp.History
    think: dp.Think
    action_prompt: dp.ActionPrompt
    abstract_example: Optional[str]
    concrete_example: Optional[str]

    def get_message(self) -> HumanMessage:
        message = HumanMessage(self.instructions.prompt)
        if self.screenshot is not None:
            message.add_text("# Screenshot:\n")
            message.add_image(self.screenshot)
        message.add_text(self.observation.prompt)
        message.add_text(self.history.prompt)
        message.add_text(self.think.prompt)
        message.add_text(self.action_prompt.prompt)
        if self.abstract_example is not None:
            message.add_text(self.abstract_example)
        if self.concrete_example is not None:
            message.add_text(self.concrete_example)
        return message

    def answer_parser(self, answer_text: str) -> dict:
        answer_dict = {}
        answer_dict.update(self.think.parse_answer(answer_text))
        answer_dict.update(self.action_prompt.parse_answer(answer_text))
        return answer_dict


@dataclass
class UIPromptArgs(VLPromptArgs):
    obs_flags: dp.ObsFlags
    action_flags: dp.ActionFlags
    extra_instructions: Optional[str]
    enable_chat: bool
    use_thinking: bool
    use_abstract_example: bool
    use_concrete_example: bool

    def make_prompt(
        self, obs_history: list[dict], actions: list[str], thoughts: list[str]
    ) -> UIPrompt:
        if self.enable_chat:
            instructions = dp.ChatInstructions(
                chat_messages=obs_history[-1]["chat_messages"],
                extra_instructions=self.extra_instructions,
            )
        else:
            instructions = dp.GoalInstructions(
                goal_object=obs_history[-1]["goal_object"],
                extra_instructions=self.extra_instructions,
            )
        if self.obs_flags.use_screenshot:
            if self.obs_flags.use_som:
                screenshot = obs_history[-1]["screenshot_som"]
            else:
                screenshot = obs_history[-1]["screenshot"]
        else:
            screenshot = None
        observation = dp.Observation(obs=obs_history[-1], flags=self.obs_flags)
        history = dp.History(
            history_obs=obs_history,
            actions=actions,
            memories=None,
            thoughts=thoughts,
            flags=self.obs_flags,
        )
        think = dp.Think(visible=self.use_thinking)
        action_prompt = dp.ActionPrompt(
            action_set=self.action_flags.action_set.make_action_set(),
            action_flags=self.action_flags,
        )
        if self.use_abstract_example:
            abstract_example = (
                f"# Abstract Example:\n{think.abstract_ex}{action_prompt.abstract_ex}"
            )
        else:
            abstract_example = None
        if self.use_concrete_example:
            concrete_example = (
                f"# Concrete Example:\n{think.concrete_ex}{action_prompt.concrete_ex}"
            )
        else:
            concrete_example = None

        return UIPrompt(
            instructions=instructions,
            screenshot=screenshot,
            observation=observation,
            history=history,
            think=think,
            action_prompt=action_prompt,
            abstract_example=abstract_example,
            concrete_example=concrete_example,
        )
