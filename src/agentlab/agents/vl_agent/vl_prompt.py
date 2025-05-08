from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage
from browsergym.core.action.base import AbstractActionSet
from dataclasses import dataclass
from typing import Optional


@dataclass
class VLPromptFlags(dp.Flags):
    obs_flags: dp.ObsFlags
    action_flags: dp.ActionFlags
    use_thinking: bool
    use_concrete_example: bool
    use_abstract_example: bool
    enable_chat: bool
    extra_instructions: Optional[str]


class VLPrompt(dp.PromptElement):
    def __init__(
        self,
        vl_prompt_flags: VLPromptFlags,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        thoughts: list[str],
        actions: list[str],
    ):
        super().__init__()
        self.vl_prompt_flags = vl_prompt_flags
        self.obs_history = obs_history
        if self.vl_prompt_flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                chat_messages=self.obs_history[-1]["chat_messages"],
                extra_instructions=self.vl_prompt_flags.extra_instructions,
            )
        else:
            self.instructions = dp.GoalInstructions(
                goal_object=self.obs_history[-1]["goal_object"],
                extra_instructions=self.vl_prompt_flags.extra_instructions,
            )
        self.observation = dp.Observation(
            obs=self.obs_history[-1], flags=self.vl_prompt_flags.obs_flags
        )
        self.history = dp.History(
            history_obs=self.obs_history,
            actions=actions,
            memories=None,
            thoughts=thoughts,
            flags=self.vl_prompt_flags.obs_flags,
        )
        self.think = dp.Think(visible=self.vl_prompt_flags.use_thinking)
        self.action_prompt = dp.ActionPrompt(
            action_set=action_set, action_flags=self.vl_prompt_flags.action_flags
        )
        self._prompt = f"{self.instructions.prompt}\n{self.observation.prompt}\n{self.history.prompt}\n{self.think.prompt}\n{self.action_prompt.prompt}\n"
        if self.vl_prompt_flags.use_abstract_example:
            self._prompt += (
                f"# Abstract Example:\n{self.think.abstract_ex}\n{self.action_prompt.abstract_ex}\n"
            )
        if self.vl_prompt_flags.use_concrete_example:
            self._prompt += (
                f"# Concrete Example:\n{self.think.concrete_ex}\n{self.action_prompt.concrete_ex}\n"
            )

    def get_message(self) -> HumanMessage:
        message = HumanMessage(content=self.prompt)
        if self.vl_prompt_flags.obs_flags.use_screenshot:
            if self.vl_prompt_flags.obs_flags.use_som:
                screenshot = self.obs_history[-1]["screenshot_som"]
                message.add_text(
                    "## Screenshot:\nHere is a screenshot of the page, it is annotated with bounding boxes and corresponding bids:\n"
                )
            else:
                screenshot = self.obs_history[-1]["screenshot"]
                message.add_text("## Screenshot:\nHere is a screenshot of the page:\n")
            message.add_image(screenshot)
        return message

    def _parse_answer(self, text_answer: str) -> dict:
        answer = {}
        answer.update(self.think.parse_answer(text_answer))
        answer.update(self.action_prompt.parse_answer(text_answer))
        return answer
