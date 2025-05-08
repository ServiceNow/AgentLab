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
        actions: list[str],
        thoughts: list[str],
    ):
        super().__init__()
        if vl_prompt_flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                chat_messages=obs_history[-1]["chat_messages"],
                extra_instructions=vl_prompt_flags.extra_instructions,
            )
        else:
            self.instructions = dp.GoalInstructions(
                goal_object=obs_history[-1]["goal_object"],
                extra_instructions=vl_prompt_flags.extra_instructions,
            )
        self.observation = dp.Observation(obs=obs_history[-1], flags=vl_prompt_flags.obs_flags)
        self.history = dp.History(
            history_obs=obs_history,
            actions=actions,
            memories=None,
            thoughts=thoughts,
            flags=vl_prompt_flags.obs_flags,
        )
        self.think = dp.Think(visible=vl_prompt_flags.use_thinking)
        self.action_prompt = dp.ActionPrompt(
            action_set=action_set, action_flags=vl_prompt_flags.action_flags
        )
        self._prompt = HumanMessage(content=self.instructions.prompt)
        self._prompt.add_text(
            f"""\
{self.observation.prompt}
{self.history.prompt}
{self.think.prompt}
{self.action_prompt.prompt}
"""
        )
        if vl_prompt_flags.use_abstract_example:
            self._prompt.add_text(
                f"""\
# Abstract Example:
{self.think.abstract_ex}
{self.action_prompt.abstract_ex}
"""
            )
        if vl_prompt_flags.use_concrete_example:
            self._prompt.add_text(
                f"""\
# Concrete Example:
{self.think.concrete_ex}
{self.action_prompt.concrete_ex}
"""
            )
        self.observation.add_screenshot(self._prompt)

    def _parse_answer(self, text_answer: str) -> dict:
        answer = {}
        answer.update(self.think.parse_answer(text_answer))
        answer.update(self.action_prompt.parse_answer(text_answer))
        return answer
