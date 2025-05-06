import logging
from dataclasses import dataclass
from browsergym.core.action.base import AbstractActionSet
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage


@dataclass
class VLPromptFlags(dp.Flags):
    obs_flags: dp.ObsFlags = None
    action_flags: dp.ActionFlags = None
    use_thinking: bool = True
    use_concrete_example: bool = False
    use_abstract_example: bool = True
    enable_chat: bool = False
    extra_instructions: str | None = None


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
        self.vl_prompt_flags = vl_prompt_flags
        if vl_prompt_flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"],
                extra_instructions=vl_prompt_flags.extra_instructions,
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = dp.GoalInstructions(
                obs_history[-1]["goal_object"],
                extra_instructions=vl_prompt_flags.extra_instructions,
            )
        self.observation = dp.Observation(obs_history[-1], vl_prompt_flags.obs_flags)
        self.history = dp.History(obs_history, actions, None, thoughts, vl_prompt_flags.obs_flags)
        self.action_prompt = dp.ActionPrompt(action_set, action_flags=vl_prompt_flags.action_flags)
        self.think = dp.Think(visible=lambda: vl_prompt_flags.use_thinking)

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.observation.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.think.prompt}\
"""
        )

        if self.vl_prompt_flags.use_abstract_example:
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

        if self.vl_prompt_flags.use_concrete_example:
            prompt.add_text(
                f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.think.concrete_ex}\
{self.action_prompt.concrete_ex}\
"""
            )
        return self.observation.add_screenshot(prompt)

    def _parse_answer(self, text_answer):
        answer = {}
        answer.update(self.think.parse_answer(text_answer))
        answer.update(self.action_prompt.parse_answer(text_answer))
        return answer
