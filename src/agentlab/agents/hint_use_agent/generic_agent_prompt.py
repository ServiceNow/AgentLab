"""
Prompt builder for GenericAgent

It is based on the dynamic_prompting module from the agentlab package.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.tool_use_agent.tool_use_agent import HintsSource
from agentlab.llm.chat_api import ChatModel
from agentlab.llm.llm_utils import HumanMessage, parse_html_tags_raise

logger = logging.getLogger(__name__)


@dataclass
class GenericPromptFlags(dp.Flags):
    """
    A class to represent various flags used to control features in an application.

    Attributes:
        use_plan (bool): Ask the LLM to provide a plan.
        use_criticise (bool): Ask the LLM to first draft and criticise the action before producing it.
        use_thinking (bool): Enable a chain of thoughts.
        use_concrete_example (bool): Use a concrete example of the answer in the prompt for a generic task.
        use_abstract_example (bool): Use an abstract example of the answer in the prompt.
        use_hints (bool): Add some human-engineered hints to the prompt.
        use_task_hint (bool): Enable task-specific hints from hint database.
        hint_db_path (str): Path to the hint database file.
        enable_chat (bool): Enable chat mode, where the agent can interact with the user.
        max_prompt_tokens (int): Maximum number of tokens allowed in the prompt.
        be_cautious (bool): Instruct the agent to be cautious about its actions.
        extra_instructions (Optional[str]): Extra instructions to provide to the agent.
        add_missparsed_messages (bool): When retrying, add the missparsed messages to the prompt.
        flag_group (Optional[str]): Group of flags used.
    """

    obs: dp.ObsFlags
    action: dp.ActionFlags
    use_plan: bool = False  #
    use_criticise: bool = False  #
    use_thinking: bool = False
    use_memory: bool = False  #
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    use_task_hint: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    max_trunc_itr: int = 20
    flag_group: str = None

    # hint related
    use_task_hint: bool = False
    hint_db_path: str | None = None
    hint_retrieval_mode: Literal["direct", "llm", "emb"] = "direct"
    hint_level: Literal["episode", "step"] = "episode"
    hint_type: str = "docs"
    hint_index_type: str = "sparse"
    hint_query_type: str = "direct"
    hint_index_path: str = "indexes/servicenow-docs-bm25"
    hint_retriever_path: str = "google/embeddinggemma-300m"
    hint_num_results: int = 5
    n_retrieval_queries: int = 1


class MainPrompt(dp.Shrinkable):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: GenericPromptFlags,
        llm: ChatModel,
        task_hints: list[str] = [],
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        goal = obs_history[-1]["goal_object"]
        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = dp.GoalInstructions(
                goal, extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(
            obs_history[-1],
            self.flags.obs,
        )

        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

        def time_for_caution():
            # no need for caution if we're in single action mode
            return flags.be_cautious and (
                flags.action.action_set.multiaction or flags.action.action_set == "python"
            )

        self.be_cautious = dp.BeCautious(visible=time_for_caution)
        self.think = dp.Think(visible=lambda: flags.use_thinking)
        self.hints = dp.Hints(visible=lambda: flags.use_hints)
        self.task_hints = TaskHint(visible=lambda: flags.use_task_hint, task_hints=task_hints)
        self.plan = Plan(previous_plan, step, lambda: flags.use_plan)  # TODO add previous plan
        self.criticise = Criticise(visible=lambda: flags.use_criticise)
        self.memory = Memory(visible=lambda: flags.use_memory)

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)

        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.hints.prompt}\
{self.task_hints.prompt}\
{self.be_cautious.prompt}\
{self.think.prompt}\
{self.plan.prompt}\
{self.memory.prompt}\
{self.criticise.prompt}\
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
{self.plan.abstract_ex}\
{self.memory.abstract_ex}\
{self.criticise.abstract_ex}\
{self.task_hints.abstract_ex}\
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
{self.plan.concrete_ex}\
{self.memory.concrete_ex}\
{self.criticise.concrete_ex}\
{self.task_hints.concrete_ex}\
{self.action_prompt.concrete_ex}\
"""
            )
        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def set_task_name(self, task_name: str):
        """Set the task name for task hints functionality."""
        self.task_name = task_name

    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.think.parse_answer(text_answer))
        ans_dict.update(self.plan.parse_answer(text_answer))
        ans_dict.update(self.memory.parse_answer(text_answer))
        ans_dict.update(self.criticise.parse_answer(text_answer))
        ans_dict.update(self.action_prompt.parse_answer(text_answer))
        return ans_dict


class Memory(dp.PromptElement):
    _prompt = ""  # provided in the abstract and concrete examples

    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions. Some tasks require to
remember hints from previous steps in order to solve it.
</memory>
"""

    _concrete_ex = """
<memory>
I clicked on bid "32" to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
</memory>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["memory"], merge_multiple=True)


class Plan(dp.PromptElement):
    def __init__(self, previous_plan, plan_step, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self.previous_plan = previous_plan
        self._prompt = f"""
# Plan:

You just executed step {plan_step} of the previously proposed plan:\n{previous_plan}\n
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary.
"""

    _abstract_ex = """
<plan>
Provide a multi step plan that will guide you to accomplish the goal. There
should always be steps to verify if the previous action had an effect. The plan
can be revisited at each steps. Specifically, if there was something unexpected.
The plan should be cautious and favor exploring befor submitting.
</plan>

<step>Integer specifying the step of current action
</step>
"""

    _concrete_ex = """
<plan>
1. fill form (failed)
    * type first name
    * type last name
2. Try to activate the form
    * click on tab 2
3. fill form again
    * type first name
    * type last name
4. verify and submit
    * verify form is filled
    * submit if filled, if not, replan
</plan>

<step>2</step>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["plan", "step"])


class Criticise(dp.PromptElement):
    _prompt = ""

    _abstract_ex = """
<action_draft>
Write a first version of what you think is the right action.
</action_draft>

<criticise>
Criticise action_draft. What could be wrong with it? Enumerate reasons why it
could fail. Did your past actions had the expected effect? Make sure you're not
repeating the same mistakes.
</criticise>
"""

    _concrete_ex = """
<action_draft>
click("32")
</action_draft>

<criticise>
click("32") might not work because the element is not visible yet. I need to
explore the page to find a way to activate the form.
</criticise>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["action_draft", "criticise"])


class TaskHint(dp.PromptElement):
    def __init__(self, visible: bool, task_hints: list[str]) -> None:
        super().__init__(visible=visible)
        self.task_hints = task_hints

    @property
    def _prompt(self):
        task_hint_str = "# Hints:\nHere are some hints for the task you are working on:\n"
        for hint in self.task_hints:
            task_hint_str += f"{hint}\n"
        return task_hint_str

    _abstract_ex = """
<task_hint>
What hint can be relevant for the next action? Only chose from the hints provided in the task description. Or select none.
</task_hint>
"""

    _concrete_ex = """
<task_hint>
Relevant hint: Based on the hints provided, I should focus on the form elements and use the
accessibility tree to identify interactive elements before taking actions.
</task_hint>
"""


class StepWiseContextIdentificationPrompt(dp.Shrinkable):
    def __init__(
        self,
        obs_history: list[dict],
        actions: list[str],
        thoughts: list[str],
        obs_flags: dp.ObsFlags,
        n_queries: int = 1,
    ) -> None:
        super().__init__()
        self.obs_flags = obs_flags
        self.n_queries = n_queries
        self.history = dp.History(obs_history, actions, None, thoughts, obs_flags)
        self.instructions = dp.GoalInstructions(obs_history[-1]["goal_object"])
        self.obs = dp.Observation(obs_history[-1], obs_flags)

        self.think = dp.Think(visible=True)  # To replace with static text maybe

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)

        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
"""
        )

        example_queries = [
            "The user has started sorting a table and needs to apply multiple column criteria simultaneously.",
            "The user is attempting to configure advanced sorting options but the interface is unclear.",
            "The user has selected the first sort column and is now looking for how to add a second sort criterion.",
            "The user is in the middle of a multi-step sorting process and needs guidance on the next action.",
        ]

        example_queries_str = json.dumps(example_queries[: self.n_queries], indent=2)

        prompt.add_text(
            f"""
# Querying memory

Before choosing an action, let's search our available documentation and memory for relevant context.
Generate a brief, general summary of the current status to help identify useful hints. Return your answer in the following format:
<think>chain of thought</think>
<queries>json list of strings of queries</queries>

Additional instructions: List of queries should contain up to {self.n_queries} queries. Both the think and the queries blocks are required!

# Concrete Example
```
<think>
I have to sort by client and country. I could use the built-in sort on each column but I'm not sure if
I will be able to sort by both at the same time.
</think>

<queries>
{example_queries_str}
</queries>
```
Note: do not generate backticks.
Now proceed to generate your own thoughts and queries.
Always return non-empty answer, its very important!
"""
        )

        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        try:
            ans_dict = parse_html_tags_raise(
                text_answer, keys=["think", "queries"], merge_multiple=True
            )
        except Exception as e:
            t = text_answer.replace("\n", "\\n")
            logger.warning(f"Failed to parse llm answer: {e}. RAW answer: '{t}'. Will retry")
            raise e
        raw_queries = ans_dict.get("queries", "[]")
        try:
            ans_dict["queries"] = json.loads(raw_queries)
        except Exception as e:
            t = text_answer.replace("\n", "\\n")
            logger.warning(
                f"Failed to parse queries: {e}. Queries block content: '{ans_dict['queries']}'. RAW llm answer: '{t}'. Will retry"
            )
            raise e
        return ans_dict
