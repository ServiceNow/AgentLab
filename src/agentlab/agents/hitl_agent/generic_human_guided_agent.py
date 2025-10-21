import base64
import copy
import io
import re
from dataclasses import Field, asdict, dataclass
from typing import Dict, List

import bgym
import numpy as np
from browsergym.experiments.agent import AgentInfo
from PIL import Image

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_utils import overlay_action
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt
from agentlab.agents.hitl_agent.hint_labelling import (
    HintLabeling,
    HintLabelingInputs,
)
from agentlab.llm.llm_utils import (
    Discussion,
    HumanMessage,
    SystemMessage,
    img_to_base_64,
)
from agentlab.llm.tracking import cost_tracker_decorator


class CandidatesGeneration(dp.PromptElement):
    # Ask for multiple alternatives; each candidate must contain <think> and <action>.
    def __init__(self, hint: list[str] | None = None, n_candidates=3) -> None:
        self.hint = hint
        self.n_candidates = n_candidates
        self.hint_prompt = "\n".join(f"{i}. {c}" for i, c in enumerate(hint, 1)) if hint else ""
        super().__init__(True)
        self._prompt = [
            dict(
                type="text",
                text=f"""
    You are a web agent. Propose {self.n_candidates} alternative next steps for the current page.
    {('Use the Hints:' + self.hint_prompt) if self.hint else ""}\n
    Return EACH candidate wrapped as numbered tags:
    <candidate_generation_1>...</candidate_generation_1>
    <candidate_generation_2>...</candidate_generation_2>

    Inside every candidate you MUST include:
    <think>...why this action is appropriate now...</think>
    <action>...ONE atomic, executable action string...</action>

    Do not include any extra text outside the candidate tags.
    Use this format:
    <candidate_generation_1>
    <think>Explain why Candidate One is chosen</think>
    <action>Candidate One Action</action>
    </candidate_generation_1>

    <candidate_generation_2>
    <think>Explain why Candidate Two is chosen</think>
    <action>Candidate Two Action</action>
    </candidate_generation_2>
    # Example 
    <candidate_generation_1>
    <think>The login button is visible and proceeding will reveal the auth form.</think>
    <action>click(role="button", name="Log in")</action>
    </candidate_generation_1>

    <candidate_generation_2>
    <think>User might need to enter email first; the email field is focused and visible.</think>
    <action>fill(bid="a112", text="user@example.com")</action>
    </candidate_generation_2>
    """,
            )
        ]

    # Regex patterns for numbered candidates only
    _NUM_BLOCK = re.compile(
        r"<\s*candidate[_ ]generation[_ ](?P<idx>[0-9]+)\s*>(?P<body>.*?)<\s*/\s*candidate[_ ]generation[_ ](?P=idx)\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _THINK_PATTERN = re.compile(
        r"<\s*think\s*>(?P<think>.*?)<\s*/\s*think\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _ACTION_PATTERN = re.compile(
        r"<\s*action\s*>(?P<action>.*?)<\s*/\s*action\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _parse_answer(self, text_answer: str) -> Dict[str, Dict[str, str]]:
        """Extract up to n_candidates candidates, using numbered tags only.

        Args:
            text_answer: The text response containing candidate generation tags.

        Returns:
            Dictionary mapping candidate names to their think and action content.
            Format: {"candidate_generation_1": {"think": "...", "action": "..."}, ...}
        """
        result = {
            f"candidate_generation_{i+1}": {"think": "", "action": ""}
            for i in range(self.n_candidates)
        }

        if not isinstance(text_answer, str):
            return result

        matches: List[re.Match] = list(self._NUM_BLOCK.finditer(text_answer))
        # Sort by numeric index
        matches_sorted = sorted(matches, key=lambda m: int(m.group("idx")))
        for i, m in enumerate(matches_sorted[: self.n_candidates]):
            body = m.group("body").strip()
            think_m = self._THINK_PATTERN.search(body)
            action_m = self._ACTION_PATTERN.search(body)
            result[f"candidate_generation_{i+1}"] = {
                "think": (think_m.group("think").strip() if think_m else ""),
                "action": (action_m.group("action").strip() if action_m else ""),
            }

        return result


@dataclass
class MultipleProposalGenericAgentArgs(GenericAgentArgs):

    def make_agent(self):
        return MultipleProposalGenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )

    def __post_init__(self):
        """Prefix subagent name with 'HITL-'."""
        super().__post_init__()
        if hasattr(self, "agent_name") and self.agent_name:
            self.agent_name = "HITL-" + self.agent_name


class MultipleProposalGenericAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args,
        flags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.ui = None  # Single HintLabeling instance

    def get_candidate_generation(
        self,
        sys_prompt: SystemMessage,
        human_prompt: HumanMessage,
        hint: list[str] | None = None,
        n_candidates=3,
    ) -> tuple[Dict[str, Dict[str, str]], Discussion]:

        cg = CandidatesGeneration(hint=hint, n_candidates=n_candidates)
        candidates_prompt = HumanMessage(cg.prompt)
        chat_messages = Discussion([sys_prompt, human_prompt, candidates_prompt])
        output = self.chat_llm(chat_messages)
        candidates = cg._parse_answer(output["content"])
        self.step_n_human_intervention_rounds += 1
        msg_to_add_to_xray = Discussion([sys_prompt, human_prompt])

        return candidates, msg_to_add_to_xray

    @cost_tracker_decorator
    def get_action(self, obs):
        # reset vars
        step_hint = []
        self.step_n_human_intervention_rounds = 0
        self.obs_history.append(obs)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        # Initialize UI once outside the loop
        if self.ui is None:
            self.ui = HintLabeling(headless=False)
            # Show initial waiting state
            initial_inputs = HintLabelingInputs(
                goal=(
                    obs.get("goal_object", [{}])[0].get("text", "")
                    if obs.get("goal_object")
                    else ""
                ),
                error_feedback="",
                screenshot=(img_to_base_64(obs["screenshot"]) if "screenshot" in obs else ""),
                screenshots=[],  # no overlay screenshots yet
                axtree=obs.get("axtree_txt", ""),
                hints=[],
                suggestions=[],  # no suggestions yet
            )
            self.ui.update_context(initial_inputs)

        # Generate first candidates
        candidates, chat_messages = self.get_candidate_generation(
            sys_prompt=system_prompt,
            human_prompt=human_prompt,
            hint=step_hint if step_hint else None,
        )
        suggestions = [
            {
                "id": key.split("_")[-1],
                "action": candidate["action"],
                "think": candidate["think"],
            }
            for key, candidate in candidates.items()
        ]
        # List of Images as base64 - create overlay screenshots for each suggestion
        screenshots = [overlay_action(obs, choice["action"]) for choice in suggestions]

        while True:
            try:
                hint_labeling_inputs = HintLabelingInputs(
                    goal=(
                        obs.get("goal_object", [{}])[0].get("text", "")
                        if obs.get("goal_object")
                        else ""
                    ),
                    error_feedback=obs.get("last_action_error", ""),
                    screenshot=(img_to_base_64(obs["screenshot"]) if "screenshot" in obs else ""),
                    screenshots=screenshots,  # list of overlay screenshots for hover
                    axtree=obs.get("axtree_txt", ""),
                    hints=step_hint,
                    suggestions=suggestions,
                )

                self.ui.update_context(hint_labeling_inputs)
                response = self.ui.wait_for_response(timeout=None)

                if response["type"] == "reprompt":
                    new_hints = response["payload"].get("hints", [])
                    step_hint = list(new_hints) if isinstance(new_hints, list) else step_hint
                    candidates, chat_messages = self.get_candidate_generation(
                        sys_prompt=system_prompt,
                        human_prompt=human_prompt,
                        hint=step_hint if step_hint else None,
                    )
                    suggestions = [
                        {
                            "id": key.split("_")[-1],
                            "action": candidate["action"],
                            "think": candidate["think"],
                        }
                        for key, candidate in candidates.items()
                    ]
                    # Regenerate screenshots for new suggestions
                    screenshots = [overlay_action(obs, choice["action"]) for choice in suggestions]
                    # Continue the loop to show new suggestions
                elif response["type"] == "step":
                    selected_action = response["payload"]["action"]
                    choice_idx = None
                    for i, candidate in enumerate(suggestions, 1):
                        if candidate["action"] == selected_action:
                            choice_idx = i
                            break
                    if choice_idx is None:
                        choice_idx = 1
                    ans_dict = candidates[f"candidate_generation_{choice_idx}"]
                    break
                else:
                    ans_dict = candidates["candidate_generation_1"]
                    break

            except KeyboardInterrupt:
                print("User cancelled the operation")
                if self.ui:
                    self.ui.close()
                raise
            except Exception as e:
                print(f"Error in human intervention UI: {e}")
                if self.ui:
                    self.ui.close()
                    self.ui = None
                # Raise exception instead of falling back to console input
                raise RuntimeError(f"Human intervention UI failed: {e}") from e

        # TODO: Refactor as discussed with ALAC.
        stats = self.chat_llm.get_stats()
        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))
        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={
                "chat_model_args": asdict(self.chat_model_args),
                "step_hints": step_hint,
                "n_human_intervention_rounds": self.step_n_human_intervention_rounds,
                "candidates": candidates,
                "suggestions": suggestions,
            },
        )
        return ans_dict["action"], agent_info


def get_base_agent(llm_config):
    """Creates and returns a MultipleProposalGenericAgentArgs instance with
    specified LLM configuration from CHAT_MODEL_ARGS_DICT.

    Args:
        llm_config: The LLM configuration key to use from CHAT_MODEL_ARGS_DICT.

    Returns:
        MultipleProposalGenericAgentArgs: Configured agent arguments instance.
    """

    from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
    from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

    return MultipleProposalGenericAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[llm_config],
        flags=BASE_FLAGS,
    )


HUMAN_GUIDED_GENERIC_AGENT = get_base_agent("openai/gpt-5-mini-2025-08-07")

if __name__ == "__main__":
    import logging

    from agentlab.agents.hitl_agent.generic_human_guided_agent import (
        HUMAN_GUIDED_GENERIC_AGENT,
    )
    from agentlab.experiments.study import Study

    agent_configs = [HUMAN_GUIDED_GENERIC_AGENT]
    benchmark = bgym.DEFAULT_BENCHMARKS["miniwob"]()
    benchmark = benchmark.subset_from_glob("task_name", "*book*")
    benchmark.env_args_list = benchmark.env_args_list[3:4]

    for env_args in benchmark.env_args_list:
        env_args.max_steps = 100  # max human steps
        env_args.headless = True

    Study(agent_configs, benchmark, logging_level=logging.WARNING).run(
        n_jobs=1,
        parallel_backend="sequential",
        n_relaunch=1,
    )
