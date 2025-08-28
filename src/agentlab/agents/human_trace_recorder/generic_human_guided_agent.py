from dataclasses import asdict, dataclass

import bgym

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt
from agentlab.llm.llm_utils import Discussion, SystemMessage
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.experiments.agent import AgentInfo
from agentlab.llm.llm_utils import HumanMessage


import re
from typing import Dict, List, Tuple


class CandidatesGeneration(dp.PromptElement):
    # Ask for multiple alternatives; each candidate must contain <think> and <action>.
    def __init__(self, hint: list[str] | None=None, n_candidates=3) -> None:
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
        """
        Extract up to n_candidates candidates, using numbered tags only.

        Returns:
        {
            "candidate_generation_1": {"think": "...", "action": "..."},
            "candidate_generation_2": {"think": "...", "action": "..."},
            ...
        }
        """
        result = {f"candidate_generation_{i+1}": {"think": "", "action": ""} for i in range(self.n_candidates)}

        if not isinstance(text_answer, str):
            return result

        matches: List[re.Match] = list(self._NUM_BLOCK.finditer(text_answer))
        # Sort by numeric index
        matches_sorted = sorted(matches, key=lambda m: int(m.group("idx")))
        for i, m in enumerate(matches_sorted[:self.n_candidates]):
            body = m.group("body").strip()
            think_m = self._THINK_PATTERN.search(body)
            action_m = self._ACTION_PATTERN.search(body)
            result[f"candidate_generation_{i+1}"] = {
                "think": (think_m.group("think").strip() if think_m else ""),
                "action": (action_m.group("action").strip() if action_m else ""),
            }

        return result

def get_human_intervention(candidates: Dict[str, Dict[str, str]]) -> Tuple[int, str]:
    """
    Get the user's choice of candidate and any hints they provide.

    Args:
        candidates (Dict[str, Dict[str, str]]): The candidates to choose from.
    """
    for i, candidate in candidates.items():
        think = candidate['think']
        action = candidate['action']
        print(f"{i}:\n Think: {think}\n Action: {action}\n")

    choice_idx = int(input('Select choice: or Provide a hint (P): '))
    hint = input('Provide any hints (optional): ')
    return choice_idx, hint


@dataclass
class MultipleProposalGenericAgentArgs(GenericAgentArgs):

    def make_agent(self):
        return MultipleProposalGenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class MultipleProposalGenericAgent(GenericAgent):

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
        candidates, chat_messages = self.get_candidate_generation(
                sys_prompt=system_prompt,
                human_prompt=human_prompt,
                hint=step_hint if step_hint else None,
            )
        while True:
            choice_idx, hint = get_human_intervention(candidates)
            if hint: # Get new candidates based on hint.
                step_hint.append(hint)
                candidates, chat_messages = self.get_candidate_generation(
                    sys_prompt=system_prompt,
                    human_prompt=human_prompt,
                    hint=step_hint if step_hint else None,
                )
            else:
                ans_dict = candidates[f'candidate_generation_{choice_idx}']
                break

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
            extra_info={"chat_model_args": asdict(self.chat_model_args),
                        "step_hints": step_hint,
                        "n_human_intervention_rounds": self.step_n_human_intervention_rounds,
                        "candidates": candidates
                    },
        )
        return ans_dict["action"], agent_info

    def get_candidate_generation(self, 
                                 sys_prompt: SystemMessage, 
                                 human_prompt: HumanMessage, 
                                 hint: list[str] | None=None, 
                                 n_candidates=3) -> tuple[Dict[str, Dict[str, str]], Discussion]:
        cg = CandidatesGeneration(hint=hint, n_candidates=n_candidates)
        candidates_prompt = HumanMessage(cg.prompt)
        chat_messages = Discussion([sys_prompt, human_prompt, candidates_prompt])
        output = self.chat_llm(chat_messages)
        candidates = cg._parse_answer(output["content"])
        self.step_n_human_intervention_rounds += 1
        return candidates, chat_messages


def get_base_agent(llm_config):
    from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
    from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

    return MultipleProposalGenericAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[llm_config],
        flags=BASE_FLAGS,
    )


HUMAN_GUIDED_GENERIC_AGENT = get_base_agent("openai/gpt-5-mini-2025-08-07")

if __name__ == "__main__":
    import logging

    from agentlab.agents.human_trace_recorder.generic_human_guided_agent import (
        HUMAN_GUIDED_GENERIC_AGENT,
    )
    from agentlab.experiments.study import Study

    agent_configs = [HUMAN_GUIDED_GENERIC_AGENT]
    benchmark = bgym.DEFAULT_BENCHMARKS["miniwob"]()
    benchmark = benchmark.subset_from_glob("task_name", "*book*")
    benchmark.env_args_list = benchmark.env_args_list[2:3]

    for env_args in benchmark.env_args_list:
        env_args.max_steps = 100  # max human steps
        env_args.headless = False

    Study(agent_configs, benchmark, logging_level=logging.WARNING).run(
        n_jobs=1,
        parallel_backend="sequential",
        n_relaunch=1,
    )
