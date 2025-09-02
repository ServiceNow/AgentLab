import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from browsergym.experiments.agent import AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt
from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage


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


class MultiCandidateGenericAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args,
        flags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)

    def get_candidate_generations(
        self,
        obs,
        hint: list[str] | None = None,
        n_candidates=3,
    ) -> list[dict]:
        # Append obs to history only if it's not already the last entry
        # Important to handle cases when get_candidate_generation is called multiple times in a single step.
        if not self.obs_history or self.obs_history[-1] is not obs:
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

        cg = CandidatesGeneration(hint=hint, n_candidates=n_candidates)
        candidates_prompt = HumanMessage(cg.prompt)
        chat_messages = Discussion([system_prompt, human_prompt, candidates_prompt])
        output = self.chat_llm(chat_messages)
        candidates = cg._parse_answer(output["content"])
        # Not adding the generate candidate prompt to xray.
        msg_to_add_to_xray = Discussion([system_prompt, human_prompt])
        suggestions = [
            {
                "action": candidate["action"],
                "think": candidate["think"],
            }
            for key, candidate in candidates.items()
        ]
        output = []
        for candidate in suggestions:
            agent_info = AgentInfo(
                think=candidate.get("think", None),
                chat_messages=msg_to_add_to_xray,
                stats=self.chat_llm.get_stats(),
                extra_info={
                    "chat_model_args": asdict(self.chat_model_args),
                    "think": candidate.get("think", None),
                    "plan": candidate.get("plan", None),
                    "step": candidate.get("step", None),
                    "memory": candidate.get("memory", None),
                },
            )
            output.append({"action": candidate["action"], "agent_info": agent_info})

        return output

    def update_agent_state_from_selected_candidate(self, output):
        """Updates the agent's internal state based on the selected candidate from human feedback.

        Args:
            output: Dictionary containing 'action' and 'agent_info' keys from selected candidate.
        """
        action, agent_info = output["action"], output["agent_info"]
        self.plan = agent_info.extra_info.get("plan", self.plan)
        self.plan_step = agent_info.extra_info.get("step", self.plan_step)
        self.memories.append(agent_info.extra_info.get("memory", None))
        self.thoughts.append(agent_info.extra_info.get("think", None))
        self.actions.append(action)

    def get_action(self, obs):
        """Generates multiple candidates and always returns the first one.
        This allows to use this agent as a drop-in replacement for a single-candidate agent.

        Args:
            obs: The observation from the environment.

        Returns:
            tuple: A tuple containing (action, agent_info).
        """
        candidates = self.get_candidate_generations(obs, hint=None, n_candidates=2)
        selection = candidates[0]  # always select the first option.
        self.update_agent_state_from_selected_candidate(selection)
        action, agent_info = selection["action"], selection["agent_info"]

        return action, agent_info


@dataclass
class MultiCandidateGenericAgentArgs(GenericAgentArgs):
    def make_agent(self):
        return MultiCandidateGenericAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
        )

    def __post_init__(self):
        """Prefix subagent name with 'MC-'."""
        super().__post_init__()
        if hasattr(self, "agent_name") and self.agent_name:
            self.agent_name = "MC-" + self.agent_name
