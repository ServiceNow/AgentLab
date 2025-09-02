from typing_extensions import Protocol

from agentlab.agents.agent_args import AgentArgs


class MultiCandidateAgent(Protocol):
    """
    Protocol for agents that generate multiple candidates for get_action.

    This protocol defines the contract for agents that can generate
    multiple candidate actions and allow selection of one of them for execution.
    """

    def get_candidate_generations(
        self, obs: dict, hint: list[str] | None = None, n_candidates: int = 3
    ) -> "list[dict]":
        """
        Generate multiple candidate actions for the given observation.

        You can pass extra info in agent_info to update internal state of the
        agent based on the selected candidate. Your internal state management
        should be robust to multiple calls to the get_candidate_generations method
        in a single step.

        Args:
            obs: The current observation dictionary containing environment state
            hint: Optional list of hint strings to guide candidate generation
            n_candidates: Number of candidate actions to generate
        """
        ...

    def update_agent_state_from_selected_candidate(self, output: dict):
        """
        Update the agent's internal state based on the selected candidate.
        This can include any memory or planning updates.

        Args:
            output: The selected candidate action dictionary
        """
        pass


class MultiCandidateAgentArgs(AgentArgs):
    def make_agent(self) -> MultiCandidateAgent: ...

    def __post_init__(self):
        """Prefix subagent name with 'MC-'."""
        super().__post_init__()
        if hasattr(self, "agent_name") and self.agent_name:
            self.agent_name = "MC-" + self.agent_name
