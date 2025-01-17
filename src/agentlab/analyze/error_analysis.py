from dataclasses import dataclass
from bgym import StepInfo


def _diff(past_obs, current_obs):
    """TODO: Implement the diff function.

    Returns a diff version of current_obs compares to past_obs, unless there is too many changes.
    """
    raise ValueError("Not implemented yet.")


@dataclass
class ChangeSummarizer:

    llm: callable  # language model
    obs_formatter: callable
    use_diff: bool = False

    def summarize(
        self, past_obs: dict, action: str, current_obs: dict, past_summaries: list[str]
    ) -> str:
        """Produces, a summary of the effect of an action."""
        past_obs_message = self.obs_formatter(past_obs)
        current_obs_message = self.obs_formatter(current_obs)
        if self.use_diff:
            current_obs_message = _diff(past_obs_message, current_obs_message)

        return self.llm(self.make_prompt(past_obs_message, current_obs_message, action))

    def make_prompt(self, past_obs_message, action, current_obs_message, past_summaries):
        """TODO: Implement the prompt."""
        return f"{past_obs_message} {action} {current_obs_message}"


@dataclass
class EpisodeAnalysis:
    analysis: str  # complete analysis of the episode
    summary: str  # short summary of the analysis
    categories: dict[str, float]  # score for each category e.g. type of error or difficulty levels


@dataclass
class EpisodeSummarizer:

    cange_summarizer: ChangeSummarizer = None

    def summarize(episode: list[StepInfo]) -> EpisodeAnalysis:
        """Run Change Summarizer for every step in the episode or extract a pre-computed one."""
        pass
