from dataclasses import dataclass
from bgym import StepInfo

CHANGE_SUMMARIZER_PROMPT = """
You are a specialized 'change summarizer' model. At a given step in the agent's interaction with the website, 
you will receive the following pieces of information:

1. The user's MAIN GOAL (e.g., "Open a GitLab issue with label 'help wanted'").
2. The AGENT'S PREVIOUS OBSERVATION (HTML or AX Tree snippet) or a 'DIFF' that shows what changed since the last step, and the corresponding change summaries.
3. The AGENT'S CURRENT OBSERVATION (HTML or AX Tree snippet).
4. The ACTION the agent just took (e.g., "Clicked the button labeled 'Show report'").
5. (Optionally) The agent's CHAIN OF THOUGHT or short planning notes for this single step, if available.

YOUR TASK (each step):
A) SUMMARIZE THE CHANGE
   - Describe what visibly changed between the previous observation (or diff) and the current observation. 
     For example, did a new panel open, did the form reset, did nothing happen, etc.?

B) ASSESS THE ACTION
   - Decide whether the agent's action seems helpful or correct given the user's main goal, 
     or if it appears incorrect/unhelpful. 
   - Briefly explain why.

OUTPUT FORMAT (per step):
Return your analysis as a JSON-like structure, for example:

{
  "changeSummary": "A new search results panel appeared on the right side.",
  "actionAssessment": "Correct",
  "explanation": "Clicking 'Search' was appropriate to display the results."
}

Or for an incorrect action:

{
  "changeSummary": "The page reloaded but the date fields were reset to defaults.",
  "actionAssessment": "Incorrect",
  "explanation": "The agent should have fixed the date format first instead of re-clicking 'Show report'.",
  "suggestion": "Correct the date format or check for error messages."
}

Please follow this structure at every step. Keep your responses concise and clear. Below are the details.

Goal: {goal}

LLM Plan: {plan}

Previous Observation: {past_observation}

Current Observation: {current_observation}

Past summaries: {past_summaries}

Action: {action}
"""


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
        goal = past_obs["goal"]
        plan = past_obs["plan"]
        if self.use_diff:
            current_obs_message = _diff(past_obs_message, current_obs_message)

        return self.llm(
            self.make_prompt(
                past_obs_message, action, current_obs_message, past_summaries, goal, plan
            )
        )

    def make_prompt(
        self, past_obs_message, action, current_obs_message, past_summaries, goal, plan
    ):
        """TODO: Implement the prompt."""
        return CHANGE_SUMMARIZER_PROMPT.format(
            goal=goal,
            plan=plan,
            past_observation=past_obs_message,
            current_observation=current_obs_message,
            past_summaries=past_summaries,
            action=action,
        )


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
