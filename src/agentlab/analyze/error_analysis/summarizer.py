from dataclasses import dataclass

from bgym import ExpResult, StepInfo

from agentlab.analyze.error_analysis.summarizer_prompts import (
    CHANGE_SUMMARIZER_PROMPT,
    ERROR_CLASSIFICATION_PROMPT,
    ERROR_CLASSIFICATION_PROMPT_SUCCESS_OR_NOT,
)
from agentlab.llm.llm_utils import json_parser, parse_html_tags
from agentlab.llm.tracking import set_tracker


def _diff(past_obs, current_obs):
    """TODO: Implement the diff function.

    Returns a diff version of current_obs compares to past_obs, unless there is too many changes.

    Args:
        past_obs: The past observation.
        current_obs: The current observation.

    Raises:
        ValueError: Not implemented yet.
    """
    raise ValueError("Not implemented yet.")


@dataclass
class ChangeSummarizer:

    llm: callable  # language model
    obs_formatter: callable = lambda x: x.get("dom_txt", "No AXTREE available")
    use_diff: bool = False

    def summarize(self, obs: StepInfo, next_obs: StepInfo, past_summaries: list[str]) -> str:
        """Produces, a summary of the effect of an action."""
        obs_message = self.obs_formatter(obs.obs)
        next_obs_message = self.obs_formatter(next_obs.obs)

        action = obs.action

        goal = obs.obs["goal"]  # Use goal object from agentlab
        # TODO(thibault): switch to 'goal_object'
        # Outsource everything to formatter

        if self.use_diff:
            next_obs_message = _diff(obs_message, next_obs_message)

        return self.parse(
            self.llm(
                self.make_prompt(
                    obs_message,
                    action,
                    next_obs_message,
                    past_summaries,
                    goal,
                    obs.obs.get("plan", "No plan available"),
                )
            )["content"]
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

    def parse(self, raw_output: str) -> dict:
        parsed_result = parse_html_tags(
            raw_output, keys=["changeSummary", "actionAssessment", "explanation", "suggestion"]
        )[0]
        return parsed_result


@dataclass
class EpisodeAnalysis:
    analysis: str  # complete analysis of the episode
    summary: str  # short summary of the analysis
    categories: dict[str, float]  # score for each category e.g. type of error or difficulty levels


@dataclass
class EpisodeSummarizer:

    change_summarizer: ChangeSummarizer = None
    llm: callable = None
    parser: callable = lambda x: json_parser(x)[0]
    guess_success: bool = False

    def make_prompt(self, exp_results: ExpResult, summaries: list[str]): ...

    def __call__(self, exp_results: ExpResult) -> EpisodeAnalysis:
        """Run Change Summarizer for every step in the episode or extract a pre-computed one."""

        if not self.guess_success:
            if exp_results.steps_info[-1].reward == 1:
                return {"analysis": "Success", "summaries": {}}

        with set_tracker("summary") as summaries_tracker:
            summaries = self.make_change_summaries(exp_results)
        prompt = self.make_prompt(exp_results, summaries)

        with set_tracker("analysis") as analysis_tracker:
            raw_analysis = self.llm(prompt)["content"]
        analysis = self.parse(raw_analysis)
        res = {
            "analysis": analysis,
            "summaries": {i: a for i, a in enumerate(summaries)},
        }
        res.update(analysis_tracker.stats)
        res.update(summaries_tracker.stats)
        return res

    def make_change_summaries(self, exp_result: ExpResult) -> list[str]:
        summaries = []  # type: list[str]
        # this assumes that there is always an extra step at the end of the episode
        # it is generally the case, but exps can sometimes fail in a weird way and not save the last step_info
        # TODO:(thibault) make some checks or w/e
        for step, next_step in zip(exp_result.steps_info[:-1], exp_result.steps_info[1:]):
            summaries.append(self.change_summarizer.summarize(step, next_step, summaries))
        return summaries

    def parse(self, raw_output: str) -> dict:
        parsed_result = parse_html_tags(raw_output, keys=["explanation", "errorCategory"])[0]
        return parsed_result


@dataclass
class EpisodeErrorSummarizer(EpisodeSummarizer):

    change_summarizer: ChangeSummarizer = None

    def make_prompt(self, exp_results: ExpResult, summaries: list[str]):
        """TODO: Implement the prompt."""
        goal = exp_results.steps_info[0].obs["goal"]

        def format_summary(summary):
            res = ""
            for key, value in summary.items():
                res += f"{key}: {value}\n"
            return res

        txt_summaries = "\n".join([format_summary(summary) for summary in summaries])

        actions = [step.action for step in exp_results.steps_info[:-1]]
        action_errors = "\n".join(
            [step.obs["last_action_error"] for step in exp_results.steps_info[1:]]
        )

        txt_actions = "\n".join(
            [
                f"Action: {action}\nAction Error: {action_error}"
                for action, action_error in zip(actions, action_errors)
            ]
        )

        extra_info = exp_results.steps_info[-1].task_info

        prompt = (
            ERROR_CLASSIFICATION_PROMPT_SUCCESS_OR_NOT
            if self.guess_success
            else ERROR_CLASSIFICATION_PROMPT
        )

        return prompt.format(
            goal=goal,
            historical_summaries=txt_summaries,
            action_history=txt_actions,
            extra_info=extra_info,
        )
