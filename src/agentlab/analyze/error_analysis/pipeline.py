import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from bgym import ExpResult

from agentlab.analyze.error_analysis.summarizer import ChangeSummarizer, EpisodeSummarizer
from agentlab.analyze.inspect_results import yield_all_exp_results


@dataclass
class Analyzer:
    prompt: str
    llm = None

    def __call__(self, *args, **kwds):
        return "analysis"


@dataclass
class ErrorAnalysisPipeline:
    exp_dir: Path
    filter: str = None
    step_summarizer: ChangeSummarizer = None
    episode_summarizer: EpisodeSummarizer = None
    analyzer: Analyzer = None

    def filter_exp_results(self) -> Generator[ExpResult, None, None]:
        # TODO:(thibault) improve filtering
        exp_results = yield_all_exp_results(self.exp_dir)
        for exp_result in exp_results:
            if self.filter is None or self.filter in str(exp_result.exp_dir):
                yield exp_result

    def run_analysis(self):
        filtered_results = self.filter_exp_results()

        for exp_result in filtered_results:
            step_analysis = self.analyze_step(exp_result)
            episode_analysis = self.analyze_episode(exp_result, step_analysis)
            error_analysis = self.analyze_errors(exp_result, episode_analysis, step_analysis)
            self.save_analysis(exp_result, error_analysis)

    def analyze_step(self, exp_result: ExpResult) -> list[str]:
        step_summaries = []  # type: list[str]
        # this assumes that there is always an extra step at the end of the episode
        # it is generally the case, but exps can sometimes fail in a weird way and not save the last step_info
        # TODO:(thibault) make some checks
        for step, next_step in zip(exp_result.steps_info[:-1], exp_result.steps_info[1:]):
            step_summaries.append(
                self.step_summarizer.summarize(step, step.action, next_step, step_summaries)
            )
        return step_summaries

    def analyze_episode(self, exp_result: ExpResult, step_analysis: list[str]) -> str:
        episode_summary = self.episode_summarizer.summarize(exp_result, step_analysis)
        return episode_summary

    def analyze_errors(
        self, exp_result: ExpResult, episode_analysis: str, step_analysis: list[str]
    ) -> str:
        error_analysis = self.analyzer(exp_result, episode_analysis, step_analysis)
        return error_analysis

    def save_analysis(self, exp_result: ExpResult, error_analysis: dict, exists_ok=True):
        """Save the analysis to json"""
        analysis_path = exp_result.exp_dir / "error_analysis.json"
        if not exists_ok and analysis_path.exists():
            raise FileExistsError(f"{analysis_path} already exists")
        with analysis_path.open("w") as f:
            json.dump(error_analysis, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str)
    parser.add_argument("-f", "--filter", type=str, default=None)

    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)

    pipeline = ErrorAnalysisPipeline(
        exp_dir=exp_dir,
        filter=None,
        episode_summarizer=EpisodeSummarizer(),
        step_summarizer=ChangeSummarizer(),
        analyzer=Analyzer("prompt"),
    )

    pipeline.run_analysis()
