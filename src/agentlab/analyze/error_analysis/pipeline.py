import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from bgym import ExpResult

from agentlab.analyze.error_analysis.summarizer import (
    ChangeSummarizer,
    EpisodeErrorSummarizer,
    EpisodeSummarizer,
)
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
    episode_summarizer: EpisodeSummarizer = None

    def filter_exp_results(self) -> Generator[ExpResult, None, None]:
        # TODO:(thibault) improve filtering
        exp_results = yield_all_exp_results(self.exp_dir)
        for exp_result in exp_results:
            if self.filter is None or self.filter in str(exp_result.exp_dir):
                yield exp_result

    def run_analysis(self):
        filtered_results = self.filter_exp_results()

        for exp_result in filtered_results:
            error_analysis = self.episode_summarizer(exp_result)
            self.save_analysis(exp_result, error_analysis)

    def save_analysis(self, exp_result: ExpResult, error_analysis: dict, exists_ok=True):
        """Save the analysis to json"""
        analysis_path = exp_result.exp_dir / "error_analysis.json"
        if not exists_ok and analysis_path.exists():
            raise FileExistsError(f"{analysis_path} already exists")
        with analysis_path.open("w") as f:
            json.dump(error_analysis, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str)
    parser.add_argument("-f", "--filter", type=str, default=None)

    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    filter = args.filter

    from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

    llm = CHAT_MODEL_ARGS_DICT["azure/gpt-4o-mini-2024-07-18"].make_model()

    step_summarizer = ChangeSummarizer(llm, lambda x: x)
    episode_summarizer = EpisodeSummarizer()

    pipeline = ErrorAnalysisPipeline(
        exp_dir=exp_dir,
        filter=filter,
        episode_summarizer=EpisodeErrorSummarizer(ChangeSummarizer(llm), llm),
    )

    pipeline.run_analysis()
