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


def analyze(exp_result, episode_summarizer, save_analysis_func):
    error_analysis = episode_summarizer(exp_result)
    save_analysis_func(exp_result, error_analysis)


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

    def run_analysis(self, parallel=False, jobs=-1):
        filtered_results = self.filter_exp_results()

        if parallel:
            import joblib

            joblib.Parallel(n_jobs=jobs, backend="threading")(
                joblib.delayed(analyze)(exp_result, self.episode_summarizer, self.save_analysis)
                for exp_result in filtered_results
            )

        else:
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


AXTREE_FORMATTER = lambda x: x.get("axtree_txt", "No AXTREE available")
HTML_FORMATTER = lambda x: x.get("pruned_html", "No HTML available")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str)
    parser.add_argument("-f", "--filter", type=str, default=None)
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-g", "--guess_success", action="store_true")

    args = parser.parse_args()

    assert args.exp_dir is not None, "Please provide an exp_dir, e.g., -e /path/to/exp_dir"

    exp_dir = Path(args.exp_dir)
    filter = args.filter
    parallel = args.parallel
    jobs = args.jobs
    guess_success = args.guess_success

    from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

    llm = CHAT_MODEL_ARGS_DICT["azure/gpt-4o-2024-08-06"].make_model()

    pipeline = ErrorAnalysisPipeline(
        exp_dir=exp_dir,
        filter=filter,
        episode_summarizer=EpisodeErrorSummarizer(
            ChangeSummarizer(llm, AXTREE_FORMATTER), llm, guess_success=guess_success
        ),
    )

    pipeline.run_analysis(parallel=parallel, jobs=jobs)


if __name__ == "__main__":

    main()
