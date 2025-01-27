from pathlib import Path

import pytest
from bgym import ExpResult, StepInfo

from agentlab.analyze.error_analysis.pipeline import ErrorAnalysisPipeline

exp_dir = Path(__file__).parent.parent.parent / "data/error_analysis"


class MockStepSummarizer:
    def summarize(
        self, step: StepInfo, action: str, next_step: StepInfo, step_summaries: list[str]
    ) -> str:
        return f"Agent took action {action} at step {len(step_summaries)}"


class MockEpisodeSummarizer:
    def summarize(self, exp_result: ExpResult, step_analysis: list[str]) -> str:
        return f"Agent did actions {', '.join(step.action for step in exp_result.steps_info if step.action)}"


class MockAnalyzer:
    def __call__(
        self, exp_result: ExpResult, episode_analysis: str, step_analysis: list[str]
    ) -> str:
        return {"error": "analysis", "episode": episode_analysis}


@pytest.fixture(scope="module")
def pipeline() -> ErrorAnalysisPipeline:
    return ErrorAnalysisPipeline(
        exp_dir=exp_dir,
        filter=None,
        episode_summarizer=MockEpisodeSummarizer(),
        step_summarizer=MockStepSummarizer(),
        analyzer=MockAnalyzer(),
    )


def test_yield_no_filter(pipeline: ErrorAnalysisPipeline):
    assert len(list(pipeline.filter_exp_results())) == 4


def test_yield_with_filter(pipeline: ErrorAnalysisPipeline):
    pattern = "click-dialog"
    pipeline.filter = pattern
    assert len(list(pipeline.filter_exp_results())) == 2
    pipeline.filter = None


def test_analyze_step(pipeline: ErrorAnalysisPipeline):
    exp_result = next(pipeline.filter_exp_results())
    step_analysis = pipeline.analyze_step(exp_result)

    assert len(exp_result.steps_info) == len(step_analysis) + 1
    assert step_analysis[0] == f"Agent took action {exp_result.steps_info[0].action} at step 0"


def test_analyze_episode(pipeline: ErrorAnalysisPipeline):
    exp_result = next(pipeline.filter_exp_results())
    step_analysis = pipeline.analyze_step(exp_result)
    episode_analysis = pipeline.analyze_episode(exp_result, step_analysis)

    for step_info in exp_result.steps_info:
        if step_info.action:
            assert step_info.action in episode_analysis


def test_save_analysis(pipeline: ErrorAnalysisPipeline):
    exp_result = next(pipeline.filter_exp_results())
    step_analysis = pipeline.analyze_step(exp_result)
    episode_analysis = pipeline.analyze_episode(exp_result, step_analysis)
    error_analysis = pipeline.analyze_errors(exp_result, episode_analysis, step_analysis)

    pipeline.save_analysis(exp_result, error_analysis, exists_ok=False)

    assert (exp_result.exp_dir / "error_analysis.json").exists()

    # remove the file
    (exp_result.exp_dir / "error_analysis.json").unlink()


if __name__ == "__main__":
    test_yield_with_filter()
