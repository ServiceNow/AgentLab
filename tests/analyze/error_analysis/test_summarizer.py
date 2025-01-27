from pathlib import Path

import pytest
from bgym import ExpResult, StepInfo

from agentlab.analyze.error_analysis.summarizer import ChangeSummarizer
from agentlab.analyze.inspect_results import yield_all_exp_results


@pytest.fixture(scope="module")
def exp_results() -> list[ExpResult]:
    exp_dir = Path(__file__).parent.parent.parent / "data/error_analysis"
    return list(yield_all_exp_results(exp_dir))


def test_change_summarizer(exp_results: list[ExpResult]):
    summarizer = ChangeSummarizer(llm=lambda x: x)
    step = exp_results[0].steps_info[0]
    next_step = exp_results[0].steps_info[1]
    past_summaries = []
    summary = summarizer.summarize(step, next_step, past_summaries)
    assert isinstance(summary, str)
