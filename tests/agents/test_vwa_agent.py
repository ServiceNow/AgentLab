import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

from agentlab.agents.visualwebarena.agent import VWAAgent, VWAAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments.study import Study
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

mock_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="


class MockGoalImageAgent(VWAAgent):
    def obs_preprocessor(self, obs: dict) -> Any:
        res = super().obs_preprocessor(obs)
        assert isinstance(res["goal_object"], tuple)
        assert len(res["goal_object"]) == 1
        assert isinstance(res["goal_object"][0], dict)
        assert "type" in res["goal_object"][0]
        assert res["goal_object"][0]["type"] == "text"
        assert "text" in res["goal_object"][0]
        res["goal_object"] = (
            res["goal_object"][0],
            {"type": "image_url", "image_url": {"url": mock_image}},
        )
        return res


@dataclass
class MockGoalImageAgentArgs(VWAAgentArgs):
    agent_name: str = "debug_vwa"
    temperature: float = 0.1
    chat_model_args = None

    def make_agent(self) -> MockGoalImageAgent:
        return MockGoalImageAgent(
            chat_model_args=self.chat_model_args,
            n_retry=3,
        )


@pytest.mark.pricy
def test_mock_goal_image_agent():

    with tempfile.TemporaryDirectory() as tmp_dir:
        study = Study(
            [
                MockGoalImageAgentArgs(
                    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"]
                )
            ],
            benchmark="miniwob_tiny_test",
        )
        study.run(n_jobs=1, parallel_backend="sequential")

        results_df = inspect_results.load_result_df(study.dir, progress_fn=None)

        for row in results_df.iterrows():
            if row[1].err_msg:
                print(row[1].err_msg)
                print(row[1].stack_trace)

        assert len(results_df) == len(study.exp_args_list)
        summary = inspect_results.summarize_study(results_df)
        print(summary)
        assert len(summary) == 1
        reward = summary.avg_reward.iloc[0]
        assert reward == 1.0


if __name__ == "__main__":
    test_mock_goal_image_agent()
