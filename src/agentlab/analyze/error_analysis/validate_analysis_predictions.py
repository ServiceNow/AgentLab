from pathlib import Path
from agentlab.analyze.inspect_results import (
    load_result_df,
)
import json


def get_aggregate_statistics(exp_dir: Path):
    """Get aggregate statistics for the experiment results."""
    results = load_result_df(exp_dir, filter=filter)


if __name__ == "__main__":
    path = Path(
        "/mnt/colab_public/data/ui_copilot/thibault/tmlr_exps/2024-10-23_14-17-47_5_agents_on_workarena_l1"
    )
    results = load_result_df(path).reset_index()
    results = results.loc[results["agent.chat_model.model_name"].str.contains("anthropic")]
    success_predictions = []
    for dir in results["exp_dir"]:
        error_analysis = Path(dir) / "error_analysis.json"
        if error_analysis.exists():
            with open(error_analysis, "r") as f:
                error_analysis = json.load(f)
            task_success_prediction_str = error_analysis["analysis"]["success"]
            task_success_prediction = True if task_success_prediction_str == "True" else False
            success_predictions.append(task_success_prediction)
        else:
            success_predictions.append(None)
    results["success_predictions"] = success_predictions
    a = 1
