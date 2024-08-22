from agentlab.experiments.launch_exp import relaunch_study

from pathlib import Path


def test_relaunch_study():
    study_dir = Path(__file__).parent.parent / "data" / "test_study"
    exp_args_list, study_dir_ = relaunch_study(study_dir, relaunch_mode="incomplete_only")

    assert study_dir_ == study_dir
    assert len(exp_args_list) == 1
    assert exp_args_list[0].env_args.task_name == "miniwob.ascending-numbers"

    exp_args_list, study_dir_ = relaunch_study(study_dir, relaunch_mode="incomplete_or_error")

    assert study_dir_ == study_dir
    assert len(exp_args_list) == 2


if __name__ == "__main__":
    test_relaunch_study()
