import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from agentlab.analyze import inspect_results
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.study import Study


@dataclass
class StudyInfo:
    study_dir: Path
    study: Study
    summary_df: pd.DataFrame
    should_delete: bool = False
    reason: str = ""


def search_for_reasons_to_archive(result_dir: Path, min_study_size: int = 0) -> list[StudyInfo]:

    study_info_list = []
    study_dirs = list(result_dir.iterdir())
    progress = tqdm(study_dirs, desc="Processing studies")
    for study_dir in progress:

        progress.set_postfix({"study_dir": study_dir})
        if not study_dir.is_dir():
            progress.set_postfix({"status": "skipped"})
            continue

        try:
            study = Study.load(study_dir)
        except Exception:
            study = None
        # get summary*.csv files and find the most recent
        summary_files = list(study_dir.glob("summary*.csv"))

        if len(summary_files) != 0:
            most_recent_summary = max(summary_files, key=os.path.getctime)
            summary_df = pd.read_csv(most_recent_summary)

        else:
            try:
                result_df = inspect_results.load_result_df(study_dir, progress_fn=None)
                summary_df = inspect_results.summarize_study(result_df)
            except Exception as e:
                print(f"  Error processing {study_dir}: {e}")
                continue

        study_info = StudyInfo(
            study_dir=study_dir,
            study=study,
            summary_df=summary_df,
        )

        if len(study_info.summary_df) == 0:
            study_info.should_delete = True
            study_info.reason = "Empty summary DataFrame"

        n_completed, n_total, n_err = 0, 0, 0

        for _, row in study_info.summary_df.iterrows():
            n_comp, n_tot = row["n_completed"].split("/")
            n_completed += int(n_comp)
            n_total += int(n_tot)
            n_err += int(row.get("n_err"))

        n_finished = n_completed - n_err

        # print(summary_df)
        # print(f"  {n_completed} / {n_total}, {n_err} errors")

        if "miniwob-tiny-test" in study_dir.name:
            study_info.should_delete = True
            study_info.reason += "Miniwob tiny test\n"
        if n_total == 0:
            study_info.should_delete = True
            study_info.reason += "No tasks\n"
        if n_completed == 0:
            study_info.should_delete = True
            study_info.reason += "No tasks completed\n"
        if float(n_finished) / float(n_total) < 0.5:
            study_info.should_delete = True
            study_info.reason += f"Less than 50% tasks finished, n_err: {n_err}, n_total: {n_total}, n_finished: {n_finished}, n_completed: {n_completed}\n"

        if n_total <= min_study_size:
            study_info.should_delete = True
            study_info.reason += (
                f"Too few tasks. n_total ({n_total}) <= min_study_size ({min_study_size})\n"
            )

        study_info_list.append(study_info)
    return study_info_list


if __name__ == "__main__":
    study_list_info = search_for_reasons_to_archive(RESULTS_DIR, min_study_size=5)
    archive_dir = RESULTS_DIR.parent / "archived_agentlab_results"  # type: Path
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Uncomment the line below to prevent moving studies to archive
    archive_dir = None

    for study_info in study_list_info:
        if not study_info.should_delete:
            continue

        print(f"Study: {study_info.study_dir.name}")
        print(f"  Reason: {study_info.reason}")
        print(study_info.summary_df)
        print()

        if archive_dir is not None:
            # move to new dir
            new_path = archive_dir / study_info.study_dir.name
            study_info.study_dir.rename(new_path)
            # save reason in a file
            reason_file = new_path / "reason_to_archive.txt"
            reason_file.write_text(study_info.reason)
