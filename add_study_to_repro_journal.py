import os
from pathlib import Path
from agentlab.experiments.study import Study


base_dir = "/home/toolkit/ui_copilot_results"

exp_paths = [
    "2025-01-31_22-08-34_genericagent-o3-mini-2025-01-31-on-workarena-l1",
    #  '2025-02-02_01-53-45_genericagent-openai-o1-mini-2024-09-12-on-workarena-l1',
    "2025-02-02_01-55-04_genericagent-openai-o1-mini-2024-09-12-on-workarena-l1",
]
full_paths = [os.path.join(base_dir, exp_path) for exp_path in exp_paths]

for full_path in full_paths:
    study = Study.load(Path(full_path))

    study.append_to_journal(strict_reproducibility=False)
