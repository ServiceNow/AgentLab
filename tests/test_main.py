import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.pricy
def test_main_script_execution():
    # this should trigger agent_4o_mini on miniwob_tiny_test unless this was
    # reconfigured differently.
    root_path = Path(__file__).parent.parent

    sys.path.insert(0, str(root_path))

    # just make sure it's in the right state
    main = __import__("main")
    assert main.benchmark == "miniwob_tiny_test"
    assert main.reproducibility_mode == False
    assert main.relaunch == False
    assert main.n_jobs <= 10

    result = subprocess.run(["python", "main.py"], capture_output=True, text=True, timeout=5 * 60)
    assert result.returncode == 0


if __name__ == "__main__":
    test_main_script_execution()
