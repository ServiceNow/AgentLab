from pathlib import Path
import subprocess
import pytest


@pytest.mark.pricy
def test_main_script_execution():
    # this should trigger agent_4o_mini on miniwob_tiny_test unless this was
    # reconfigured differently.
    script_path = Path(__file__).parent.parent / "main.py"
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    assert result.returncode == 0


if __name__ == "__main__":
    test_main_script_execution()
