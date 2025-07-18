import os
import uuid
from pathlib import Path

try:
    from tapeagents.steps import ImageObservation

    from agentlab.agents.tapeagent.agent import TapeAgent, TapeAgentArgs, load_config
    from agentlab.benchmarks.gaia import GaiaBenchmark, GaiaQuestion
except ModuleNotFoundError:
    import pytest

    pytest.skip("Skipping test due to missing dependencies", allow_module_level=True)


def mock_dataset() -> dict:
    """Mock dataset for testing purposes."""
    data = [{"task_id": str(uuid.uuid4()), "file_name": "", "file_path": ""} for i in range(165)]
    data[5] = {
        "task_id": "32102e3e-d12a-4209-9163-7b3a104efe5d",
        "Question": """The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet.""",
        "Level": "2",
        "Final answer": "Time-Parking 2: Parallel Universe",
        "file_name": "32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx",
        "file_path": "tests/data/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx",
        "Annotator Metadata": {
            "Steps": """1. Open the attached file.\n2. Compare the years given in the Blu-Ray section to find the oldest year, 2009.\n3. Find the title of the Blu-Ray disc that corresponds to the year 2009: Time-Parking 2: Parallel Universe.""",
            "Number of steps": "3",
            "How long did this take?": "1 minute",
            "Tools": "1. Microsoft Excel",
            "Number of tools": "1",
        },
    }
    data[20] = {
        "task_id": "df6561b2-7ee5-4540-baab-5095f742716a",
        "Question": "When you take the average of the standard population deviation of the red numbers and the standard sample deviation of the green numbers in this image using the statistics module in Python 3.11, what is the result rounded to the nearest three decimal points?",
        "Level": "2",
        "Final answer": "17.056",
        "file_name": "df6561b2-7ee5-4540-baab-5095f742716a.png",
        "file_path": "tests/data/df6561b2-7ee5-4540-baab-5095f742716a.png",
        "Annotator Metadata": {
            "Steps": "1. Opened the PNG file.\n2. Made separate lists of the red numbers and green numbers.\n3. Opened a Python compiler.\n4. Ran the following code:\n```\nimport statistics as st\nred = st.pstdev([24, 74, 28, 54, 73, 33, 64, 73, 60, 53, 59, 40, 65, 76, 48, 34, 62, 70, 31, 24, 51, 55, 78, 76, 41, 77, 51])\ngreen = st.stdev([39, 29, 28, 72, 68, 47, 64, 74, 72, 40, 75, 26, 27, 37, 31, 55, 44, 64, 65, 38, 46, 66, 35, 76, 61, 53, 49])\navg = st.mean([red, green])\nprint(avg)\n```\n5. Rounded the output.",
            "Number of steps": "5",
            "How long did this take?": "20 minutes",
            "Tools": "1. Python compiler\n2. Image recognition tools",
            "Number of tools": "2",
        },
    }
    return {"validation": data}


def test_agent_creation():
    config = load_config("gaia_val")
    args = TapeAgentArgs(config=config)
    agent = args.make_agent()
    assert isinstance(agent, TapeAgent)
    assert agent.agent.name == "gaia_agent"


def test_gaia_bench():
    config = load_config("gaia_val")
    bench = GaiaBenchmark.from_config(config, dataset=mock_dataset())
    assert bench.name == "gaia"
    assert bench.split == "validation"
    assert len(bench.env_args_list) == 165

    task = bench.env_args_list[5].task
    question = """The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet."""
    steps = """1. Open the attached file.\n2. Compare the years given in the Blu-Ray section to find the oldest year, 2009.\n3. Find the title of the Blu-Ray disc that corresponds to the year 2009: Time-Parking 2: Parallel Universe."""
    assert task["task_id"] == "32102e3e-d12a-4209-9163-7b3a104efe5d"
    assert task["Question"] == question
    assert task["Level"] == "2"
    assert task["Final answer"] == "Time-Parking 2: Parallel Universe"
    assert task["file_name"] == "32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx"
    assert task["Annotator Metadata"]["Steps"] == steps
    assert task["Annotator Metadata"]["Number of steps"] == "3"
    assert task["Annotator Metadata"]["How long did this take?"] == "1 minute"
    assert task["Annotator Metadata"]["Tools"] == "1. Microsoft Excel"
    assert task["Annotator Metadata"]["Number of tools"] == "1"


def test_gaia_gym_reset():
    exp_dir = "/tmp/gaia_unit_test"
    os.makedirs(exp_dir, exist_ok=True)

    config = load_config("gaia_val")
    bench = GaiaBenchmark.from_config(config, dataset=mock_dataset())
    args = bench.env_args_list[5]
    env = args.make_env(Path(exp_dir))
    steps, _ = env.reset()
    assert len(steps) == 1
    assert isinstance(steps[0], GaiaQuestion)
    assert steps[0].content.startswith(args.task["Question"])

    args = bench.env_args_list[20]
    env = args.make_env(Path(exp_dir))
    steps, _ = env.reset()
    assert len(steps) == 2
    assert isinstance(steps[0], GaiaQuestion)
    assert steps[0].content == args.task["Question"]
    assert isinstance(steps[1], ImageObservation)
    assert os.path.basename(steps[1].image_path) == args.task["file_name"]
