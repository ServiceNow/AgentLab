import os

from tapeagents.steps import ImageObservation

from agentlab.agents.tapeagent.agent import TapeAgent, TapeAgentArgs
from agentlab.benchmarks.gaia import GaiaBenchmark, GaiaQuestion


def test_agent_creation():
    args = TapeAgentArgs(agent_name="gaia_agent")
    agent = args.make_agent()
    assert isinstance(agent, TapeAgent)
    assert agent.agent.name == "gaia_agent"


def test_gaia_bench():
    bench = GaiaBenchmark(split="validation")
    assert bench.name == "gaia"
    assert bench.split == "validation"
    assert len(bench.env_args_list) == 165

    assert bench.env_args_list[5].viewport_chars == 64000
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
    bench = GaiaBenchmark(split="validation")
    exp_dir = "/tmp/"

    args = bench.env_args_list[5]
    env = args.make_env(exp_dir)
    steps, _ = env.reset()
    assert len(steps) == 1
    assert isinstance(steps[0], GaiaQuestion)
    assert steps[0].content == args.task["Question"]

    args = bench.env_args_list[20]
    env = args.make_env(exp_dir)
    steps, _ = env.reset()
    assert len(steps) == 2
    assert isinstance(steps[0], GaiaQuestion)
    assert steps[0].content == args.task["Question"]
    assert isinstance(steps[1], ImageObservation)
    assert os.path.basename(steps[1].image_path) == args.task["file_name"]
