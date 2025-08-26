"""Basic Recorder Agent - Minimal version for recording human interactions."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
import bgym
from playwright.sync_api import Page
from agentlab.agents.agent_args import AgentArgs


@dataclass
class BasicRecorderAgentArgs(AgentArgs):
    agent_name: str = "BasicRecorderAgent"
    trace_dir: str = "basic_traces"
    use_raw_page_output: bool = True

    def make_agent(self) -> bgym.Agent:
        return BasicRecorderAgent(self.trace_dir)

    def set_reproducibility_mode(self):
        pass


class BasicRecorderAgent(bgym.Agent):
    def __init__(self, trace_dir: str):
        self.action_set = bgym.PythonActionSet()
        self._trace_dir = Path(trace_dir)
        self._page: Page | None = None
        self._recorded = False

    def obs_preprocessor(self, obs: dict):
        if isinstance(obs, dict):
            self._page = obs.get("page")
        del obs["page"] # unpickable
        return obs

    def get_action(self, obs: dict):
        if not self._recorded:
            self._record_and_test()
            self._recorded = True
            exit()  # Exit after recording

        return "", bgym.AgentInfo(think="Recording complete", chat_messages=[], stats={})

    def test_recorded_pw_script(self, output_file):
        # Test the recorded script
        try:
            result = subprocess.run(["python", str(output_file)], capture_output=True, text=True)
            if result.returncode == 0:
                return True
            if result.stderr:
                print(f"Errors: {result.stderr}")
            return False
        except Exception as e:
            print(f"❌ Test failed: {e}")

    def _record_and_test(self):
        """Record actions and test the script."""
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        output_file = self._trace_dir / "recorded_script.py"
        storage_file = self._trace_dir / "storage_state.json"
        self._page.context.storage_state(path=str(storage_file))
        done = False
        while not done:
            # Record with codegen
            cmd = ["python", "-m", "playwright", "codegen", "--target", "python", "--output",
                    str(output_file), "--load-storage", str(storage_file), self._page.url]
            subprocess.run(cmd, check=True)
            print(f"🎥 Recorded script saved to: {output_file}")
            success = self.test_recorded_pw_script(output_file)
            while not success: # edit the PW script
                subprocess.run(["code", "--new-window", "--wait", str(output_file)], check=True)
                success = self.test_recorded_pw_script(output_file)
            done  = input('Record Again (y/n): ').strip().lower() == 'y' 


BASIC_RECORDER_AGENT = BasicRecorderAgentArgs()


if __name__ == "__main__":
    from agentlab.experiments.study import Study
    agent_configs = [BASIC_RECORDER_AGENT]
    benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l1"]()
    benchmark = benchmark.subset_from_glob("task_name", "*filter*")
    benchmark.env_args_list = benchmark.env_args_list[2:3]
    
    for env_args in benchmark.env_args_list:
        env_args.max_steps = 10
        env_args.headless = False

    Study(agent_configs, benchmark).run(n_jobs=1, parallel_backend="sequential", n_relaunch=1)
