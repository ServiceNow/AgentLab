"""
GenericAgent implementation for AgentLab

This module defines a `GenericAgent` class and its associated arguments for use in the AgentLab framework. \
The `GenericAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `GenericAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from warnings import warn
import json

from typing import Dict, List, Optional

import bgym
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .privaleged_agent_prompt import PrivalegedPrompt, PrivalegedPromptFlags
from functools import partial


@dataclass
class PrivalegedAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: PrivalegedPromptFlags = None
    max_retry: int = 4
    privaleged_actions_path: Path = None
    use_privileged_actions: bool = True

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"GenericAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        # for backward compatibility with old traces
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        # verify if we can remove this
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return PrivalegedAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
            privaleged_actions_path=self.privaleged_actions_path,
            use_privileged_actions=self.use_privileged_actions,
        )


@dataclass
class PrivilegedObservation:
    """Represents a single privileged observation with action and metadata."""

    action: str
    reward: int
    task_name: str
    model_name: str
    output: str
    goal: Optional[str]   
    action_name: Optional[str]
    action_value: Optional[str]
    bid_line: Optional[str]




@dataclass
class PrivilegedRun:
    """Represents a single run with multiple steps for a goal."""

    steps: Dict[str, PrivilegedObservation]

    def __init__(self):
        self.steps = {}

    def add_step(self, step_id: str, obs: PrivilegedObservation):
        """Add a step to this run."""

        self.steps[step_id] = obs

    def get_step(self, step_id: str) -> Optional[PrivilegedObservation]:
        """Get a specific step."""
        return self.steps.get(step_id)

    def get_all_steps(self) -> List[PrivilegedObservation]:
        """Get all steps in order."""
        return [
            self.steps[str(i)]
            for i in sorted([int(k) for k in self.steps.keys()])
            if str(i) in self.steps
        ]


@dataclass
class PrivilegedObservationCollection:
    """Container for privileged observations with query capabilities."""

    observations: Dict[str, Dict[str, Dict[str, PrivilegedRun]]]

    def __init__(self):
        self.observations = {}

    def add_run(self, task: str, goal: str, trajectory_id: str, run: PrivilegedRun):
        """Add a privileged run for a specific task and goal."""
        if task not in self.observations:
            self.observations[task] = {}
        if goal not in self.observations[task]:
            self.observations[task][goal] = {}
        self.observations[task][goal][trajectory_id] = run

    def get_run(self, task: str, goal: str, trajectory_id: str) -> Optional[PrivilegedRun]:
        """Get a specific run for a task and goal."""
        return self.observations.get(task, {}).get(goal, {}).get(trajectory_id)

    def get_all_runs(self, task: str, goal: str = None) -> List[PrivilegedRun]:
        """Get all runs for a specific task and optionally goal."""
        if task not in self.observations:
            return []
        if goal is None:
            runs = []
            for goal_data in self.observations[task].values():
                runs.extend(goal_data.values())
            return runs
        return list(self.observations[task].get(goal, {}).values())

    def get_all_tasks(self) -> List[str]:
        """Get all available tasks."""
        return list(self.observations.keys())

    def get_all_goals(self, task: str) -> List[str]:
        """Get all available goals for a task."""
        return list(self.observations.get(task, {}).keys())

    def has_task(self, task: str) -> bool:
        """Check if a task exists in the collection."""
        return task in self.observations

    def has_goal(self, task: str, goal: str) -> bool:
        """Check if a goal exists for a task."""
        return task in self.observations and goal in self.observations[task]

    def get_random(
        self, task: str, goal: Optional[str] = None, get_step: bool = False
    ) -> Optional[PrivilegedObservation]:
        """Get a random step from a random trajectory. If goal is specified and exists, use it. Otherwise, pick a random goal."""
        import random

        if not self.has_task(task):
            return None

        # Determine the goal to use
        goal_to_use = goal
        if goal_to_use is None or not self.has_goal(task, goal_to_use):
            # Pick a random goal if not specified or not found
            available_goals = self.get_all_goals(task)
            if not available_goals:
                return None
            goal_to_use = random.choice(available_goals)

        # Get all trajectories for this goal
        trajectories = list(self.observations[task][goal_to_use].values())
        if not trajectories:
            return None

        # Pick a random trajectory
        random_trajectory = random.choice(trajectories)
        steps = random_trajectory.get_all_steps()
        if not steps:
            return None

        return random.choice(steps) if get_step else steps


class PrivalegedAgent(Agent):

    def __init__(
        self,
        chat_model_args: PrivalegedAgentArgs,
        flags: PrivalegedPromptFlags,
        max_retry: int = 4,
        privaleged_actions_path: Path = None,
        use_privileged_actions: bool = True,
    ):

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry
        self.use_privileged_actions = use_privileged_actions
        self.task = None
        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self.privileged_observations = PrivilegedObservationCollection()
        self.privalaged_path = privaleged_actions_path
        self.load_privaleged_actions(self.privalaged_path)
        self.goal = None
        self._check_flag_constancy()
        self.reset(seed=None)
        self.trajectory = None

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def load_privaleged_actions(self, privaleged_actions_path: Path):
        """Load privileged actions from a JSON file."""
        try:
            with open(privaleged_actions_path, "r") as f:
                data = json.load(f)

            for goal, trajectories_data in data.items():
                for trajectory_id, steps_data in trajectories_data.items():
                    run = PrivilegedRun()
                    task_name = None

                    for step_id, obs_data in steps_data.items():
                        privileged_obs = PrivilegedObservation(
                            action=obs_data["action"],
                            reward=obs_data["reward"],
                            task_name=obs_data["task_name"],
                            model_name=obs_data["model_name"],
                            output=obs_data["output"],
                            goal = goal,
                            action_name=obs_data["action_name"],
                            action_value=obs_data["action_value"],
                            bid_line=obs_data["bid_line"],
                            
                        )
                        run.add_step(step_id, privileged_obs)

                        # Extract task_name from the first step
                        if task_name is None:
                            task_name = obs_data["task_name"]

                    # Add run indexed by task_name
                    if task_name:
                        self.privileged_observations.add_run(task_name, goal, trajectory_id, run)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            warn(f"Failed to load privileged actions from {privaleged_actions_path}: {e}")

    def sample_trajectory(
        self, task: str, goal: str, trajectory_id: str
    ) -> List[PrivilegedObservation]:
        """Sample a trajectory of privileged actions for a specific task, goal, and trajectory."""
        if not self.privileged_observations.has_task(task):
            warn(f"No privileged actions found for task: {task}")
            return []

        if not self.privileged_observations.has_goal(task, goal):
            warn(f"No goal found for task: {task}, goal: {goal}")
            return []

        run = self.privileged_observations.get_run(task, goal, trajectory_id)
        if not run:
            warn(
                f"No trajectory found for task: {task}, goal: {goal}, trajectory_id: {trajectory_id}"
            )
            return []

        return run.get_all_steps()

    def set_goal(self, goal):
        self.goal = goal
        self.trajectory = self.privileged_observations.get_random(task=self.task, goal=self.goal)

    @cost_tracker_decorator
    def get_action(self, obs):

        self.obs_history.append(obs)

        main_prompt = PrivalegedPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            goal=self.goal,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
            trajectory=self.trajectory,
            use_privileged_actions=self.use_privileged_actions,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr

    def set_task(self, task: str):
        """
        Set the task for the agent. This method can be used to change the task
        during an episode.

        Parameters:
        -----------
        task: str
            The new task for the agent.
        """
        self.task = task
