from dataclasses import dataclass
from typing import Optional

import bgym
import playwright
from browsergym.experiments.agent import Agent

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.agent_utils import overlay_action
from agentlab.agents.hitl_agent.base_multi_candidate_agent import MultiCandidateAgent
from agentlab.agents.hitl_agent.hint_labelling import (
    HintLabeling,
    HintLabelingInputs,
)
from agentlab.llm.llm_utils import img_to_base_64
from agentlab.llm.tracking import cost_tracker_decorator


class HumanInTheLoopAgent(Agent):

    def __init__(
        self,
        subagent_args,  # Type: any object with MultiCandidateAgent interface
    ):
        self.subagent: MultiCandidateAgent = subagent_args.make_agent()
        super().__init__()
        self.ui = None

    @cost_tracker_decorator
    def get_action(self, obs):
        # reset vars
        step_n_human_intervention_rounds = 0
        step_hint = []

        # Initialize UI once outside the loop
        if self.ui is None:
            self.ui = HintLabeling(headless=False)
            # Show initial waiting state
            initial_inputs = HintLabelingInputs(
                goal=(
                    obs.get("goal_object", [{}])[0].get("text", "")
                    if obs.get("goal_object")
                    else ""
                ),
                error_feedback="",
                screenshot=(img_to_base_64(obs["screenshot"]) if "screenshot" in obs else ""),
                screenshots=[],  # no overlay screenshots yet
                axtree=obs.get("axtree_txt", ""),
                hints=[],
                suggestions=[],  # no suggestions yet
            )
            self.ui.update_context(initial_inputs)

        # Generate first candidates
        candidates = self.subagent.get_candidate_generations(obs, hint=None, n_candidates=3)
        step_n_human_intervention_rounds += 1
        suggestions = [{"action": c["action"], "think": c["agent_info"].think} for c in candidates]
        # List of Images as base64 - create overlay screenshots for each suggested action
        screenshots = [overlay_action(obs, choice["action"]) for choice in suggestions]

        while True:
            try:
                hint_labeling_inputs = HintLabelingInputs(
                    goal=(
                        obs.get("goal_object", [{}])[0].get("text", "")
                        if obs.get("goal_object")
                        else ""
                    ),
                    error_feedback=obs.get("last_action_error", ""),
                    screenshot=(img_to_base_64(obs["screenshot"]) if "screenshot" in obs else ""),
                    screenshots=screenshots,  # list of overlay screenshots for hover
                    axtree=obs.get("axtree_txt", ""),
                    hints=step_hint,
                    suggestions=suggestions,
                )

                self.ui.update_context(hint_labeling_inputs)
                response = self.ui.wait_for_response(timeout=None)

                if response["type"] == "reprompt":
                    new_hints = response["payload"].get("hints", [])
                    # Replace with the new list from UI, or extend if needed
                    step_hint = list(new_hints) if isinstance(new_hints, list) else step_hint
                    candidates = self.subagent.get_candidate_generations(
                        obs, hint=step_hint if step_hint else None, n_candidates=3
                    )
                    step_n_human_intervention_rounds += 1
                    suggestions = [
                        {"action": c["action"], "think": c["agent_info"].think} for c in candidates
                    ]
                    screenshots = [overlay_action(obs, choice["action"]) for choice in suggestions]

                elif response["type"] == "step":
                    selected_action = response["payload"]["action"]
                    choice_idx = None
                    for i, candidate in enumerate(suggestions):
                        if candidate["action"] == selected_action:
                            choice_idx = i
                            break
                    selected_candidate = candidates[choice_idx]
                    self.subagent.update_agent_state_from_selected_candidate(selected_candidate)
                    action = selected_candidate["action"]
                    agent_info = selected_candidate["agent_info"]
                    return action, agent_info

            except KeyboardInterrupt:
                print("User cancelled the operation")
                if self.ui:
                    self.ui.close()
                raise
            except playwright.sync_api.TimeoutError:
                # Handle timeout specifically: fall back to first candidate
                print("UI timeout; falling back to first candidate.")
                selected_candidate = candidates[0]
                self.subagent.update_agent_state_from_selected_candidate(selected_candidate)
                action = selected_candidate["action"]
                agent_info = selected_candidate["agent_info"]
                return action, agent_info
            except Exception as e:
                print(f"Error in human intervention UI: {e}")
                if self.ui:
                    self.ui.close()
                    self.ui = None
                # Raise exception instead of falling back to console input
                raise RuntimeError(f"Human intervention UI failed: {e}") from e


@dataclass
class HumanInTheLoopAgentArgs(AgentArgs):
    subagent_args: Optional[AgentArgs] = None  # args for the underlying multiple proposal agent

    def make_agent(self):
        assert self.subagent_args is not None
        return HumanInTheLoopAgent(subagent_args=self.subagent_args)

    def __post_init__(self):
        """Prefix subagent name with 'HITL-'."""
        super().__post_init__()
        if self.subagent_args and self.subagent_args.agent_name:
            self.agent_name = "HITL-" + self.subagent_args.agent_name

    def set_benchmark(self, benchmark, demo_mode):
        """Delegate set_benchmark to the subagent if it has the method."""
        if hasattr(self.subagent_args, "set_benchmark"):
            self.subagent_args.set_benchmark(benchmark, demo_mode)

    def set_reproducibility_mode(self):
        """Delegate set_reproducibility_mode to the subagent if it has the method."""
        if hasattr(self.subagent_args, "set_reproducibility_mode"):
            self.subagent_args.set_reproducibility_mode()


def get_base_human_in_the_loop_genericagent(llm_config):
    """
    Create a base human-in-the-loop generic agent configuration using the key from CHAT_MODEL_ARGS_DICT.

    This function creates a HumanInTheLoopAgentArgs instance with a MultiCandidateGenericAgent
    as the subagent, configured with the specified LLM configuration and base flags.

    Args:
        llm_config (str): The LLM configuration key to use from CHAT_MODEL_ARGS_DICT.

    Returns:
        HumanInTheLoopAgentArgs: Configured human-in-the-loop agent arguments with
                                a multi-candidate generic agent as the subagent.
    """
    from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
    from agentlab.agents.hitl_agent.hitl_agent import HumanInTheLoopAgentArgs
    from agentlab.agents.hitl_agent.multi_candidate_generic_agent import (
        MultiCandidateGenericAgentArgs,
    )
    from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

    return HumanInTheLoopAgentArgs(
        subagent_args=MultiCandidateGenericAgentArgs(
            chat_model_args=CHAT_MODEL_ARGS_DICT[llm_config],
            flags=BASE_FLAGS,
        )
    )


HUMAN_GUIDED_GENERIC_AGENT = get_base_human_in_the_loop_genericagent("openai/gpt-5-mini-2025-08-07")

if __name__ == "__main__":
    import logging

    from agentlab.agents.hitl_agent.hitl_agent import (
        HUMAN_GUIDED_GENERIC_AGENT,
    )
    from agentlab.experiments.study import Study

    agent_configs = [HUMAN_GUIDED_GENERIC_AGENT]
    benchmark = bgym.DEFAULT_BENCHMARKS["miniwob"]()
    benchmark = benchmark.subset_from_glob("task_name", "*book*")
    benchmark.env_args_list = benchmark.env_args_list[2:3]

    for env_args in benchmark.env_args_list:
        env_args.max_steps = 100  # max human steps
        env_args.headless = False

    Study(agent_configs, benchmark, logging_level=logging.WARNING).run(
        n_jobs=1,
        parallel_backend="sequential",
        n_relaunch=1,
    )
