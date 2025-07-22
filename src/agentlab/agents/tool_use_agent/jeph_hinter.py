import os
import json
import random
from typing import Any, List, Dict, Optional
from pathlib import Path
import pandas as pd
import hashlib
from datetime import datetime
import gzip
import pickle
import re 
from functools import partial

import bgym
from agentlab.agents.agent_args import AgentArgs
from agentlab.experiments.loop import StepInfo
from agentlab.llm.response_api import MessageBuilder, OpenAIResponseModelArgs
from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt
from agentlab.agents import dynamic_prompting as dp
import logging
from agentlab.agents.generic_agent.generic_agent import GenericAgent
from agentlab.llm.llm_utils import ParseError, get_tokenizer
from agentlab_configs import HintPromptConfig, JephHinterConfig, MineHintsConfig

class SimpleDiscussion:
    """Minimal message grouping for hint prompting."""
    def __init__(self):
        self.messages: List[MessageBuilder] = []
    def append(self, message: MessageBuilder):
        self.messages.append(message)
    def flatten(self) -> List[MessageBuilder]:
        return self.messages

def load_all_step_pickles(root_dir):
    step_data = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pkl.gz") and fname.startswith("step_"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with gzip.open(fpath, "rb") as f:
                        compressed_data = f.read()
                    data = pickle.loads(compressed_data)  # type: ignore
                    step_data.append({"file": fpath, "data": data})
                except Exception as e:
                    print(f"Failed to load {fpath}: {e}")
    return step_data

def extract_trace_info(step_data):
    """Extract meaningful information from step data for hint generation."""
    trace_info = []
    for step in step_data:
        step_info = step["data"]
        if hasattr(step_info, 'obs') and hasattr(step_info, 'agent_info'):
            # Extract observation and agent info
            obs = step_info.obs
            agent_info = step_info.agent_info
            axtree_txt = obs['axtree_txt']

            # Get the agent's thinking process
            think = getattr(agent_info, 'think', '')
            
            # Get the action taken
            action = getattr(step_info, 'action', '')
            
            # Get any error messages
            last_action_error = obs.get('last_action_error', '') if isinstance(obs, dict) else ''
            
            # Get goal information
            goal = obs.get('goal', []) if isinstance(obs, dict) else []

            # Get the current reward
            reward = getattr(step_info, 'reward', 0)


            trace_info.append({
                'axtree_txt': axtree_txt,
                'think': think,
                'action': action,
                'error': last_action_error,
                'goal': goal,
                'reward': reward,
                'step_file': step["file"]
            })
    
    return trace_info

def _format_trace_steps(trace_info: list, hint_prompt_config: HintPromptConfig):
    """Helper function to format trace steps into prompt parts."""
    prompt_parts = []
    for i, step in enumerate(trace_info):
        prompt_parts.append(f"\nStep {i+1}:")
        if not hint_prompt_config.exclude_axtree and step.get('axtree_txt'):
            prompt_parts.append(f"AXTree: {step['axtree_txt']}")
        if not hint_prompt_config.exclude_think and step.get('think'):
            prompt_parts.append(f"Agent's reasoning: {step['think']}")
        if not hint_prompt_config.exclude_actions and step.get('action'):
            prompt_parts.append(f"Action taken: {step['action']}")
        if step.get('error'):
            prompt_parts.append(f"Error encountered: {step['error']}")
        if not hint_prompt_config.exclude_reward:
            prompt_parts.append(f"Current reward: {step['reward']}")
    return prompt_parts

def construct_hint_prompt(trace_info_list, task_name, hint_prompt_config: HintPromptConfig):
    """Construct a comprehensive prompt for hint generation based on trace analysis."""
    prompt_parts = []
    
    # Add system instruction
    prompt_parts.append(f"Task: {task_name}")
    
    # Determine the scenario based on the number and type of traces
    if len(trace_info_list) == 1:
        # Single trace scenario
        trace_info = trace_info_list[0]
        prompt_parts.append("\n=== EXECUTION TRACE ===")
        prompt_parts.extend(_format_trace_steps(trace_info, hint_prompt_config))
        
        # Add outcome summary
        if not hint_prompt_config.exclude_reward:
            cum_reward = sum(step.get('reward', 0) for step in trace_info)
            if cum_reward > 0:
                prompt_parts.append(f"\nThis was a successful trace.")
            else:
                prompt_parts.append(f"\nThis was a failed trace.")
        
        # Add analysis request for single trace
        prompt_parts.append("\n=== HINT GENERATION ===")
        prompt_parts.append("Based on this trace, provide a concise, actionable hint that would help an agent avoid common mistakes and succeed at this task.")
        
    elif len(trace_info_list) == 2:
        # Failed and successful trace pair scenario
        for trace_idx, trace_info in enumerate(trace_info_list):
            cum_reward = sum(step.get('reward', 0) for step in trace_info)
            outcome = "SUCCESSFUL" if cum_reward > 0 else "FAILED"
            prompt_parts.append(f"\n--- Trace {trace_idx + 1} ({outcome}) ---")
            prompt_parts.extend(_format_trace_steps(trace_info, hint_prompt_config))
        # Add analysis request for comparison
        prompt_parts.append("\n=== HINT GENERATION ===")
        prompt_parts.append("Compare the failed and successful traces above. Provide a concise, actionable hint that explains the key differences and what the agent should do to succeed.")
        
    else:
        # Random set of traces scenario
        prompt_parts.append(f"\n=== MULTIPLE EXECUTION TRACES ({len(trace_info_list)} traces) ===")
        
        for trace_idx, trace_info in enumerate(trace_info_list):
            cum_reward = sum(step.get('reward', 0) for step in trace_info)
            outcome = "SUCCESSFUL" if cum_reward > 0 else "FAILED"
            prompt_parts.append(f"\n--- Trace {trace_idx + 1} ({outcome}) ---")
            prompt_parts.extend(_format_trace_steps(trace_info, hint_prompt_config))
        
        # Add analysis request for multiple traces
        prompt_parts.append("\n=== HINT GENERATION ===")
        prompt_parts.append("Based on the multiple traces above, provide a concise, actionable hint that identifies common patterns, successful strategies, and key insights for completing this task.")
    
    # Common instructions for all scenarios
    prompt_parts.append("IMPORTANT: Keep your hint SHORT (1-2 sentences maximum) and write it as a SINGLE LINE without line breaks.")
    prompt_parts.append("Focus on:")
    prompt_parts.append("- Common pitfalls or errors to avoid")
    prompt_parts.append("- Specific strategies that work well")
    prompt_parts.append("- Important details to pay attention to")
    prompt_parts.append("- Step-by-step guidance if applicable")
    
    return "\n".join(prompt_parts)

def summarize_trace_for_important_steps(trace_info, summarizer_llm, msg_builder, hint_prompt_config):
    """
    Use an LLM to summarize the full trace and select the most important step(s).
    Returns a list of step indices (0-based) to zoom in on.
    """
    # Build a summary prompt
    prompt_parts = []
    prompt_parts.append("You are a trace summarizer. Given the following execution trace, identify the step or steps that are most important for understanding success or failure. Return the step numbers (starting from 1) and a brief reason why they are important.")
    prompt_parts.append("\n=== EXECUTION TRACE ===")
    for i, step in enumerate(trace_info):
        # Use _format_trace_steps for consistent formatting
        prompt_parts.extend(_format_trace_steps([step], hint_prompt_config))
    prompt_parts.append("\n=== STEP SELECTION ===")
    prompt_parts.append("List the most important step numbers (comma separated) and a brief reason for each.")
    summary_prompt = "\n".join(prompt_parts)

    discussion = SimpleDiscussion()
    sys_msg = msg_builder.system().add_text("You are a trace summarizer.")
    discussion.append(sys_msg)
    user_msg = msg_builder.user().add_text(summary_prompt)
    discussion.append(user_msg)
    from agentlab.llm.response_api import APIPayload
    payload = APIPayload(messages=discussion.flatten())
    response = summarizer_llm(payload)
    # Parse step numbers from response.think or str(response)
    import re
    text = response.think if hasattr(response, "think") else str(response)
    # Look for numbers (steps) in the response
    step_nums = re.findall(r"Step\s*(\d+)", text)
    if not step_nums:
        # fallback: look for any numbers
        step_nums = re.findall(r"\b(\d+)\b", text)
    # Convert to 0-based indices
    indices = [int(num)-1 for num in step_nums if num.isdigit() and 0 < int(num) <= len(trace_info)]
    # If nothing found, fallback to last step
    if not indices:
        indices = [len(trace_info)-1]
    return indices

def construct_hint_prompt_step_zoom(trace_info, important_indices, task_name, hint_prompt_config):
    """
    Construct a prompt focusing on the most important step(s) for the judge model.
    """
    prompt_parts = []
    prompt_parts.append(f"Task: {task_name}")
    prompt_parts.append("\n=== ZOOMED-IN STEPS ===")
    for idx in important_indices:
        if 0 <= idx < len(trace_info):
            step = trace_info[idx]
            prompt_parts.extend(_format_trace_steps([step], hint_prompt_config))
    prompt_parts.append("\n=== HINT GENERATION ===")
    prompt_parts.append("Based on the most important step(s) above, provide a concise, actionable hint that would help an agent avoid common mistakes and succeed at this task. IMPORTANT: Keep your hint SHORT (1-2 sentences maximum) and write it as a SINGLE LINE without line breaks.")
    return "\n".join(prompt_parts)

def _select_traces_for_hint(traces, n_traces_to_hinter):
    """Select traces for hint generation based on the scenario."""
    if n_traces_to_hinter == 1:
        # Single trace scenario
        return [random.choice(traces)["trace_info"]]
    
    elif n_traces_to_hinter == 2:
        # Two traces scenario - ideally one failed and one successful
        successful_traces = []
        failed_traces = []
        
        for trace in traces:
            cum_reward = sum(step.get('reward', 0) for step in trace["trace_info"])
            if cum_reward > 0:
                successful_traces.append(trace)
            else:
                failed_traces.append(trace)
        
        selected_traces = []
        
        if successful_traces and failed_traces:
            selected_traces = [random.choice(successful_traces), random.choice(failed_traces)]
        else:
            pool = successful_traces or failed_traces or traces
            selected_traces = random.sample(pool, min(2, len(pool)))
        return [trace["trace_info"] for trace in selected_traces]
    
    else:
        # Random set of traces scenario
        n_available = len(traces)
        n_to_sample = min(n_traces_to_hinter, n_available)
        selected_traces = random.sample(traces, n_to_sample)
        return [trace["trace_info"] for trace in selected_traces]

class JephHinter(bgym.Agent):
    """
    Agent that processes traces and builds/updates a hint database (hint_db.csv) from them.
    """
    HINT_DB_COLUMNS = [
        "time_stamp", "task_name", "task_seed", "base_llm", "agent_name", "domain_name", "user_name", "source", "semantic_keys", "hint"
    ]

    def __init__(self, model_args: Any, config: Optional[JephHinterConfig] = None):
        self.model_args = model_args
        self.config = config or JephHinterConfig()
        self.llm = model_args.make_model()
        self.msg_builder = model_args.get_message_builder()
        self.traces = []

    def load_traces(self) -> List[Dict[str, Any]]:
        """Load traces from pickle files in the traces folder."""
        traces = []
        if not self.config or not self.config.traces_folder:
            return traces
        
        # Load all step pickle files
        step_data = load_all_step_pickles(self.config.traces_folder)
        # Group steps by experiment/task and seed
        experiments = {}
        for step in step_data:
            # Extract experiment path from file path
            file_path = step["file"]
            print("loading file", file_path)
            # Extract task name and seed from path (assuming structure like .../task_name_seed/step_*.pkl.gz)
            path_parts = file_path.split(os.sep)
            task_name = "unknown_task"
            seed = "unknown_seed"
            
            for part in path_parts:
                if "miniwob." in part or "workarena." in part:
                    # Handle new format: "miniwob.use-colorwheel-2_1" or "workarena.servicenow.sort-user-list_85"
                    if "miniwob." in part:
                        prefix = "miniwob."
                    else:  # workarena.
                        prefix = "workarena."
                    
                    # Extract everything after the prefix and split by last underscore
                    task_part = part.split(prefix)[1]
                    # Find the first underscore to separate task name from seed
                    first_underscore_idx = task_part.find("_")
                    if first_underscore_idx != -1:
                        task_name = prefix + task_part[:first_underscore_idx]
                        seed = task_part[first_underscore_idx + 1:]
                    else:
                        task_name = prefix + task_part
                        seed = "unknown_seed"
                    break
            
            # Create unique key for task+seed combination
            experiment_key = f"{task_name}_{seed}"
            print("experiment_key", experiment_key)
            if experiment_key not in experiments:
                experiments[experiment_key] = {
                    "task_name": task_name,
                    "seed": seed,
                    "steps": []
                }
            experiments[experiment_key]["steps"].append(step)
        
        # Convert to trace format
        for experiment_key, experiment_data in experiments.items():
            trace_info = extract_trace_info(experiment_data["steps"])
            if trace_info:
                traces.append({
                    "task_name": experiment_data["task_name"],
                    "seed": experiment_data["seed"],
                    "trace_info": trace_info,
                    "step_count": len(experiment_data["steps"])
                })
        
        self.traces = traces
        return traces

    def build_hint_db(self, output_path: Optional[str] = None):
        """
        Loads (or creates) a hint database CSV, adds new hints for new traces/tasks, and avoids duplicates.
        Each row matches the columns and order of the current hint_db.csv.
        """
        if not self.traces:
            self.load_traces()
        output_path = output_path or self.config.hint_db_path
        
        # Load existing DB if exists
        if os.path.exists(output_path):
            db = pd.read_csv(output_path, dtype=str)
        else:
            db = pd.DataFrame({col: [] for col in self.HINT_DB_COLUMNS})
        
        # Build a set of existing hints to avoid duplicates
        existing = set()
        if not db.empty:
            for i, row in db.iterrows():
                key = (row.get("task_name", ""), row.get("hint", ""))
                existing.add(key)
        
        # Group traces by task name
        tasks_traces = {}
        for trace in self.traces:
            task_name = trace.get("task_name", "unknown_task")
            if task_name not in tasks_traces:
                tasks_traces[task_name] = []
            tasks_traces[task_name].append(trace)
        
        new_rows = []
        for task_name, task_traces in tasks_traces.items():
            if not task_traces:
                continue
            n_traces_to_hinter = self.config.hint_prompt_config.n_traces_to_hinter
            n_hints_per_task = getattr(self.config.hint_prompt_config, 'n_hints_per_task', 1)
            hints_for_this_task = set()
            for _ in range(n_hints_per_task):
                selected_trace_infos = _select_traces_for_hint(task_traces, n_traces_to_hinter)
                if not selected_trace_infos:
                    continue
                # If step-zoom is enabled and only one trace is selected, use the new mechanism
                if self.config.hint_prompt_config.use_step_zoom and len(selected_trace_infos) == 1:
                    trace_info = selected_trace_infos[0]
                    # Use the same LLM for summarizer and judge for now, or allow config
                    summarizer_llm = self.llm
                    msg_builder = self.msg_builder
                    important_indices = summarize_trace_for_important_steps(trace_info, summarizer_llm, msg_builder, self.config.hint_prompt_config)
                    print("Importany indices: ", important_indices)
                    hint_prompt = construct_hint_prompt_step_zoom(trace_info, important_indices, task_name, self.config.hint_prompt_config)
                else:
                    hint_prompt = construct_hint_prompt(
                        selected_trace_infos, task_name,
                        hint_prompt_config=self.config.hint_prompt_config
                    )
                discussion = SimpleDiscussion()
                sys_msg = self.msg_builder.system().add_text(
                    "You are a hint-generating agent. Analyze the following execution trace and propose a helpful hint that is not only useful for this specific task, but is also generalizable to other goals and, ideally, to other tasks. Focus on extracting strategies or principles that could help in similar situations, rather than task-specific details."
                )
                discussion.append(sys_msg)
                trace_msg = self.msg_builder.user().add_text(hint_prompt)
                discussion.append(trace_msg)
                from agentlab.llm.response_api import APIPayload
                payload = APIPayload(messages=discussion.flatten())
                response = self.llm(payload)
                hint = response.think if hasattr(response, "think") else str(response)
                hint = hint.strip()
                hint = " ".join(hint.split())
                if len(hint) > 300:
                    hint = hint[:297] + "..."
                key = (task_name, hint)
                if key in existing or hint in hints_for_this_task:
                    continue
                first_trace = task_traces[0]
                total_steps = sum(len(trace_info) for trace_info in selected_trace_infos)
                row = {
                    "time_stamp": datetime.now().strftime("%b %d"),
                    "task_name": task_name,
                    "task_seed": first_trace.get("seed", "unknown_seed"),
                    "base_llm": getattr(self.model_args, "model_name", "unknown_llm"),
                    "agent_name": self.config.agent_name,
                    "domain_name": self.config.domain_name,
                    "user_name": self.config.user_name,
                    "source": self.config.source,
                    "semantic_keys": f"trace_analysis_{len(selected_trace_infos)}_traces_{total_steps}_steps",
                    "hint": hint,
                }
                new_rows.append(row)
                existing.add(key)
                hints_for_this_task.add(hint)
        
        # Append new rows and save
        if new_rows:
            db = pd.concat([db, pd.DataFrame(new_rows)], ignore_index=True)
            db = db[self.HINT_DB_COLUMNS]  # ensure column order
            # Save with proper CSV formatting to handle quotes and special characters
            db.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL
            print(f"Hint database updated at {output_path} with {len(new_rows)} new rows (total {len(db)}).")
        else:
            print(f"No new hints to add. Database at {output_path} is up to date.")

    def get_action(self, obs: Any) -> Any:
        """
        Stub implementation to satisfy bgym.Agent interface. Not used for hinting.
        """
        raise NotImplementedError("JephHinter does not implement get_action; use build_hint_db instead.")


class MineHints:
    """
    Class to mine hints from execution traces using JephHinter.
    """
    def __init__(self, config: MineHintsConfig):
        self.config = config
        self.root_dir = config.root_dir
        self.output_path = config.output_path or f"{self.root_dir}/hint_db_updated.csv"
        self.exclude_axtree = config.exclude_axtree
        self.exclude_actions = config.exclude_actions
        self.exclude_think = config.exclude_think
        self.exclude_reward = config.exclude_reward
        self.n_traces = config.n_traces
        self.n_hints_per_task = config.n_hints_per_task
        self.use_step_zoom = config.use_step_zoom
        self._setup_model_and_config()
    def _setup_model_and_config(self):
        self.model_args = OpenAIResponseModelArgs(
            model_name="gpt-4o",
            max_total_tokens=128_000,
            max_input_tokens=128_000,
            max_new_tokens=16_384,
            vision_support=False,
        )
        hint_prompt_config = HintPromptConfig(
            exclude_axtree=self.exclude_axtree,
            exclude_actions=self.exclude_actions,
            exclude_think=self.exclude_think,
            exclude_reward=self.exclude_reward,
            n_traces_to_hinter=self.n_traces,
            n_hints_per_task=self.n_hints_per_task,
            use_step_zoom=self.use_step_zoom,
        )
        self.jeph_config = JephHinterConfig(
            traces_folder=self.root_dir,
            max_traces=100,
            hint_db_path="/hint_db.csv",
            agent_name="gpt-4o",
            user_name="auto",
            source="jeph_hinter",
            domain_name="miniwob",
            hint_prompt_config=hint_prompt_config,
        )
    def run(self):
        print(f"Starting hint mining from: {self.root_dir}")
        print(f"Output will be saved to: {self.output_path}")
        agent = JephHinter(model_args=self.model_args, config=self.jeph_config)
        agent.build_hint_db(output_path=self.output_path)
        print("Hint database update complete.")

# Example usage
if __name__ == "__main__":
    config = MineHintsConfig()
    mine_hints = MineHints(config)
    mine_hints.run()