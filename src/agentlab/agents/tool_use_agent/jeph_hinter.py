import os
import json
from dataclasses import dataclass, field
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
from agentlab.llm.response_api import MessageBuilder, OpenAIResponseModelArgs
from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt
from agentlab.agents import dynamic_prompting as dp
import logging
from agentlab.agents.generic_agent.generic_agent import GenericAgent
from agentlab.llm.llm_utils import ParseError, get_tokenizer

@dataclass
class HintPromptConfig:
    include_axtree: bool = False
    include_actions: bool = False
    include_think: bool = False
    include_reward: bool = False

@dataclass
class JephHinterConfig:
    traces_folder: str = "/Users/had.nekoeiqachkanloo/"
    max_traces: int = 100
    hint_db_path: str = "hint_db_updated.csv"
    agent_name: str = "JephHinter"
    user_name: str = "auto"
    source: str = "jeph_hinter"
    domain_name: str = ""
    hint_prompt_config: HintPromptConfig = field(default_factory=HintPromptConfig)

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

def construct_hint_prompt(trace_info, task_name, hint_prompt_config: HintPromptConfig):
    """Construct a comprehensive prompt for hint generation based on trace analysis."""
    prompt_parts = []
    
    # Add system instruction
    prompt_parts.append(f"Task: {task_name}")
    prompt_parts.append("\n=== EXECUTION TRACE ===")
    cum_reward = 0
    # Add trace information
    for i, step in enumerate(trace_info):
        prompt_parts.append(f"\nStep {i+1}:")
        if hint_prompt_config.include_axtree and step.get('axtree_txt'):
            prompt_parts.append(f"AXTree: {step['axtree_txt']}")
        if hint_prompt_config.include_think and step.get('think'):
            prompt_parts.append(f"Agent's reasoning: {step['think']}")
        if hint_prompt_config.include_actions and step.get('action'):
            prompt_parts.append(f"Action taken: {step['action']}")
        if step.get('error'):
            prompt_parts.append(f"Error encountered: {step['error']}")
        if hint_prompt_config.include_reward:
            prompt_parts.append(f"Current reward: {step['reward']}")
        if hint_prompt_config.include_reward:
            cum_reward += step['reward']

    if hint_prompt_config.include_reward:
        if cum_reward == 1:
            prompt_parts.append(f"\nThis was a succeful trace.")
        else:
            prompt_parts.append(f"\nThis was a failed trace.")
    
    # Add analysis request
    prompt_parts.append("\n=== HINT GENERATION ===")
    prompt_parts.append("Based on this trace, provide a concise, actionable hint that would help an agent avoid common mistakes and succeed at this task.")
    prompt_parts.append("IMPORTANT: Keep your hint SHORT (1-2 sentences maximum) and write it as a SINGLE LINE without line breaks.")
    prompt_parts.append("Focus on:")
    prompt_parts.append("- Common pitfalls or errors to avoid")
    prompt_parts.append("- Specific strategies that work well")
    prompt_parts.append("- Important details to pay attention to")
    prompt_parts.append("- Step-by-step guidance if applicable")
    
    return "\n".join(prompt_parts)

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
            # Extract task name and seed from path (assuming structure like .../task_name_seed/step_*.pkl.gz)
            path_parts = file_path.split(os.sep)
            task_name = "unknown_task"
            seed = "unknown_seed"
            
            for part in path_parts:
                if "miniwob." in part or "webarena." in part:
                    # Extract task name from format like "miniwob.drag-items-grid_17"
                    if "miniwob." in part:
                        task_parts = part.split("miniwob.")[1].split("_")
                        task_name = "miniwob." + task_parts[0]  # Extract "drag-items-grid"
                        if len(task_parts) > 1:
                            seed = task_parts[1]  # Extract "17"
                    elif "webarena." in part:
                        task_parts = part.split("webarena.")[1].split("_")
                        task_name = "webarena." + task_parts[0]  # Extract task name
                        if len(task_parts) > 1:
                            seed = task_parts[1]  # Extract seed
                    break
            
            # Create unique key for task+seed combination
            experiment_key = f"{task_name}_{seed}"
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
        
        new_rows = []
        for trace in self.traces:
            task_name = trace.get("task_name", "unknown_task")
            trace_info = trace.get("trace_info", [])
            
            if not trace_info:
                continue
            
            # Construct comprehensive prompt for hint generation
            hint_prompt = construct_hint_prompt(
                trace_info, task_name,
                hint_prompt_config=self.config.hint_prompt_config
            )
            
            # Use LLM to generate a hint
            discussion = SimpleDiscussion()
            sys_msg = self.msg_builder.system().add_text(
                "You are a hint-generating agent. Analyze the following execution trace and propose a helpful hint that is not only useful for this specific task, but is also generalizable to other goals and, ideally, to other tasks. Focus on extracting strategies or principles that could help in similar situations, rather than task-specific details."
            )
            discussion.append(sys_msg)
            trace_msg = self.msg_builder.user().add_text(hint_prompt)
            discussion.append(trace_msg)
            
            response = self.llm(messages=discussion.flatten())
            hint = response.think if hasattr(response, "think") else str(response)
            
            # Clean up hint to ensure it's short and single-line
            hint = hint.strip()
            # Remove line breaks and extra whitespace
            hint = " ".join(hint.split())
            # Truncate if too long (keep under 200 characters)
            if len(hint) > 200:
                hint = hint[:197] + "..."
            
            # Check for duplicates
            key = (task_name, hint)
            if key in existing:
                continue
            
            # Create new row
            row = {
                "time_stamp": datetime.now().strftime("%b %d"),
                "task_name": task_name,
                "task_seed": trace.get("seed", "unknown_seed"),
                "base_llm": getattr(self.model_args, "model_name", "unknown_llm"),
                "agent_name": self.config.agent_name,
                "domain_name": self.config.domain_name,
                "user_name": self.config.user_name,
                "source": self.config.source,
                "semantic_keys": f"trace_analysis_{len(trace_info)}_steps",
                "hint": hint,
            }
            new_rows.append(row)
            existing.add(key)
        
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


# Example usage with OpenAI API
if __name__ == "__main__":
    import os
    import argparse
    from agentlab.llm.response_api import OpenAIResponseModelArgs
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run JephHinter to generate hints from traces')
    parser.add_argument('--root-dir', type=str, 
                       default="/Users/had.nekoeiqachkanloo/hadi/AgentLab/agentlab_results_no_hint_miniwob10",
                       help='Root directory containing trace files')
    parser.add_argument('--output-path', type=str,
                       default=None,
                       help='Output path for hint database (defaults to root_dir/hint_db_updated.csv)')
    parser.add_argument('--openai-key', type=str,
                       default="OPENAI_KEY",
                       help='OpenAI API key')
    parser.add_argument('--include-axtree', action='store_true', help='Include axtree in the hint prompt')
    parser.add_argument('--include-actions', action='store_true', help='Include actions in the hint prompt')
    parser.add_argument('--include-think', action='store_true', help='Include think in the hint prompt')
    parser.add_argument('--include-reward', action='store_true', help='Include reward in the hint prompt')

    args = parser.parse_args()
    
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = args.openai_key

    # Initialize the model arguments
    model_args = OpenAIResponseModelArgs(
        model_name="gpt-4o",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=False,
    )

    # Configure the JephHinter agent
    root_dir = args.root_dir
    hint_prompt_config = HintPromptConfig(
        include_axtree=args.include_axtree,
        include_actions=args.include_actions,
        include_think=args.include_think,
        include_reward=args.include_reward,
    )
    config = JephHinterConfig(
        traces_folder=root_dir,
        max_traces=100,
        hint_db_path="/hint_db.csv",
        agent_name="gpt-4o",
        user_name="auto",
        source="jeph_hinter",
        domain_name="miniwob",
        hint_prompt_config=hint_prompt_config,
    )

    # Create and run the agent
    agent = JephHinter(model_args=model_args, config=config)
    output_path = args.output_path or f"{root_dir}/hint_db_updated.csv"
    agent.build_hint_db(output_path=output_path)
    
    print("Hint database update complete.")