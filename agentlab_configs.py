"""
Configuration dataclasses for AgentLab JephHinter workflow.
Centralized configuration management for all components.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HintPromptConfig:
    """Configuration for hint prompt generation."""
    exclude_axtree: bool = False
    exclude_actions: bool = False
    exclude_think: bool = False
    exclude_reward: bool = False
    n_traces_to_hinter: int = 1
    n_hints_per_task: int = 1
    use_step_zoom: bool = False


@dataclass
class JephHinterConfig:
    """Configuration for JephHinter agent."""
    traces_folder: str = "/home/toolkit/AgentLab/agentlab_results_miniwib/agentlab_results_miniwib10"
    max_traces: int = 100
    hint_db_path: str = "hint_db.csv"
    agent_name: str = "JephHinter"
    user_name: str = "auto"
    source: str = "jeph_hinter"
    domain_name: str = ""
    hint_prompt_config: HintPromptConfig = field(default_factory=HintPromptConfig)


@dataclass
class AgentLabRunConfig:
    """Configuration for AgentLab experiment runs."""
    exp_root: str = "/home/toolkit/AgentLab/agentlab_results_no_hint/"
    use_task_hint: bool = False
    hint_db_path: Optional[str] = None
    reproducibility_mode: bool = False
    relaunch: bool = False
    n_jobs: int = 5


@dataclass
class MineHintsConfig:
    """Configuration for hint mining process."""
    root_dir: str = "/home/toolkit/AgentLab/agentlab_results_miniwib/agentlab_results_miniwib10"
    output_path: Optional[str] = None
    exclude_axtree: bool = False
    exclude_actions: bool = False
    exclude_think: bool = False
    exclude_reward: bool = False
    n_traces: int = 2
    n_hints_per_task: int = 1
    use_step_zoom: bool = False


@dataclass
class JephHinterWorkflowConfig:
    """Configuration for the complete JephHinter workflow."""
    exp_root: str = "/home/toolkit/AgentLab/agentlab_results_no_hint/"
    snow_pwd: str = 'SNOW_INSTANCE_PWD'
    snow_url: str = 'https://researchworkarenademo.service-now.com'
    snow_username: str = 'admin'
    openai_key: str = 'OPENAI_API_KEY'
    anthropic_key: str = 'ANTHROPIC_API_KEY'
    miniwob_url: str = 'MINIWOB_URL' 