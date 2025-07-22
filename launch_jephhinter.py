#!/usr/bin/env python3
"""
Python launcher for JephHinter workflow
This script runs the complete workflow:
1. Run experiments without hints
2. Generate hints from traces using JephHinter
3. Run experiments with hints enabled
"""

import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_jephhinter import AgentLabRun
from src.agentlab.agents.tool_use_agent.jeph_hinter import MineHints
from agentlab_configs import AgentLabRunConfig, MineHintsConfig, JephHinterWorkflowConfig

class JephHinterWorkflow:
    """
    Complete workflow for running JephHinter experiments.
    """
    
    def __init__(self, config: JephHinterWorkflowConfig):
        """
        Initialize the workflow.
        
        Args:
            config: Configuration object containing all workflow parameters
        """
        self.config = config
        self.exp_root = config.exp_root
        self.exp_root_no_hint = config.exp_root
        self.exp_root_with_hint = f"{config.exp_root}-with_jeph"
        self.hint_db_path = f"{config.exp_root}/hint_db_updated.csv"
        
        # Environment variables
        self.env_vars = {
            'SNOW_INSTANCE_PWD': config.snow_pwd,
            'SNOW_INSTANCE_URL': config.snow_url,
            'SNOW_INSTANCE_UNAME': config.snow_username,
            'OPENAI_API_KEY': config.openai_key,
            'ANTHROPIC_API_KEY': config.anthropic_key,
            'MINIWOB_URL': config.miniwob_url,
        }
        
        # Set environment variables
        for key, value in self.env_vars.items():
            os.environ[key] = value
    
    def _print_header(self, step_name: str):
        """Print a formatted header for each step."""
        print(f"\n{'='*50}")
        print(f"=== {step_name} ===")
        print(f"{'='*50}")
    
    def _print_config(self):
        """Print the current configuration."""
        print("Configuration:")
        print(f"  No-hint experiments: {self.exp_root_no_hint}")
        print(f"  With-hint experiments: {self.exp_root_with_hint}")
        print(f"  Hint database: {self.hint_db_path}")
        print()
    
    def _create_directories(self):
        """Create necessary directories."""
        print("Creating necessary directories...")
        os.makedirs(self.exp_root_no_hint, exist_ok=True)
        os.makedirs(self.exp_root_with_hint, exist_ok=True)
        print("Directories created successfully!")
        print()
    
    def step1_run_no_hints(self):
        """Step 1: Run experiments without hints."""
        self._print_header("Step 1: Running experiments without hints")
        print(f"Running experiments with exp-root={self.exp_root_no_hint} (hints disabled by default)")
        
        # Create and run AgentLab experiments without hints
        agentlab_run = AgentLabRun(AgentLabRunConfig(
            exp_root=self.exp_root_no_hint,
            use_task_hint=False,
            hint_db_path=self.hint_db_path
        ))
        
        agentlab_run.run()
        print("Step 1 complete!")
    
    def step2_generate_hints(self):
        """Step 2: Generate hints using JephHinter."""
        self._print_header("Step 2: Generating hints from traces")
        print(f"Running JephHinter with root-dir={self.exp_root_no_hint}")
        
        # Create and run hint mining
        mine_hints = MineHints(MineHintsConfig(
            root_dir=self.exp_root_no_hint,
            output_path=self.hint_db_path
        ))
        
        mine_hints.run()
        print("Step 2 complete!")
    
    def step3_run_with_hints(self):
        """Step 3: Run experiments with hints enabled."""
        self._print_header("Step 3: Running experiments with hints")
        
        # Check if hint database exists
        if not os.path.exists(self.hint_db_path):
            print(f"Error: Hint database not found at {self.hint_db_path}")
            print("Please ensure Step 2 completed successfully.")
            return False
        
        print(f"Running experiments with exp-root={self.exp_root_with_hint} (hints enabled)")
        
        # Create and run AgentLab experiments with hints
        agentlab_run = AgentLabRun(AgentLabRunConfig(
            exp_root=self.exp_root_with_hint,
            use_task_hint=True,
            hint_db_path=self.hint_db_path
        ))
        
        agentlab_run.run()
        print("Step 3 complete!")
        return True
    
    def run_complete_workflow(self):
        """Run the complete JephHinter workflow."""
        self._print_header("JephHinter Workflow")
        self._print_config()
        self._create_directories()
        
        # Step 1: Run experiments without hints
        self.step1_run_no_hints()
        
        # Step 2: Generate hints using JephHinter
        self.step2_generate_hints()
        
        # Step 3: Run experiments with hints enabled
        success = self.step3_run_with_hints()
        
        # Summary
        self._print_header("JephHinter Workflow Complete")
        print("Results without hints:", self.exp_root_no_hint)
        print("Results with hints:", self.exp_root_with_hint)
        print("Hint database:", self.hint_db_path)
        
        if success:
            print("\n✅ All steps completed successfully!")
        else:
            print("\n❌ Some steps failed. Please check the output above.")


# Example usage
if __name__ == "__main__":
    config = JephHinterWorkflowConfig()
    workflow = JephHinterWorkflow(config)
    workflow.run_complete_workflow() 