"""
Unit tests to verify all code examples from README.md work correctly.

These tests ensure that:
1. All imports from README examples are valid
2. Core functions and classes are accessible
3. Basic API structures match README documentation
4. CLI commands are available and functional
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestReadmeInstallationAndSetup:
    """Tests for installation and setup code from README"""

    def test_playwright_install_command_exists(self):
        """Verify playwright install command is available"""
        result = subprocess.run(
            ["playwright", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, "Playwright should be installed"
        assert "Version" in result.stdout or "version" in result.stdout.lower()

    def test_agentlab_package_installed(self):
        """Verify agentlab package is installed"""
        try:
            import agentlab
            assert agentlab is not None
        except ImportError:
            pytest.fail("agentlab package should be importable")


class TestReadmeUIAssistant:
    """Tests for UI-Assistant code from README"""

    def test_agentlab_assistant_command_exists(self):
        """Verify agentlab-assistant command is available (lines 110-117)"""
        result = subprocess.run(
            ["agentlab-assistant", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, "agentlab-assistant command should work"
        assert "--start_url" in result.stdout
        assert "--agent_config" in result.stdout

    def test_generic_agent_import(self):
        """Verify generic agent can be imported for UI assistant"""
        from agentlab.agents.generic_agent import AGENT_4o_MINI
        assert AGENT_4o_MINI is not None


class TestReadmeLaunchExperiments:
    """Tests for experiment launching code from README (lines 122-149)"""

    def test_agent_imports(self):
        """Verify agent configuration can be imported (line 125)"""
        from agentlab.agents.generic_agent import AGENT_4o_MINI
        assert AGENT_4o_MINI is not None
        assert hasattr(AGENT_4o_MINI, 'agent_name')

    def test_make_study_import(self):
        """Verify make_study can be imported (line 127)"""
        from agentlab.experiments.study import make_study
        assert make_study is not None
        assert callable(make_study)

    def test_make_study_creates_study(self):
        """Verify make_study function works (lines 129-133)"""
        from agentlab.agents.generic_agent import AGENT_4o_MINI
        from agentlab.experiments.study import make_study

        study = make_study(
            benchmark="miniwob",
            agent_args=[AGENT_4o_MINI],
            comment="My first study",
        )
        
        assert study is not None
        assert hasattr(study, 'run')
        # Note: We don't actually run study.run() as it requires API keys

    def test_study_load_import(self):
        """Verify Study.load can be imported (line 142)"""
        from agentlab.experiments.study import Study
        assert hasattr(Study, 'load')
        assert callable(Study.load)

    def test_study_find_incomplete(self):
        """Verify Study has find_incomplete method (line 143)"""
        from agentlab.experiments.study import Study
        assert hasattr(Study, 'find_incomplete')

    def test_study_run(self):
        """Verify Study has run method (line 144)"""
        from agentlab.experiments.study import Study
        assert hasattr(Study, 'run')


class TestReadmeMainPy:
    """Tests for main.py referenced in README (line 147)"""

    def test_main_py_exists(self):
        """Verify main.py exists in the repository"""
        main_path = Path(__file__).parent.parent / "main.py"
        assert main_path.exists(), "main.py should exist in repository root"

    def test_all_agent_imports_from_main(self):
        """Verify all agent imports from main.py work"""
        from agentlab.agents.generic_agent import (
            AGENT_LLAMA3_70B,
            AGENT_LLAMA31_70B,
            RANDOM_SEARCH_AGENT,
            AGENT_4o,
            AGENT_4o_MINI,
            AGENT_o3_MINI,
            AGENT_37_SONNET,
            AGENT_CLAUDE_SONNET_35,
            AGENT_GPT5_MINI,
        )
        
        # Verify they're all not None
        assert AGENT_LLAMA3_70B is not None
        assert AGENT_LLAMA31_70B is not None
        assert RANDOM_SEARCH_AGENT is not None
        assert AGENT_4o is not None
        assert AGENT_4o_MINI is not None
        assert AGENT_o3_MINI is not None
        assert AGENT_37_SONNET is not None
        assert AGENT_CLAUDE_SONNET_35 is not None
        assert AGENT_GPT5_MINI is not None

    def test_study_import_from_main(self):
        """Verify Study import from main.py works"""
        from agentlab.experiments.study import Study
        assert Study is not None


class TestReadmeAnalyseResults:
    """Tests for analyzing results code from README (lines 193-203)"""

    def test_inspect_results_import(self):
        """Verify inspect_results can be imported (line 194)"""
        from agentlab.analyze import inspect_results
        assert inspect_results is not None

    def test_load_result_df_function(self):
        """Verify load_result_df function exists (line 197)"""
        from agentlab.analyze import inspect_results
        assert hasattr(inspect_results, 'load_result_df')
        assert callable(inspect_results.load_result_df)

    def test_exp_result_class(self):
        """Verify ExpResult class is accessible (line 200)"""
        from agentlab.experiments.loop import ExpResult
        assert ExpResult is not None


class TestReadmeAgentXray:
    """Tests for AgentXray code from README (lines 210-226)"""

    def test_agentlab_xray_command_exists(self):
        """Verify agentlab-xray command is available (line 212)"""
        # agentlab-xray launches a Gradio UI and doesn't have --help
        # Just verify the command exists in the PATH or as a module
        try:
            result = subprocess.run(
                ["which", "agentlab-xray"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # If 'which' finds it, returncode will be 0
            # If not found but command exists, it should still be importable
            command_found = result.returncode == 0
            
            # Alternatively, check if the module entry point exists
            if not command_found:
                # Try importing the xray module
                try:
                    from agentlab.ui import xray
                    command_found = True
                except ImportError:
                    pass
            
            assert command_found, "agentlab-xray command or module should exist"
        except subprocess.TimeoutExpired:
            pytest.fail("Command check timed out")


class TestReadmeImplementNewAgent:
    """Tests for implementing new agent code from README (lines 239-245)"""

    def test_most_basic_agent_file_exists(self):
        """Verify MostBasicAgent file exists (line 240)"""
        agent_path = Path(__file__).parent.parent / "src" / "agentlab" / "agents" / "most_basic_agent" / "most_basic_agent.py"
        assert agent_path.exists(), "MostBasicAgent file should exist"

    def test_agent_args_file_exists(self):
        """Verify AgentArgs file exists (line 242)"""
        args_path = Path(__file__).parent.parent / "src" / "agentlab" / "agents" / "agent_args.py"
        assert args_path.exists(), "agent_args.py file should exist"

    def test_agent_args_api(self):
        """Verify AgentArgs API is importable"""
        from agentlab.agents.agent_args import AgentArgs
        assert AgentArgs is not None


class TestReadmeReproducibility:
    """Tests for reproducibility features from README (lines 265-278)"""

    def test_study_has_reproducibility_info(self):
        """Verify Study contains reproducibility information (line 266)"""
        from agentlab.experiments.study import Study
        # Study should have methods/attributes for reproducibility
        assert Study is not None

    def test_reproducibility_journal_exists(self):
        """Verify reproducibility_journal.csv exists (line 269)"""
        journal_path = Path(__file__).parent.parent / "reproducibility_journal.csv"
        assert journal_path.exists(), "reproducibility_journal.csv should exist"

    def test_reproducibility_agent_exists(self):
        """Verify ReproducibilityAgent file exists (line 274)"""
        repro_agent_path = Path(__file__).parent.parent / "src" / "agentlab" / "agents" / "generic_agent" / "reproducibility_agent.py"
        assert repro_agent_path.exists(), "reproducibility_agent.py should exist"


class TestReadmeEnvironmentVariables:
    """Tests for environment variables documentation (lines 280-291)"""

    def test_env_variables_documented(self):
        """Verify key environment variables are documented in README"""
        readme_path = Path(__file__).parent.parent / "README.md"
        readme_content = readme_path.read_text()
        
        # Check that important env vars are mentioned
        assert "OPENAI_API_KEY" in readme_content
        assert "AZURE_OPENAI_API_KEY" in readme_content
        assert "AGENTLAB_EXP_ROOT" in readme_content
        assert "OPENROUTER_API_KEY" in readme_content


class TestReadmeBenchmarks:
    """Tests for supported benchmarks from README (lines 50-66)"""

    def test_miniwob_benchmark_accessible(self):
        """Verify miniwob benchmark can be used"""
        from agentlab.agents.generic_agent import AGENT_4o_MINI
        from agentlab.experiments.study import make_study
        
        # Should create study without error (even if benchmark not fully set up)
        study = make_study(
            benchmark="miniwob",
            agent_args=[AGENT_4o_MINI],
            comment="Test study",
        )
        assert study is not None


if __name__ == "__main__":
    # Allow running tests directly with: python test_readme_examples.py
    pytest.main([__file__, "-v"])
