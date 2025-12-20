# Testing README Examples

This document describes how to run unit tests that verify all code examples from the main [README.md](README.md) work correctly.

## Overview

The test file `tests/test_readme_examples.py` contains comprehensive unit tests for all code snippets and examples shown in the README. These tests ensure that:

- All imports are valid and accessible
- CLI commands exist and are functional  
- Core functions and classes work as documented
- File paths referenced in README exist
- API structures match the documentation

## Prerequisites

Before running the tests, ensure you have:

1. **Python 3.11 or 3.12** (or use the uv virtual environment)
2. **AgentLab installed** with all dependencies:
   ```bash
   uv sync
   ```

3. **Playwright browsers installed**:
   ```bash
   uv run playwright install
   ```

## Running the Tests

### Run All README Tests

To run all tests for README examples:

```bash
uv run pytest tests/test_readme_examples.py -v
```

### Run Specific Test Classes

You can run tests for specific README sections:

```bash
# Test installation and setup
uv run pytest tests/test_readme_examples.py::TestReadmeInstallationAndSetup -v

# Test UI-Assistant examples
uv run pytest tests/test_readme_examples.py::TestReadmeUIAssistant -v

# Test experiment launching examples
uv run pytest tests/test_readme_examples.py::TestReadmeLaunchExperiments -v

# Test analysis examples
uv run pytest tests/test_readme_examples.py::TestReadmeAnalyseResults -v

# Test AgentXray examples
uv run pytest tests/test_readme_examples.py::TestReadmeAgentXray -v

# Test new agent implementation examples
uv run pytest tests/test_readme_examples.py::TestReadmeImplementNewAgent -v

# Test reproducibility features
uv run pytest tests/test_readme_examples.py::TestReadmeReproducibility -v

# Test benchmark examples
uv run pytest tests/test_readme_examples.py::TestReadmeBenchmarks -v
```

### Run Individual Tests

To run a specific test function:

```bash
uv run pytest tests/test_readme_examples.py::TestReadmeLaunchExperiments::test_make_study_creates_study -v
```

## Test Coverage by README Section

### ✅ Installation and Setup (Lines 72-87)

Tests verify:
- `pip install agentlab` works
- `playwright install` command is available
- Package imports succeed

**Related tests:**
- `TestReadmeInstallationAndSetup::test_agentlab_package_installed`
- `TestReadmeInstallationAndSetup::test_playwright_install_command_exists`

### ✅ UI-Assistant (Lines 110-117)

Tests verify:
- `agentlab-assistant` CLI command exists
- Command accepts `--start_url` and `--agent_config` flags
- Generic agent imports work

**Related tests:**
- `TestReadmeUIAssistant::test_agentlab_assistant_command_exists`
- `TestReadmeUIAssistant::test_generic_agent_import`

### ✅ Launch Experiments (Lines 122-149)

Tests verify:
- `make_study()` function works correctly
- `Study.load()` method exists
- `study.find_incomplete()` method exists
- `study.run()` method exists
- All agent imports are valid

**Related tests:**
- `TestReadmeLaunchExperiments::test_make_study_creates_study`
- `TestReadmeLaunchExperiments::test_study_load_import`
- `TestReadmeLaunchExperiments::test_study_find_incomplete`
- `TestReadmeLaunchExperiments::test_agent_imports`

### ✅ main.py Examples (Line 147)

Tests verify:
- `main.py` file exists in repository
- All agent imports from `main.py` work
- Study class can be imported

**Related tests:**
- `TestReadmeMainPy::test_main_py_exists`
- `TestReadmeMainPy::test_all_agent_imports_from_main`
- `TestReadmeMainPy::test_study_import_from_main`

### ✅ Analyse Results (Lines 193-203)

Tests verify:
- `inspect_results` module imports correctly
- `load_result_df()` function exists
- `ExpResult` class is accessible

**Related tests:**
- `TestReadmeAnalyseResults::test_inspect_results_import`
- `TestReadmeAnalyseResults::test_load_result_df_function`
- `TestReadmeAnalyseResults::test_exp_result_class`

### ✅ AgentXray (Lines 210-226)

Tests verify:
- `agentlab-xray` CLI command exists and is runnable

**Related tests:**
- `TestReadmeAgentXray::test_agentlab_xray_command_exists`

### ✅ Implement a New Agent (Lines 239-245)

Tests verify:
- `MostBasicAgent` file exists at documented path
- `AgentArgs` API file exists
- `AgentArgs` class can be imported

**Related tests:**
- `TestReadmeImplementNewAgent::test_most_basic_agent_file_exists`
- `TestReadmeImplementNewAgent::test_agent_args_file_exists`
- `TestReadmeImplementNewAgent::test_agent_args_api`

### ✅ Reproducibility (Lines 265-278)

Tests verify:
- `reproducibility_journal.csv` exists
- `ReproducibilityAgent` file exists at documented path
- Study class supports reproducibility features

**Related tests:**
- `TestReadmeReproducibility::test_reproducibility_journal_exists`
- `TestReadmeReproducibility::test_reproducibility_agent_exists`
- `TestReadmeReproducibility::test_study_has_reproducibility_info`

### ✅ Supported Benchmarks (Lines 50-66)

Tests verify:
- Benchmark names (like "miniwob") work with `make_study()`

**Related tests:**
- `TestReadmeBenchmarks::test_miniwob_benchmark_accessible`

## Understanding Test Results

### Successful Test Output

When all tests pass, you'll see:
```
tests/test_readme_examples.py::TestReadmeInstallationAndSetup::test_agentlab_package_installed PASSED
tests/test_readme_examples.py::TestReadmeLaunchExperiments::test_make_study_creates_study PASSED
...
======================== XX passed in X.XXs ========================
```

### Failed Test Output

If a test fails, you'll see detailed error information:
```
tests/test_readme_examples.py::TestReadmeUIAssistant::test_agentlab_assistant_command_exists FAILED

FAILED tests/test_readme_examples.py::TestReadmeUIAssistant::test_agentlab_assistant_command_exists
AssertionError: agentlab-assistant command should work
```

This indicates that the README example may be outdated or there's an installation issue.

## Notes

- **API Keys Not Required**: These tests verify code structure and imports, not actual experiment execution. You don't need API keys (OPENAI_API_KEY, etc.) to run these tests.

- **No Actual Experiments**: Tests that call `make_study()` verify the function works but don't call `study.run()`, which would require:
  - Configured API keys
  - Set up benchmark environments
  - Significant time and resources

- **CLI Command Tests**: Tests for `agentlab-assistant` and `agentlab-xray` verify the commands exist and respond to `--help`, but don't actually launch the UIs.

- **File Existence Tests**: Some tests verify that files mentioned in README (like `main.py`, `reproducibility_journal.csv`) exist at their documented locations.

## Continuous Integration

These tests are ideal for CI/CD pipelines to ensure README examples stay up-to-date with code changes.

Example GitHub Actions workflow:
```yaml
- name: Test README Examples
  run: uv run pytest tests/test_readme_examples.py -v
```

## Troubleshooting

### Test fails with "ModuleNotFoundError"

Make sure you've installed all dependencies:
```bash
uv sync
```

### Test fails with "playwright not found"

Install Playwright browsers:
```bash
uv run playwright install
```

### Test fails with "File not found"

Ensure you're running tests from the repository root directory:
```bash
cd /path/to/AgentLab
uv run pytest tests/test_readme_examples.py -v
```

## Contributing

When updating the README:

1. **Update code examples** in `README.md`
2. **Update corresponding tests** in `tests/test_readme_examples.py`
3. **Run tests** to verify:
   ```bash
   uv run pytest tests/test_readme_examples.py -v
   ```
4. **Update this document** if test coverage changes

## Related Documentation

- [Main README](README.md) - Complete AgentLab documentation
- [BrowserGym Documentation](https://github.com/ServiceNow/BrowserGym)
- [Contributing Guidelines](CONTRIBUTING.md) (if applicable)
