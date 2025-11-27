import importlib, sys, warnings

OLD = __name__
NEW = "agentlab.agents.hint_use_agent"
SUBS = ("agent_configs", "generic_agent_prompt", "generic_agent", "tmlr_config")

warnings.warn(
    f"{OLD} is renamed to {NEW}. {OLD} will be removed in future",
    DeprecationWarning,
    stacklevel=2,
)

# Alias the top-level
new_mod = importlib.import_module(NEW)
sys.modules[OLD] = new_mod

# Alias known submodules
for sub in SUBS:
    sys.modules[f"{OLD}.{sub}"] = importlib.import_module(f"{NEW}.{sub}")
