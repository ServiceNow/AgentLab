## Building LLM Agents in Your Library

This tutorial will guide you through the process of subclassing the `Agent` class to create agents that can interact with a `browsergym` environment. We'll cover the following steps:

1. Subclassing the `Agent` class
2. Implementing the `get_action` method
3. Customizing the `obs_preprocessor` method (optional)
4. Creating a compatible action set
5. Defining agent arguments
6. Running experiments with your agent

### Step 1: Subclassing the `Agent` Class

To create a custom agent, you need to subclass the `Agent` class and implement the abstract `get_action` method.

```python
from browsergym.experiment.loop import AbstractActionSet, DEFAULT_ACTION_SET
from browsergym.experiment.agent import Agent

class CustomAgent(Agent):
    def __init__(self, action_set: AbstractActionSet = DEFAULT_ACTION_SET):
        self.action_set = action_set

    def obs_preprocessor(self, obs: dict) -> Any:
        # Optionally override this method to customize observation preprocessing
        return super().obs_preprocessor(obs)

    def get_action(self, obs: Any) -> tuple[str, dict]:
        # Implement your custom logic here
        action = "your_action"
        info = {"custom_info": "details"}
        return action, info
```

### Step 2: Implementing the `get_action` Method

The `get_action` method updates the agent with the current observation and returns the next action along with optional additional information.

```python
def get_action(self, obs: Any) -> tuple[str, dict]:
    # Example implementation
    output = self.LLM(obs)
    action = output["action"]
    info = {
        "think": output["think"],
        "messages": output["messages"],
        "stats": output["stats"]
    }
    return action, info
```

The info dictionnary is saved in the logs of the experiment. It can be used to store any information you want to keep track of during the experiment.

### Step 3: Customizing the `obs_preprocessor` Method (Optional)

The `obs_preprocessor` method preprocesses observations before they are fed to the `get_action` method. You can override this method to implement custom preprocessing logic.

```python
def obs_preprocessor(self, obs: dict) -> Any:
    # Example preprocessing logic
    obs["custom_key"] = "custom_value"
    return obs
```


### Step 4: Creating a Compatible Action Set

Your agent must use an action set that conforms to the `AbstractActionSet` class. The library provides a `HighLevelActionSet` with pre-implemented actions that you can use directly or customize.

```python
class CustomActionSet(AbstractActionSet):
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        return "Custom action set description."

    def example_action(self, abstract: bool) -> str:
        return "Example actions for in context learning."

    def to_python_code(self, action) -> str:
        return "executable python code"
```

### Step 5: Defining Agent Arguments

Define a class that inherits from `AbstractAgentArgs` to specify the arguments required to instantiate your agent.

```python
from dataclasses import dataclass
from browsergym.experiment.agent import AbstractAgentArgs, Agent

@dataclass
class CustomAgentArgs(AbstractAgentArgs):
    custom_params: dict

    def make_agent(self) -> Agent:
        return CustomAgent(self.custom_params)
```

### Step 6: Running Experiments with Your Agent

To run experiments with your custom agent, define an instance of `ExpArgs` with the required parameters.

```python
from browsergym.experiment.loop import ExpArgs

exp_args = ExpArgs(
    agent_args=CustomAgentArgs(custom_param="value"),
    env_args=env_args,
    exp_dir="./experiments",
    exp_name="custom_experiment",
    enable_debug=True
)

# Run the experiment
exp_args.prepare()
exp_args.run()
```
