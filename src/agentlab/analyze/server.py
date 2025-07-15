# server.py
import base64
import copy
import importlib
from typing import Any

import dotenv
import numpy as np
import uvicorn

# Import your BrowserEnv and any task setup you need
from bgym import DEFAULT_BENCHMARKS
from fastapi import FastAPI
from pydantic import BaseModel

dotenv.load_dotenv()

app = FastAPI()


def import_from_path(path: str) -> callable:
    """
    Util function to import and instantiate a class, then return a specific method.

    Args:
        path (str): Path to the method, e.g., 'browsergym.core.action.highlevel.HighLevelActionSet.to_python_code'.

    Raises:
        ModuleNotFoundError: If the module cannot be imported.

    Returns:
        callable: The method.
    """

    parts = path.split(".")
    # Find the module (the longest prefix that can be imported)
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue
    else:
        raise ModuleNotFoundError(f"Could not import module from path: {path}")

    obj = module
    for attr in parts[i:]:
        obj = getattr(obj, attr)

    # If the final object is a method, and its __qualname__ contains a class, instantiate the class
    if callable(obj) and hasattr(obj, "__qualname__") and "." in obj.__qualname__:
        class_name = obj.__qualname__.split(".")[0]
        cls = getattr(module, class_name)
        instance = cls()
        method = getattr(instance, obj.__name__)
        return method

    return obj


def make_json_safe(obj: Any) -> Any:
    """
    Util function to convert numpy arrays and other non-JSON-serializable objects to JSON-serializable objects.
    Specifically, we convert numpy arrays to base64 encoded strings so that payloads are of reasonable size.

    Args:
        obj (Any): Object to convert

    Returns:
        Any: JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        # convert to base64
        return {
            "data": base64.b64encode(obj.tobytes()).decode("utf-8"),
            "shape": obj.shape,
            "dtype": str(obj.dtype),
        }
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return make_json_safe(vars(obj))
    else:
        return obj


# --- Models for requests ---
class SetInfoRequest(BaseModel):
    benchmark_name: str
    task_name: str
    seed: int
    action_mapping_fn: str
    exp_dir: str


class StepRequest(BaseModel):
    action: str


# --- Persistent Environment State ---
class EnvWrapper:
    def __init__(self):

        # env info
        self.benchmark_name = None
        self.task_name = None
        self.seed = None
        self.action_mapping_fn = None
        self.exp_dir = None
        self.info_set = False

        # env state
        self.env = None
        self.last_obs = None
        self.last_info = None

        # used to reload task
        self.start_info = None
        self.start_url = None

    def set_info(
        self,
        benchmark_name: str,
        task_name: str,
        seed: int,
        action_mapping_fn: str,
        exp_dir: str,
    ) -> dict:
        """
        Set the environment info.

        Args:
            benchmark_name (str): Name of the benchmark
            task_name (str): Name of the task
            seed (int): Seed of the task
            action_mapping_fn (str): Action mapping function
            exp_dir (str): Directory for experiment

        Returns:
            dict: Dictionary with status
        """
        if self.info_set:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment info already set. Please unset the environment info first.",
                }
            )
        if self.env is not None:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment already created. Close the current environment proceeding.",
                }
            )
        self.benchmark_name = benchmark_name
        self.task_name = task_name
        self.seed = seed
        self.action_mapping_fn = action_mapping_fn
        self.exp_dir = exp_dir
        self.info_set = True

        return make_json_safe(
            {
                "status": "success",
                "message": "Environment info set successfully.",
            }
        )

    def get_info(self) -> dict:
        """
        Get the environment info.

        Returns:
            dict: Dictionary with info
        """
        if not self.info_set:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment info not set. Please set the environment info first.",
                }
            )
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment info retrieved successfully.",
                "benchmark_name": self.benchmark_name,
                "task_name": self.task_name,
                "seed": self.seed,
                "action_mapping_fn": self.action_mapping_fn,
                "exp_dir": self.exp_dir,
            }
        )

    def unset_info(self) -> dict:
        """
        Unset the environment info.

        Returns:
            dict: Dictionary with status
        """
        if not self.info_set:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment info not set. Please set the environment info first.",
                }
            )
        self.info_set = False
        self.benchmark_name = None
        self.task_name = None
        self.seed = None
        self.action_mapping_fn = None
        self.exp_dir = None
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment info unset successfully.",
            }
        )

    def status(self) -> dict:
        """
        Get the environment status.

        Returns:
            dict: Dictionary with status
        """
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment status retrieved successfully.",
                "obs": self.last_obs,
                "reward": 0,
                "terminated": False,
                "truncated": False,
                "info_set": self.info_set,
                "env_created": self.env is not None,
            }
        )

    def prepare_benchmark(self) -> dict:
        """
        Prepare the benchmark environment.

        Returns:
            dict: Dictionary with status
        """
        if not self.info_set:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment info not set. Please set the environment info first.",
                }
            )

        if self.env is not None:
            # close the current environment first
            self.env.close()
            self.env = None

        # prepare backends
        benchmark = DEFAULT_BENCHMARKS[self.benchmark_name]()
        benchmark.env_args_list = [
            elem
            for elem in benchmark.env_args_list
            if elem.task_name == self.task_name and str(elem.task_seed) == str(self.seed)
        ]
        benchmark.prepare_backends()

        env_args = benchmark.env_args_list[0]
        self.action_mapping = import_from_path(self.action_mapping_fn)

        # create environment
        self.env = env_args.make_env(self.action_mapping, self.exp_dir)
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment prepared successfully.",
            }
        )

    def reload_task(self) -> dict:
        """
        Reload the task.

        Returns:
            dict: Dictionary with status
        """
        if not self.info_set:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment info not set. Please set the environment info first.",
                }
            )
        elif not self.env:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment not created. Please create an environment first.",
                }
            )

        # instead of resetting the whole environment, we go back to the original webpage and clear localStorage and sessionStorage
        # NOTE: this is not guaranteed to result in the exact same state, but we find that it works most of the time, is much
        # faster than resetting the whole environment, and ensures the seed of the environment remains the same
        self.env.unwrapped.page.goto(self.start_url, wait_until="load")
        self.env.unwrapped.page.evaluate(
            "window.localStorage.clear(); window.sessionStorage.clear();"
        )
        obs = self.env.unwrapped._get_obs()

        self.last_obs = copy.deepcopy(obs)
        self.last_info = copy.deepcopy(self.start_info)
        return make_json_safe(
            {
                "status": "success",
                "message": "Task reloaded successfully.",
                "obs": self.last_obs,
                "reward": 0,
                "terminated": False,
                "truncated": False,
                "info": self.last_info,
            }
        )

    def reset(self) -> dict:
        """
        Reset the environment.

        Returns:
            dict: Dictionary with obs and info
        """
        if not self.info_set:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment info not set. Please set the environment info first.",
                }
            )
        elif not self.env:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment not created. Please create an environment first.",
                }
            )

        # reset the environment
        obs, info = self.env.reset(seed=self.seed)

        self.last_obs = copy.deepcopy(obs)
        self.last_info = copy.deepcopy(info)
        self.start_info = copy.deepcopy(info)
        self.start_url = copy.deepcopy(self.env.unwrapped.page.url)
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment reset successfully",
                "obs": self.last_obs,
                "reward": 0,
                "terminated": False,
                "truncated": False,
                "info": self.last_info,
            }
        )

    def step(self, action: str) -> dict:
        """
        Step the environment.

        Args:
            action (str): Action to take

        Returns:
            dict: Dictionary with obs, reward, terminated, truncated and info
        """
        if self.env is None:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment not created. Please create an environment first.",
                }
            )
        # step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.last_obs = copy.deepcopy(obs)
        self.last_info = copy.deepcopy(info)
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment stepped successfully.",
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )

    def get_obs(self) -> dict:
        """
        Get the last observation.

        Returns:
            dict: Dictionary with obs and info
        """
        if self.env is None:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment not created. Please create an environment first.",
                }
            )
        return make_json_safe(
            {
                "status": "success",
                "message": "Observation retrieved successfully.",
                "obs": self.last_obs,
                "reward": 0,
                "terminated": False,
                "truncated": False,
                "info": self.last_info,
            }
        )

    def close(self) -> dict:
        """
        Close the environment.

        Returns:
            dict: Dictionary with status
        """
        if self.env is None:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment not created. Please create an environment first.",
                }
            )
        self.env.close()
        self.env = None
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment closed successfully.",
            }
        )


env = EnvWrapper()


# --- FastAPI endpoints ---
@app.post("/set_info")
def set_info(req: SetInfoRequest) -> dict:
    """
    Set the environment info.

    Args:
        req (SetInfoRequest): Request containing environment info

    Returns:
        dict: Dictionary with status
    """
    return env.set_info(
        benchmark_name=req.benchmark_name,
        task_name=req.task_name,
        seed=req.seed,
        action_mapping_fn=req.action_mapping_fn,
        exp_dir=req.exp_dir,
    )


@app.get("/get_info")
def get_info() -> dict:
    """
    Get the environment info.

    Returns:
        dict: Dictionary with info
    """
    return env.get_info()


@app.post("/unset_info")
def unset_info() -> dict:
    """
    Unset the environment info.

    Returns:
        dict: Dictionary with status
    """
    return env.unset_info()


@app.get("/status")
def status() -> dict:
    """
    Get the status of the environment.

    Returns:
        dict: Dictionary with status
    """
    return env.status()


@app.post("/prepare_benchmark")
def prepare_benchmark() -> dict:
    """
    Prepare the benchmark.

    Returns:
        dict: Dictionary with status
    """
    return env.prepare_benchmark()


@app.post("/reload_task")
def reload_task() -> dict:
    """
    Reload the task.

    Returns:
        dict: Dictionary with status
    """
    return env.reload_task()


@app.post("/reset")
def reset() -> dict:
    """
    Reset the environment.

    Returns:
        dict: Dictionary with status
    """
    return env.reset()


@app.post("/step")
def step(req: StepRequest) -> dict:
    """
    Step the environment.

    Args:
        req (StepRequest): Request containing action

    Returns:
        dict: Dictionary with obs, reward, terminated, truncated and info
    """
    return env.step(action=req.action)


@app.get("/get_obs")
def get_obs() -> dict:
    """
    Get the last observation.

    Returns:
        dict: Dictionary with obs and info
    """
    return env.get_obs()


@app.post("/close")
def close() -> dict:
    """
    Close the environment.

    Returns:
        dict: Dictionary with status
    """
    return env.close()


def main():
    uvicorn.run("agentlab.analyze.server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
