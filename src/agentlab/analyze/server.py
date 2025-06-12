# server.py
import base64
import copy
import importlib
import logging
import time
from typing import Any, Dict, Optional

import dotenv
import numpy as np
import uvicorn

# Import your BrowserEnv and any task setup you need
from bgym import DEFAULT_BENCHMARKS
from browsergym.core.env import BrowserEnv
from browsergym.core.task import AbstractBrowserTask
from fastapi import FastAPI, Request
from pydantic import BaseModel

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()


# Utils to import the action mapping fn
def import_from_path(path):
    """
    Import and instantiate a class, then return its 'to_python_code' method.
    For example, given 'browsergym.core.action.highlevel.HighLevelActionSet.to_python_code',
    this will instantiate HighLevelActionSet and return its to_python_code method.
    """
    import importlib

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


## Utils to convert to safe JSON response
def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        # convert to base64
        return {"data": base64.b64encode(obj.tobytes()).decode("utf-8"), "shape": obj.shape, "dtype": str(obj.dtype)}
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

    def set_info(
        self,
        benchmark_name: str,
        task_name: str,
        seed: int,
        action_mapping_fn: str,
        exp_dir: str,
    ):
        """Set the environment info.

        :param benchmark_name: Name of the benchmark
        :type benchmark_name: str
        :param task_name: Name of the task
        :type task_name: str
        :param seed: Seed of the task.
        :type seed: int
        :param action_mapping_fn: Action mapping function
        :type action_mapping_fn: str
        :param exp_dir: Directory for experiment
        :type exp_dir: str
        :return: Dictionary with status
        :rtype: dict
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
        """Get the environment info

        :return: Dictionary with info
        :rtype: dict
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
        """Unset the environment info

        :return: Dictionary with status
        :rtype: dict
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
        """Get the environment status

        :return: Dictionary with status
        :rtype: dict
        """
        return make_json_safe(
            {
                "status": "success",
                "message": "Environment status retrieved successfully.",
                "info_set": self.info_set,
                "env_created": self.env is not None,
            }
        )

    def reset(self) -> dict:
        """Reset the environment

        :return: Dictionary with obs and info
        :rtype: dict
        """
        start = time.time()
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

        # then create the new environment
        benchmark = DEFAULT_BENCHMARKS[self.benchmark_name]()
        benchmark.env_args_list = [
            elem for elem in benchmark.env_args_list if elem.task_name == self.task_name and str(elem.task_seed) == str(self.seed)
        ]
        benchmark.prepare_backends()

        env_args = benchmark.env_args_list[0]
        # env_args.headless = False

        self.action_mapping = import_from_path(self.action_mapping_fn)
        end = time.time()
        logger.info(f"init reset done in {end - start}")
        start = time.time()
        self.env = env_args.make_env(self.action_mapping, self.exp_dir)
        end = time.time()
        logger.info(f"make_env done in {end - start}")
        start = time.time()
        # finally, reset the environment
        obs, info = self.env.reset(seed=self.seed)
        self.last_obs = copy.deepcopy(obs)
        self.last_info = copy.deepcopy(info)
        end = time.time()
        logger.info(f"env reset done in {end - start}")
        start = time.time()
        # out = make_json_safe(
        out = make_json_safe(
            {
                "status": "success",
                "message": "Environment reset successfully",
                "obs": self.last_obs,
                "info": self.last_info,
            }
        )
        end = time.time()
        logger.info(f"payload cleaned in {end - start}")
        # log payload size
        from pympler import asizeof

        logger.info(f"Payload size: {asizeof.asizeof(out)} bytes")
        # print(out)
        # return {"status": "success", "message": "Environment reset successfully"}
        return out

    def step(self, action: str) -> dict:
        """Step the environment

        :param action: Action to take
        :type action: str
        :return: Dictionary with obs, reward, terminated, truncated and info
        :rtype: dict
        """
        if self.env is None:
            return make_json_safe(
                {
                    "status": "error",
                    "message": "Environment not created. Please create an environment first.",
                }
            )
        start = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        end = time.time()
        logger.info(f"env step done in {end - start}")
        start = time.time()
        self.last_obs = copy.deepcopy(obs)
        self.last_info = copy.deepcopy(info)
        out = make_json_safe(
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
        end = time.time()
        logger.info(f"obs copied in {end - start}")
        return out

    def get_obs(self) -> dict:
        """Get the last observation

        :return: Dictionary with obs and info
        :rtype: dict
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
                "info": self.last_info,
            }
        )

    def close(self) -> dict:
        """Close the environment

        :return: Dictionary with status
        :rtype: dict
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
def set_info(req: SetInfoRequest):
    return env.set_info(
        benchmark_name=req.benchmark_name,
        task_name=req.task_name,
        seed=req.seed,
        action_mapping_fn=req.action_mapping_fn,
        exp_dir=req.exp_dir,
    )


@app.get("/get_info")
def get_info():
    return env.get_info()


@app.post("/unset_info")
def unset_info():
    return env.unset_info()


@app.get("/status")
def status():
    return env.status()


@app.post("/reset")
def reset():
    return env.reset()


@app.post("/step")
def step(req: StepRequest):
    return env.step(action=req.action)


@app.get("/get_obs")
def get_obs():
    return env.get_obs()


@app.post("/close")
def close():
    return env.close()


def main():
    uvicorn.run("agentlab.analyze.server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
