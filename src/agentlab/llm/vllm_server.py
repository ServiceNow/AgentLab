import json
import logging
import time
import os
import subprocess
from abc import abstractmethod
from dataclasses import dataclass
import urllib.request
import urllib.error
from finetuning.configs import toolkit_configs
from finetuning.toolkit_utils.toolkit_servers import (
    auto_launch_server,
    check_server_status,
    kill_server,
    get_job_status,
    get_n_cpus_for_inference,
)

from agentlab.llm.chat_api import BaseModelArgs, HuggingFaceURLChatModel, SelfHostedModelArgs

from agentlab.llm.logging_config import logger



@dataclass
class ServerModelArgs(BaseModelArgs):
    """Abstract class for server-based models, with methods for preparing and closing the server."""

    model_url: str = None

    registry = {}  # static variable holder all servers

    def __post_init__(self):
        if self.max_total_tokens is None:
            self.max_total_tokens = 4096

        if self.max_new_tokens is None and self.max_input_tokens is not None:
            self.max_new_tokens = self.max_total_tokens - self.max_input_tokens
        elif self.max_new_tokens is not None and self.max_input_tokens is None:
            self.max_input_tokens = self.max_total_tokens - self.max_new_tokens
        elif self.max_new_tokens is None and self.max_input_tokens is None:
            raise ValueError("max_new_tokens or max_input_tokens must be specified")
        pass

    @abstractmethod
    def prepare_server(self):
        pass

    @abstractmethod
    def close_server(self):
        pass

    def key(self):
        return json.dumps(
            {
                "model_name": self.model_name,
                "max_total_tokens": self.max_total_tokens,
                "max_input_tokens": self.max_input_tokens,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        )

    @abstractmethod
    def get_server(self):
        """Method to get the server information and wait for it to be ready."""
        pass

    def prepare_server(self):
        key = self.key()
        if key in ServerModelArgs.registry:
            server_info = ServerModelArgs.registry[key][1]

            # TODO: make sure this is okay, i.e. not checking if the server is cancelled
            self.get_server(server_info)
            # return server_info, wait_func
        else:
            server_info = self.get_server()
            ServerModelArgs.registry[key] = (self, server_info)

    def close_all_servers(self):  # class method ?
        # need to find a place to call this
        for key, (model_args, _) in ServerModelArgs.registry.items():
            model_args.close_server()
            del ServerModelArgs.registry[key]


@dataclass
class ToolkitModelArgs(ServerModelArgs):
    """Serializable object for instantiating a generic chat model.

    Attributes
    ----------
    model_name : str
        The name or path of the model to use.
    model_path: str, optional
        Sometimes the model is stored locally. This is the path to the model.
    model_size: str, optional
        The size of the model to use. Relevant for TGI serving.
    model_url : str, optional
        The url of the model to use. If None, then model_name or model_name must
        be specified.
    temperature : float
        The temperature to use for the model.
    max_new_tokens : int
        The maximum number of tokens to generate.
    max_total_tokens : int
        The maximum number of total tokens (input + output). Defaults to 4096.
    hf_hosted : bool
        Whether the model is hosted on HuggingFace Hub. Defaults to False.
    is_model_operational : bool
        Whether the model is operational or there are issues with it.
    sliding_window: bool
        Whether the model uses a sliding window during training. Defaults to False.
    n_retry_server: int, optional
        The number of times to retry the TGI server if it fails to respond. Defaults to 4.
    info : dict, optional
        Any other information about how the model was finetuned.
    """

    exp_name: str = None
    model_name: str = None
    model_path: str = None
    base_model_path: str = None
    model_url: str = None
    model_size: str = None
    training_total_tokens: int = None
    hf_hosted: bool = False
    is_model_operational: str = False
    sliding_window: bool = False
    n_retry_server: int = 4
    infer_tokens_length: bool = False
    vision_support: bool = False
    shard_support: bool = True
    n_gpus: int = None
    info: dict = None

    def __post_init__(self):
        if self.model_url is not None and self.hf_hosted:
            raise ValueError("model_url cannot be specified when hf_hosted is True")

        if self.infer_tokens_length:

            if self.max_total_tokens is None:
                if self.training_total_tokens is not None:
                    self.max_total_tokens = self.training_total_tokens
                else:
                    logging.debug(
                        "max_total_tokens is not specified. Setting it to 8_096 (default value)."
                    )
                    self.max_total_tokens = 8_096
            if self.max_new_tokens is None and self.max_input_tokens is not None:
                self.max_new_tokens = self.max_total_tokens - self.max_input_tokens
            elif self.max_new_tokens is not None and self.max_input_tokens is None:
                self.max_input_tokens = self.max_total_tokens - self.max_new_tokens
            elif self.max_new_tokens is None and self.max_input_tokens is None:
                self.max_new_tokens = 512 if self.max_total_tokens <= 8_096 else 2_048
                self.max_input_tokens = self.max_total_tokens - self.max_new_tokens
                logging.debug(
                    f"max_new_tokens is not specified. Setting it to {self.max_input_tokens}."
                )

        if not self.max_total_tokens:
            self.max_total_tokens = self.training_total_tokens

        if self.model_path is None:
            self.model_path = self.model_name

    def set_base_model_path(self):
        # TODO clean this up
        # go get the value in adapter_config.json
        if self.base_model_path is not None:
            # go in adapter config and set the base_model_name_or_path key to the base_model_path
            with open(os.path.join(self.model_path, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            adapter_config["base_model_name_or_path"] = str(self.base_model_path)
            # rw path
            rw_model_path = self.model_path.replace(
                "/mnt/ui_copilot/data/", "/mnt/ui_copilot/data_rw/"
            )
            with open(os.path.join(rw_model_path, "adapter_config.json"), "w") as f:
                json.dump(adapter_config, f)

        elif self.model_path is not None and os.path.isdir(self.model_path):
            # check if path is a full fine-tuned path or a path to a dir of adapters
            files = os.listdir(self.model_path)
            if "adapter_config.json" in files:
                with open(os.path.join(self.model_path, "adapter_config.json")) as f:
                    adapter_config = json.load(f)
                self.base_model_path = adapter_config["base_model_name_or_path"].replace(
                    "/mnt/ui_copilot/data_rw/", "/mnt/ui_copilot/data/"
                )

    def make_model(self):
        # TODO: eventually check if the path is either a valid repo_id or a valid path

        hf_model_args = HuggingFaceURLChatModel(
            model_name=self.model_path,
            base_model_name=self.base_model_path,
            model_url=self.model_url,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            n_retry_server=self.n_retry_server,
            log_probs=self.log_probs,
        )
        # NOTE: for OLD offline experiments, we will wait for the servers here during evaluation
        # because n_finetuning_seeds * n_ckpts servers are launched
        # so deadlocking could be an issues
        # self.wait_server()

        return hf_model_args

    def get_server(self, server_info=None):

        if server_info is not None:
            self.job_id, self.model_url = server_info
            # TODO what should we do if the server crashed? between rounds
            # job_status = get_job_status(self.job_id)
            # if job_status != "RUNNING":
            #     if job_status in ["CANCELLED", "CANCELLING", "FAILED"] and self.num_resets < 3:
            #         logging.info("Server has been cancelled. Relaunching...")
            #         self.job_id, self.model_url = auto_launch_server(self)
            #     self.wait_server()
        else:
            self.job_id, self.model_url = auto_launch_server(self)
            # NOTE: for on_policy experiments, we wait wait for the server here before evaluation
            # because only n_finetuning_seeds servers are launched
            # so deadlocking is not an issue
            self.wait_server()

        return self.job_id, self.model_url

    def close_server(self):
        # NOTE: instead of having each job kill their server, which are shared,
        # we will kill the servers when the script ends
        return

        kill_server(self.job_id)
        # remove key from registry with error handling

        if self.key() in ServerModelArgs.registry:
            del ServerModelArgs.registry[self.key()]

    def wait_server(self):
        if self.model_url == "dry_run":
            return
        is_ready = False
        logging.info("Waiting for the server to be ready for 12h ...")
        for _ in range(int(12 * 60 * 60 / 10)):
            is_ready = check_server_status(self.job_id, self.model_url)
            if is_ready:
                return
            time.sleep(10)
        raise TimeoutError("Server is not ready after 5h - Killing the job")

    def key(self):
        return json.dumps(
            {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "max_total_tokens": self.max_total_tokens,
                "max_input_tokens": self.max_input_tokens,
                "max_new_tokens": self.max_new_tokens,
                "info": self.info,  # NOTE: useful to create multiple servers with the same model
            }
        )


@dataclass
class VLLMModelArgs(SelfHostedModelArgs):
    """Model args for launching a vllm serve subprocess with parallelism and optional features."""

    port: int = 8000
    tensor_parallel_size: int = 1
    sliding_window: bool = False
    _process: subprocess.Popen = None  # holds the vllm serve process
    exp_name: str = None
    model_name: str = None
    model_path: str = None
    base_model_path: str = None
    model_url: str = None
    model_size: str = None
    max_total_tokens: int = 4096
    max_input_tokens: int = None
    max_new_tokens: int = None
    training_total_tokens: int = None
    infer_tokens_length: bool = False  # Added to control token calculation logic
    max_batch_prefill_tokens: int = 512
    max_batch_total_tokens: int = 4096
    sliding_window: bool = False
    shard_support: bool = True
    n_gpus: int = None
    vllm_cpus: int = None
    reasoning_parser: str = (
        None  # Explicitly set reasoning parser (e.g., 'qwen3'), or None to auto-detect
    )

    def __post_init__(self):
        os.environ["VLLM_API_KEY"] = "EMPTY"  # VLLM requires this, set to "EMPTY" if no key
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # Enable FlashAttention
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"  # Reduce vLLM logging verbosity

        if self.infer_tokens_length:

            if self.max_total_tokens is None:
                if self.training_total_tokens is not None:
                    self.max_total_tokens = self.training_total_tokens
                else:
                    logging.debug(
                        "max_total_tokens is not specified. Setting it to 8_096 (default value)."
                    )
                    self.max_total_tokens = 8_096
            if self.max_new_tokens is None and self.max_input_tokens is not None:
                self.max_new_tokens = self.max_total_tokens - self.max_input_tokens
            elif self.max_new_tokens is not None and self.max_input_tokens is None:
                self.max_input_tokens = self.max_total_tokens - self.max_new_tokens
            elif self.max_new_tokens is None and self.max_input_tokens is None:
                self.max_new_tokens = 512 if self.max_total_tokens <= 8_096 else 2_048
                self.max_input_tokens = self.max_total_tokens - self.max_new_tokens
                logging.debug(
                    f"max_new_tokens is not specified. Setting it to {self.max_input_tokens}."
                )

        if not self.max_total_tokens:
            self.max_total_tokens = self.training_total_tokens

        if self.model_path is None:
            self.model_path = self.model_name

        # Auto-detect reasoning parser based on model name if not explicitly set
        if self.reasoning_parser is None:
            self.reasoning_parser = self._detect_reasoning_parser()

    def _detect_reasoning_parser(self):
        """
        Automatically detect the appropriate reasoning parser based on model name.

        Returns:
            str or None: The reasoning parser name ('qwen3', 'llama', etc.) or None if not applicable
        """
        if self.model_name is None and self.model_path is None:
            return None

        # Use model_path as it's the actual model being loaded
        model_identifier = (self.model_path or self.model_name).lower()

        # Qwen3 models use qwen3 reasoning parser
        if "qwen3" in model_identifier:
            return "qwen3"
        # Qwen2.5 models may also support reasoning (check vLLM docs for your version)
        else:
            return None

    def _terminate_and_reset_process(self, reason_msg=""):
        """Helper to terminate the VLLM process if running and reset attributes."""
        if self._process and self._process.poll() is None:  # If process exists and is running
            logger.warning(
                f"Terminating VLLM server for {self.model_name} (PID: {self._process.pid}). Reason: {reason_msg}"
            )
            self._process.terminate()
            try:
                self._process.wait(timeout=5)  # Wait for graceful shutdown
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"VLLM server for {self.model_name} (PID: {self._process.pid}) did not terminate gracefully, killing."
                )
                self._process.kill()  # Force kill
                self._process.wait()  # Ensure it's killed
        self._process = None
        self.model_url = None

    def prepare_server(self, verbose: bool = False):
        """Launches the VLLM server as a subprocess if not already running."""

        # NOTE: get n_gpus here to simplify code and make it robust to relaunchs w/ varying n_gpus
        self.n_gpus = get_available_gpus()
        assert self.n_gpus > 0, "No GPUs available"
        self.tensor_parallel_size = self.n_gpus
        self.vllm_cpus = get_n_cpus_for_inference(self.n_gpus)

        if verbose:
            logger.info(
                f"Preparing VLLM server for {self.model_name} on port {self.port} with tensor_parallel_size {self.tensor_parallel_size}..."
            )

        # First, check if this instance already manages a running process.
        if self._process and self._process.poll() is None:
            if verbose:
                logger.info(
                    f"VLLM server for {self.model_name} is already running on port {self.port}."
                )
            return

        # Next, check if a server is already running on the port (e.g., from another process).
        readiness_url = f"http://127.0.0.1:{self.port}/health"
        try:
            with urllib.request.urlopen(readiness_url, timeout=3) as response:
                if response.status == 200:
                    if verbose:
                        logger.info(
                            f"Found an existing VLLM server on port {self.port}. Reusing it."
                        )
                    self.model_url = f"http://127.0.0.1:{self.port}"
                    return  # Exit if server is already up
        except (urllib.error.URLError, ConnectionRefusedError):
            # This is expected if the server is not running, so we can proceed to launch.
            logger.info(
                f"No running VLLM server detected on port {self.port}. Launching a new one."
            )
        except Exception as e:
            logger.warning(
                f"An unexpected error occurred while checking for an existing server on port {self.port}: {e}"
            )

        logger.info(f"Preparing VLLM server for {self.model_name} on port {self.port}...")

        cmd_parts = [
            toolkit_configs.VLLM_ENV_PATH,  # TODO change to a toolkit_configs variable
            "serve",
            self.model_path,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--port",
            str(self.port),
            "--served-model-name",
            self.model_name,
            "--max-log-len",
            "0",
            "--disable-log-stats",
            "--uvicorn-log-level",
            "warning",
            "--disable-uvicorn-access-log",
            "--async-scheduling",  # TODO: make sure when this shouldn't be used.
        ]

        # Only add reasoning parser if applicable for this model
        if self.reasoning_parser:
            cmd_parts.extend(["--reasoning-parser", self.reasoning_parser])
            if verbose:
                logger.info(f"Using reasoning parser: {self.reasoning_parser}")

        if self.sliding_window:
            cmd_parts.append("--sliding-window")

        cmd_parts.extend(["--max-model-len", str(self.max_total_tokens)])
        cmd_parts.extend(["--max-num-batched-tokens", str(self.max_batch_total_tokens)])

        logger.info(f"Launching VLLM server with command: {' '.join(cmd_parts)}")

        # Set up environment with CPU allocation if specified
        env = os.environ.copy()
        # Set CPU cores for vLLM (0 to cpu_cores-1)
        cpu_range = f"0-{self.vllm_cpus-1}"
        env["VLLM_CPU_OMP_THREADS_BIND"] = cpu_range
        if verbose:
            logger.info(
                f"Setting VLLM_CPU_OMP_THREADS_BIND={cpu_range} to allocate {self.vllm_cpus} CPU cores to vLLM"
            )

        try:
            self._process = subprocess.Popen(
                cmd_parts, env=env
            )  # Pass environment with CPU constraints
            self.model_url = f"http://127.0.0.1:{self.port}"
            if verbose:
                logger.info(
                    f"VLLM server for {self.model_name} process started with PID {self._process.pid}. Model URL: {self.model_url}"
                )

            # Implement a proper readiness check
            readiness_url = f"{self.model_url}/health"  # Standard VLLM health check endpoint
            timeout_seconds = 1620  # Total time to wait for server readiness
            poll_interval_seconds = 2  # How often to check

            logger.info(
                f"Waiting for VLLM server at {readiness_url} to be ready for up to {timeout_seconds}s..."
            )
            start_time = time.time()
            server_ready = False
            while time.time() - start_time < timeout_seconds:
                process_poll_result = self._process.poll()
                if process_poll_result is not None:
                    # Process terminated prematurely
                    pid_val = self._process.pid
                    logger.error(
                        f"VLLM server for {self.model_name} (PID: {pid_val}) terminated prematurely while waiting for readiness. Exit code: {process_poll_result}"
                    )
                    self._process = None  # Already terminated, just reset attributes
                    self.model_url = None
                    raise RuntimeError(
                        f"VLLM server for {self.model_name} failed to start (terminated before ready)."
                    )

                try:
                    with urllib.request.urlopen(
                        readiness_url, timeout=poll_interval_seconds
                    ) as response:
                        if response.status == 200:
                            logger.info(
                                f"VLLM server for {self.model_name} is ready at {self.model_url}."
                            )
                            server_ready = True
                            break
                        else:
                            logger.debug(
                                f"VLLM server for {self.model_name} at {readiness_url} responded with status {response.status}. Retrying..."
                            )
                except urllib.error.URLError as e:
                    logger.debug(
                        f"VLLM server for {self.model_name} not yet ready (poll {readiness_url}): {e}. Retrying in {poll_interval_seconds}s..."
                    )
                except Exception as e:  # Catch other potential errors like socket.timeout
                    logger.debug(
                        f"VLLM server for {self.model_name} not yet ready (poll {readiness_url}, unexpected error): {e}. Retrying in {poll_interval_seconds}s..."
                    )

                time.sleep(poll_interval_seconds)

            if not server_ready:
                logger.error(
                    f"VLLM server for {self.model_name} at {readiness_url} did not become ready within {timeout_seconds} seconds."
                )
                self._terminate_and_reset_process(reason_msg="timeout")
                raise TimeoutError(
                    f"VLLM server for {self.model_name} timed out waiting for readiness at {readiness_url}."
                )

        except Exception as e:
            logger.error(f"Failed to launch or prepare VLLM server for {self.model_name}: {e}")
            self._terminate_and_reset_process(reason_msg=f"launch failure: {e}")
            raise

    def close_server(self, verbose: bool = False):
        """Terminates the VLLM server subprocess."""
        if self._process and self._process.poll() is None:  # Check if process exists and is running
            logger.info(
                f"Closing VLLM server for {self.model_name} (PID: {self._process.pid}) on port {self.port}..."
            )
            try:
                self._process.terminate()  # Send SIGTERM
                self._process.wait(timeout=10)  # Wait for graceful shutdown
                logger.info(f"VLLM server for {self.model_name} terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"VLLM server for {self.model_name} did not terminate gracefully, sending SIGKILL."
                )
                self._process.kill()  # Force kill
                self._process.wait()  # Ensure it's killed
                logger.info(f"VLLM server for {self.model_name} killed.")
            except Exception as e:
                logger.error(f"Error during VLLM server termination for {self.model_name}: {e}")
            finally:
                self._process = None
                self.model_url = None
        elif self._process:  # Process exists but is not running
            logger.info(f"VLLM server for {self.model_name} was already stopped.")
            self._process = None
            self.model_url = None
        else:
            if verbose:
                logger.info(f"No VLLM server process found for {self.model_name} to close.")

    # The make_model method from SelfHostedModelArgs will be inherited.
    # It typically creates a client (like HuggingFaceURLChatModel) using self.model_url.
    # Ensure self.model_url is correctly set in prepare_server.
    def set_base_model_path(self):
        # TODO clean this up
        # go get the value in adapter_config.json
        if self.base_model_path is not None:
            # go in adapter config and set the base_model_name_or_path key to the base_model_path
            with open(os.path.join(self.model_path, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            adapter_config["base_model_name_or_path"] = str(self.base_model_path)
            # rw path
            rw_model_path = self.model_path.replace(
                "/mnt/ui_copilot/data/", "/mnt/ui_copilot/data_rw/"
            )
            with open(os.path.join(rw_model_path, "adapter_config.json"), "w") as f:
                json.dump(adapter_config, f)

        elif self.model_path is not None and os.path.isdir(self.model_path):
            # check if path is a full fine-tuned path or a path to a dir of adapters
            files = os.listdir(self.model_path)
            if "adapter_config.json" in files:
                with open(os.path.join(self.model_path, "adapter_config.json")) as f:
                    adapter_config = json.load(f)
                self.base_model_path = adapter_config["base_model_name_or_path"].replace(
                    "/mnt/ui_copilot/data_rw/", "/mnt/ui_copilot/data/"
                )

    def __getstate__(self):
        """Custom pickling state: exclude _process."""
        state = self.__dict__.copy()
        if "_process" in state:
            del state["_process"]
        return state

    def __setstate__(self, state):
        """Custom unpickling: restore state and initialize _process to None."""
        self.__dict__.update(state)
        self._process = None  # Ensure _process is reset after unpickling


def get_available_gpus():
    """
    Get the number of available GPUs.

    Returns:
        int: Number of available GPUs. Returns 0 if no GPUs are available or CUDA is not available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    except ImportError:
        # Fallback if torch is not available
        try:
            import subprocess

            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Count lines that contain "GPU"
                return len([line for line in result.stdout.strip().split("\n") if "GPU" in line])
            else:
                return 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return 0
        

