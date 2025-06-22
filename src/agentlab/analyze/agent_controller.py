import base64
import copy
import importlib
import json
import logging
import os
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import PIL.Image
import requests
import streamlit as st
from agentlab.agents.generic_agent import __all__ as ALL_AGENTS
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.llm.llm_utils import Discussion
from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv
from transformers import AutoTokenizer

# used to display prompt. simple chat template from apache 2.0 model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
load_dotenv()

DEFAULT_AGENT = "AGENT_AZURE_4o"
DEFAULT_BENCHMARK = "workarena_l1"

SERVER_URL = "http://127.0.0.1:8000"

# region Sidebar Text
SIDEBAR_TEXT = """
# AgentLab Controller

AgentLab Controller is a tool used to help control and debug an agent deployed in an environment.

AgentLab Controller works by connecting a Streamlit UI that handles the agent to a FastAPI backend server that handles the environment.

---

## Instructions

1. ⚙️ Setup the task
    - Select an agent, benchmark, task, and subtask you want to work on.
    - Select "🔄" to reset the environment. This includes resetting the environment server.
    - Select "▶️" to start the environment. This will start the environment by opening a browser in the background. This step might take some time

2. 🎮 Control the environment
    - Look at the goal set for the task, the thought of the model, and the action taken.
    - If the action looks right, select the "▶️ Next Step" button to step the environment.
        + The action will then be executed and the environment will be updated.
    - If the action is wrong and you want to re-prompt, select the "🔄 Regenerate Action".
        + You can also expand the "Prompt Modifier" menu to change the prompt used to generate the thoughts / actions.
    - If you want to backtrack and undo the previous actions, select the "⬅️ Previous Step" button.
        + Note: This is a slow process as we need to reset the environment server and perform the previous actions one by one.

3. 🔎 Investigate the environment
    - Look at the screenshot of the current environment state
    - Verify that the action selected by the model matches the AxTree
    - Ensure that the prompt is properly build. If there are issues with the prompt yielding the wrong action, modify them using the "Prompt Modifier" above.
"""
# endregion


class Constants:
    STATUS = "status"
    STATUS_SUCCESS = "success"
    STATUS_ERROR = "error"
    MESSAGE = "message"

    OBS = "obs"
    SCREENSHOT = "screenshot"
    AXTREE_TXT = "axtree_txt"


class IgnoreMessageFilter(logging.Filter):
    def filter(self, record):
        return "but it does not exist!" not in record.getMessage()


streamlit_logger = st.watcher.local_sources_watcher._LOGGER
streamlit_logger.setLevel(logging.ERROR)


def make_hashable(obj):
    if isinstance(obj, np.ndarray):
        # Use shape, dtype, and bytes for uniqueness
        return (obj.shape, obj.dtype.str, obj.tobytes())
    elif isinstance(obj, (tuple, list)):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        # Sort keys to ensure consistent order
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    else:
        return obj  # Assume it's already hashable


def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def get_import_path(obj):
    return f"{obj.__module__}.{obj.__qualname__}"


def deserialize_response(response_json):
    if Constants.OBS in response_json:
        if Constants.SCREENSHOT in response_json[Constants.OBS]:
            screenshot_data = response_json[Constants.OBS][Constants.SCREENSHOT]
            # convert base64 to numpy array
            screenshot = np.frombuffer(
                base64.b64decode(screenshot_data["data"]), dtype=np.dtype(screenshot_data["dtype"])
            )
            screenshot = screenshot.reshape(screenshot_data["shape"])
            response_json[Constants.OBS][Constants.SCREENSHOT] = screenshot
    return response_json


def reset_env_history():
    logger.info("Resetting env history")
    st.session_state.last_obs = None
    st.session_state.obs_history = []
    st.session_state.screenshot_history = []
    st.session_state.axtree_history = []


def reset_agent_history():
    logger.info("Resetting agent history")
    st.session_state.action = None
    st.session_state.action_info = None
    st.session_state.action_history = []
    st.session_state.action_info_history = []
    st.session_state.thought_history = []
    st.session_state.prompt_history = []
    st.session_state.memory_history = []


def reset_agent_state():
    logger.info("Resetting agent state")
    st.session_state.agent.reset()


def step_env_history(obs):
    logger.info("Stepping env history")
    st.session_state.last_obs = copy.deepcopy(obs)
    st.session_state.obs_history.append(obs)
    st.session_state.screenshot_history.append(obs[Constants.SCREENSHOT])
    st.session_state.axtree_history.append(obs[Constants.AXTREE_TXT])


def step_agent_history(action, action_info):
    logger.info("Stepping agent history")
    st.session_state.action = copy.deepcopy(action)
    st.session_state.action_info = copy.deepcopy(action_info)
    st.session_state.action_history.append(action)
    st.session_state.action_info_history.append(action_info)
    st.session_state.thought_history.append(action_info.think)
    st.session_state.prompt_history.append(get_prompt(action_info))

    # HACK: memory history can only be obtained via the agent
    st.session_state.memory_history.append(st.session_state.agent.memories[-1])


def set_agent_state():
    logger.info("Setting agent state")
    st.session_state.agent.obs_history = copy.deepcopy(st.session_state.obs_history)
    st.session_state.agent.actions = copy.deepcopy(st.session_state.action_history)
    st.session_state.agent.thoughts = copy.deepcopy(st.session_state.thought_history)
    st.session_state.agent.memories = copy.deepcopy(st.session_state.memory_history)


def revert_env_history():
    logger.info("Reverting env history")
    st.session_state.obs_history.pop()
    st.session_state.screenshot_history.pop()
    st.session_state.axtree_history.pop()


def revert_agent_history():
    logger.info("Reverting agent history")
    st.session_state.action_history.pop()
    st.session_state.action_info_history.pop()
    st.session_state.thought_history.pop()
    st.session_state.prompt_history.pop()
    st.session_state.memory_history.pop()


def revert_agent_state():
    logger.info("Reverting agent state")
    st.session_state.agent.obs_history.pop()
    st.session_state.agent.actions.pop()
    st.session_state.agent.thoughts.pop()
    st.session_state.agent.memories.pop()


def restore_env_history(step: int):
    logger.info(f"Restoring env history to step {step}")
    st.session_state.obs_history = copy.deepcopy(st.session_state.obs_history[:step])
    st.session_state.screenshot_history = copy.deepcopy(st.session_state.screenshot_history[:step])
    st.session_state.axtree_history = copy.deepcopy(st.session_state.axtree_history[:step])


def restore_agent_history(step: int):
    logger.info(f"Restoring agent history to step {step}")
    st.session_state.action_history = copy.deepcopy(st.session_state.action_history[:step])
    st.session_state.action_info_history = copy.deepcopy(
        st.session_state.action_info_history[:step]
    )
    st.session_state.thought_history = copy.deepcopy(st.session_state.thought_history[:step])
    st.session_state.prompt_history = copy.deepcopy(st.session_state.prompt_history[:step])
    st.session_state.memory_history = copy.deepcopy(st.session_state.memory_history[:step])


def get_prompt(info):
    if info is not None and isinstance(info.chat_messages, Discussion):
        chat_messages = info.chat_messages.messages
        new_chat_messages = []
        for message in chat_messages:
            if isinstance(message["content"], list):
                # concatenate all text elements
                new_chat_messages.append(
                    {
                        "role": message["role"],
                        "content": "\n\n".join(
                            [elem["text"] for elem in message["content"] if elem["type"] == "text"]
                        ),
                    }
                )
            else:
                new_chat_messages.append(message)
        prompt = tokenizer.apply_chat_template(
            new_chat_messages, add_special_tokens=True, tokenize=False
        )
        return prompt


def setup_sidebar():
    with st.sidebar:
        st.markdown(SIDEBAR_TEXT)


def set_session_state():

    # args used to instantiate agent / environment
    if "has_submitted_configs" not in st.session_state:
        st.session_state.has_submitted_configs = False
    if "agent_args" not in st.session_state:
        st.session_state.agent_args = None
    if "benchmark" not in st.session_state:
        st.session_state.benchmark = None
    if "task" not in st.session_state:
        st.session_state.task = None
    if "subtask" not in st.session_state:
        st.session_state.subtask = None

    # current state
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "action" not in st.session_state:
        st.session_state.action = None
    if "action_info" not in st.session_state:
        st.session_state.action_info = None
    if "last_obs" not in st.session_state:
        st.session_state.last_obs = None

    # track history
    if "prompt_history" not in st.session_state:
        st.session_state.prompt_history = None
    if "screenshot_history" not in st.session_state:
        st.session_state.screenshot_history = None
    if "axtree_history" not in st.session_state:
        st.session_state.axtree_history = None
    if "thought_history" not in st.session_state:
        st.session_state.thought_history = None
    if "memory_history" not in st.session_state:
        st.session_state.memory_history = None
    if "action_history" not in st.session_state:
        st.session_state.action_history = None
    if "action_info_history" not in st.session_state:
        st.session_state.action_info_history = None
    if "obs_history" not in st.session_state:
        st.session_state.obs_history = None

    if "has_clicked_prev" not in st.session_state:
        st.session_state.has_clicked_prev = False
    if "has_clicked_next" not in st.session_state:
        st.session_state.has_clicked_next = False
    if "has_clicked_multiple_reprompt" not in st.session_state:
        st.session_state.has_clicked_multiple_reprompt = False


def select_agent():
    """Dropdown to select an agent."""
    agent_str = st.selectbox("Select Agent", ALL_AGENTS, index=ALL_AGENTS.index(DEFAULT_AGENT))
    agents_module = importlib.import_module("agentlab.agents.generic_agent")
    agent = getattr(agents_module, agent_str)
    return agent


def select_benchmark() -> str:
    """Dropdown to select a benchmark."""
    all_benchmarks = list(DEFAULT_BENCHMARKS.keys())
    benchmark_str = st.selectbox(
        "Select Benchmark", all_benchmarks, index=all_benchmarks.index(DEFAULT_BENCHMARK)
    )
    return benchmark_str


def select_task(benchmark):
    """Dropdown to select a task based on the benchmark."""
    all_tasks = sorted(list(set([elem.task_name for elem in benchmark.env_args_list])))
    task_str = st.selectbox("Select Task", all_tasks)
    return task_str


def select_subtask(benchmark, task_str) -> str:
    """Dropdown to select a subtask based on the task name."""
    all_subtasks = sorted(
        [str(elem.task_seed) for elem in benchmark.env_args_list if elem.task_name == task_str]
    )
    subtask_str = st.selectbox("Select Subtask", all_subtasks)
    return subtask_str


def set_task_selector():
    """Create task selector form. Allows the user to select the agent, benchmark, task, and subtask to run."""
    with st.container(border=True):
        st.markdown("##### ⚙️ Select")
        with st.form("Task Selector"):
            col1, col2, col3, col4, col5, col6 = st.columns(
                [2, 2, 4, 2, 1, 1], vertical_alignment="bottom"
            )
            with col1:
                selected_agent_args = select_agent()
            with col2:
                selected_benchmark_str = select_benchmark()
                selected_benchmark = DEFAULT_BENCHMARKS[selected_benchmark_str]()
            with col3:
                selected_task_str = select_task(selected_benchmark)
            with col4:
                selected_subtask_str = select_subtask(selected_benchmark, selected_task_str)
            with col5:
                if st.form_submit_button("🔄", use_container_width=True):
                    clean_session()
            with col6:
                if st.form_submit_button("▶️", use_container_width=True):

                    # saving configs related to agent and task
                    st.session_state.has_submitted_configs = True
                    st.session_state.agent_args = selected_agent_args
                    st.session_state.benchmark = selected_benchmark_str
                    st.session_state.task = selected_task_str
                    st.session_state.subtask = selected_subtask_str

                    reset_env_history()
                    reset_agent_history()

                    prepare_agent()
                    set_environment_info()
                    prepare_benchmark()
                    reset_environment()


def clean_session():
    logger.info("Cleaning session...")
    start = datetime.now()
    requests.post(f"{SERVER_URL}/unset_info")
    requests.post(f"{SERVER_URL}/close")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    end = datetime.now()
    logger.info(f"Done in {end - start}")
    st.rerun()


def prepare_agent():
    st.session_state.agent_args.prepare()
    st.session_state.agent = st.session_state.agent_args.make_agent()


def set_environment_info():
    action_mapping_fn = get_import_path(st.session_state.agent.action_set.to_python_code)
    payload = {
        "benchmark_name": st.session_state.benchmark,
        "task_name": st.session_state.task,
        "seed": st.session_state.subtask,
        "action_mapping_fn": action_mapping_fn,
        "exp_dir": str(RESULTS_DIR),
    }
    resp = requests.post(f"{SERVER_URL}/set_info", json=payload)
    if resp.status_code != 200 or resp.json().get(Constants.STATUS) != Constants.STATUS_SUCCESS:
        st.error(resp.json())


def prepare_benchmark():
    logger.info("Preparing benchmark...")
    start = datetime.now()
    resp = requests.post(f"{SERVER_URL}/prepare_benchmark")
    if resp.status_code != 200 or resp.json().get(Constants.STATUS) != Constants.STATUS_SUCCESS:
        st.error(resp.json())
    end = datetime.now()
    logger.info(f"Done in {end - start}")


def reset_environment():
    logger.info("Restarting environment...")
    start = datetime.now()
    resp = requests.post(f"{SERVER_URL}/reset")
    end = datetime.now()
    logger.info(f"Done request in {end - start}")
    if resp.status_code != 200 or resp.json().get(Constants.STATUS) != Constants.STATUS_SUCCESS:
        logger.error(resp.status_code)
        logger.error(resp.json()[Constants.STATUS])
        logger.error(resp.json()[Constants.MESSAGE])
    response_json = resp.json()
    response_json = deserialize_response(response_json)
    obs = response_json[Constants.OBS]
    if st.session_state.agent.obs_preprocessor:
        obs = st.session_state.agent.obs_preprocessor(obs)
    step_env_history(obs)
    st.session_state.action = None
    st.session_state.action_info = None


def reload_task():
    logger.info("Reloading task...")
    start = datetime.now()
    resp = requests.post(f"{SERVER_URL}/reload_task")
    end = datetime.now()
    logger.info(f"Done request in {end - start}")
    if resp.status_code != 200 or resp.json().get(Constants.STATUS) != Constants.STATUS_SUCCESS:
        logger.error(resp.status_code)
        logger.error(resp.json()[Constants.STATUS])
        logger.error(resp.json()[Constants.MESSAGE])
    response_json = resp.json()
    response_json = deserialize_response(response_json)
    obs = response_json[Constants.OBS]
    if st.session_state.agent.obs_preprocessor:
        obs = st.session_state.agent.obs_preprocessor(obs)
    step_env_history(obs)
    st.session_state.action = None
    st.session_state.action_info = None


def step_environment(action):
    logger.info("Stepping environment...")
    start = datetime.now()
    payload = {"action": action}
    resp = requests.post(f"{SERVER_URL}/step", json=payload)
    end = datetime.now()
    logger.info(f"Done request in {end - start}")
    if resp.status_code != 200 or resp.json().get(Constants.STATUS) != Constants.STATUS_SUCCESS:
        logger.error(resp.status_code)
        logger.error(resp.json()[Constants.STATUS])
        logger.error(resp.json()[Constants.MESSAGE])
    response_json = resp.json()
    response_json = deserialize_response(response_json)
    obs = response_json[Constants.OBS]
    if st.session_state.agent.obs_preprocessor:
        obs = st.session_state.agent.obs_preprocessor(obs)
    step_env_history(obs)
    st.session_state.action = None
    st.session_state.action_info = None


def restore_environment():
    reload_task()
    for action in st.session_state.action_history[:-1]:
        step_environment(action)
    st.session_state.action = st.session_state.action_history[-1]
    st.session_state.action_info = st.session_state.action_info_history[-1]
    set_agent_state()


def get_action():
    logger.info("Getting action...")
    start = datetime.now()
    action, info = st.session_state.agent.get_action(copy.deepcopy(st.session_state.last_obs))
    step_agent_history(action, info)
    end = datetime.now()
    logger.info(f"Done in {end - start}")


def set_agent_state_box():

    # Custom CSS to set textarea style same as code block
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Handlee&family=IBM+Plex+Mono:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&family=Sedgwick+Ave&display=swap');
        textarea, .stTextArea textarea {
            font-family: "IBM Plex Mono", monospace !important;
            font-size: 14px !important;
            font-weight: 400;
            font-style: normal;
            line-height: 1.6 !important;
            padding-top: 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # set agent state and goal box
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            with st.container(border=True, height=250):
                st.markdown("**Goal**")
                st.code(
                    st.session_state.last_obs["goal"],
                    wrap_lines=True,
                    language=None,
                    height=175,
                )
        with col2:
            with st.container(border=True, height=250):
                st.markdown("**Think**")
                initial_think = copy.deepcopy(st.session_state.action_info.think)
                st.session_state.action_info.think = st.text_area(
                    "Think",
                    st.session_state.action_info.think,
                    height=172,
                    label_visibility="collapsed",
                )
                if st.session_state.action_info.think != initial_think:
                    # if thought has been updated, update thought history
                    st.session_state.thought_history[-1] = copy.deepcopy(
                        st.session_state.action_info.think
                    )
                    st.session_state.agent.thoughts[-1] = copy.deepcopy(
                        st.session_state.action_info.think
                    )
        with col3:
            with st.container(border=True, height=250):
                st.markdown("**Action**")
                initial_action = copy.deepcopy(st.session_state.action)
                st.session_state.action = st.text_area(
                    "Action", st.session_state.action, height=172, label_visibility="collapsed"
                )
                if st.session_state.action != initial_action:
                    # if action has been updated, update action history
                    st.session_state.action_history[-1] = copy.deepcopy(st.session_state.action)
                    st.session_state.agent.actions[-1] = copy.deepcopy(st.session_state.action)


def set_prompt_modifier():
    with st.expander("**Prompt Modifier**", expanded=False):
        st.markdown("**Observation Flags**")
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        with col1:
            st.session_state.agent.flags.obs.use_html = st.checkbox(
                "use_html", value=st.session_state.agent.flags.obs.use_html
            )
            st.session_state.agent.flags.obs.use_action_history = st.checkbox(
                "use_action_history", value=st.session_state.agent.flags.obs.use_action_history
            )
        with col2:
            st.session_state.agent.flags.obs.use_ax_tree = st.checkbox(
                "use_ax_tree", value=st.session_state.agent.flags.obs.use_ax_tree
            )
            st.session_state.agent.flags.obs.use_think_history = st.checkbox(
                "use_think_history", value=st.session_state.agent.flags.obs.use_think_history
            )
        with col3:
            st.session_state.agent.flags.obs.use_focused_element = st.checkbox(
                "use_focused_element", value=st.session_state.agent.flags.obs.use_focused_element
            )
            st.session_state.agent.flags.obs.use_diff = st.checkbox(
                "use_diff", value=st.session_state.agent.flags.obs.use_diff
            )
        with col4:
            st.session_state.agent.flags.obs.use_error_logs = st.checkbox(
                "use_error_logs", value=st.session_state.agent.flags.obs.use_error_logs
            )
            st.session_state.agent.flags.obs.use_screenshot = st.checkbox(
                "use_screenshot", value=st.session_state.agent.flags.obs.use_screenshot
            )
        with col5:
            st.session_state.agent.flags.obs.use_history = st.checkbox(
                "use_history", value=st.session_state.agent.flags.obs.use_history
            )
            st.session_state.agent.flags.obs.use_som = st.checkbox(
                "use_som", value=st.session_state.agent.flags.obs.use_som
            )
        with col6:
            st.session_state.agent.flags.obs.use_past_error_logs = st.checkbox(
                "use_past_error_logs", value=st.session_state.agent.flags.obs.use_past_error_logs
            )
            st.session_state.agent.flags.obs.use_tabs = st.checkbox(
                "use_tabs", value=st.session_state.agent.flags.obs.use_tabs
            )
        st.markdown("**Other Flags**")
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        with col1:
            st.session_state.agent.flags.use_plan = st.checkbox(
                "use_plan", value=st.session_state.agent.flags.use_plan
            )
            st.session_state.agent.flags.use_hints = st.checkbox(
                "use_hints", value=st.session_state.agent.flags.use_hints
            )
        with col2:
            st.session_state.agent.flags.use_criticise = st.checkbox(
                "use_criticise", value=st.session_state.agent.flags.use_criticise
            )
            st.session_state.agent.flags.be_cautious = st.checkbox(
                "be_cautious", value=st.session_state.agent.flags.be_cautious
            )
        with col3:
            st.session_state.agent.flags.use_thinking = st.checkbox(
                "use_thinking", value=st.session_state.agent.flags.use_thinking
            )
            st.session_state.agent.flags.enable_chat = st.checkbox(
                "enable_chat", value=st.session_state.agent.flags.enable_chat
            )
        with col4:
            st.session_state.agent.flags.use_memory = st.checkbox(
                "use_memory", value=st.session_state.agent.flags.use_memory
            )
        with col5:
            st.session_state.agent.flags.use_abstract_example = st.checkbox(
                "use_abstract_example", value=st.session_state.agent.flags.use_abstract_example
            )
        with col6:
            st.session_state.agent.flags.use_concrete_example = st.checkbox(
                "use_concrete_example", value=st.session_state.agent.flags.use_concrete_example
            )
        extra_instructions = st.text_area(
            "extra_instructions", value=st.session_state.agent.flags.extra_instructions
        )
        if extra_instructions == "":
            extra_instructions = None
        st.session_state.agent.flags.extra_instructions = extra_instructions


def set_go_back_to_step_n_section():
    with st.container(border=True):
        st.markdown("**Go Back to Step N**")
        col1, col2 = st.columns([1, 1], vertical_alignment="bottom")
        is_go_back_to_step_n_disabled = len(st.session_state.action_history) <= 1
        with col1:
            step = st.number_input(
                "Step",
                value=1,
                min_value=1,
                max_value=len(st.session_state.action_history),
                disabled=is_go_back_to_step_n_disabled,
            )
        with col2:
            if st.button(
                "⬅️ Go Back",
                help="Go back to step N",
                use_container_width=True,
                disabled=is_go_back_to_step_n_disabled,
            ):
                logger.info(f"Going back to step {step}")
                reset_agent_state()
                restore_agent_history(step=step)
                reset_env_history()
                restore_environment()
                st.rerun()


def set_regenerate_action_n_times_section():
    with st.container(border=True):
        st.markdown("**Regenerate Action N Times**")
        col1, col2 = st.columns([1, 1], vertical_alignment="bottom")
        with col1:
            n = st.number_input(
                "Number of Actions to Generate",
                value=5,
                min_value=1,
                max_value=25,
            )
        with col2:
            st.session_state.has_clicked_multiple_reprompt = st.button(
                "🔄 Regenerate",
                help="Reprompt the agent K times to get a distribution of actions to take",
                use_container_width=True,
            )
        if st.session_state.has_clicked_multiple_reprompt:
            logger.info(f"Regenerating action {n} times...")
            reprompt_actions = []
            action_to_info_mapping = {}
            action_to_memory_mapping = {}
            progress_bar = st.progress(0, text=f"Regenerating action {n} times...")
            for i in range(n):
                progress_bar.progress((i + 1) / n, text=f"Regenerating action {i + 1} of {n}...")
                revert_agent_history()
                revert_agent_state()
                get_action()
                reprompt_actions.append(st.session_state.action)
                action_to_info_mapping[st.session_state.action] = copy.deepcopy(
                    st.session_state.action_info
                )
                action_to_memory_mapping[st.session_state.action] = copy.deepcopy(
                    st.session_state.agent.memories[-1]
                )
            progress_bar.progress(1, text=f"Regenerating action {n} times...")
            progress_bar.empty()
            # show all unique actions found in reprompt actions along with their probability
            unique_actions_counter = Counter(reprompt_actions)
            unique_actions = sorted(
                unique_actions_counter.items(), key=lambda x: x[1], reverse=True
            )
            st.markdown("**Unique Actions**")
            for action, count in unique_actions:
                has_clicked_reprompted_action = st.button(f"`{action}` ({count / n * 100:.2f}%)")
                if has_clicked_reprompted_action:
                    logger.info(f"Selected action: {action} -- stepping")
                    st
                    revert_agent_history()
                    revert_agent_state()

                    # manually step agent state
                    st.session_state.agent.obs_history.append(
                        copy.deepcopy(st.session_state.last_obs)
                    )
                    st.session_state.agent.actions.append(action)
                    st.session_state.agent.thoughts.append(action_to_info_mapping[action].think)
                    st.session_state.agent.memories.append(action_to_memory_mapping[action])

                    step_agent_history(action, action_to_info_mapping[action])
                    # step_environment(action)
                    st.session_state.has_clicked_multiple_reprompt = False
                    st.rerun()


def set_act_k_times_section():
    with st.container(border=True):
        st.markdown("**Go Forward N Steps**")
        col1, col2 = st.columns([1, 1], vertical_alignment="bottom")
        with col1:
            n = st.number_input("Number of Steps", value=5, min_value=1, max_value=10)
        with col2:
            has_clicked_act = st.button(
                "➡️ Go Forward",
                help="Let the agent autonomously perform actions for N steps",
                use_container_width=True,
            )
        if has_clicked_act:
            logger.info(f"Going forward {n} steps...")
            progress_bar = st.progress(0, text=f"Going forward {n} steps...")
            for i in range(n):
                if st.session_state.action is None:  # so that we don't do it for first step
                    get_action()
                step_environment(st.session_state.action)
                progress_bar.progress((i + 1) / n, text=f"Going forward {i + 1} of {n}...")
            progress_bar.empty()
            st.rerun()


def set_advanced_controller():
    with st.expander("**Advanced**", expanded=False):
        col_go_back_to, col_reprompt_k, col_act_k = st.columns([1, 1, 1])
        with col_go_back_to:
            set_go_back_to_step_n_section()
        with col_reprompt_k:
            set_regenerate_action_n_times_section()
        with col_act_k:
            set_act_k_times_section()


def set_previous_step_section():
    prev_disabled = len(st.session_state.action_history) <= 1
    if st.button("⬅️ Previous Step", disabled=prev_disabled, use_container_width=True):
        if not prev_disabled:
            logger.info("Clicked previous step")
            st.session_state.action = (
                None
                if len(st.session_state.action_history) == 0
                else st.session_state.action_history[-1]
            )
            reset_agent_state()
            revert_agent_history()
            reset_env_history()
            restore_environment()
            st.rerun()


def set_regenerate_action_section():
    if st.button("🔄 Regenerate Action", use_container_width=True):
        logger.info("Clicked regenerate action")
        revert_agent_history()
        revert_agent_state()
        get_action()
        st.rerun()


def set_next_step_section():
    if st.button("➡️ Next Step", use_container_width=True):
        logger.info("Clicked next step")
        step_environment(st.session_state.action)
        st.rerun()


def set_controller():
    with st.container(border=True):
        st.markdown("##### 🎮 Control")
        set_agent_state_box()
        set_prompt_modifier()
        col_prev, col_redo, col_next = st.columns([1, 1, 1])
        with col_prev:
            set_previous_step_section()
        with col_redo:
            set_regenerate_action_section()
        with col_next:
            set_next_step_section()
        set_advanced_controller()


def get_base64_serialized_image(img_arr):
    if isinstance(img_arr, list):
        img_arr = np.array(img_arr)
    if isinstance(img_arr, np.ndarray):
        im = PIL.Image.fromarray(img_arr)
        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        return img_b64
    return None


def display_image(img_arr):
    img_b64 = get_base64_serialized_image(img_arr)
    if img_b64:
        st.markdown(
            f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{img_b64}" style="max-width: 80vw; height: auto;" /></div>',
            unsafe_allow_html=True,
        )


def set_screenshot_tab():
    display_image(st.session_state.screenshot_history[-1])


def set_axtree_tab():
    st.code(st.session_state.axtree_history[-1], language=None, wrap_lines=True)


def set_prompt_tab():
    st.code(st.session_state.prompt_history[-1], language=None, wrap_lines=True)


def set_previous_steps_tab():
    for i in range(len(st.session_state.action_history) - 1):
        with st.expander(f"### Step {i + 1}", expanded=False):
            if st.button(f"Go back to step {i + 1}"):
                logger.info(f"Go back to step {i + 1}")
                reset_agent_state()
                restore_agent_history(step=i + 1)
                reset_env_history()
                restore_environment()
                st.rerun()
            screenshot_tab, axtree_tab, prompt_tab = st.tabs(["Screenshot", "AxTree", "Prompt"])
            with screenshot_tab:
                display_image(st.session_state.screenshot_history[i])
            with axtree_tab:
                st.code(st.session_state.axtree_history[i], language=None, wrap_lines=True)
            with prompt_tab:
                st.code(st.session_state.prompt_history[i], language=None, wrap_lines=True)
            st.markdown("**Thought**")
            st.code(st.session_state.thought_history[i], language=None, wrap_lines=True)
            st.markdown("**Action**")
            st.code(st.session_state.action_history[i], language=None, wrap_lines=True)


def set_save_tab():
    # dump full session_state to json
    save_dir = st.text_input("Save Directory", value="~/Downloads")
    save_dir = os.path.expanduser(save_dir)
    if st.button("Save Session State for Current Run"):
        now_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        filename = f"agentlab_controller_state_{now_str}.json"

        # prepare payload for saving
        payload = {}
        payload["timestamp"] = now_str
        payload["benchmark"] = st.session_state.benchmark
        payload["task"] = st.session_state.task
        payload["subtask"] = st.session_state.subtask
        payload["agent_args"] = {
            k: v for k, v in vars(st.session_state.agent_args).items() if is_json_serializable(v)
        }
        payload["agent_flags"] = {
            k: v for k, v in vars(st.session_state.agent.flags).items() if is_json_serializable(v)
        }
        payload["agent_flags"]["obs"] = {
            k: v
            for k, v in vars(st.session_state.agent.flags.obs).items()
            if is_json_serializable(v)
        }
        payload["agent_flags"]["action"] = {
            k: v
            for k, v in vars(st.session_state.agent.flags.action).items()
            if is_json_serializable(v)
        }
        payload["goal"] = st.session_state.last_obs["goal"]
        payload["steps"] = []
        for i in range(len(st.session_state.action_history)):
            step = {}
            step["action"] = st.session_state.action_history[i]
            step["thought"] = st.session_state.thought_history[i]
            step["prompt"] = st.session_state.prompt_history[i]
            step["screenshot"] = get_base64_serialized_image(st.session_state.screenshot_history[i])
            step["axtree"] = st.session_state.axtree_history[i]
            payload["steps"].append(step)

        with open(os.path.join(save_dir, filename), "w") as f:
            json.dump(payload, f)


def set_info_tabs():
    with st.container(border=True):
        st.markdown("##### 🔎 Analyze")
        # Display only if everything is now ready
        if len(st.session_state.action_history) > 1:
            screenshot_tab, axtree_tab, prompt_tab, previous_steps_tab, save_tab = st.tabs(
                ["Screenshot", "AxTree", "Prompt", "Previous Steps", "Save"]
            )
        else:
            screenshot_tab, axtree_tab, prompt_tab = st.tabs(["Screenshot", "AxTree", "Prompt"])

        with screenshot_tab:
            set_screenshot_tab()
        with axtree_tab:
            set_axtree_tab()
        with prompt_tab:
            set_prompt_tab()
        if len(st.session_state.action_history) > 1:
            with previous_steps_tab:
                set_previous_steps_tab()
            with save_tab:
                set_save_tab()


def run_streamlit():

    # config page
    st.set_page_config(
        page_title="AgentLab Controller",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        '<h1 style="text-align: center;">🎮 AgentLab Controller 🎮</h1>', unsafe_allow_html=True
    )

    setup_sidebar()

    set_session_state()
    set_task_selector()

    if st.session_state.agent is not None:
        if st.session_state.action is None:
            get_action()

        set_controller()
        set_info_tabs()


def main():
    run_streamlit()


if __name__ == "__main__":
    main()
