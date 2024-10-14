import logging
import re
from dataclasses import asdict, dataclass
from functools import partial
from warnings import warn

from browsergym.experiments.agent import Agent
from langchain.schema import HumanMessage, SystemMessage

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.utils import openai_monitored_agent
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import RetryError, retry_raise, ParseError
from agentlab.llm.tracking import cost_tracker_decorator

from .generic_agent_prompt import GenericPromptFlags, MainPrompt


@dataclass
class GenericAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"GenericAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark):
        if benchmark == "miniwob":
            self.flags.obs.use_html = True

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class GenericAgent(Agent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
    ):

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.action_set = dp.make_action_set(self.flags.action)
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        self._check_flag_constancy()
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):

        self.obs_history.append(obs)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = dp.SystemPrompt().prompt

        prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        stats = {}
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
            ans_dict = retry_raise(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            # inferring the number of retries, TODO: make this less hacky
            stats["n_retry"] = (len(chat_messages) - 3) / 2
            stats["busted_retry"] = 0
        except RetryError as e:
            ans_dict = {"action": None}
            stats["busted_retry"] = 1

            stats["n_retry"] = self.max_retry + 1

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = dict(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr


from functools import partial


def get_action_post_hoc(agent: GenericAgent, obs: dict, ans_dict: dict):
    """
    Get the action post-hoc for the agent.

    This function is used to get the action after the agent has already been run.
    Its goal is to recreate the prompt and the output of the agent a posteriori.
    The purpose is to build datasets for training the agents.

    Args:
        agent (GenericAgent): The agent for which the action is being determined.
        obs (dict): The observation dictionary to append to the agent's history.
        ans_dict (dict): The answer dictionary containing the plan, step, memory, think, and action.

    Returns:
        Tuple[str, str]: The complete prompt used for the agent and the reconstructed output based on the answer dictionary.
    """
    system_prompt = dp.SystemPrompt().prompt

    agent.obs_history.append(obs)

    main_prompt = MainPrompt(
        action_set=agent.action_set,
        obs_history=agent.obs_history,
        actions=agent.actions,
        memories=agent.memories,
        thoughts=agent.thoughts,
        previous_plan=agent.plan,
        step=agent.plan_step,
        flags=agent.flags,
    )

    max_prompt_tokens, max_trunc_itr = agent._get_maxes()

    fit_function = partial(
        dp.fit_tokens,
        max_prompt_tokens=max_prompt_tokens,
        model_name=agent.chat_model_args.model_name,
        max_iterations=max_trunc_itr,
    )

    instruction_prompt = fit_function(shrinkable=main_prompt)

    if isinstance(instruction_prompt, list):
        # NOTE: this is when we have images
        instruction_prompt = instruction_prompt[0]["text"]

    # TODO: make sure the bid is in the prompt

    output = ""

    # TODO: validate this
    agent.plan = ans_dict.get("plan", agent.plan)
    if agent.plan != "No plan yet":
        output += f"\n<plan>\n{agent.plan}\n</plan>\n"

    # TODO: is plan_step something that the agent's outputs?
    agent.plan_step = ans_dict.get("step", agent.plan_step)

    memory = ans_dict.get("memory", None)
    agent.memories.append(memory)
    if memory is not None:
        output += f"\n<memory>\n{memory}\n</memory>\n"

    thought = ans_dict.get("think", None)
    agent.thoughts.append(thought)
    if thought is not None:
        output += f"\n<think>\n{thought}\n</think>\n"

    action = ans_dict["action"]
    agent.actions.append(action)
    if action is not None:
        output += f"\n<action>\n{action}\n</action>"

    return system_prompt, instruction_prompt, output


def get_action_post_hoc(agent: GenericAgent, step_info):
    """
    Get the action post-hoc for the agent.

    This function is used to get the action after the agent has already been run.
    Its goal is to recreate the prompt and the output of the agent a posteriori.
    The purpose is to build datasets for training the agents.

    Args:
        agent (GenericAgent): The agent for which the action is being determined.
        obs (dict): The observation dictionary to append to the agent's history.
        ans_dict (dict): The answer dictionary containing the plan, step, memory, think, and action.

    Returns:
        Tuple[str, str]: The complete prompt used for the agent and the reconstructed output based on the answer dictionary.
    """
    system_prompt = dp.SystemPrompt().prompt

    agent.obs_history.append(step_info.obs)

    main_prompt = MainPrompt(
        action_set=agent.action_set,
        obs_history=agent.obs_history,
        actions=agent.actions,
        memories=agent.memories,
        thoughts=agent.thoughts,
        previous_plan=agent.plan,
        step=agent.plan_step,
        flags=agent.flags,
    )

    max_prompt_tokens, max_trunc_itr = agent._get_maxes()

    fit_function = partial(
        dp.fit_tokens,
        max_prompt_tokens=max_prompt_tokens,
        model_name=agent.chat_model_args.model_name,
        max_iterations=max_trunc_itr,
    )

    instruction_prompt = fit_function(shrinkable=main_prompt)

    if isinstance(instruction_prompt, list):
        # NOTE: this is when we have images
        instruction_prompt = instruction_prompt[0]["text"]

    def parser(text):
        try:
            ans_dict = main_prompt._parse_answer(text)
        except ParseError as e:
            # these parse errors will be caught by the retry function and
            # the chat_llm will have a chance to recover
            return None, False, str(e)
        return ans_dict, True, ""

    og_agent_output = step_info.agent_info["chat_messages"][-1].content
    if og_agent_output.startswith("assistant\n"):
        og_agent_output = og_agent_output[10:]

    ans_dict = parser(og_agent_output)[0]

    # self.plan = ans_dict.get("plan", self.plan)
    # self.plan_step = ans_dict.get("step", self.plan_step)
    # self.actions.append(ans_dict["action"])
    # self.memories.append(ans_dict.get("memory", None))
    # self.thoughts.append(ans_dict.get("think", None))

    agent_output = ""

    # TODO: validate this
    thought = ans_dict.get("think", None)
    agent.thoughts.append(thought)
    if thought is not None:
        agent_output += f"\n<think>\n{thought}\n</think>\n"

    agent.plan = ans_dict.get("plan", agent.plan)
    if agent.plan != "No plan yet":
        agent_output += f"\n<plan>\n{agent.plan}\n</plan>\n"

    agent.plan_step = ans_dict.get("step", agent.plan_step)
    if agent.plan_step != -1:
        agent_output += f"\n<step>{agent.plan_step}</step>\n"

    memory = ans_dict.get("memory", None)
    agent.memories.append(memory)
    if memory is not None:
        agent_output += f"\n<memory>\n{memory}\n</memory>\n"

    action = step_info.action
    agent.actions.append(action)
    if action is not None:
        agent_output += f"\n<action>\n{action}\n</action>"

    def find_bid(string):
        # Try to find 'a' followed by digits within single or double quotes
        match = re.search(r"[\"'](a\d+)[\"']", string)

        # If not found, search digits within single or double quotes
        if not match:
            match = re.search(r"[\"'](\d+)[\"']", string)

        # Return the matched pattern or None if no match found
        if match:
            return match.group(1)  # Return the match inside the quotes
        else:
            return None

    # TODO: finish this
    bid = find_bid(action)
    if bid is not None:
        if bid not in instruction_prompt:
            logging.info("Bid is not in the instruction prompt.")
            return "missing_bid"

    # NOTE: keep in mind the original agent output can be more verbose
    if agent_output not in og_agent_output:
        logging.info("Agent output does exactly not match the last chat message.")
        if not set(agent_output.split()).issubset(set(og_agent_output.split())):
            logging.info("Agent output does not match the last chat message.")
            return "action_output_mismatch"

    # TODO: make sure the bid is in the prompt
    return (system_prompt, instruction_prompt, agent_output)
