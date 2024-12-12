from copy import deepcopy
from dataclasses import dataclass

import bgym
from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.utils.obs import overlay_som

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.dynamic_prompting import Tabs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import (
    AIMessage,
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    extract_html_tags,
    retry,
    retry_multiple,
)
from agentlab.llm.tracking import LLMTracker, set_tracker


@dataclass
class WebDreamerFlags:
    action_set: bgym.HighLevelActionSetArgs = None
    use_refiner: bool = True
    obs_key: bool = "axtree_txt"
    use_tabs: bool = False
    long_description: bool = False
    examples: bool = False
    num_samples: int = 10
    use_axtree_in_wm: bool = False
    use_som: bool = True


@dataclass
class WebDreamerArgs(AgentArgs):
    flags: WebDreamerFlags = None
    chat_model_args: BaseModelArgs = None

    def __post_init__(self):
        try:
            self.agent_name = f"WebDreamerAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs_key = "pruned_html"

        self.flags.use_tabs = benchmark.is_multi_tab
        self.flags.action_set = deepcopy(benchmark.high_level_action_set_args)

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0.0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self) -> Agent:
        return WebDreamerAgent(self.chat_model_args, self.flags)


class WebDreamerAgent(Agent):
    def __init__(self, chat_model_args: BaseModelArgs, flags: WebDreamerFlags):
        self.chat_model_args = chat_model_args
        self.flags = flags

        self.action_set = flags.action_set.make_action_set()

        if self.flags.use_refiner:
            self.refiner = Refiner(self.chat_model_args.make_model(), flags, self.action_set)
        self.world_model = WorldModel(self.chat_model_args.make_model(), flags)
        self.value_model = ValueModel(self.chat_model_args.make_model(), flags)

        controller_chat_model_args = deepcopy(chat_model_args)
        controller_chat_model_args.temperature = 1.0
        controller_chat_model_args.top_p = 0.95

        self.controller = Controller(
            controller_chat_model_args.make_model(), flags, self.action_set
        )
        self.history = []

    def obs_preprocessor(self, obs):
        obs = super().obs_preprocessor(obs)
        if self.flags.use_som:
            obs["som"] = overlay_som(
                obs["screenshot"], extra_properties=obs["extra_element_properties"]
            )
        return obs

    def get_action(self, obs: dict) -> tuple[str, AgentInfo]:
        self.history.append(obs)
        markdown_summary = ""
        stats_summary = {}
        with set_tracker() as global_tracker:
            messages = Discussion()
            trackers = []  # type: list[LLMTracker]

            # call Controller for possible actions
            with set_tracker("controller") as controller_tracker:
                possible_actions, content, markdown, stats = self.controller(self.history)
            trackers.append(controller_tracker)
            for c in content:
                messages.append(c)
            markdown_summary += "\n## Controller\n" + markdown
            stats_summary.update(stats)

            # call Refiner to refine the possible actions
            if self.flags.use_refiner:
                with set_tracker("refiner") as refiner_tracker:
                    refined_actions, content, markdown, stats = self.refiner(
                        possible_actions, self.history
                    )
                trackers.append(refiner_tracker)
                for c in content:
                    messages.append(c)
            markdown_summary += "\n## Refiner\n" + markdown
            stats_summary.update(stats)

            # call WorldModel predict state changes
            with set_tracker("world_model") as world_model_tracker:
                world_model_output, content, markdown, stats = self.world_model(
                    refined_actions, self.history
                )
            trackers.append(world_model_tracker)
            for c in content:
                messages.append(c)
            markdown_summary += "\n## WorldModel\n" + markdown
            stats_summary.update(stats)

            # call ValueModel to predict the value of the resulting states
            with set_tracker("value_model") as value_model_tracker:
                value_model_output, content, markdown, stats, txt_values = self.value_model(
                    world_model_output, refined_actions, self.history
                )
            trackers.append(value_model_tracker)
            for c in content:
                messages.append(c)
            markdown_summary += "\n## ValueModel\n" + markdown
            stats_summary.update(stats)

        markdown_summary += "\n## WorldModel/Value interaction\n"
        for action, state, txt_value in zip(refined_actions, world_model_output, txt_values):
            markdown_summary += f"\n### Action: {action}\n"
            markdown_summary += f"\n{state}\n"
            markdown_summary += f"\nValue: \n{txt_value}\n"

        markdown_summary = self.process_md(markdown_summary)

        # get all model stats
        stats_summary.update(global_tracker.stats)
        for tracker in trackers:
            stats_summary.update(tracker.stats)

        action = self.get_best_action(value_model_output, refined_actions)
        return action, AgentInfo(
            think="", chat_messages=messages, stats=stats_summary, markdown_page=markdown_summary
        )

    def process_md(self, markdown_summary: str) -> str:
        markdown_summary = markdown_summary.replace("<think>", "\n<think>").replace(
            "</think>", "</think>\n"
        )
        markdown_summary = markdown_summary.replace("<status>", "\n<status>").replace(
            "</status>", "</status>\n"
        )
        markdown_summary = markdown_summary.replace("<track>", "\n<track>").replace(
            "</track>", "</track>\n"
        )

        markdown_summary = markdown_summary.replace("<", "&lt;").replace(">", "&gt;")
        return markdown_summary

    def get_best_action(self, value_model_output: list[float], refined_actions: list[str]) -> str:
        return refined_actions[value_model_output.index(max(value_model_output))]


class Controller:
    def __init__(self, model, flags: WebDreamerFlags, action_set: bgym.HighLevelActionSet):
        self.model = model
        self.flags = flags
        self.action_set = action_set

        self.system_prompt = SystemMessage(
            f"""You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue."""
        )

    def __call__(self, history: list[dict]) -> tuple[list[str], Discussion, str, dict]:
        obs = history[-1]
        prompt = Discussion(self.system_prompt)
        prompt.append(HumanMessage("""Here is the information you will have:\n"""))
        prompt.add_text(obs[self.flags.obs_key])
        prompt.add_text("\nThe user's objective:\n")
        for goal in obs["goal_object"]:
            prompt.add_content(type=goal["type"], content=goal[goal["type"]])
        prompt.add_text("\nThis is the task you are trying to complete.")
        prompt.add_text("\nThe current webpage screenshot:\n")
        if self.flags.use_som:
            prompt.add_image(obs["som"])
        else:
            prompt.add_image(obs["screenshot"])
        prompt.add_text(
            "The observation you are given has a [bid] for each element in the webpage. These are used to interact with the webpage. `[a77] button 'Manage Attachments', clickable, visible` means that the element with bid `a77` is a button with the text 'Manage Attachments' that is clickable and visible on the webpage."
        )
        prompt.add_text(Tabs(obs).prompt)
        prompt.add_text("\nThe previous actions taken by the agent, this can be useful:\n")
        prompt.add_text(obs["last_action"])
        prompt.add_text("\n")
        prompt.add_text(
            "Here are the possible actions you can take, they are mostly playwright actions, that you can leverage using the [bid] in the page:\n"
        )
        prompt.add_text(
            self.action_set.describe(
                with_long_description=self.flags.long_description,
                with_examples=self.flags.examples,
            )
        )

        prompt.add_text(
            "\nSpecifically, the action `send_msg_to_user` ends the episode, and allows you to provide required information if needed. Do not use this action to ask for help."
        )
        prompt.add_text(
            """
To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Generate the action in the correct format:
<think>Put any reflections or reasoning here</think>
<action>Your selected action using a [bid] from the page</action>
"""
        )
        answers, tries = retry_multiple(
            self.model,
            prompt,
            n_retry=4,
            parser=self.parser,
            num_samples=self.flags.num_samples,
        )
        markdown = f"```\n{"\n".join(answers)}\n```"
        return answers, prompt, markdown, {}  # TODO log more info

    def parser(self, answer: str) -> str:
        res = extract_html_tags(answer, ["action"])
        if "action" not in res:
            raise ParseError("No actions found")
        return res["action"][0]


class Refiner:
    def __init__(self, model, flags: WebDreamerFlags, action_set: bgym.HighLevelActionSet):
        self.model = model
        self.flags = flags

        self.system_prompt = SystemMessage(
            f"""
You are assisting a web navigation agent to help a human user navigate a website to complete a task. Given the user's intent, the action history, and the current state of the webpage, the agent has proposed a set of candidate actions to take at the current step. 
Your role is not to determine a best action for the agent at this step, but to filter out the actions that are very likely not relevant or helpful for the agent to accomplish the task.
Please select all actions that you think that could possibly lead the agent to accomplish the task. It's important to note that to accomplish a task, the agent will execute a sequence of actions. So the action to take at this step does not have to immediately lead to the completion of the task. You should select any action that could be relevant for the agent to take in the current state of the webpage. Try to be as thoughtful and comprehensive as you can! Don't miss any possible action. If there is one action that is clearly the best, and all other actions are clearly not very relevant, you can only select one action. Please do this sparely, since some actions may be helpful in a longer horizon. 
A action should be included as long as it could be relevant to the task, even if it may not be the most direct action to take at this step!! Some relevant actions might seem indirect at the first glance, but could be helpful in a longer horizon. Please also include those actions.
Please at least select one action.

*IMPORTANT*
Format your response into two lines as shown below:

<think>your thoughts and reasoning process. You must explicitly evaluate each action one by one and imagine whether it could be relevant to the task following the format: actoin:... rationale:... </think>
<action>action1</action>
<action>action2</action>
<action>as many actions as you think are relevant</action>
(please return the actions from the candidate actions list. Don't output the action description itself.)
"""
        )

    def __call__(
        self, possible_actions: list[str], history: list[dict]
    ) -> tuple[list[str], Discussion, str, dict]:
        # remove identical actions first
        orgn_len = len(possible_actions)
        possible_actions = list(set(possible_actions))
        len_diff = orgn_len - len(possible_actions)
        stats = {"removed_identical_actions": len_diff}
        possible_actions_prompt = "</action>\n<action>".join(possible_actions)
        possible_actions_prompt = f"<action>{possible_actions_prompt}</action>"

        last_action_history = "\n".join([h["last_action"] for h in history])

        prompt = Discussion(self.system_prompt)
        prompt.append(HumanMessage("Here is the goal of the user you must accomplish:\n"))
        for goal in history[-1]["goal_object"]:
            prompt.add_content(type=goal["type"], content=goal[goal["type"]])
        prompt.add_text("\nHere are the previous actions taken by the agent:\n")
        prompt.add_text(last_action_history)
        prompt.add_text(Tabs(history[-1]).prompt)
        prompt.add_text("\nHere are the possible actions, evaluate and select the relevant ones:\n")
        prompt.add_text(possible_actions_prompt)

        try:
            answers = retry(self.model, prompt, n_retry=4, parser=self.parser)
            refined_actions = [a for a in answers["action"]]
            markdown = f"Selected actions:\n```\n{"\n".join(refined_actions)}\n```"
        except ParseError as e:
            refined_actions = possible_actions
            markdown = "No actions selected, using all possible actions"
        stats["removed_actions"] = len(possible_actions) - len(refined_actions)
        return refined_actions, prompt, markdown, stats  # TODO log more info

    def parser(self, answer: str) -> dict[list[str]]:
        res = extract_html_tags(answer, ["think", "action"])
        if "action" not in res:
            raise ParseError("No refinements found")
        return res


class WorldModel:
    def __init__(self, model, flags: WebDreamerFlags):
        self.model = model
        self.flags = flags
        self.system_prompt = SystemMessage(
            "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage and an operation to perform on the webpage. You are required to predict the changes that will occur on the webpage after the operation is performed, such as the appearance of new elements, the disappearance of existing elements, or changes in the content of existing elements. The operation type and the element to operate will be provided in the prompt. Directly output 'State changes: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        )

    def __call__(
        self, refined_actions, history: list[dict]
    ) -> tuple[list[str], Discussion, str, dict]:
        states = []
        markdown = ""
        for i, action in enumerate(refined_actions):
            prompt = Discussion(self.system_prompt)
            prompt.append(HumanMessage("Here is the current webpage screenshot:\n"))
            if self.flags.use_som:
                prompt.add_image(history[-1]["som"])
            else:
                prompt.add_image(history[-1]["screenshot"])
            if self.flags.use_axtree_in_wm:
                prompt.add_text("The current webpage accessibility tree:\n")
                prompt.add_text(history[-1]["axtree_txt"])
            prompt.add_text(
                "\nBased on those observation, please predict the changes after action:\n"
            )
            prompt.add_text(action)
            answer = retry(self.model, prompt, n_retry=4, parser=self.parser)
            states.append(answer)
            markdown += f"\n### Action {i+1}: {action}\n{answer}\n"
        return states, prompt, markdown, {}  # TODO log more info

    def parser(self, answer: str) -> str:
        return answer


class ValueModel:
    def __init__(self, model, flags: WebDreamerFlags):
        self.model = model
        self.flags = flags
        self.system_prompt = SystemMessage(
            """
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, your goal is to decide **whether the proposed action successfully accomplish the task**. If it does not but is on the right track towards success, you should also output as such. You don't have the ability to predict the future, so you should only consider the current state of the webpage and the proposed action.

*IMPORTANT*
Format your response into three lines as shown below:

<think>your thoughts and reasoning process</think>
<status>success or failure</status>
<track>yes or no</track> if it is on track to success 
"""
        )

    def __call__(
        self, world_model_output, imagined_actions, history: list[dict]
    ) -> tuple[list[float], Discussion, str, dict, list[str]]:
        values = []
        txt_values = []
        markdown = ""
        for state, action in zip(world_model_output, imagined_actions):
            prompt = Discussion(self.system_prompt)
            prompt.append(HumanMessage("Here is the goal of the user the agent must accomplish:\n"))
            for goal in history[-1]["goal_object"]:
                prompt.add_content(type=goal["type"], content=goal[goal["type"]])
            prompt.add_text("\nHere is the current webpage screenshot:\n")
            if self.flags.use_som:
                prompt.add_image(history[-1]["som"])
            else:
                prompt.add_image(history[-1]["screenshot"])
            prompt.add_text("\nHere is the action taken by the agent:\n")
            prompt.add_text(action)
            prompt.add_text("\nHere is the predicted state of the webpage:\n")
            prompt.add_text(state)
            prompt.add_text("\nEnd of prediction\n")
            answer = retry(self.model, prompt, n_retry=4, parser=self.parser)
            values.append(self.process_value(answer))
            txt_values.append(answer["full"])
            markdown += f"\n### Action {action}\n{str(prompt.messages[-1])}\nValue: {values[-1]}\n"
        return values, prompt, markdown, {}, txt_values  # TODO log more info

    def parser(self, answer: str) -> dict[str, str]:
        res = extract_html_tags(answer, ["think", "status", "track"])
        if "status" not in res or "track" not in res:
            raise ParseError("Missing status or track")
        if len(res["status"]) != 1 or len(res["track"]) != 1:
            raise ParseError("Multiple status or track")
        res["status"] = res["status"][0]
        res["track"] = res["track"][0]
        res["full"] = answer
        return res

    def process_value(self, answer: dict) -> float:
        if answer["status"] == "success":
            return 1.0
        elif answer["track"] == "yes":
            return 0.5
        else:
            return 0.0
