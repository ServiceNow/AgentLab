from dataclasses import dataclass
import re
from typing import Any, Dict, List

from browsergym.experiments.agent import Agent as BrowserGymAgent
from browsergym.experiments.loop import AbstractAgentArgs

from .prompt_agent import PromptAgent
from .agent import Agent
from .utils.stack import Stack, Element
from .prompts.miniwob import step_fewshot_template


class StepAgent(Agent):
    def __init__(self, max_actions: int = 10, verbose: bool = False, logging: bool = False,
                 debug: bool = False,
                 root_action: str = None,
                 action_to_prompt_dict: Dict = None,
                 low_level_action_list: List = None,
                 model: str = "gpt-3.5-turbo",
                 prompt_mode: str = "chat",
                 previous_actions: List = None):
        super().__init__(
            max_actions=max_actions,
            previous_actions=previous_actions,
        )
        self.debug = debug
        self.root_action = root_action
        self.action_to_prompt_dict = {} if action_to_prompt_dict is None else action_to_prompt_dict
        self.low_level_action_list = [] if low_level_action_list is None else low_level_action_list
        self.model = model
        self.prompt_mode = prompt_mode
        self.stack = Stack()

    def is_done(self, action):
        if "stop" in action:
            return True
        return False

    def is_low_level_action(self, action):
        action_type = action.split()[0]
        return (action_type in self.low_level_action_list)

    def is_high_level_action(self, action):
        action_type = action.split()[0]
        return (action_type in self.action_to_prompt_dict)

    def init_root_agent(self, objective):
        root_prompt_template = self.action_to_prompt_dict[self.root_action]
        agent = PromptAgent(
            prompt_template=root_prompt_template,
            model=self.model,
            prompt_mode=self.prompt_mode,
            max_actions=self.max_actions,
            verbose=self.verbose,
            logging=self.logging,
            debug=self.debug,
            previous_actions=[],
            previous_reasons=[],
            previous_responses=[]
        )
        return Element(agent=agent, objective=objective)

    def init_agent(self, action):
        pattern = r'(\w+)\s+\[(.*?)\]'
        matches = re.findall(pattern, action)
        action_type, _ = matches[0]
        objective = action
        prompt_template = self.action_to_prompt_dict[action_type]
        agent = PromptAgent(
            prompt_template=prompt_template,
            model=self.model,
            prompt_mode=self.prompt_mode,
            max_actions=self.max_actions,
            verbose=self.verbose,
            logging=self.logging,
            debug=self.debug,
            previous_actions=[],
            previous_reasons=[],
            previous_responses=[]
        )
        return Element(agent=agent, objective=objective)

    def predict_action(self, objective: str, observation: str, url: str = None) -> tuple[str, dict]:
        print(type(observation), observation)
        
        if self.stack.is_empty():
            new_element = self.init_root_agent(objective=objective)
            self.stack.push(new_element)

        action, reason = None, None
        while not self.stack.is_empty():
            element = self.stack.peek()
            action, reason = element.agent.predict_action(
                objective=element.objective, observation=observation, url=url)
            if (not self.is_done(action)) and self.is_low_level_action(action):
                element.agent.receive_response("")
                return action, reason
            if (not self.is_done(action)) and self.is_high_level_action(action):
                new_element = self.init_agent(action)
                self.stack.push(new_element)
                if self.logging:
                    self.log_step(objective=element.objective, url=url,
                                  observation=observation, action=action, reason=reason, status={})
                continue
            if self.is_done(action):
                self.stack.pop()
                if not self.stack.is_empty():
                    self.stack.peek().agent.receive_response(
                        re.search(r"\[(.*?)\]", action).group(1))
                if self.logging:
                    self.log_step(objective=element.objective, url=url,
                                  observation=observation, action=action, reason=reason, status={})
                continue
        return action, reason


@dataclass
class BrowserGymStepAgentArgs(AbstractAgentArgs):
    agent_name: str = "StepAgent"
    max_actions: int = 10
    verbose: bool = False
    logging: bool = False
    debug: bool = False
    root_action: str = None
    action_to_prompt_dict: Dict = None
    low_level_action_list: List = None
    model: str = "gpt-3.5-turbo"
    prompt_mode: str = "chat"
    previous_actions: List = None

    def make_agent(self):
        return BrowserGymStepAgent(
            max_actions=self.max_actions,
            verbose=self.verbose,
            logging=self.logging,
            debug=self.debug,
            root_action=self.root_action,
            action_to_prompt_dict=self.action_to_prompt_dict,
            low_level_action_list=self.low_level_action_list,
            model=self.model,
            prompt_mode=self.prompt_mode,
            previous_actions=self.previous_actions
        )


class BrowserGymStepAgent(BrowserGymAgent):
    def __init__(self, max_actions: int = 10, verbose: bool = False, logging: bool = False,
                 debug: bool = False,
                 root_action: str = None,
                 action_to_prompt_dict: Dict = None,
                 low_level_action_list: List = None,
                 model: str = "gpt-3.5-turbo",
                 prompt_mode: str = "chat",
                 previous_actions: List = None
                 ):
        action_to_prompt_dict = {k: v for k, v in step_fewshot_template.__dict__.items() if isinstance(v, dict)}
        root_action="miniwob_agent"
        self.agent = StepAgent(
            max_actions=max_actions, verbose=verbose, logging=logging,
            debug=debug, root_action=root_action,
            action_to_prompt_dict=action_to_prompt_dict,
            low_level_action_list=low_level_action_list,
            model=model, prompt_mode=prompt_mode,
            previous_actions=previous_actions)
        super().__init__()

    def reset(self):
        pass

    def get_action(self, obs: dict) -> tuple[str, dict]:
        url = obs["url"] if "url" in obs else None
        objective = obs["goal"] if "goal" in obs else None
        observation = obs["axtree_object"] if "axtree_object" in obs else None
        return self.agent.predict_action(objective=objective, observation=observation, url=url)
