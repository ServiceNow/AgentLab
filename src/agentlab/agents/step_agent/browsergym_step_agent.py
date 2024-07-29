from dataclasses import dataclass
import re
from typing import Literal, Dict, List

import bs4

from browsergym.experiments.agent import Agent
from browsergym.experiments.loop import AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from agentlab.llm.chat_api import BaseModelArgs
from .step_agent import StepAgent


@dataclass
class BrowserGymStepAgentArgs(AbstractAgentArgs):
    agent_name: str = "StepAgent"
    max_actions: int = 10
    verbose: int = 0
    logging: bool = False
    root_action: str = None
    action_to_prompt_dict: Dict = None
    low_level_action_list: List = None
    model: BaseModelArgs = None
    prompt_mode: str = "chat"
    previous_actions: List = None
    use_dom: bool = False  # or AXTree
    benchmark: str = "miniwob"
    website_name: str = None  # To use with WorkArena only

    def make_agent(self):
        return BrowserGymStepAgent(
            max_actions=self.max_actions,
            verbose=self.verbose,
            logging=self.logging,
            root_action=self.root_action,
            action_to_prompt_dict=self.action_to_prompt_dict,
            low_level_action_list=self.low_level_action_list,
            model=self.model,
            prompt_mode=self.prompt_mode,
            previous_actions=self.previous_actions,
            use_dom=self.use_dom,
            benchmark=self.benchmark,
            website_name=self.website_name
        )
    
    def prepare(self):
        pass

    def close(self):
        pass


class BrowserGymStepAgent(Agent):
    BENCHMARKS = Literal["miniwob", "webarena"]
    WEBARENA_AGENTS = {
        "gitlab": "github_agent",
        "reddit": "reddit_agent",
        "shopping": "shopping_agent",
        "shopping_admin": "shopping_admin_agent",
        "maps": "maps_agent",
    }

    def __init__(self,
                 model: BaseModelArgs,
                 max_actions: int = 10, verbose: int = 0, logging: bool = False,
                 root_action: str = None,
                 action_to_prompt_dict: Dict = None,
                 low_level_action_list: List = None,
                 prompt_mode: str = "chat",
                 previous_actions: List = None,
                 use_dom: bool = True,
                 benchmark: BENCHMARKS = "miniwob",
                 website_name: str = None
                 ):
        match benchmark:
            case "miniwob":
                from .prompts.miniwob import step_fewshot_template
                root_action = "miniwob_agent"
            case "webarena":
                from .prompts.webarena import step_fewshot_template
                root_action = self.WEBARENA_AGENTS[website_name] if website_name in self.WEBARENA_AGENTS else None

        action_to_prompt_dict = {
            k: v for k, v in step_fewshot_template.__dict__.items() if isinstance(v, dict)}

        self.model = model.make_model()
        self.use_dom = use_dom
        self.benchmark = benchmark
        self.logging = logging
        self.agent = StepAgent(
            model=self.model,
            max_actions=max_actions, verbose=verbose, logging=logging,
            root_action=root_action,
            action_to_prompt_dict=action_to_prompt_dict,
            low_level_action_list=low_level_action_list,
            prompt_mode=prompt_mode,
            previous_actions=previous_actions
        )
        super().__init__()

    def reset(self):
        self.agent.reset()
    
    def obs_preprocessor(self, obs: Dict) -> dict:
        obs = obs.copy()  # shallow copy to avoid modifying the original dict
        # augment the observation with text versions of the DOM and AXTree
        obs["dom_txt"] = flatten_dom_to_str(obs["dom_object"])
        obs["axtree_txt"] = flatten_axtree_to_str(obs["axtree_object"])
        obs["pruned_html"] = prune_html(obs["dom_txt"])
        if self.benchmark == "miniwob":
            # Apply specific processing to miniwob to match SteP prompts
            obs["pruned_html"] = self.preprocess_dom(obs["pruned_html"])
        # remove raw entries that the agent won't use, and we don't want to record
        del obs["dom_object"]
        del obs["axtree_object"]
        return obs

    def get_action(self, obs: dict) -> tuple[str, dict]:
        url = obs["url"] if "url" in obs else None
        objective: str = obs["goal"] if "goal" in obs else None
        if self.use_dom:
            observation: str = obs["pruned_html"] if "pruned_html" in obs else None
        else:
            observation: str = obs["axtree_txt"] if "axtree_txt" in obs else None
        action, reason = self.agent.predict_action(
            objective=objective, observation=observation, url=url)
        if self.logging:
            self.agent.log_step(objective=objective, url=url, observation=observation, action=action, reason=reason, status={})
            # print(f"Action: {action}\nReason: {reason}")
        return self.parse_action(action), {}

    def preprocess_dom(self, dom: str) -> str:
        """Preprocess the DOM before passing it to the model. Keep only 'id' and 'val' 
        attributes to match original SteP prompt for MiniWoB.
        """
        attrs_to_keep = ["bid", "ref", "val"]

        parsed_dom = bs4.BeautifulSoup(dom, 'html.parser')
        all_elements = parsed_dom.find_all()
        for tag in all_elements:
            if isinstance(tag, bs4.NavigableString):
                continue
            
            if "text" in tag.attrs and len(tag["text"]) > 0:
                tag["val"] = tag["text"]
            elif "value" in tag.attrs and len(tag["value"]) > 0:
                tag["val"] = tag["value"]
            elif "id" in tag.attrs:
                tag["val"] = tag["id"]
                        
            tag_attrs = tag.attrs.copy()
            for attr in tag_attrs:
                if attr not in attrs_to_keep:
                    del tag[attr]
                    
            if "ref" in tag.attrs:
                tag["id"] = tag["ref"]
                del tag["ref"]
            elif "bid" in tag.attrs:
                tag["id"] = tag["bid"]
                del tag["bid"]
        return str(parsed_dom)

    def parse_action(self, action: str) -> str:
        """Parse the action to a string from BrowserGym action space."""
        if "click" in action:
            click_match = re.search(r'click\s*\[(\d+)\]', action, re.DOTALL)
            bid = click_match.group(1) if click_match else None
            return f"click(\"{bid}\")"

        if "type" in action:
            type_match = re.search(
                r'type\s*\[(\d+)\]\s*\[(.*?)\](\s*\[(0|1)\])?', action, re.DOTALL)
            bid = type_match.group(1) if type_match else None
            text = type_match.group(2) if type_match else None
            has_enter_option = type_match.group(3) if type_match else None
            press_enter = type_match.group(4) if has_enter_option else None
            # TODO: need to handle "press_enter" option: returns 2 actions instead of one
            return f"fill('{bid}', '{text}')"

        if "scroll" in action:
            scroll_match = re.search(r'scroll\s*\[(.*?)\]', action, re.DOTALL)
            direction = scroll_match.group(1) if scroll_match else None
            # TODO: Better handling of scroll
            if direction == "up":
                dy = -5
                return f"scroll('{dy}')"
            elif direction == "down":
                dy = 5
                return f"scroll('{dy}')"

        if "goto" in action:
            goto_match = re.search(r'goto\s*\[(.*?)\]', action, re.DOTALL)
            url = goto_match.group(1) if goto_match else None
            return f"goto('{url}')"

        if "hover" in action:
            hover_match = re.search(r'hover\s*\[(\d+)\]', action, re.DOTALL)
            bid = hover_match.group(1) if hover_match else None
            return f"hover('{bid}')"

        if "go_back" in action:
            return "go_back()"

        if "note" in action:
            note_match = re.search(r'note\s*\[(.*?)\]', action, re.DOTALL)
            note = note_match.group(1) if note_match else None
            # Save note to previous actions history
            self.agent.update_history(action=note, reason=None)
            return "noop()"
