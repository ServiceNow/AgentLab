from typing import List


class Agent:
    def __init__(
        self,
        max_actions,
        verbose=0,
        logging=False,
        previous_actions: List[str] = None,
        previous_reasons: List[str] = None,
        previous_responses: List[str] = None,
    ):
        self.previous_actions = [] if previous_actions is None else previous_actions 
        self.previous_reasons = [] if previous_reasons is None else previous_reasons
        self.previous_responses = [] if previous_responses is None else previous_responses
        self.max_actions = max_actions
        self.verbose = verbose
        self.logging = logging
        self.trajectory = []
        self.data_to_log = {}

    def reset(self):
        self.previous_actions = []
        self.previous_reasons = []
        self.previous_responses = []
        self.trajectory = []
        self.data_to_log = {}

    def get_trajectory(self) -> list[str]:
        return self.trajectory
    
    def update_history(self, action: str, reason: str):
        if action:
            self.previous_actions += [action]
        if reason:
            self.previous_reasons += [reason]    

    def predict_action(self, objective: str, observation: str, url: str=None):
        pass

    def receive_response(self, response: str):
        self.previous_responses += [response]

    def log_step(self, objective: str, url: str, observation: str, action: str, reason: str, status: str):
        self.data_to_log['objective'] = objective
        self.data_to_log['url'] = url
        self.data_to_log['observation'] = observation
        self.data_to_log['previous_actions'] = self.previous_actions[:-1]
        self.data_to_log['previous_responses'] = self.previous_responses[:-1]
        self.data_to_log['previous_reasons'] = self.previous_reasons[:-1]
        self.data_to_log['action'] = action
        self.data_to_log['reason'] = reason
        for (k, v) in status.items():
            self.data_to_log[k] = v
        self.trajectory.append(self.data_to_log)
        self.data_to_log = {}