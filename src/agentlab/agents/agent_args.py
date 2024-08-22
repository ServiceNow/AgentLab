from browsergym.experiments.loop import AbstractAgentArgs


class AgentArgs(AbstractAgentArgs):

    def set_benchmark(self, benchmark: str):
        """Optional method to set benchmark specific flags."""
        pass
