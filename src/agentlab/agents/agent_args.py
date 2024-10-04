from bgym import AbstractAgentArgs


class AgentArgs(AbstractAgentArgs):

    def set_benchmark(self, benchmark: str, demo_mode: bool):
        """Optional method to set benchmark specific flags.

        Args:
            benchmark: str
                Name of the benchmark.
            demo_mode: bool
                If True, the agent should adapt to demo mode. E.g. it can set
                the demo_mode flag in the browsergym action space.
        """
        pass

    def set_reproducibility_mode(self):
        """Optional method to set the agent in a reproducibility mode.

        This should adjust the agent configuration to make it as deterministic
        as possible e.g. setting the temperature of the model to 0.
        """
        pass
