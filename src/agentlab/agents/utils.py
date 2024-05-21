from langchain_community.callbacks import get_openai_callback


def openai_monitored_agent(get_action_func):
    def wrapper(self, obs):
        with get_openai_callback() as openai_cb:
            action, agent_info = get_action_func(self, obs)

        stats = {
            "openai_total_cost": openai_cb.total_cost,
            "openai_total_tokens": openai_cb.total_tokens,
            "openai_completion_tokens": openai_cb.completion_tokens,
            "openai_prompt_tokens": openai_cb.prompt_tokens,
        }

        if "stats" in agent_info:
            agent_info["stats"].update(stats)
        else:
            agent_info["stats"] = stats

        return action, agent_info

    return wrapper
