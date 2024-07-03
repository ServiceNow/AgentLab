from browsergym.experiments.loop import ExpArgs, EnvArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
from agentlab.llm.chat_api import APIChatModelArgs
from agentlab.experiments.exp_utils import RESULTS_DIR

start_url = "https://www.google.com"

exp_args = ExpArgs(
    agent_args=GenericAgentArgs(
        agent_name="DemoAgent",
        chat_model_args=APIChatModelArgs(
            model_name="openai/gpt-4o-2024-05-13",
            # model_name="openai/gpt-4-vision-preview",
            max_total_tokens=128_000,
            max_input_tokens=124_000,  # YOLO
            max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
            vision_support=True,
        ),
        flags=GenericPromptFlags(
            obs=ObsFlags(
                use_html=False,
                use_ax_tree=True,
                use_focused_element=True,
                use_error_logs=True,
                use_history=True,
                use_past_error_logs=True,
                use_action_history=True,
                use_think_history=True,
                use_diff=False,
                html_type="pruned_html",
                use_screenshot=True,
                use_som=False,
                extract_visible_tag=True,
                extract_clickable_tag=True,
                extract_coords="False",
                filter_visible_elements_only=False,  # True
            ),
            action=ActionFlags(
                action_set="bid+nav",
                multi_actions=False,
                demo_mode="default",
                is_strict=False,
            ),
            use_plan=False,
            use_criticise=False,
            use_thinking=True,
            use_memory=False,
            use_concrete_example=True,
            use_abstract_example=True,
            use_hints=False,  # True?
            enable_chat=True,
            max_prompt_tokens=None,
            be_cautious=True,
            extra_instructions="When you are asked for directions, use Google Maps. If Google Maps cannot locate something, use Google Search to find the location.",
        ),
    ),
    env_args=EnvArgs(
        max_steps=100,
        task_seed=None,
        task_name="openended",
        task_kwargs={
            "start_url": start_url,
            "goal": "Give me directions to French Tech Toronto 2024 from Marriott Eaton Center",
        },
        headless=False,
        record_video=True,
        wait_for_user_message=True,
        # aiming for a 1920*1200 combined video (chat+viewport)
        viewport={
            # "width": 1920 - 500,  # room for chat
            # "height": 1200 + 94,  # extra space for navigation bar
            "width": 1638 - 500,  # room for chat
            "height": 1024 + 94,  # extra space for navigation bar
        },
        slow_mo=500,
    ),
    enable_debug=True,
)

exp_args.prepare(RESULTS_DIR / "directions_to_frenchtech")
exp_args.run()
