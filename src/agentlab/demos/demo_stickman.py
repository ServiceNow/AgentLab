from agentlab.agents.dynamic_prompting import Flags
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from browsergym.experiments.loop import ExpArgs
from agentlab.llm.chat_api import ChatModelArgs
from agentlab.experiments.exp_utils import RESULTS_DIR

start_url = "https://miro.com/online-whiteboard/"

exp_args = ExpArgs(
    agent_args=GenericAgentArgs(
        chat_model_args=ChatModelArgs(
            model_name="openai/gpt-4-vision-preview",
            max_total_tokens=128_000,
            max_input_tokens=40_000,  # make sure we don't bust budget
            max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
            vision_support=True,
        ),
        flags=Flags(
            use_plan=False,
            use_html=False,
            use_ax_tree=True,
            use_focused_element=True,
            use_criticise=False,
            use_thinking=True,
            use_error_logs=True,
            use_memory=False,
            use_history=True,
            use_diff=False,
            use_past_error_logs=True,
            use_action_history=True,
            use_think_history=True,
            use_hints=True,
            use_abstract_example=True,
            use_concrete_example=True,
            enable_chat=True,
            demo_mode="default",
            extract_coords="box",  # "False", "center", "box"
            use_screenshot=True,
            multi_actions=True,
            action_space="bid+coord+nav",
            filter_visible_elements_only=True,
            extra_instructions="On this whiteboard you can draw between coordinates 100 and 900 on the x-axis, and 100 and 900 on the y-axis. You can draw any shape but with straight line segments only. Use mouse_drag_and_drop() to draw line segments.",
        ),
    ),
    max_steps=100,
    task_seed=None,
    task_name="openended",
    task_kwargs={
        "start_url": start_url,
        # "goal": "Give me directions to Knowledge 24 from the Courtyard Henderson hotel in Vegas"
    },
    enable_debug=True,
    headless=False,
    record_video=True,
    wait_for_user_message=True,
    # aiming for a 1920*1200 combined video (chat+viewport)
    viewport={
        "width": 1524 - 500,  # room for chat
        "height": 1024 + 94,  # extra space for navigation bar
    },
    slow_mo=500,
)

exp_args.prepare(RESULTS_DIR / "stickman")
exp_args.run()
