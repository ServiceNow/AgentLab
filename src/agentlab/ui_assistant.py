from agentlab.agents.dynamic_prompting import Flags
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from browsergym.experiments.loop import ExpArgs
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.llm.chat_api import ChatModelArgs

start_url = "https://www.google.com"
# start_url = "https://en.wikipedia.org/wiki/Main_Page"
# start_url =
# "https://surf.service-now.com/tcp?sysparm_domain_restore=false&sysparm_stack=no"
# start_url = "https://example.com/"
# start_url = "https://docs.google.com/document/d/1WmwtI_OnKNL8CuQyegNO6xGfjYAMSWowLoEwGpuRSHs/edit"
# start_url = "https://miro.com/online-whiteboard/"

exp_args = ExpArgs(
    agent_args=GenericAgentArgs(
        chat_model_args=ChatModelArgs(
            model_name="openai/gpt-4-vision-preview",
            max_total_tokens=128_000,
            max_input_tokens=40_000,  # make sure we don't bust budget
            max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
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
            multi_actions=True,
            use_hints=True,
            use_abstract_example=True,
            use_concrete_example=True,
            use_screenshot=True,
            enable_chat=True,
            demo_mode="default",
            action_space="bid+coord+nav",
            extract_coords="center",
        ),
    ),
    max_steps=100,
    task_seed=None,
    task_name="openended",
    task_kwargs={
        "start_url": start_url,
    },
    enable_debug=True,
    headless=False,
    record_video=True,
    wait_for_user_message=True,
    viewport={"width": 1500, "height": 1280},
    slow_mo=1000,
)

exp_args.prepare(RESULTS_DIR / "ui_assistant_logs")
exp_args.run()
