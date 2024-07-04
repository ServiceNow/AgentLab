from agentlab.agents.dynamic_prompting import Flags
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from browsergym.experiments.loop import ExpArgs
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.llm.chat_api import APIModelArgs
from agentlab.experiments import args

import json

start_url = "https://us2.concursolutions.com/home.asp"

goal = """
I finished my trip at Knowledge 24 for presenting a keynote. Can you create a report with my flight and hotel receipts on the desktop.
"""

fligt_info = {
    "Departure Date": "05/05/2024",
    "Ticket Number": "0062029101347",
    "Vendor": "Air Canada",
    "Airline Travel Service Code": "coach",
    "Amount": "1786.36",
    "Currency": "CAD",
    "Trip Type": "Knowledge",
    "Project List": "2024 Knowledge",
    "Payment Type": "Out of Pocket",
}

hotel_info = {
    "Date Range": "05/05/2024 - 05/10/2024",
    "Vendor": "The Venetian Las Vegas",
    "Amount": "955.00",
    "Currency": "USD",
    "Trip Type": "Knowledge",
    "Project List": "2024 Knowledge",
    "Payment Type": "Out of Pocket",
}

info = {"flight": fligt_info, "hotel": hotel_info}

load_expense_instructions = f"""\
Secret information to help you solve the task:
Note: dont' put any of this information in youre <think> or chain of thoughts.
e.g. don't say as requested by the user or pre user's request.
1- Load expenses into system from the desktop /Users/alexandre.lacoste/Desktop.
Files are hotel_receipt.png and flight_receipt.pdf. Don't click on the "upload
receipt", use the upload_file action.
2- Create a new report for knowledge 24. Make sure to fill all required fields.
3- add the flight and hotel expenses from available expanses
4- be as autonomous as possible and infer as much information as possible but
don't submit the report.
"""

correct_flight_instructions = f"""\
Use this information to correct the flight information in the report:
Important note: Be concise when describing your thinking process.
If information is not explicitly provided, be autonomous, clever and creative.
* Report is already created and expenses are added.
* Correct all the flight informations, and make sure there are no error lefts
* this is the information that an agent extracted from the receipts
{json.dumps(fligt_info, indent=2)}
* correct previously filled information if needed. Don't change values if they
are already correct.
* Don't use the select_option function it doesn't work in concur. Dropdown
selection box may be opened using click to help chosing the right option.
* Don't change transaction date.
* be as autonomous as possible and infer as much information as possible but
don't submit the report.
* your task is not done until all flight errors are resolved. You can ignore the
warning about Project Type.
* don't correct any information about hotel yet.
"""

correct_hotel_instructions = f"""\
Use this information to correct the flight information in the report:
Important note: Be concise when describing your thinking process.
If information is not explicitly provided, be autonomous, clever and creative.* Correct all the hotel informations, and make sure there are no error lefts
* Correct all the hotel informations, and make sure there are no error lefts
* this is the information that an agent extracted from the receipts
{json.dumps(hotel_info, indent=2)}
* correct previously filled information if needed. Don't change values if they
are already correct.
* Don't use the select_option function it doesn't work in concur. Dropdown
selection box may be opened using click to help chosing the right option.
* Don't change transaction date.
* be as autonomous as possible and infer as much information as possible but
don't submit the report.
* your task is not done until all hotel errors are resolved.
* finally save the repot but don't submit it.
"""


exp_args = ExpArgs(
    agent_args=GenericAgentArgs(
        chat_model_args=APIModelArgs(
            model_name="openai/gpt-4-vision-preview",
            max_total_tokens=128_000,
            max_input_tokens=124_000,  # YOLO
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
            multi_actions=False,
            use_hints=False,
            use_abstract_example=True,
            use_concrete_example=True,
            use_screenshot=True,
            enable_chat=True,
            demo_mode="default",
            action_space="bid",
            extract_coords="False",
            extra_instructions=correct_hotel_instructions,
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
    storage_state=RESULTS_DIR / "concur_state.json",
    slow_mo=1000,
    viewport={"width": 1500, "height": 1280},
)


def make_exp_args_list(n_seeds=4):
    exp_args.task_seed = args.CrossProd([None] * n_seeds)
    exp_args.headless = True
    exp_args.wait_for_user_message = False
    exp_args.task_kwargs["goal"] = goal
    return "concur_demo", args.expand_cross_product(exp_args)


if __name__ == "__main__":
    exp_args.prepare(RESULTS_DIR / "concur_assistant_logs")
    exp_args.run()
