SYSTEM_PROMPT = """You are an expert AI Agent, your goal is to help the user perform tasks using a web browser.
Your role is to understand user queries and respond in a helpful and accurate manner.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
You will be provided with the content of the current page and a task from the user.
Do not express your emotions or opinions about the user question."""

ALLOWED_STEPS = """
You are allowed to produce ONLY steps with the following json schemas:
{allowed_steps}
Do not reproduce schema when producing the steps, use it as a reference.
"""

hints = """
HINTS:
- You can use the BIDs of the elements to interact with them.
- To select value in the dropdown or combobox, ALWAYS use select_action.
- To click on the checkbox or radio button, ALWAYS use BID of corresponding LabelText and not the BID of the element itself.
- Press enter key to submit the search query.
- Always produce only one step at a time.
- Step kind is always lowercase and underscore separated.
"""

START = """
Produce reasoning_thought step that describes the intended solution to the task. In the reasoning lines:
- review the instructions from the user and the content of the page.
- outline the main task to be accomplished and the steps to be taken to achieve it.
- produce definiton of done, that will be checked later to verify if the task was completed.
Produce only one step!
"""

REFLECT = f"""
Review the current state of the page and previous steps to find the best possible next action to accomplish the task.
Produce reflection_thought.
{hints}
"""

ACT = f"""
Produce the next action to be performed with the current page.
If the task is already solved, produce final_answer_action and stop.
You can interact with the page elements using their BIDs as arguments for actions.
Be very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions using reasoning first. For example
you can fill a few elements of a form, but don't click submit before verifying
that everything was filled correctly.
{hints}
"""

### Baseline Prompts ###

BASELINE_SYSTEM_PROMPT = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You can interact with the page and explore, and send messages to the user. Each time you
submit an action it will be sent to the browser and you will receive a new page."""

GOAL_INSTRUCTIONS = """
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}
"""

HINTS = """\
Note:
* Some tasks may be game like and may require to interact with the mouse position
in x, y coordinates.
* Some text field might have auto completion. To see it, you have to type a few
characters and wait until next step.
* If you have to cut and paste, don't forget to select the text first.
* Coordinate inside an SVG are relative to it's top left corner.
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky,
sometimes you need to use select_option, while other times you need to use fill
or click and wait for the reaction of the page.
"""

BE_CAUTIOUS = """
Be very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions first. For example
you can fill a few elements of a form, but don't click submit before verifying
that everything was filled correctly.
"""

ABSTRACT_EXAMPLE = """
# Abstract Example

Here is an abstract version of the answer with description of the content of
each step. Only one single action can be executed after the thoughts. You can only use one action at a time.
Make sure you follow this structure, but replace the step fields content with your
answer:
[
    {
        "kind": "reasoning_thought",
        "reasoning": "<Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect that your previous action had on the current content of the page.>"
    },
    {
        "kind": "<some_of_the_allowed_action>",
        "<field_name>": "<field_content>"
    }
]
"""
CONCRETE_EXAMPLE = """
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper steps:
[
    {
        "kind": "reasoning_thought",
        "reasoning": "From previous action I tried to set the value of year to "2022",\
using select_option, but it doesn't appear to be in the form. It may be a\
dynamic dropdown, I will try using click with the bid "a324" and look at the\
response from the page."
    },
    {
        "kind": "click_action",
        "bid": "a324"
    }
]
"""
BASELINE_STEPS_PROMPT = """# Action space:
Note: This action set allows you to interact with your environment. The primary way of referring to
elements in the page is through bid which are specified in your observations.
"""

MAC_HINT = "\nNote: you are on mac so you should use Meta instead of Control for Control+C etc.\n"


class PromptRegistry:
    system_prompt = SYSTEM_PROMPT
    baseline_system_prompt = BASELINE_SYSTEM_PROMPT
    allowed_steps = ALLOWED_STEPS
    baseline_steps_prompt = BASELINE_STEPS_PROMPT
    goal_instructions = GOAL_INSTRUCTIONS
    hints = HINTS
    be_cautious = BE_CAUTIOUS
    concrete_example = CONCRETE_EXAMPLE
    abstract_example = ABSTRACT_EXAMPLE
    mac_hint = MAC_HINT
    reflect = REFLECT
    act = ACT
    start = START
