from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.core import Action, LLMOutputParsingFailureAction, Observation, SetNextNode, StopStep, Tape, Thought
from tapeagents.dialog_tape import DialogContext

################### Base Step Classes ###################
class WorkArenaThought(Thought):
    pass


class WorkArenaAction(Action):
    pass


class WorkArenaObservation(Observation):
    pass


################### Steps ###################


class WorkArenaTask(WorkArenaObservation):
    kind: Literal["task"] = "task"
    task: str


class PageObservation(WorkArenaObservation):
    kind: Literal["page_observation"] = "page_observation"
    text: str
    current_page: int
    total_pages: int
    last_action_error: str = ""


class ReasoningThought(WorkArenaThought):
    """
    Thoughts produced by the agent during the reasoning process.
    """

    kind: Literal["reasoning_thought"] = "reasoning_thought"
    reasoning: str = Field(description="chain of thoughts")


class ReflectionThought(WorkArenaThought):
    """
    Review the current state of the page and previous steps to find the best possible next action to accomplish the task:
    1. Produce reasoning thougt explaining the observed state, think about which blocks could be relevant to the given task and its current state, note relevant BIDs.
    2. Produce list of things to do to accomplish the task. Mention all the fields that need to be filled, buttons that need to be clicked, etc.
    3. Reflect if you last action succes, check if it produced the expected effect on the page.
    4. Reflect on you last actions to check if you are being stuck with repeating the same action. If you repeatedly failed to perform the same action with the page element, propose a new way of interacting.
    5. Describe the next action to be performed and expected effect on the page.
    """

    kind: Literal["reflection_thought"] = "reflection_thought"
    page_state: str = Field(
        description="description of the current page state, think about which blocks could be relevant to the given task and its current state, note relevant BIDs"
    )
    last_action_kind: str = Field(description="describe the kind of the last action")
    last_action_achieved_effect: str = Field(
        description="describe the the changes in the page state achieved by the last action that you can observe"
    )
    last_action_expected_effect: str = Field(
        description="describe the expected changes in the page state that should be achieved by the last action"
    )
    last_action_success: bool = Field(
        description="success of the last action, True if it produced the expected effect on the page"
    )
    last_action_repetitions: str = Field(
        description="check if you are being stuck with repeating the same action and describe here"
    )
    are_we_stuck: bool = Field(description="if you are being stuck with repeating the same action")
    task_solved: bool = Field(description="if the main task given by used is solved")
    todo_list: list[str] = Field(
        description="detailed list of things to do to accomplish the task, down to the fields level"
    )
    next_action: str = Field(description="describe the next action to be performed and expected effect on the page")
    page_line: str = Field(
        description="verbatim quote of the line of the page that contains required element. Empty if next action is not related to specific page element"
    )


class ClickAction(WorkArenaAction):
    """
    Action that clicks the element on the page with the provided BID
    """

    kind: Literal["click_action"] = "click_action"
    bid: str = Field(description="BID of the element to click")
    button: Literal["left", "middle", "right"] = Field(description="button to click", default="left")
    modifiers: list[Literal["Alt", "Control", "Meta", "Shift"]] = Field(
        description="modifier keys to press", default_factory=list
    )


class SelectOptionAction(WorkArenaAction):
    """
    Action that selects option in the dropdown or combobox element with the provided BID.
    ONLY applicable to dropdowns and comboboxes!
    """

    kind: Literal["select_option_action"] = "select_option_action"
    bid: str = Field(description="BID of the dropdown or combobox to select from")
    element_description: str = Field(description="brief description of the dropdown or combobox")
    option: str = Field(description="option to select")


class HoverAction(WorkArenaAction):
    """
    Action that hovers over the element on the page with the provided BID
    """

    kind: Literal["hover_action"] = "hover_action"
    bid: str = Field(description="BID of the element to hover")


class InputTextAction(WorkArenaAction):
    """
    Action that fills out the input element identified by BID with the provided text
    """

    kind: Literal["input_text_action"] = "input_text_action"
    bid: str = Field(description="BID of the input element to fill")
    text: str = Field(description="text to put into the element")


class PressAction(WorkArenaAction):
    """
    Action that puts focus on the element with a given BID and presses a combination of keys.
    Accepts the logical key names: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc.
    Following modification, shortcuts are also supported: Shift, Control, Alt, Meta.
    """

    kind: Literal["press_action"] = "press_action"
    bid: str = Field(description="BID of the input element to focus")
    key_comb: str = Field(description="keys combination to press")


class ScrollAction(WorkArenaAction):
    """
    Action that scrolls the page in the provided direction
    """

    kind: Literal["scroll_action"] = "scroll_action"
    direction: str = Field(description="direction to scroll")


class TabFocusAction(WorkArenaAction):
    """
    Action that focuses the tab with the provided index
    """

    kind: Literal["tab_focus_action"] = "tab_focus_action"
    index: int = Field(description="index of the tab to focus")


class NewTabAction(WorkArenaAction):
    """
    Action that opens a new browser tab
    """

    kind: Literal["new_tab_action"] = "new_tab_action"


class CloseTabAction(WorkArenaAction):
    """
    Action that closes the browser tab
    """

    kind: Literal["close_tab_action"] = "close_tab_action"


class GoBackAction(WorkArenaAction):
    """
    Action that goes back in the browser history
    """

    kind: Literal["go_back_action"] = "go_back_action"


class GoForwardAction(WorkArenaAction):
    """
    Action that goes forward in the browser history
    """

    kind: Literal["go_forward_action"] = "go_forward_action"


class GotoPageAction(WorkArenaAction):
    """
    Action that opens the page with the provided URL in the current tab
    """

    kind: Literal["goto_page_action"] = "goto_page_action"
    url: str = Field(description="url to go to")


class FinalAnswerAction(WorkArenaAction, StopStep):
    """
    Action that provides the final answer to the user after completing the task.
    Should be produced when the agent has finished the task.
    When the task has question about numerical value, the answer should contain only one number!
    """

    kind: Literal["final_answer_action"] = "final_answer_action"
    text: str = Field(description="final answer to the user")

class ActionExecutionFailure(WorkArenaObservation):
    kind: Literal["action_execution_failure"] = "action_execution_failure"
    error: str

WorkArenaStep = Union[
    WorkArenaTask,
    PageObservation,
    ActionExecutionFailure,
    LLMOutputParsingFailureAction,
    # thoughts
    ReasoningThought,
    ReflectionThought,
    # browser actions
    ClickAction,
    SelectOptionAction,
    HoverAction,
    InputTextAction,
    PressAction,
    ScrollAction,
    # TabFocusAction,
    # NewTabAction,
    # CloseTabAction,
    GoBackAction,
    GoForwardAction,
    # GotoPageAction,
    FinalAnswerAction,
    SetNextNode,
]


class WorkArenaTape(Tape[DialogContext, WorkArenaStep]):
    context: DialogContext = DialogContext(tools=[])


WorkArenaBaselineStep: TypeAlias = Annotated[
    Union[
        # thoughts
        ReasoningThought,
        # browser actions
        ClickAction,
        SelectOptionAction,
        HoverAction,
        InputTextAction,
        PressAction,
        ScrollAction,
        GoBackAction,
        GoForwardAction,
        FinalAnswerAction,
    ],
    Field(discriminator="kind"),
]

WorkArenaAgentStep: TypeAlias = Annotated[
    Union[
        # thoughts
        ReasoningThought,
        ReflectionThought,
        # browser actions
        ClickAction,
        SelectOptionAction,
        HoverAction,
        InputTextAction,
        PressAction,
        ScrollAction,
        # TabFocusAction,
        # NewTabAction,
        # CloseTabAction,
        GoBackAction,
        GoForwardAction,
        # GotoPageAction,
        FinalAnswerAction,
    ],
    Field(discriminator="kind"),
]
