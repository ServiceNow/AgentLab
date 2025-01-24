from dataclasses import dataclass

from bgym import ExpResult, StepInfo

CHANGE_SUMMARIZER_PROMPT = """
You are a specialized 'change summarizer' model. At a given step in the agent's interaction with the website, 
you will receive the following pieces of information:

1. The user's MAIN GOAL (e.g., "Open a GitLab issue with label 'help wanted'").
2. The AGENT'S PREVIOUS OBSERVATION (HTML or AX Tree snippet) or a 'DIFF' that shows what changed since the last step, and the corresponding change summaries.
3. The AGENT'S CURRENT OBSERVATION (HTML or AX Tree snippet).
4. The ACTION the agent just took (e.g., "Clicked the button labeled 'Show report'").
5. (Optionally) The agent's CHAIN OF THOUGHT or short planning notes for this single step, if available.

YOUR TASK (each step):
A) SUMMARIZE THE CHANGE
   - Describe what visibly changed between the previous observation (or diff) and the current observation. 
     For example, did a new panel open, did the form reset, did nothing happen, etc.?

B) ASSESS THE ACTION
   - Decide whether the agent's action seems helpful or correct given the user's main goal, 
     or if it appears incorrect/unhelpful. 
   - Briefly explain why.

OUTPUT FORMAT (per step):
Return your analysis as a JSON-like structure, for example:

{
  "changeSummary": "A new search results panel appeared on the right side.",
  "actionAssessment": "Correct",
  "explanation": "Clicking 'Search' was appropriate to display the results."
}

Or for an incorrect action:

{
  "changeSummary": "The page reloaded but the date fields were reset to defaults.",
  "actionAssessment": "Incorrect",
  "explanation": "The agent should have fixed the date format first instead of re-clicking 'Show report'.",
  "suggestion": "Correct the date format or check for error messages."
}

Please follow this structure at every step. Keep your responses concise and clear. Below are the details.

Goal: {goal}

LLM Plan: {plan}

Previous Observation: {past_observation}

Current Observation: {current_observation}

Past summaries: {past_summaries}

Action: {action}
"""

ERROR_CLASSIFICATION_PROMPT = """
You are an expert evaluator that classifies web agent failures according to a predefined taxonomy. 
Below are the high-level definitions of each top-level category (Agent Errors, Language Model Errors, and Benchmark/Environment Errors),
followed by an explanation of the inputs you will receive (planning history, chain of thought, etc.), 
a set of labeled examples for reference (few-shot), and finally the classification task you must complete.

--------------------------------------------------------------------------------
TAXONOMY DEFINITIONS
--------------------------------------------------------------------------------

1. AGENT ERRORS
These errors arise when agents interact with web interfaces and fail due to limitations in perception, navigation, or manipulation.

   - Navigation & Planning Errors
     The agent cannot construct or execute a correct sequence of actions to reach its goal 
     (e.g., getting lost on a website, failing to recover from missteps, or using incorrect search terms).

   - Interaction Execution Errors
     The agent enters data in the wrong format, forgets to click "Submit" after typing, 
     repeats the same failing action without adaptation, or loses track of the changing webpage state.

   - Information Processing Errors
     The agent misreads or misinterprets visible data (e.g., extracting the wrong field values), 
     misconstrues relationships between pieces of information, or fails to validate data against task requirements.

   - Observation & Action Errors
     The agent fails to observe important updates in the environment (e.g., not noticing the page reloaded)
     or misaligns its actions (clicks the wrong element or stale link).

2. LANGUAGE MODEL ERRORS
These errors result from the model's inability to correctly interpret or reason about the task at a higher level, 
independent of the low-level web interactions.

   - Task Understanding Errors
     The agent misreads or misunderstands the user's objective (goal interpretation), 
     loses crucial context (context loss), or performs actions beyond or short of the intended scope.

   - Reasoning Failures
     The agent's logic is flawed (logical inference errors), behaves inconsistently across multiple steps, 
     or fails to prioritize important subtasks when handling complex goals.

3. BENCHMARK & ENVIRONMENT ERRORS
These errors are external to the agent's logic and the language model's reasoning, 
arising from flaws in the system, network, or evaluation framework itself.

   - System Errors
     Network failures, API downtime, or dynamic web changes that break the agent's assumptions (e.g., layout shifts).

   - Benchmark Design Errors
     Ambiguous or contradictory task specifications, incorrect validation criteria (where correct solutions are flagged as failures), 
     or inflexible evaluation systems that fail to account for valid alternative solutions.

--------------------------------------------------------------------------------
INPUT DESCRIPTION
--------------------------------------------------------------------------------

You will receive the following for each scenario:
1. User Goal
   - The original objective provided by the user (e.g., "Open a GitLab issue labeled 'help wanted'").
   
2. Planning / Thought History
   - The internal reasoning or plan the agent considered. May include branches of logic or key decision points.

3. Current Observation (HTML / AX Tree Snippet)
   - The webpage structure or state that the agent sees at a given point in time.

4. Historical change summaries
   - A list of summaries of changes in the observation that the agent has seen during the course of actions.

5. Action History
   - A record of the agent's step-by-step actions in the web environment (clicks, form entries, navigations, etc.) 
     along with immediate outcomes or errors.

Using these inputs, you must categorize the observed failure (or success) under the appropriate category or categories.

--------------------------------------------------------------------------------
FEW-SHOT CLASSIFICATION EXAMPLES
--------------------------------------------------------------------------------

1) EXAMPLE A (Benchmark Error - Benchmark Design Error)
   • Context: The agent correctly finds a cheaper product meeting the user's criteria, 
     but the benchmark expects a more expensive product and marks the solution as wrong.
   • Classification: ["Benchmark Design Error"]
   • Justification: The agent's solution is objectively valid, but the evaluation framework is too rigid 
     and does not allow an alternative correct solution.

2) EXAMPLE B (Agent Error - Interaction Execution)
   • Context: The agent repeatedly clicks "Show report" after entering dates in the wrong format. 
     Each time, the site resets to default dates. The agent never notices and keeps doing the same thing.
   • Classification: ["Agent Error - Interaction Execution"]
   • Justification: The agent used an invalid input format ("Format Errors"), then repeated the failing action 
     without adaptation ("Action Repetition").

3) EXAMPLE C (Benchmark Error - Benchmark Design Error)
   • Context: The user asks, "Where is the nearest In-N-Out to Upitts?" 
     The query is ambiguous because "Upitts" is not a standard location. 
     The agent flounders, eventually returning "No In-N-Out found," which is incorrect for the region.
   • Classification: ["Benchmark Design Error"]
   • Justification: The task goal is poorly specified ("Upitts" is ambiguous or unrealistic), 
     leading the agent astray due to unclear context.

4) EXAMPLE D (Language Model Error - Task Understanding)
   • Context: The user says, "In the repository myorg/myrepo, locate any issues labeled 'help wanted' 
     that are older than 30 days and add a comment saying 'I can help fix this.'" 
     The agent's planning notes mention searching for existing issues but quickly pivot to creating a brand-new issue 
     with label 'help wanted,' ignoring the user's actual request to find and comment on old issues.
   • Classification: ["Language Model Error - Task Understanding"]
   • Justification: The agent misunderstood the user's goal. Instead of searching for and commenting on existing issues, 
     it focused on creating a new issue. This is a misinterpretation of the instructions, 
     not a mechanical error in clicking or input format.

--------------------------------------------------------------------------------
CLASSIFICATION TASK
--------------------------------------------------------------------------------

1. Read through:
   - The planning and thought history
   - The action history
   - The current HTML or AX Tree observation
   - The user goal

2. Decide if the failure is:
   - An Agent Error (which subcategory/subcategories),
   - A Language Model Error (which subcategory/subcategories),
   - A Benchmark/Environment Error (which subcategory/subcategories),
   - Or a combination thereof (multi-label if needed).

3. Provide a brief explanation justifying your classification, referencing specific steps if helpful.

4. If the agent succeeds (no error), label the errorCategory accordingly as "Success".

Output Format Example:
{
  "errorCategory": ["Agent Error - Navigation & Planning"],
  "explanation": "The agent opened the wrong GitLab page and never recovered..."
}

Please follow this structure at every step. Keep your responses concise and clear. Below are the details.

Overall goal: {goal}

LLM Plan and thought history: {plan}

Current Observation: {current_observation}

Historical change summaries: {historical_summaries}

Action history: {action_history}
"""


def _diff(past_obs, current_obs):
    """TODO: Implement the diff function.

    Returns a diff version of current_obs compares to past_obs, unless there is too many changes.
    """
    raise ValueError("Not implemented yet.")


@dataclass
class ChangeSummarizer:

    llm: callable  # language model
    obs_formatter: callable
    use_diff: bool = False

    def summarize(
        self, past_obs: dict, action: str, current_obs: dict, past_summaries: list[str]
    ) -> str:
        """Produces, a summary of the effect of an action."""
        past_obs_message = self.obs_formatter(past_obs)
        current_obs_message = self.obs_formatter(current_obs)

        goal = past_obs["goal"]  # Use goal object from agentlab
        # Outsource everything to formatter
        plan = past_obs["plan"]
        if self.use_diff:
            current_obs_message = _diff(past_obs_message, current_obs_message)

        return self.llm(
            self.make_prompt(
                past_obs_message, action, current_obs_message, past_summaries, goal, plan
            )
        )

    def make_prompt(
        self, past_obs_message, action, current_obs_message, past_summaries, goal, plan
    ):
        """TODO: Implement the prompt."""
        return CHANGE_SUMMARIZER_PROMPT.format(
            goal=goal,
            plan=plan,
            past_observation=past_obs_message,
            current_observation=current_obs_message,
            past_summaries=past_summaries,
            action=action,
        )


@dataclass
class EpisodeAnalysis:
    analysis: str  # complete analysis of the episode
    summary: str  # short summary of the analysis
    categories: dict[str, float]  # score for each category e.g. type of error or difficulty levels


@dataclass
class EpisodeSummarizer:

    change_summarizer: ChangeSummarizer = None

    def summarize(exp_results: list[ExpResult], change_summaries: list[str]) -> EpisodeAnalysis:
        """Run Change Summarizer for every step in the episode or extract a pre-computed one."""
        pass


@dataclass
class EpisodeErrorSummarizer(EpisodeSummarizer):

    change_summarizer: ChangeSummarizer = None

    def make_prompt(self, current_observation, action_history, historical_summaries, goal, plan):
        """TODO: Implement the prompt."""
        return ERROR_CLASSIFICATION_PROMPT.format(
            goal=goal,
            plan=plan,
            current_observation=current_observation,
            historical_summaries=historical_summaries,
            action_history=action_history,
        )
