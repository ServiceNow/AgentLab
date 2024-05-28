from pathlib import Path
import pandas as pd
from agentlab.analyze import inspect_results
from agentlab.experiments.task_collections import webgum_tasks
from browsergym.experiments.loop import ExpResult, yield_all_exp_results

from browsergym.core.action.parsers import highlevel_action_parser
from tqdm import tqdm
from joblib import Memory


def select_single_agent(result_df: pd.DataFrame, agent_index) -> pd.DataFrame:
    """
    Selects the rows of the dataframe that correspond to a single agent
    """
    new_df = result_df.reset_index(level="task_name", inplace=False).sort_index()
    agent_result_df = new_df.loc[agent_index]

    inspect_results.set_index_from_variables(agent_result_df)
    return agent_result_df


MODEL_NAME_MAP = {
    "openai/gpt-4-1106-preview": "gpt-4",
    "openai/gpt-4-vision-preview": "gpt-4-v",
    "openai/gpt-3.5-turbo-1106": "gpt-3.5",
}

TASK_CATEGORY_MAP = {
    "workarena.servicenow.all-menu": "menu",
    "workarena.servicenow.create-change-request": "form",
    "workarena.servicenow.create-hardware-asset": "form",
    "workarena.servicenow.create-incident": "form",
    "workarena.servicenow.create-problem": "form",
    "workarena.servicenow.create-user": "form",
    "workarena.servicenow.filter-asset-list": "list-filter",
    "workarena.servicenow.filter-change-request-list": "list-filter",
    "workarena.servicenow.filter-hardware-list": "list-filter",
    "workarena.servicenow.filter-incident-list": "list-filter",
    "workarena.servicenow.filter-service-catalog-item-list": "list-filter",
    "workarena.servicenow.filter-user-list": "list-filter",
    "workarena.servicenow.impersonation": "menu",
    "workarena.servicenow.knowledge-base-search": "knowledge",
    "workarena.servicenow.order-apple-mac-book-pro15": "service catalog",
    "workarena.servicenow.order-apple-watch": "service catalog",
    "workarena.servicenow.order-developer-laptop": "service catalog",
    "workarena.servicenow.order-development-laptop-p-c": "service catalog",
    "workarena.servicenow.order-ipad-mini": "service catalog",
    "workarena.servicenow.order-ipad-pro": "service catalog",
    "workarena.servicenow.order-loaner-laptop": "service catalog",
    "workarena.servicenow.order-sales-laptop": "service catalog",
    "workarena.servicenow.order-standard-laptop": "service catalog",
    "workarena.servicenow.sort-asset-list": "list-sort",
    "workarena.servicenow.sort-change-request-list": "list-sort",
    "workarena.servicenow.sort-hardware-list": "list-sort",
    "workarena.servicenow.sort-incident-list": "list-sort",
    "workarena.servicenow.sort-service-catalog-item-list": "list-sort",
    "workarena.servicenow.sort-user-list": "list-sort",
}


def make_joint_ablation_study(result_dict):
    """Generate an ablation report for all models."""
    col_dict = {}
    for model_name, result_df in result_dict.items():
        report = inspect_results.ablation_report(result_df)
        short_model_name = MODEL_NAME_MAP.get(model_name, model_name)
        col_dict[short_model_name] = 100 * report["avg_reward"]
        col_dict[f"±{short_model_name}"] = 100 * report["uncertainty_reward"]

    return pd.DataFrame(col_dict)


# def set_task_category_as_index(result_df, TASK_CATEGORY_MAP):
#     """Create task_category index from task_name if needed and re-assign index
#     from variables using task_category."""
#     # rested index task_name (level 0)
#     new_df = result_df.reset_index(inplace=False)
#     if not "task_category" in new_df.columns:
#         new_df["task_category"] = new_df["task_name"].map(TASK_CATEGORY_MAP)
#     inspect_results.set_index_from_variables(new_df, task_key="task_category")
#     return new_df


def make_joint_report(result_dict, agent_index_dict, use_category=True):
    """Select a specific agent and generate a report for all models.

    Args:
        result_dict (dict): a dictionary of dataframes for each benchmark
        agent_index_dict (dict): a dictionary of agent index. If a single index
            is used, it will be used for all benchmarks
        use_category (bool): if True, use the task category as index. Otherwise,
            will return the report for all tasks.

    Returns:
        pd.DataFrame: a dataframe with the average reward and uncertainty for
            each model.
    """
    col_dict = {}
    for model_name, result_df in result_dict.items():
        if isinstance(agent_index_dict, dict):
            agent_index = agent_index_dict[model_name]
        else:
            agent_index = agent_index_dict
        agent_result_df = select_single_agent(result_df, agent_index)
        if use_category:
            agent_result_df = set_task_category_as_index(agent_result_df)
        report = inspect_results.global_report(agent_result_df, rename_index=None)
        short_model_name = MODEL_NAME_MAP.get(model_name, model_name)
        col_dict[short_model_name] = 100 * report["avg_reward"]
        col_dict[f"±{short_model_name}"] = 100 * report["uncertainty_reward"]

    return pd.DataFrame(col_dict)


def add_web_gum_subset(result_df):
    """Add the webgum subset to the result_df"""
    webgum_df = result_df[result_df.index.get_level_values("task_name").isin(webgum_tasks)].copy()

    webgum_df["task_category"] = "webgum"
    result_df["task_category"] = "all"
    return pd.concat([result_df, webgum_df])


def paper_stats(sub_df, quantile=0.95):
    """Extract stats to generate plot of the paper"""
    record = {
        "max DOM tokens": sub_df["stats.max_token_dom_txt"].quantile(quantile),
        "max Pruned DOM tokens": sub_df["stats.max_token_pruned_html"].quantile(quantile),
        "max AXTree tokens": sub_df["stats.max_token_axtree_txt"].quantile(quantile),
        "episode DOM tokens": sub_df["stats.cum_token_dom_txt"].mean(),
        "episode Pruned DOM tokens": sub_df["stats.cum_token_pruned_html"].mean(),
        "episode AXTree tokens": sub_df["stats.cum_token_axtree_txt"].mean(),
    }

    return pd.Series(record)


def step_action_count(action_str: str):
    """Count the number of actions in a step from an action string as parsed by
    highlevel_action_parser in browsergym."""
    function_calls = sum(highlevel_action_parser.search_string(action_str))
    if isinstance(function_calls, int):
        return function_calls
    else:
        return len(function_calls)


def episode_action_count(exp_result: ExpResult):
    """Count the number of actions in an episode, including multiple actions in
    one step."""
    episode = exp_result.steps_info
    return sum([step_action_count(step_info.action) for step_info in episode])


cache_dir = str((Path.home() / ".agentlab_cache").mkdir(exist_ok=True))
memory = Memory(cache_dir, verbose=0)


def filter_multi_action_and_sucess(exp_result: ExpResult):
    """Only keep experiments that have multi_actions and are successful."""
    info = exp_result.get_exp_record()
    try:
        success = info["cum_reward"] > 0
        return success and info["agent_args.flags.multi_actions"]
    except KeyError:
        return False


@memory.cache
def get_all_action_count(exp_dir: str | Path, filter_func=filter_multi_action_and_sucess):
    """Extract the number of actions for each episode for all experiments in a
    directory.

    Args:
        exp_dir (str | Path): Recursively search experiments from this directory.
        filter_func (function): A callable returning False if the experiment
            should be skipped.

    Returns:
        pd.DataFrame: as defined by ExpResults.get_summary, but with an added
        column n_actions.
    """
    all_results = list(yield_all_exp_results(exp_dir, use_cache=False))

    info = []
    for exp_result in tqdm(all_results):
        if not filter_func(exp_result):
            continue
        n_actions = episode_action_count(exp_result)
        summary = exp_result.get_exp_record()

        summary["n_actions"] = n_actions
        info.append(summary)
    return pd.DataFrame(info)


###########
# These are prompt to help generating the latex tables for the paper. Just
# update the results in the prompt and ask GPT to generate the latex table.

_prompt_for_main_table = """

Here is my current table:

---------------
% Define a command for the smaller, gray-scale text
\newcommand{\gpm}[1]{\textcolor{gray}{\small$\pm#1$}}

% New column types for left and right alignment
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

\begin{table}[t] % Use table for single column
\caption{Success rate\gpm{Standard Error} of all agents on MiniWoB++ and WorkArena. Bolded numbers represent the average success rate over the entire corresponding benchmark. \maxime{I think we are missing an info here: how many episodes/instances per category? How many episodes/instances total?}}
\noindent\resizebox{\columnwidth}{!}{ % Adjust to column width
\begin{tabular}{l r@{\hspace{2pt}}l r@{\hspace{2pt}}l r@{\hspace{2pt}}l}
\toprule
\textbf{Task Category} & \multicolumn{2}{c}{\textbf{GPT-4}} & \multicolumn{2}{c}{\textbf{GPT-3.5}} & \multicolumn{2}{c}{\textbf{CodeLlama}} \\
 & \textbf{Suc \%} & \gpm{SE} & \textbf{Suc \%} & \gpm{SE} & \textbf{Suc \%} & \gpm{SE} \\
\midrule
\textbf{WorkArena} & \textbf{51.0} & \gpm{1.9} & \textbf{14.5} & \gpm{1.5} & \textbf{0} & \gpm{0} \\
\quad Form            & 50.0 & \gpm{5.0} & 2.0  & \gpm{2.6} & 0 & \gpm{0} \\
\quad Knowledge       & 50.0 & \gpm{15.2} & 0.0 & \gpm{4.1} & 0 & \gpm{0} \\
\quad List-filter     & 0.0  & \gpm{1.6} & 0.0  & \gpm{1.6} & 0 & \gpm{0} \\
\quad List-sort       & 53.3 & \gpm{5.7} & 35.0 & \gpm{5.7} & 0 & \gpm{0} \\
\quad Menu            & 85.0 & \gpm{7.9} & 30.0 & \gpm{8.8} & 0 & \gpm{0} \\
\quad Service catalog & 76.7 & \gpm{3.8} & 15.6 & \gpm{2.5} & 0 & \gpm{0} \\
\midrule
\textbf{MiniWoB} {\tiny(125 tasks)}     & \textbf{71.7} & \gpm{1.0} & \textbf{43.6} & \gpm{0.9} & \textbf{25.5} & \gpm{1.4} \\
\quad WebGum Subset {\tiny(56 tasks)}   & 87.6 & \gpm{1.0} & 59.8 & \gpm{1.5} & 32.4 & \gpm{2.1} \\
\bottomrule
\end{tabular}
}
\label{tab:acc-summary}
\end{table}
-----------

make sure to keep the 0 of CodeLlama.
I need to update with new results:

MiniWob:

gpt-4    ±gpt-4    gpt-3.5    ±gpt-3.5    CodeLlama    ±CodeLlama
task_category
all    71.70    1.00    43.60    1.00    25.50    1.30
webgum    87.60    1.20    59.80    1.70    32.40    2.10

Work Arena Results:

    gpt-4    ±gpt-4    gpt-3.5    ±gpt-3.5
task_category
form    58.00    4.80    16.00    4.00
knowledge    50.00    14.80    0.00    4.30
list-filter    0.00    1.70    0.00    2.00
list-sort    58.30    5.50    38.30    6.10
menu    95.00    4.80    25.00    8.70
service catalog    78.90    3.70    20.00    3.30
[ALL TASKS]    54.80    2.10    18.60    2.20

--------

"""


_prompt_for_ablation_table = """
Getting inspiration from this table,


% Define a command for the smaller, gray-scale text
\newcommand{\gpm}[1]{\textcolor{gray}{\small$\pm#1$}}

\begin{table}[ht]
\caption{MiniWoB++\drouin{todo}}
\noindent\resizebox{\columnwidth}{!}{
\begin{tabular}{l r@{\hspace{2pt}}l r@{\hspace{2pt}}l}
\toprule
\multicolumn{1}{l}{\textbf{Configuration}} & \multicolumn{2}{c}{\textbf{GPT-4}} & \multicolumn{2}{c}{\textbf{GPT-3.5}} \\
 & \textbf{SR \%} & \gpm{SE} & \textbf{SR \%} & \gpm{SE} \\
\midrule
Initial Configuration & 65.1 & \gpm{0.9} & 29.7 & \gpm{0.9} \\
$\hookrightarrow$ use\_error\_logs=True & 66.6 & \gpm{1.0} & 32.2 & \gpm{1.0} \\
$\hookrightarrow$ use\_ax\_tree=True & 66.1 & \gpm{1.0} & 38.2 & \gpm{1.0} \\
$\hookrightarrow$ multi\_actions=True & 67.9 & \gpm{1.0} & 42.0 & \gpm{1.1} \\
$\hookrightarrow$ extract\_coords=center & 70.4 & \gpm{0.8} & 43.6 & \gpm{1.0} \\
$\hookrightarrow$ action\_space=bid+coord & 68.4 & \gpm{1.0} & 41.6 & \gpm{1.1} \\
$\hookrightarrow$ extract\_coords=box & 71.7 & \gpm{1.0} & 39.1 & \gpm{1.1} \\
$\hookrightarrow$ extract\_visible\_tag=True & 66.9 & \gpm{1.1} & 39.8 & \gpm{1.1} \\
\bottomrule
\end{tabular}
}
\label{tab:bgym-ablation-mw}
\end{table}
-------------

Format these results in latex:

gpt-4    ±gpt-4    gpt-3.5    ±gpt-3.5
change
Initial Configuration    65.10    0.90    29.70    1.00
↳ use_error_logs=True    66.60    0.90    32.20    1.00
↳ use_ax_tree=True    66.10    0.90    38.20    1.00
↳ multi_actions=True    67.90    0.80    42.00    1.00
↳ extract_coords=center    70.40    0.90    43.60    1.10
↳ action_space=bid+coord    68.40    1.00    41.60    1.20
↳ extract_coords=box    71.70    1.00    39.10    1.10
↳ extract_visible_tag=True    66.90    1.10    39.80    1.00
"""
