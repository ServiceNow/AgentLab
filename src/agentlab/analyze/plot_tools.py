from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import trim_mean

from agentlab.utils.bootstrap import bootstrap_all_benchmarks

sns.set_style("dark", {"grid.color": "0.98", "axes.facecolor": "(0.95, 0.95, 0.97)"})


def load_benchmark_masks(csv_path: Path | str, benchmark_names=list[str]):
    """Load benchmark masks from a csv file and convert it.

    To create the csv file simply use file/download/"...csv" from google
    spreadsheet.

    Args:
        csv_path (Path | str): The path to the csv file.
        benchmark_names (list[str]): The names of the benchmarks to be loaded.

    Returns:
        pd.DataFrame: The benchmark masks.
    """
    mask_df = pd.read_csv(csv_path)  # type: pd.DataFrame

    for benchmark_name in benchmark_names:
        mask_df[benchmark_name] = mask_df[benchmark_name].apply(
            lambda x: True if str(x).strip() == "x" else False
        )

    mask_df["all"] = True
    benchmark_names = ["all"] + benchmark_names

    # remove the Average row
    mask_df = mask_df[mask_df["Task Name"] != "ZAverage"]
    mask_df = mask_df[mask_df["Task Name"].isna() == False]

    # # print the excluded tasks
    # excluded_tasks = set(mask_df["Task Name"].unique()) - set(MINIWOB_TASKNAME_TO_CLASSNAME.keys())
    # print("Excluded tasks:", excluded_tasks)

    # # filter task that are not in the MINIWOB_TASKNAME_TO_CLASSNAME
    # mask_df = mask_df[mask_df["Task Name"].isin(MINIWOB_TASKNAME_TO_CLASSNAME.keys())]

    # # convert task name to class name and keep all columns
    mask_df["task_name"] = mask_df["Task Name"].apply(lambda x: f"miniwob.{x}")
    mask_df.set_index("task_name", inplace=True)

    # extract only the benchmark columns
    mask_df = mask_df[benchmark_names]

    # rename columns
    mask_df.rename(rename_func, inplace=True, axis="columns")
    return mask_df


def add_benchmark_masks(df: pd.DataFrame, benchmark_masks: pd.DataFrame):
    """Add benchmark masks to the dataframe."""
    return df.merge(benchmark_masks, left_on="task_name", right_index=True)


def sort_agents(agents, df, metric="cum_reward"):
    agent_order = (
        df.groupby("agent_model_name")[metric].mean().sort_values(ascending=False).index.tolist()
    )
    return [agent for agent in agent_order if agent in agents]


def iqm(scores):
    """Interquantile mean."""
    return trim_mean(scores, proportiontocut=0.25, axis=None)


# add memory from joblib
memory = joblib.Memory(location=Path().home() / "cache", verbose=1)
bootstrap_all_benchmarks_cached = memory.cache(bootstrap_all_benchmarks)


def bootstrap_and_plot(
    df: pd.DataFrame,
    benchmark_names: List[str],
    metric: str = "cum_reward",
    model_axis: str = "agent_model_name",
    agent_order: List[str] | None = None,
    agent_colors=None,
    agent_markers=None,
    repeat: int = 100,
    fig_size=None,
    n_legend_rows: int = 2,
):
    """Add aggregated data as a new benchmark."""

    if agent_order is None:
        agent_order = sorted(df[model_axis].unique())
    print("bootstrapping")
    # create a new df containing bootstrapped samples of iqm

    df_bootstrap = bootstrap_all_benchmarks(
        df, metric, benchmark_cols=benchmark_names, repeat=repeat
    )
    print("plotting")

    # plot results per benchmark (aggregated results is an extra benchmark)
    plot_per_benchmark(
        df_bootstrap,
        agent_order,
        model_axis,
        benchmark_order=benchmark_names,
        agent_colors=agent_colors,
        agent_markers=agent_markers,
        metric=metric,
        fig_size=fig_size,
        n_legend_rows=n_legend_rows,
    )


def remove_violin_outline(ax):
    """Remove the outline of the violin plot."""
    for pc in ax.collections:
        pc.set_edgecolor("none")


def get_violin_centers(ax: plt.Axes) -> list[float]:
    """Estimate the center of violin patches from axes.

    This is a hacky way to get the center of the violin patches.
    It works by averaging the x coordinates of the vertices of PolyCollection

    Args:
        ax (plt.Axes): The axes containing the violin plot.

    Returns:
        list[float]: The x coordinates of the centers of the violins.
    """
    violin_centers = []
    for child in ax.get_children():
        if isinstance(child, PolyCollection):
            violin_centers.append(np.mean(child.get_paths()[0].vertices[:, 0]))

    return sorted(violin_centers)


def associate_action_markers(agent_order):
    """Associate a marker to each agent, based on action space."""
    markers = {}
    for agent in agent_order:
        marker = None
        if "high" in agent:
            marker = "^"
        elif "low" in agent:
            marker = "v"
        elif "both" in agent:
            marker = "d"
        elif "code" in agent:
            marker = "o"
        else:
            marker = "*"

        markers[agent] = marker
    return markers


def rename_func(name):
    return RENAME_DICT.get(name, name)


# for display names in the paper
RENAME_DICT = {
    "easy (auto)": "easy",
    "hard (auto)": "hard",
    "long context (auto)": "long context",
    "rci-agent task (47)": "RCI\nsubset",
    "WebGUM task (56)": "WebGUM\nsubset",
    "long context": "long context",
    "pixel": "pixel",
    "2d under standing": "2D\nunderstanding",
    "x,y actions": "x, y\nactions",
    "domain specific knowledge or exploration ?": "domain specific\nknowledge",
    "long episode": "long episode",
    "rapid interaction": "rapid interaction",
    "action space limited": "action space\nlimited",
    "requies augmented HTML and/or AXTree": "requires\naugmented HTML",
}


def add_markers(ax, df, model_axis, metric, agent_order, agent_markers, agent_colors):
    """Add markers to the scatter plot."""
    v_centers = get_violin_centers(ax)
    scatter_handles = {}

    for center_x, agent in zip(v_centers, agent_order):
        median_y = df[df[model_axis] == agent][metric].mean()

        scatter_plot = ax.scatter(
            center_x,
            median_y,
            marker=agent_markers[agent],
            color=agent_colors[agent],
            edgecolor="black",
            label=agent,
            s=60,
        )

        scatter_handles[agent] = scatter_plot

    # markers overtake the legend
    legend = ax.legend(
        handles=list(scatter_handles.values()),
        labels=list(scatter_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(agent_order),
        title="",
    )

    # bigger markers!
    for handle in legend.legendHandles:
        handle.set_sizes([100])
    return scatter_handles


def plot_per_benchmark(
    df: pd.DataFrame,
    agent_order: List[str],
    model_axis: str,
    benchmark_order: List[str] | None = None,
    metric: str = "cum_reward",
    aggregated_name: str = "all",
    sharey: bool = True,
    inner: str = None,
    fig_size: tuple = None,
    n_legend_rows: int = 1,
    agent_colors: dict[str, tuple] = None,
    agent_markers: dict[str, str] = None,
):
    """Violin plots for each benchmarks and each agents.

    Args:
        df: pd.DataFrame
            The input DataFrame.
        agent_order: List[str]
            The order of the agents in the plot.
        model_axis: str
            The column containing the agent names.
        benchmark_order: List[str], optional
            The order of the benchmarks in the plot. Defaults to alhpabetical order.
        metric: str, optional
            The column containing the metric to which the function will be applied.
        aggregated_name: str, optional
            The name of the "special" benchmark. It will be placed first and
            highlighted in pale blue
        sharey: bool, optional
            Whether to share the y axis between plots.
        inner: str, optional
            The type of inner display inside of the violins. See seaborn.violinplot
        fig_size: tuple, optional
            The size of the figure. see matplotlib.pyplot.figure
        n_legend_rows: int, optional
            The number of rows in the legend.
        agent_colors: dict[str, tuple], optional
            A dictionary mapping agent names to colors. Defaults to seaborn's colorblind palette.
        agent_markers: dict[str, str], optional
            A dictionary mapping agent names to markers. Defaults to no markers.
    """
    if benchmark_order is None:
        benchmark_order = sorted(df["benchmark"].unique())

    if fig_size is None:
        fig_width = len(benchmark_order) * 2
        fig_size = (fig_width, 3)
    fig, axes = plt.subplots(1, len(benchmark_order), sharey=sharey, figsize=fig_size)

    if agent_colors is None:
        colors = sns.color_palette("colorblind", n_colors=len(agent_order))
        agent_colors = dict(zip(agent_order, colors))

    if agent_markers is not None:
        violon_palette = {agent: (0.8, 0.8, 0.8) for agent in agent_order}
    else:
        violon_palette = agent_colors

    for benchmark, ax in zip(benchmark_order, axes):
        sub_df = df[df["benchmark"] == benchmark]

        sns.violinplot(
            x="benchmark",
            y=metric,
            hue=model_axis,
            data=sub_df,
            hue_order=agent_order,
            linewidth=0.5,
            saturation=1,
            scale="count",
            inner=inner,
            palette=violon_palette,
            ax=ax,
        )
        remove_violin_outline(ax)

        if agent_markers is not None:
            add_markers(ax, sub_df, model_axis, metric, agent_order, agent_markers, agent_colors)

        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y")

        if benchmark == aggregated_name:
            ax.set_facecolor("#cff6fc")

        ax.set(xlabel=None)

        if benchmark != benchmark_order[int((len(benchmark_order) - 1) / 2)]:
            ax.get_legend().remove()
        else:
            ncols = int(np.ceil(len(agent_order) / n_legend_rows))

            sns.move_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=ncols, title="")

        if benchmark != benchmark_order[0]:
            ax.set(ylabel=None)

    if sharey:
        fig.subplots_adjust(wspace=0.02)
    else:
        fig.subplots_adjust(wspace=0.3)


@memory.cache
def modify_names(df):
    """Generate names that are paper-friendly, and fix model_name and agent_name for the different formats."""

    def process_row(row):
        model_name = row.get("model_name", np.nan)
        agent_name = row["agent_name"]

        if not isinstance(model_name, str) or len(model_name) == 0:
            # rim's type of model extract from agent name
            model_name = "gpt-4" if "gpt4" in agent_name else "gpt-3.5"
            agent_name = agent_name.replace("_gpt4", "")
        else:
            # massimo's type of model
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
            model_name = model_name.replace("-Instruct-hf", "").replace("-beta", "")

        agent_name = agent_name.replace("GenericAgent_", "")

        row["agent_name"] = agent_name
        row["model_name"] = model_name
        row["agent_model_name"] = f"{agent_name}-{model_name}"

        return row

    return df.apply(process_row, axis=1)


def find_incomplete_exp(df: pd.DataFrame, agent_columns_to_show=None):
    agents = df["agent_model_name"].unique()
    # print(agents)
    dicts = []
    for agent in agents:
        agent_df = df[df["agent_model_name"] == agent]  # type: pd.DataFrame
        tasks = agent_df["task_name"].unique()

        n_episode = []
        for task in tasks:
            task_df = agent_df[agent_df["task_name"] == task]
            n_episode.append(len(task_df))
        # find missing cum_reward
        missing_cum_reward = agent_df["cum_reward"].isna().sum()
        # num results
        num_results = len(agent_df)

        info = {
            "agent": agent,
            "n_tasks": len(tasks),
            "max_ep_count": np.max(n_episode),
            "min_ep_count": np.min(n_episode),
            "mean_ep_count": np.mean(n_episode),
            "unique cum_reward": str(agent_df["cum_reward"].unique()),
            "missing_cum_reward": missing_cum_reward,
            "total_results": num_results,
        }

        if agent_columns_to_show is not None:
            for col in agent_columns_to_show:
                info[f"{col}-unique"] = agent_df[col].unique()[:10]

        for task in tasks:
            sub_df = agent_df[agent_df["task_name"] == task]
            if len(sub_df) > 10:
                unique_exp_dir = sub_df["exp_dir"].unique()
                print(
                    f"  {agent} {task} {len(sub_df)} n unique expseeds {len(sub_df['exp_seed'].unique())}"
                )
                for i, dir in enumerate(unique_exp_dir):
                    print(f"    Unique exp_dir {i}: {dir}")

        dicts.append(info)
    df_summary = pd.DataFrame(dicts)
    # df_summary = df_summary.sort_values("max_episode_count", ascending=False)
    return df_summary
