import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd


def stratified_bootstrap(
    df: pd.DataFrame,
    group_by: Union[str, List[str]],
    strat: Union[str, List[str]],
    metric: str,
    func: Callable,
    repeat: Optional[int] = 100,
    rng: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """
    Perform stratified bootstrapping on a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_by (Union[str, List[str]]): The column(s) by which to group the DataFrame.
        strat (str): The column used to define strata within each group.
        metric (str): The column containing the metric to which the function will be applied.
        func (Callable): The function to apply to the metric within each stratum.
        repeat (int, optional): Number of bootstrap iterations. Default is 100.
        rng (np.random.Generator, optional): Random number generator. Default is NumPy's random generator.

    Returns:
        pd.DataFrame: A DataFrame containing bootstrapped samples with applied function.
    """
    if group_by is None:
        group_by = []

    if isinstance(group_by, str):
        group_by = [group_by]

    if isinstance(strat, str):
        strat = [strat]

    # extract subset of columns
    df = df[group_by + strat + [metric]]

    group = df.groupby(group_by + strat)

    df_list = []
    for i in range(repeat):
        new_df = group.sample(frac=1, replace=True, random_state=rng)
        series = new_df.groupby(group_by)[metric].apply(func)
        sub_df = series.to_frame().reset_index()
        sub_df["bootstrap_index"] = i
        df_list.append(sub_df)

    new_df = pd.concat(df_list)
    return new_df


def bootstrap_all_benchmarks(
    df: pd.DataFrame,
    metric: str,
    benchmark_cols,
    group_by=["agent_model_name"],
    repeat: int = 100,
):
    """Add aggregated data as a new benchmark."""

    # create a new df containing bootstrapped samples of iqm
    df_bootstrap = []
    for benchmark in benchmark_cols:
        # filter sub_df assuming that the benchmark column exists and is a boolean
        sub_df = df[df[benchmark]]

        bs_df = stratified_bootstrap(
            sub_df,
            group_by=group_by,
            strat=["task_name"],
            metric=metric,
            func=np.mean,
            repeat=repeat,
        )
        bs_df["benchmark"] = benchmark
        df_bootstrap.append(bs_df)
    df_bootstrap = pd.concat(df_bootstrap)
    return df_bootstrap


def bootstrap_matrix(data: np.ndarray, n_bootstrap: int, reduce_fn=np.nanmean):
    n_task, n_samples = data.shape

    indices = np.random.randint(0, n_samples, (n_bootstrap, n_task, n_samples))

    assert indices.shape == (n_bootstrap, n_task, n_samples)
    bootstrapped_samples = data[np.arange(n_task)[:, None], indices]

    with warnings.catch_warnings():
        # catches means of empty slices
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results = [reduce_fn(b_data) for b_data in bootstrapped_samples]

    return results


def convert_df_to_array(grouped, metric="cum_reward", threshold=0.9):
    max_samples_per_task = max(len(group) for _, group in grouped)

    arr = np.zeros((len(grouped), max_samples_per_task))

    for task_idx, (group_id, group) in enumerate(grouped):
        # Calculate the ratio of valid values
        valid_ratio = len(group) / max_samples_per_task
        # if valid_ratio < threshold:
        #     raise ValueError(
        #         f"Task {group_id} has insufficient data: ratio {valid_ratio} is below the threshold {threshold}"
        #     )

        # Repeat the task data cyclically to fill up to max_samples_per_task
        repeated_data = np.resize(group[metric].values, max_samples_per_task)
        arr[task_idx, :] = repeated_data

    return arr
