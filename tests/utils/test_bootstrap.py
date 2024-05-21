import pandas as pd
import numpy as np
from agentlab.utils.bootstrap import stratified_bootstrap, bootstrap_matrix, convert_df_to_array


# Generate a DataFrame to use for all unit tests
df = pd.DataFrame(
    {
        "agent": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "episode": [1, 2, 1, 2, 1, 2, 1, 2],
        "task": ["T1", "T1", "T2", "T2", "T1", "T1", "T2", "T2"],
        "cum_reward": [10, 11, 12, 13, 14, 15, 16, 17],
    }
)


df_no_var = pd.DataFrame(
    {
        "agent": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "episode": [1, 2, 1, 2, 1, 2, 1, 2],
        "task": ["T1", "T1", "T2", "T2", "T1", "T1", "T2", "T2"],
        "cum_reward": [10, 10, 12, 12, 14, 14, 16, 16],
    }
)


def stratified_bootstrap_simple(df: pd.DataFrame, n_bootstrap_samples):
    """
    Perform stratified bootstrap sampling to calculate mean cumulative rewards.

    Parameters:
        df (DataFrame): DataFrame containing the columns 'task', 'episode', 'cum_reward'.
        n_bootstrap_samples (int): Number of bootstrap samples to generate.

    Returns:
        np.array: An array of overall bootstrap means.
    """

    # Initialize an empty list to store the overall means
    overall_bootstrap_means = []

    # Perform bootstrapping
    for _ in range(n_bootstrap_samples):
        bootstrap_sample_rows = []

        # Stratified sampling for each task
        for task in df["task"].unique():
            task_data = df[df["task"] == task]

            # Generate bootstrap sample for this task (sampling with replacement)
            bootstrap_sample = task_data.sample(n=len(task_data), replace=True)
            bootstrap_sample_rows.append(bootstrap_sample)

        # Concatenate all bootstrap samples to form the overall sample
        overall_bootstrap_sample = pd.concat(bootstrap_sample_rows, ignore_index=True)

        # Compute the mean cum_reward for the overall sample
        overall_bootstrap_mean = overall_bootstrap_sample["cum_reward"].mean()

        # Store this mean
        overall_bootstrap_means.append(overall_bootstrap_mean)

    # Convert the list of overall means to a NumPy array
    return np.array(overall_bootstrap_means)


# Test for expected number of rows in the output DataFrame
def test_with_no_var():
    df_bootstrap = stratified_bootstrap(
        df_no_var, "agent", "task", "cum_reward", np.mean, repeat=10
    )
    df_mean = df_bootstrap.groupby(["agent"])["cum_reward"].mean()
    df_mean_alt = df_no_var.groupby(["agent"])["cum_reward"].mean()

    pd.testing.assert_series_equal(df_mean, df_mean_alt, rtol=1e-1, atol=0)


# Test for statistical properties (mean, in this case)
def test_statistical_properties():
    df_bootstrap = stratified_bootstrap(df, "agent", "task", "cum_reward", np.mean, repeat=1000)
    bs_mean = df_bootstrap.groupby(["agent"])["cum_reward"].mean()
    bs_std = df_bootstrap.groupby(["agent"])["cum_reward"].std()

    # alternative method for stratified bootstrap
    df_bs_alt = []
    for agent in ["A", "B"]:
        bs_mean_alt = stratified_bootstrap_simple(df[df["agent"] == agent], 1000)
        bs_mean_df = pd.DataFrame(bs_mean_alt, columns=["cum_reward"])
        bs_mean_df["agent"] = agent
        bs_mean_df["bootstrap_index"] = np.arange(1000)
        df_bs_alt.append(bs_mean_df)

    df_bs_alt = pd.concat(df_bs_alt, ignore_index=True)
    bs_mean_alt = df_bs_alt.groupby(["agent"])["cum_reward"].mean()
    bs_std_alt = df_bs_alt.groupby(["agent"])["cum_reward"].std()

    # high relative tolerance because of small sample size
    pd.testing.assert_series_equal(bs_mean, bs_mean_alt, rtol=1e-1, atol=0)
    pd.testing.assert_series_equal(bs_std, bs_std_alt, rtol=1e-1, atol=0)

    mean_alt = df.groupby(["agent"])["cum_reward"].mean()
    pd.testing.assert_series_equal(bs_mean, mean_alt, rtol=1e-1, atol=0)


def test_statistical_properties_array():
    arr = convert_df_to_array(df.groupby("task"))
    bootstrap_results = bootstrap_matrix(arr, 1000, np.mean)
    bs_mean = np.mean(bootstrap_results)
    original_mean = np.mean(arr)
    assert np.isclose(bs_mean, original_mean, rtol=1e-1, atol=0)


# Test with no variation
def test_with_no_var_array():
    df_no_var = df.copy()
    df_no_var["cum_reward"] = 10  # Setting a constant value for no variation
    arr_no_var = convert_df_to_array(df_no_var.groupby("task"))
    bootstrap_results = bootstrap_matrix(arr_no_var, 1000, np.mean)
    bs_mean = np.mean(bootstrap_results)
    assert np.isclose(bs_mean, 10, rtol=1e-1, atol=0)


def test_convert_df_to_array():
    df_uneven = df.drop(7)
    result = convert_df_to_array(df_uneven.groupby("task"), threshold=0.7)
    assert result.shape == (2, 4)  # 2 tasks, 4 samples each
    assert np.array_equal(result[1, :], [12, 13, 16, 12])


if __name__ == "__main__":
    test_with_no_var_array()
    test_statistical_properties_array()
