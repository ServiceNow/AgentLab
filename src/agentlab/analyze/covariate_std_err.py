from typing import Callable, Sequence, Tuple

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from statsmodels.genmod.families import Binomial


def aggregate_std_err(
    run_rewards: Sequence[np.ndarray],
    baseline_rewards: np.ndarray,
    std_err_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Args:
      run_rewards: list of length k, each an array of shape (n,)
      baseline_rewards: array of shape (m, n) holding m covariate-baselines
      std_err_fn: function that, given (rewards: np.ndarray of shape (n,),
             baselines: np.ndarray of shape (m, n)),
             returns (mean: float, se: float)
    Returns:
      (overall_mean, overall_se)
    """

    if std_err_fn is None:
        std_err_fn = std_err_ancova

    # 1) call the low-level routine on each run
    stats = [std_err_fn(r, baseline_rewards) for r in run_rewards]
    means = np.array([mu for mu, _ in stats])
    ses = np.array([sigma for _, sigma in stats])

    k = len(means)
    # 2) overall mean of the per-run means
    overall_mean = means.mean()

    # 3) decompose variance: between-runs + within-runs
    var_between = means.var(ddof=1) / k
    var_within = (ses**2).mean() / k

    overall_se = np.sqrt(var_between + var_within)
    return overall_mean, overall_se


def std_err_clt(rewards: np.array) -> tuple[float, float]:
    """
    Computes the mean and standard error of the rewards.

    Parameters:
    - rewards: array-like of shape (n,)
        Observed rewards for each sample.

    Returns:
    - reward_mean: float
        Mean of the rewards.
    - se: float
        Standard error of the mean.
    """
    rewards = np.asarray(rewards, dtype=float)
    n = rewards.size
    if n == 0:
        raise ValueError("The input array is empty.")

    reward_mean = rewards.mean()
    se = np.std(rewards, ddof=1) / np.sqrt(n)

    return reward_mean, se


def std_err_bootstrap(rewards: np.array, n_boot: int = 1000) -> tuple[float, float]:
    """
    Computes the mean and standard error of the rewards using bootstrap.

    Parameters:
    - rewards: array-like of shape (n,)
        Observed rewards for each sample.
    - n_boot: int, default=1000
        Number of bootstrap samples.

    Returns:
    - reward_mean: float
        Mean of the rewards.
    - se: float
        Standard error of the mean.
    """
    rewards = np.asarray(rewards, dtype=float)
    n = rewards.size
    if n == 0:
        raise ValueError("The input array is empty.")

    boot_means = []
    rng = np.random.default_rng()
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means.append(rewards[idx].mean())

    reward_mean = np.mean(boot_means)
    se = np.std(boot_means, ddof=1)

    return reward_mean, se


def _replace_nans_by_average(baselines):
    """
    Impute NaNs in each column of the baselines matrix with that column's mean.

    Parameters:
    - baselines: array-like of shape (n, k)
        Baseline estimates per sample and baseline index.

    Returns:
    - imputed: np.ndarray of shape (n, k)
        Baselines with NaNs replaced by their column means.
    """
    baselines = np.asarray(baselines, dtype=float)
    # Compute column means ignoring NaNs
    col_means = np.nanmean(baselines, axis=0)
    # Broadcast and fill NaNs
    imputed = np.where(np.isnan(baselines), col_means, baselines)
    return imputed


def _select_best_baseline(rewards, baselines):
    """
    Select the best baseline based on the total absolute error.
    """
    errors = np.abs(baselines - rewards[:, None]).sum(axis=0)
    j_star = int(np.argmin(errors))
    return baselines[:, j_star]


def std_err_diff_baselines(rewards, baselines):
    """
    Find the best baseline and compute the adjusted mean and SE.

    Parameters:
    - rewards: array-like of shape (n,)
        Observed rewards (may contain NaN).
    - baselines: array-like of shape (n, k)
        k baseline estimates per sample (may contain NaN).

    Returns:
    - adjusted_reward_mean: float
        Mean of valid rewards.
    - adjusted_se: float∫
        SE of the adjusted mean (differences) using the selected baseline.
    - selected_baseline: np.ndarray of shape (n,)
        The values of the chosen baseline with NaNs filled.
    """
    rewards = np.asarray(rewards, dtype=float)
    baselines = _replace_nans_by_average(baselines)

    if rewards.shape[0] != baselines.shape[0]:
        raise ValueError("rewards and baselines must have the same length.")

    # Identify valid reward samples
    valid = ~np.isnan(rewards)
    reward_valid = rewards[valid]
    if reward_valid.size == 0:
        return np.nan, np.nan

    selected_baseline_valid = _select_best_baseline(reward_valid, baselines[valid])
    diffs = reward_valid - selected_baseline_valid
    adjusted_se = np.std(diffs, ddof=1) / np.sqrt(diffs.size)

    # Adjusted mean reward is the raw mean of valid rewards
    adjusted_reward_mean = reward_valid.mean()

    return adjusted_reward_mean, adjusted_se


def _clean_input(rewards, baselines):
    rewards = np.asarray(rewards)
    baselines = np.asarray(baselines)
    baselines = _replace_nans_by_average(baselines)
    if rewards.shape[0] != baselines.shape[0]:
        raise ValueError("rewards and baselines must have the same length.")
    if rewards.ndim != 1:
        raise ValueError("rewards must be a 1D array.")
    if baselines.ndim != 2:
        raise ValueError("baselines must be a 2D array.")

    # remove nan rows
    valid = ~np.isnan(rewards)
    rewards = rewards[valid]
    baselines = baselines[valid]
    if rewards.size == 0:
        raise ValueError("No valid rewards after filtering.")
    if baselines.shape[0] != rewards.shape[0]:
        raise ValueError("rewards and baselines must have the same length after filtering.")

    return rewards, baselines


def std_err_ancova(rewards, baselines):
    """
    Parameters:
    - rewards: array-like of shape (n,)
        Observed rewards per sample
    - baselines: array-like of shape (n, k)
        k baseline estimates per sample

    Returns:
    - adjusted_mean: float
        Mean reward adjusted to the average baseline levels
    - standard_error: float
        Standard error of the adjusted mean
    """
    rewards, baselines = _clean_input(rewards, baselines)

    # Center the baselines
    baseline_means = baselines.mean(axis=0)
    centered_baselines = baselines - baseline_means

    # Build design matrix with intercept
    design_matrix = sm.add_constant(centered_baselines)

    # Fit the model
    results = sm.OLS(rewards, design_matrix).fit()

    # Extract the adjusted mean (intercept) and its SE
    adjusted_mean = results.params[0]
    standard_error = results.bse[0]

    # print rsquared
    print(f"R-squared: {results.rsquared:.4f}")

    return adjusted_mean, standard_error


def std_err_glm_cv_regularized(
    rewards, baselines, lambda_grid=None, n_splits=5, n_boot=200, random_state=None
):
    """
    Fit a logistic GLM with L2 regularization, selecting the penalty strength via k-fold CV,
    and estimate SE of the adjusted mean via bootstrap.

    Parameters
    ----------
    rewards : array-like, shape (n,)
        Observed binary outcomes (0 or 1).
    baselines : array-like, shape (n, k)
        k baseline estimates per sample.
    lambda_grid : list or array-like of floats
        Candidate L2 penalty strengths (alpha values) for cv.
    n_splits : int, default=5
        Number of folds for cross-validation.
    n_boot : int, default=200
        Number of bootstrap replicates for SE estimation.
    random_state : int or None, default=None
        Seed for reproducibility in CV splits and bootstrap.

    Returns
    -------
    adjusted_mean : float
        Mean predicted probability across all samples under the final model.
    adjusted_se : float
        Bootstrap-based SE of the adjusted mean.
    best_lambda : float
        The selected regularization strength (alpha).
    """

    if lambda_grid is None:
        lambda_grid = np.logspace(-4, 3, 10)

    # Prepare data
    y = np.asarray(rewards)
    B = np.asarray(baselines)
    Bc = B - B.mean(axis=0)  # center covariates
    X = sm.add_constant(Bc)  # design matrix

    # Cross-validate to pick lambda
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_lambda = None
    best_score = np.inf
    for lam in lambda_grid:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            model = sm.GLM(y[train_idx], X[train_idx], family=Binomial())
            res = model.fit_regularized(alpha=lam, L1_wt=0)
            p_val = res.predict(X[val_idx])
            p_val = np.clip(p_val, 1e-6, 1 - 1e-6)
            # Negative log-likelihood per sample
            nll = -np.mean(y[val_idx] * np.log(p_val) + (1 - y[val_idx]) * np.log(1 - p_val))
            fold_scores.append(nll)
        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score, best_lambda = avg_score, lam

    print(f"Best lambda std_err_glm_cv_regularized: {best_lambda:.4f} with NLL: {best_score:.4f}")

    # Fit final model on full data
    final_model = sm.GLM(y, X, family=Binomial()).fit_regularized(alpha=best_lambda, L1_wt=0)
    p_hat = final_model.predict(X)
    adjusted_mean = np.mean(p_hat)

    # Bootstrap SE estimation
    rng = np.random.RandomState(random_state)
    boot_means = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        res_boot = sm.GLM(y[idx], X[idx], family=Binomial()).fit_regularized(
            alpha=best_lambda, L1_wt=0
        )
        p_boot = res_boot.predict(X)
        boot_means.append(np.mean(p_boot))
    adjusted_se = np.std(boot_means, ddof=1)

    return adjusted_mean, adjusted_se


def std_err_glm_crossfit_bootstrap(
    rewards, baselines, lambda_grid=None, K=5, B=200, random_state=None
):
    """
    1) Tune lambda once on the full data via K-fold CV
    2) Bootstrap: for each replicate, cross-fit GLM with the fixed lambda and compute the mean
    3) Report the point estimate (cross-fitted mean on original data) and bootstrap SE

    Parameters
    ----------
    rewards : array-like, shape (n,)
        Binary outcomes (0 or 1).
    baselines : array-like, shape (n, k)
        Baseline covariates per observation.
    lambda_grid : array-like, optional
        Candidate L2 penalties for CV. Defaults to logspace(-4, 3, 10).
    K : int, default=5
        Number of folds for CV and cross-fit.
    B : int, default=200
        Number of bootstrap replicates.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    mu_hat : float
        Cross-fitted mean predicted probability on the original data.
    se : float
        Bootstrap-based standard error of the mean.
    best_lambda : float
        Selected regularization strength.
    """
    y = np.asarray(rewards)
    X = sm.add_constant(np.asarray(baselines) - np.asarray(baselines).mean(axis=0))
    n = len(y)

    # 1) Tune lambda on full data
    if lambda_grid is None:
        lambda_grid = np.logspace(-4, 3, 10)
    kf_inner = KFold(n_splits=K, shuffle=True, random_state=random_state)
    best_lambda, best_score = None, np.inf
    for lam in lambda_grid:
        scores = []
        for train_idx, val_idx in kf_inner.split(X):
            model = sm.GLM(y[train_idx], X[train_idx], family=sm.families.Binomial())
            res = model.fit_regularized(alpha=lam, L1_wt=0)
            p_val = np.clip(res.predict(X[val_idx]), 1e-6, 1 - 1e-6)
            nll = -np.mean(y[val_idx] * np.log(p_val) + (1 - y[val_idx]) * np.log(1 - p_val))
            scores.append(nll)
        if np.mean(scores) < best_score:
            best_score, best_lambda = np.mean(scores), lam

    # 2) Cross-fitted point estimate on original data
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    p_full = np.zeros(n)
    for train_idx, val_idx in kf.split(X):
        model = sm.GLM(y[train_idx], X[train_idx], family=sm.families.Binomial())
        res = model.fit_regularized(alpha=best_lambda, L1_wt=0)
        p_full[val_idx] = res.predict(X[val_idx])
    mu_hat = p_full.mean()

    # 3) Bootstrap SE via cross-fitted means
    rng = np.random.RandomState(random_state)
    mu_boot = np.zeros(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        yb, Xb = y[idx], X[idx]
        p_b = np.zeros(n)
        for train_idx, val_idx in kf.split(Xb):
            model = sm.GLM(yb[train_idx], Xb[train_idx], family=sm.families.Binomial())
            res = model.fit_regularized(alpha=best_lambda, L1_wt=0)
            p_b[val_idx] = res.predict(Xb[val_idx])
        mu_boot[b] = p_b.mean()

    print(
        f"Best lambda std_err_glm_crossfit_bootstrap: {best_lambda:.4f} with NLL: {best_score:.4f}"
    )

    se = mu_boot.std(ddof=1)
    return mu_hat, se


def crossfit_se_min_nll(rewards, baselines, lambda_grid=None, K=5, random_state=None):
    """
    Cross-fit predictions for each lambda, compute out-of-sample NLL,
    select lambda with min NLL, then compute SE = std(y - p)/sqrt(n).

    Parameters
    ----------
    rewards : array-like, shape (n,)
        Binary outcomes (0 or 1).
    baselines : array-like, shape (n, k)
        Baseline covariates per observation.
    lambda_grid : array-like
        Candidate L2 penalties for CV.
    K : int, default=5
        Number of folds for cross-fit.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    best_lambda : float
        Lambda with minimum out-of-sample NLL.
    best_p : array, shape (n,)
        Cross-fitted probabilities for best_lambda.
    best_se : float
        Standard error = std(y - best_p) / sqrt(n).
    nll_dict : dict
        NLL values for each lambda.
    """
    y = np.asarray(rewards)
    B = np.asarray(baselines)
    # center covariates
    Bc = B - B.mean(axis=0)
    X = sm.add_constant(Bc)
    n = len(y)
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)

    nll_dict = {}
    p_dict = {}

    if lambda_grid is None:
        lambda_grid = np.logspace(-4, 3, 10)

    for lam in lambda_grid:
        p = np.zeros(n)
        for train_idx, val_idx in kf.split(X):
            model = sm.GLM(y[train_idx], X[train_idx], family=sm.families.Binomial())
            res = model.fit_regularized(alpha=lam, L1_wt=0)
            p[val_idx] = res.predict(X[val_idx])

        p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
        nll_dict[lam] = nll
        p_dict[lam] = p

    best_lambda = min(nll_dict, key=nll_dict.get)
    best_p = p_dict[best_lambda]
    best_se = np.std(y - best_p, ddof=0) / np.sqrt(n)
    print(
        f"Best lambda crossfit_se_min_nll: {best_lambda:.4f} with NLL: {nll_dict[best_lambda]:.4f}"
    )

    adjusted_mean = best_p.mean()
    return adjusted_mean, best_se
