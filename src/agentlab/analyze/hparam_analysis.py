from collections import Counter

import ConfigSpace as CS  # type: ignore
import numpy as np
import optuna
import pandas as pd
from fanova import fANOVA as _fANOVA
from optuna.importance import PedAnovaImportanceEvaluator
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


def bootstrap_optimal_hparam(
    df: pd.DataFrame,
    hp_cols: list,
    focus_col: str,
    metric_col: str,
    n_boot: int = 2000,
    maximize: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap the probability that each value of `focus_col` is part of
    the global optimum after the other hyper-parameters are (implicitly)
    optimised.

    Parameters
    ----------
    df          : DataFrame containing one row per trial.
    hp_cols     : list of *all* hyper-parameter column names (focus included).
    focus_col   : the hyper-parameter whose optimal value distribution we want.
    metric_col  : column with the validation metric (higher is better by default).
    n_boot      : number of bootstrap replicates.
    maximize    : maximise (True) or minimise (False) the metric.
    random_state: reproducibility control.

    Returns
    -------
    values : ndarray  -- unique values of `focus_col` (order as in `df`).
    probs  : ndarray  -- probability of each value being optimal.
    """
    rng = np.random.default_rng(random_state)

    # 1.  Label each row by its “context” (all other hyper‑params)
    others = [c for c in hp_cols if c != focus_col]
    df = df.copy()
    df["ctx_id"] = df.groupby(others, sort=False).ngroup()
    ctx_counts = df["ctx_id"].value_counts()

    # 2.  Inverse‑frequency sampling probs so rare contexts aren’t drowned
    weights = 1 / ctx_counts
    p_ctx = weights / weights.sum()

    # 3.  Bootstrap loop
    focus_vals = df[focus_col].unique()
    hit = dict.fromkeys(focus_vals, 0)

    argbest = np.argmax if maximize else np.argmin

    for _ in range(n_boot):
        sampled_ctx = rng.choice(
            ctx_counts.index,
            size=len(ctx_counts),
            replace=True,
            p=p_ctx.loc[ctx_counts.index].values,
        )
        sample = df[df.ctx_id.isin(sampled_ctx)]

        best_row_idx = argbest(sample[metric_col].values)
        best_focus = sample.iloc[best_row_idx][focus_col]
        hit[best_focus] += 1

    values = np.array(list(hit.keys()))
    probs = np.array(list(hit.values()), dtype=float) / n_boot

    df_result = pd.DataFrame({"values": values, "prob": probs})
    # sort by values and set index as "values"
    df_result = df_result.sort_values(by="values").reset_index(drop=True)
    df_result["values"] = df_result["values"].astype(str)
    df_result.set_index("values", inplace=True)
    return df_result


def bootstrap_optimal_hparam_uncertainty(
    df: pd.DataFrame,
    hp_cols: list,
    focus_col: str,
    metric_col: str,
    run_id_col: str,
    n_boot_outer: int = 2000,
    n_boot_inner: int = 30,
    maximize: bool = True,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    argbest = np.argmax if maximize else np.argmin
    eps = 1e-12  # numerical safety
    H_bits = lambda p: -np.sum(p * np.log2(p + eps))

    # --- prep -----------------------------------------------------------
    others = [c for c in hp_cols if c != focus_col]
    df = df.copy()
    df["ctx_id"] = df.groupby(others, sort=False).ngroup()

    ctx_counts = df["ctx_id"].value_counts()
    p_ctx = (1 / ctx_counts) / (1 / ctx_counts).sum()

    focus_vals = np.sort(df[focus_col].unique())
    epi_hits = Counter({v: 0 for v in focus_vals})
    total_hits = Counter({v: 0 for v in focus_vals})
    alea_H_list = []  # per‑outer‑draw conditional entropy

    run2metrics = df.groupby(run_id_col)[metric_col].apply(np.asarray).to_dict()

    # --- bootstrap ------------------------------------------------------
    for _ in tqdm(range(n_boot_outer), desc="outer‑boots"):
        # 1. resample contexts  → epistemic
        sampled_ctx = rng.choice(
            ctx_counts.index,
            size=len(ctx_counts),
            replace=True,
            p=p_ctx.loc[ctx_counts.index].values,
        )
        rows_epi = df[df.ctx_id.isin(sampled_ctx)]
        rows_u = rows_epi.drop_duplicates(run_id_col, keep="first")

        # optimum without noise  (epistemic only)
        best_idx = argbest(rows_u[metric_col].values)
        best_focus = rows_u.iloc[best_idx][focus_col]
        epi_hits[best_focus] += 1

        # 2. inner: add aleatoric noise many times
        inner_hits = Counter({v: 0 for v in focus_vals})

        for _ in range(n_boot_inner):
            noisy_scores = []
            focus_tmp = []
            for rid, row in rows_u.groupby(run_id_col).first().iterrows():
                val = rng.choice(run2metrics[rid])  # one noisy draw
                noisy_scores.append(val)
                focus_tmp.append(row[focus_col])

            best_noisy = argbest(noisy_scores)
            inner_hits[focus_tmp[best_noisy]] += 1
            total_hits[focus_tmp[best_noisy]] += 1

        # conditional entropy for this outer draw
        p_inner = np.array([inner_hits[v] for v in focus_vals], dtype=float)
        p_inner /= p_inner.sum()
        alea_H_list.append(H_bits(p_inner))

    # --- convert counters -> probabilities -----------------------------
    def hits_to_p(counter):
        arr = np.array([counter[v] for v in focus_vals], float)
        return arr / arr.sum()

    p_epi = hits_to_p(epi_hits)
    p_total = hits_to_p(total_hits)

    # entropies
    H_epi = H_bits(p_epi)
    H_total = H_bits(p_total)
    H_alea = np.mean(alea_H_list)  # always ≥ 0 and ≤ H_total

    # tidy DataFrame
    df_out = (
        pd.DataFrame(
            {
                "focus_value": focus_vals,
                "P_epistemic": p_epi,
                "P_total": p_total,
                "H_epi": H_epi,
                "H_total": H_total,
                "H_alea": H_alea,
            }
        )
        .sort_values("focus_value")
        .reset_index(drop=True)
    )

    return df_out


# -----------------------------------------------------------------------------
# 1.  Primary implementation (fanova package)
# -----------------------------------------------------------------------------


def fanova_df(df: pd.DataFrame, hp_cols: list[str], metric_col: str) -> dict[str, float]:
    import ConfigSpace as CS
    import pandas as pd
    from fanova import fANOVA  # correct import
    from sklearn.preprocessing import OrdinalEncoder

    enc = OrdinalEncoder(dtype=float)
    X_enc = enc.fit_transform(df[hp_cols])
    y = df[metric_col].to_numpy(float)

    cs = CS.ConfigurationSpace()
    for col in hp_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            hp = CS.UniformFloatHyperparameter(col, float(s.min()), float(s.max()))
        else:
            hp = CS.CategoricalHyperparameter(col, sorted(map(str, s.unique())))
        cs.add(hp)

    fan = fANOVA(X_enc, y, config_space=cs)
    return {
        col: fan.quantify_importance((i,))["individual importance"] for i, col in enumerate(hp_cols)
    }


def fanova_df_(
    df: pd.DataFrame,
    hp_cols: list[str],
    metric_col: str,
) -> dict[str, float]:
    """Return first‑order importances with the official *fANOVA* package."""

    # ------------------------------------------------------------------
    # 1. Build the ConfigSpace (numeric bounds OR categorical choices)
    # ------------------------------------------------------------------
    cs = CS.ConfigurationSpace()
    cat_maps = {}  # {col: {label -> idx}}
    for col in hp_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            # int vs float → choose the matching hyper‑type
            if pd.api.types.is_integer_dtype(s):
                hp = CS.UniformIntegerHyperparameter(col, int(s.min()), int(s.max()))
            else:
                hp = CS.UniformFloatHyperparameter(col, float(s.min()), float(s.max()))
        else:
            labels = sorted(map(str, s.unique()))
            hp = CS.CategoricalHyperparameter(col, labels)
            cat_maps[col] = {lbl: i for i, lbl in enumerate(labels)}
        cs.add(hp)

    # ------------------------------------------------------------------
    # 2. Encode every trial into a numeric vector compatible with cs
    # ------------------------------------------------------------------
    X = np.empty((len(df), len(hp_cols)), dtype=float)
    for j, col in enumerate(hp_cols):
        if col in cat_maps:  # categorical → index
            X[:, j] = df[col].astype(str).map(cat_maps[col]).to_numpy()
        else:  # numeric → as‑is
            X[:, j] = df[col].to_numpy(float)

    y = df[metric_col].to_numpy(float)

    # ------------------------------------------------------------------
    # 3. Run fANOVA
    # ------------------------------------------------------------------
    fan = _fANOVA(X, y, config_space=cs)

    return {
        col: fan.quantify_importance((i,))["individual importance"] for i, col in enumerate(hp_cols)
    }


def fanova_mc(
    df: pd.DataFrame,
    hp_cols: list[str],
    metric_col: str,
    n_mc: int = 10_000,
    random_state: int | None = 0,
) -> dict[str, float]:
    """Return first‑order importance via Monte‑Carlo + Random‑Forest surrogate.

    Uses the same API as `run_fanova` but does **not** rely on external packages
    beyond scikit‑learn and numpy.
    """
    enc = OrdinalEncoder(dtype=float)
    X_enc = enc.fit_transform(df[hp_cols])
    y = df[metric_col].to_numpy(float)

    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_enc, y)

    rng = np.random.default_rng(random_state)
    lo, hi = X_enc.min(0), X_enc.max(0)
    X_base = rng.random((n_mc, X_enc.shape[1])) * (hi - lo) + lo

    f0 = rf.predict(X_base).mean()
    var_total = ((rf.predict(X_base) - f0) ** 2).mean()

    importances: dict[str, float] = {}
    for j, col in enumerate(hp_cols):
        X_pert = X_base.copy()
        # Fix column j, resample others (5× for variance reduction)
        idxs = rng.choice(n_mc, size=n_mc)
        X_pert[:, :] = X_base[idxs]
        X_pert[:, j] = X_base[:, j]
        y_bar = rf.predict(X_pert)
        importances[col] = ((y_bar - f0) ** 2).mean() / var_total

    return importances


def pedanova_importance(
    df: pd.DataFrame,
    hp_cols: list[str],
    metric_col: str,
    baseline_quantile: float = 0.10,
    evaluate_on_local: bool = True,
) -> dict[str, float]:
    """Return PED‑ANOVA importance scores for each hyper‑parameter."""

    # 1. Build a single, global distribution per hyper‑parameter
    search_space: dict[str, optuna.distributions.BaseDistribution] = {}
    for hp in hp_cols:
        s = df[hp]
        if pd.api.types.is_numeric_dtype(s):
            lo, hi = s.min(), s.max()
            if pd.api.types.is_integer_dtype(s):
                search_space[hp] = optuna.distributions.IntDistribution(int(lo), int(hi))
            else:
                search_space[hp] = optuna.distributions.FloatDistribution(float(lo), float(hi))
        else:  # categorical
            search_space[hp] = optuna.distributions.CategoricalDistribution(
                sorted(s.unique().tolist())
            )

    # 2. Re‑create a Study and add each row as a completed trial
    study = optuna.create_study(direction="maximize")
    for _, row in df.iterrows():
        params = {hp: row[hp] for hp in hp_cols}
        trial = optuna.trial.create_trial(
            params=params,
            distributions=search_space,
            values=[row[metric_col]],
        )
        study.add_trial(trial)

    # 3. Run PED‑ANOVA
    evaluator = PedAnovaImportanceEvaluator(
        baseline_quantile=baseline_quantile,
        evaluate_on_local=evaluate_on_local,
    )
    return optuna.importance.get_param_importances(study, evaluator=evaluator)
