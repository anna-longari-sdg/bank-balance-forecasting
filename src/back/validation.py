import numpy as np
import pandas as pd

from src.back.utils import MAPE


# Cross validation kpi
def cross_validation_stats(cv_df: pd.DataFrame, time_col="DATE_RIF", value_col="VALUE") -> pd.DataFrame:
    """
    Compute per-series, per-model MAPE statistics from cross-validation output.
    Balances accuracy and variance using multiple scoring approaches.
    """
    # Melt CV dataframe to (unique_id, cutoff, y, ds, model, y_hat)
    cv_df = cv_df.melt(
        id_vars=["unique_id", "cutoff", value_col, time_col],
        var_name="model",
        value_name=f"{value_col}_hat",
    )

    # Compute MAPE per (unique_id, model, cutoff)
    mape_df = (
        cv_df.groupby(["unique_id", "model", "cutoff"], as_index=False)
        .apply(lambda g: pd.Series({"mape": MAPE(g[value_col], g[f"{value_col}_hat"])}), include_groups=False)
        .reset_index(drop=True)
    )

    # Aggregate across per each model-series
    mape_stats = mape_df.groupby(["unique_id", "model"], as_index=False).agg(
        mape_mean=("mape", "mean"), 
        mape_std=("mape", "std"), 
        mape_max=("mape", "max")
    )
    mape_stats["mape_std"] = mape_stats["mape_std"].fillna(0)

    # Pessimistic score (mean + 2 * sd)
    # Penalise unstable models linearly.
    mape_stats["mape_pessimistic"] = mape_stats["mape_mean"] + (2 * mape_stats["mape_std"])

    # Stability penalty  (Mean * (1 + var coeff))
    # Multiply the error by an instability factor
    mape_stats["mape_stability"] = mape_stats["mape_mean"] * (
        1 + (mape_stats["mape_std"] / (mape_stats["mape_mean"] + 1e-6))
    )

    # Balanced L2 (L2 standard on normalised values)
    # Solves the problem of different scales between mean and std
    def min_max_norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)
    mape_stats["mean_norm"] = mape_stats.groupby("unique_id")["mape_mean"].transform(min_max_norm)
    mape_stats["std_norm"] = mape_stats.groupby("unique_id")["mape_std"].transform(min_max_norm)
    mape_stats["mape_L2"] = np.sqrt(
        mape_stats["mean_norm"]**2 + mape_stats["std_norm"]**2
    )

    # # Norma L2 di prima (Mean e std non normalizzati)
    # mape_stats["mape_L2"] = np.sqrt(
    #     mape_stats["mape_mean"] ** 2 + mape_stats["mape_std"] ** 2
    # )

    # mape_stats["mape_L2"] = mape_stats["mape_stability"]

    # Rimuoviamo le colonne di appoggio prima del return
    return mape_stats.drop(columns=["mean_norm", "std_norm"])
