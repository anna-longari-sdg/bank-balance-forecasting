import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from loguru import logger

from src.back.configuration import PIPELINE_DEFAULT_CONFIGURATION as config
from src.back.data import DataLoader, split_dataset
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def calculate_cause_effect(data: DataLoader):
    """

    Parameters
    ----------

    Returns
    -------

    """
    # From data loader get drivers selected for almost one of economics and retrieve anagrafical information
    drv_used = data.driver_selected_df["DRV_COD"].unique()
    drv_used_anag = data.driver_anag[data.driver_anag["DRV_COD"].isin(drv_used)]
    # From mapping table contruct whatif group mapping
    drv_mapped = data.mapped_df[data.mapped_df["DRV_COD"].isin(drv_used)]
    drv_mapped = drv_mapped.merge(data.eco_anag, on="ECO_COD", how="left")[
        ["DRV_COD", "ECO_GRP_0", "ECO_GRP_1"]
    ].drop_duplicates()
    drv_mapped = (
        drv_mapped.groupby("DRV_COD")
        .agg(
            {
                "ECO_GRP_0": lambda x: "|".join(sorted(set(map(str, x)))),
                "ECO_GRP_1": lambda x: "|".join(sorted(set(map(str, x)))),
            }
        )
        .reset_index()
    )
    drv_mapped["whatif_grp"] = drv_mapped["ECO_GRP_0"] + "|" + drv_mapped["ECO_GRP_1"]
    # Loop on whatif groups and calculate cause effect matrix for each group
    cause_effect_dict = {}
    for grp in drv_mapped["whatif_grp"].unique():
        drv_in_grp = drv_mapped[drv_mapped["whatif_grp"] == grp]["DRV_COD"].tolist()
        df = data.driver_df[data.driver_df["DRV_COD"].isin(drv_in_grp)].reset_index(drop=True)
        df_drv_hist, _ = split_dataset(df, cutoff_month=config.dataset.cutoff_month, time_col="DATE_RIF", val_col = "VALUE", group_col="DRV_COD", fillna_method="zero")
        df_drv_hist = df_drv_hist.pivot(index="DATE_RIF", columns="DRV_COD", values="VALUE").sort_index()
        df_drv_hist.fillna(0, inplace=True)
        ce_matrix = cause_effect_matrix(df_drv_hist, max_lag=1)
        ce_matrix_long = ce_matrix.reset_index().melt(id_vars="DRV_COD", var_name="DRV_COD_CE", value_name="COEF_CE")
        cause_effect_dict[grp] = ce_matrix_long

    cause_effect_df = pd.concat(cause_effect_dict.values()).reset_index(drop=True)
    data.cause_effect_df = cause_effect_df
    return cause_effect_df


def propagate_shock(
    drivers_forecast_df: pd.DataFrame, cause_effect_df: pd.DataFrame, shock_driver, shock_month_index, shock_value, col_forecast= "FORECAST"
):
    """
    Dynamic simulation: the shock propagates into the future through historical relationships.
    """
    ce_matrix_df = cause_effect_df.copy()
    ce_matrix = ce_matrix_df.pivot(index="DRV_COD", columns="DRV_COD_CE", values="COEF_CE")
    ce_matrix.fillna(0, inplace=True)
    df_forecast = drivers_forecast_df.pivot(index="DATE_RIF", columns="DRV_COD", values=col_forecast).sort_index()

    sim_df = df_forecast.copy()
    n_months = len(sim_df)
    driver_names = sim_df.columns

    # 1. INITIAL SHOCK APPLICATION (e.g., March)
    # Compute the delta relative to the original value to scale impacts
    original_value = sim_df.iloc[shock_month_index].get(shock_driver)
    sim_df.iloc[shock_month_index, sim_df.columns.get_loc(shock_driver)] = shock_value

    # Instant propagation (L0) only for the shock month
    l0_row = f"{shock_driver}_L0"    
    if l0_row in ce_matrix.index:        
        for target_col in driver_names:            
            coeff = ce_matrix.loc[l0_row, target_col]
            if abs(coeff) > 0 and target_col != shock_driver:                
                # logger.info(f"Applying shock from {shock_driver} to {target_col} with coefficient {coeff:.4f}")
                # Apply the instantaneous shock based on the percentage change
                delta_pct = (shock_value - original_value) / original_value
                sim_df.iloc[shock_month_index, sim_df.columns.get_loc(target_col)] *= 1 + delta_pct * coeff
    
    # 2. RECURSIVE PROPAGATION (from the next month onward)
    # Each month t depends on values recomputed at month t-1
    for t in range(shock_month_index + 1, n_months):
        for target_col in driver_names:
            # Compute month t change based on Lag 1 (L1) of ALL drivers at month t-1
            cumulative_change = 0
            for input_driver in driver_names:
                lag_row = f"{input_driver}_L1"
                if lag_row in ce_matrix.index:
                    coeff_l1 = ce_matrix.loc[lag_row, target_col]
                    if abs(coeff_l1) > 0:
                        # logger.info(f"Month {t}: Applying L1 effect from {input_driver} to {target_col} with coefficient {coeff_l1:.4f}")
                        # Compute how much input_driver at time t-1
                        # deviated relative to its original forecast
                        current_value = sim_df.iloc[t - 1].get(input_driver)
                        base_value = df_forecast.iloc[t - 1].get(input_driver)
                        if base_value != 0:
                            deviation_pct = (current_value - base_value) / base_value
                        else:
                            deviation_pct = 0

                        # deviation_pct = (current_value - base_value) / base_value
                        cumulative_change += deviation_pct * coeff_l1

            # Update the forecast value for month t
            sim_df.iloc[t, sim_df.columns.get_loc(target_col)] *= 1 + cumulative_change

    sim_df_long = sim_df.reset_index().melt(id_vars="DATE_RIF", var_name="DRV_COD", value_name="FORECAST_WHATIF")

    return sim_df_long

# def cause_effect_matrix(df, max_lag=1):
#     df_diff = df.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    
#     # Create lagged features for Lasso regression
#     X_lagged = pd.concat([df_diff.shift(i).add_suffix(f"_L{i}") for i in range(max_lag + 1)], axis=1)
#     X_lagged = X_lagged.dropna()
#     y = df_diff.loc[X_lagged.index]

#     ce_matrix = pd.DataFrame(0.0, index=X_lagged.columns, columns=y.columns)

#     for target in y.columns:
#         # Remove L0 of the target from features to avoid perfect multicollinearity
#         X = X_lagged.drop(columns=[f"{target}_L0"])
        
#         # Using LassoCV to find the best alpha for feature selection
#         model = LassoCV(cv=5, n_alphas=200, tol=1e-3).fit(X, y[target])
        
#         for i, col_name in enumerate(X.columns):
#             coeff = model.coef_[i]
            
#             # If L1 of Driver 5 on itself is small but non-zero, we set a minimal inertia
#             if col_name == f"{target}_L1":
#                 if abs(coeff) > 0.001: 
#                     ce_matrix.loc[col_name, target] = coeff
#                 else:
#                     ce_matrix.loc[col_name, target] = 0.05 
            
#             # For the other coefficients, we set a higher threshold to focus on stronger relationships (10%)
#             elif abs(coeff) > 0.01:
#                 ce_matrix.loc[col_name, target] = coeff

#     return ce_matrix

# def cause_effect_matrix(df, max_lag=1, abs_coeff=0.20):
#     df_diff = df.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    
#     # Creazione feature laggate
#     X_lagged = pd.concat([df_diff.shift(i).add_suffix(f"_L{i}") for i in range(max_lag + 1)], axis=1)
#     X_lagged = X_lagged.dropna()
#     y = df_diff.loc[X_lagged.index]

#     ce_matrix = pd.DataFrame(0.0, index=X_lagged.columns, columns=y.columns)

#     for target in y.columns:
#         # Rimuoviamo L0 del target
#         X = X_lagged.drop(columns=[f"{target}_L0"])
        
#         # LassoCV: n_alphas alto e selezione stringente
#         # Usiamo selection='random' per stabilità se i dati sono molto correlati
#         model = LassoCV(cv=5, n_alphas=200, selection='cyclic', tol=1e-3).fit(X, y[target])
        
#         # Recuperiamo i coefficienti originali del CV
#         coeffs = model.coef_
        
#         for i, col_name in enumerate(X.columns):
#             coeff = coeffs[i]
            
#             # --- LOGICA RICHIESTA ---
            
#             # 1. Focus su se stesso (Inerzia L1)
#             if col_name == f"{target}_L1":
#                 # Se LassoCV lo ha azzerato o reso quasi nullo, 
#                 # forziamo un'inerzia minima (0.05) per la stabilità del driver
#                 if abs(coeff) < 0.05:
#                     ce_matrix.loc[col_name, target] = 0.05
#                 else:
#                     ce_matrix.loc[col_name, target] = coeff
            
#             # 2. Focus su altri Driver (Effetti esterni)
#             else:
#                 # Applichiamo una "Hard Threshold": 
#                 # Se l'impatto non è almeno del 10% (0.1), lo cancelliamo.
#                 # Questo evita che il What-if si propaghi ovunque per micro-correlazioni.
#                 if abs(coeff) > abs_coeff:
#                     ce_matrix.loc[col_name, target] = coeff
#                 else:
#                     ce_matrix.loc[col_name, target] = 0.0

#     return ce_matrix

def cause_effect_matrix(df, max_lag=1, abs_coeff=0.20):
    # Standardize the data to get comparable coefficients 
    df_diff = df.pct_change().replace([np.inf, -np.inf], 0).fillna(0)    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_diff), columns=df.columns, index=df.index)
    
    # Creation feature lagged
    X_lagged = pd.concat([df_scaled.shift(i).add_suffix(f"_L{i}") for i in range(max_lag + 1)], axis=1)
    X_lagged = X_lagged.dropna()
    y = df_scaled.loc[X_lagged.index]

    ce_matrix = pd.DataFrame(0.0, index=X_lagged.columns, columns=y.columns)

    for target in y.columns:
        # Remove L0 
        X = X_lagged.drop(columns=[f"{target}_L0"])
        
        # LassoCV con parametri per alta sparsità
        # eps=1e-2 aumenta la stringenza del path di regolarizzazione
        model = LassoCV(cv=5, n_alphas=300, eps=1e-2, selection='cyclic', tol=1e-3).fit(X, y[target])
        
        coeffs = model.coef_
        
        for i, col_name in enumerate(X.columns):
            coeff = coeffs[i]
            
            # Caso A: Inerzia (Self-loop) - manteniamo un minimo se il modello lo suggerisce
            if col_name == f"{target}_L1":
                ce_matrix.loc[col_name, target] = coeff if abs(coeff) > 0.05 else 0.05
            
            # Caso B: Driver diversi - applichiamo una soglia più alta per mantenere solo relazioni forti
            else:
                # Solo se il coefficiente è "pesante" rispetto agli altri
                if abs(coeff) >= abs_coeff:
                    ce_matrix.loc[col_name, target] = coeff
                else:
                    ce_matrix.loc[col_name, target] = 0.0

    return ce_matrix

def get_drivers_wave(cause_effect_matrix, edit_drivers):
    cause_effect_matrix_appo = cause_effect_matrix.copy()
    cause_effect_matrix_appo['DRV_COD_RADIX'] = cause_effect_matrix_appo['DRV_COD'].str[:7]
    found = set(edit_drivers)
    while True:
        # Trova tutti i DRV_COD_CE collegati agli edits correnti
        new_codes = set(
            cause_effect_matrix_appo[
                (cause_effect_matrix_appo['DRV_COD_RADIX'].isin(found)) &
                (cause_effect_matrix_appo['COEF_CE'] != 0)
            ]['DRV_COD_CE'].unique()
        )
        print(new_codes)
        # Solo i codici che non sono stati già trovati
        new_codes = new_codes - found
        if not new_codes:
            break
        found.update(new_codes)

    return found
