from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from utilsforecast.preprocessing import fill_gaps


def fill_order_gaps(data, nm_unique_id, nm_ds, nm_y, last_date):
    # Rename fields into unique_id, ds, y
    data = data.rename(columns={nm_unique_id: "unique_id", nm_ds: "ds", nm_y: "y"})

    # Check for duplicates BEFORE fill_gaps
    duplicates = data.groupby(["unique_id", "ds"]).size()
    if (duplicates > 1).any():
        logger.info(f"Found {(duplicates > 1).sum()} duplicate (unique_id, ds) pairs")
        logger.info(f"Sample duplicates:\n{data[data.duplicated(subset=['unique_id', 'ds'], keep=False)].head(10)}")

    data = data.drop_duplicates(subset=["unique_id", "ds"], keep="last").reset_index(drop=True)

    # for each unique_id adds the last date avaiable
    # ANNA 20250113
    if last_date is None or last_date == "":
        last_date = str(data["ds"].max())
    else:
        last_date = str(last_date)
    df_last_date = data[["unique_id"]].drop_duplicates(ignore_index=True)
    df_last_date["ds"] = last_date
    df_last_date["ds"] = pd.to_datetime(df_last_date["ds"], format="%m%Y")
    df_last_date = df_last_date.merge(data, how="left", on=["unique_id", "ds"])

    # Apply fill_gaps

    df_complete = data[data["ds"] < pd.to_datetime(last_date, format="%m%Y")]
    df_complete = pd.concat([df_complete, df_last_date], ignore_index=True)
    df_complete = fill_gaps(df_complete, freq="MS")
    # fill missing values with zeros
    # df_complete["y"] = df_complete["y"].fillna(0) ANNA

    # Return to original column names
    df_complete = df_complete.rename(columns={"unique_id": nm_unique_id, "ds": nm_ds, "y": nm_y})
    return df_complete


def load_yml_config(path: str | Path):
    """Loads a YAML configuration file.

    Args:
        path (str | Path): The file path to the YAML config.

    Returns:
        The content of the YAML file, typically a dictionary.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    try:
        return yaml.safe_load(Path(path).read_text())

    except FileNotFoundError as error:
        message = "Error: yml config file not found."
        raise FileNotFoundError(error, message) from error


# MAPE
def MAPE(y, y_hat):
    """
    Mean Absolute Percentage Error avoiding division by zero.

    Args:
      y (array-like): actual values.
      y_hat (array-like): predicted values.

    Returns:
      float: mean absolute percentage error computed only for y != 0.
    """
    mask = y != 0
    return np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask]))


# Mape on best model
def mape_best(df_fct: pd.DataFrame, df_tst: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAPE and reconciled MAPE for test set forecasts.

    Args:
      df_fct (pd.DataFrame): forecasts containing 'yhat' and 'yhat_rec' for each unique_id/ds.
      df_tst (pd.DataFrame): test dataframe with actual 'y' values.

    Returns:
      pd.DataFrame: per-unique_id dataframe with columns ['unique_id','mape','mape_rec'].
    """
    merged_df = pd.merge(
        df_tst,  # test data with actual y
        df_fct,  # forecasted data with y_hat and yhat_rec
        on=["unique_id", "ds"],  # join keys
        how="left",  # keep all test rows
    )

    # compute MAPE and reconciled MAPE per series
    mape_df = (
        merged_df.groupby("unique_id", as_index=False)
        .apply(
            lambda g: pd.Series({"mape": MAPE(g["y"], g["yhat"]), "mape_rec": MAPE(g["y"], g["yhat_rec"])}),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    return mape_df

def remove_outliers(df, column='y', coef=1.5):
    """
    Rimuove gli outlier trasformandoli in NaN.
    Basato sulla logica del boxplot (Interquartile Range).
    """
    df = df.copy()
    # Identifica i limiti tramite boxplot_outliers
    outliers_mask = boxplot_outliers(df, column=column, coef=coef)
    
    # Applica la maschera: se è un outlier, metti NaN
    df.loc[outliers_mask, column] = np.nan
    return df

def boxplot_outliers(df, column='y', coef=1.5):
    """
    Calcola i limiti superiore e inferiore per ogni unique_id.
    """
    # Calcola i quartili Q1 (25%) e Q3 (75%) per ogni serie
    qs = df.groupby('unique_id')[column].quantile([0.25, 0.75]).unstack()
    iqr = qs[0.75] - qs[0.25]
    
    # Definisce i "baffi" del boxplot
    low = qs[0.25] - coef * iqr
    high = qs[0.75] + coef * iqr
    
    # Riaggancia i limiti al DataFrame originale per il confronto riga per riga
    df = df.merge(low.rename('low'), on='unique_id', how='left')
    df = df.merge(high.rename('high'), on='unique_id', how='left')
    
    # Restituisce una maschera booleana (True dove è outlier)
    return (df[column] < df['low']) | (df[column] > df['high'])