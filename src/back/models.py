from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from statsforecast.core import StatsForecast
from statsforecast.models import HistoricAverage

from src.back.configuration import get_instantiated_nixtla_models
from src.back.validation import cross_validation_stats

# __MODEL_DICT_BACK = {
#     "AutoARIMA12": AutoARIMA(season_length=12),
#     "AutoARIMA01": AutoARIMA(season_length=1),
#     "HistoricAverage": HistoricAverage(),
#     "AutoETS12": AutoETS(season_length=12),
#     "AutoETS01": AutoETS(season_length=1),
# }
#


class ForecastingModel:
    def __init__(self, configuration_path: str | Path, freq: int | str, time_col="ds", value_col="y") -> None:
        self.__models = get_instantiated_nixtla_models(configuration_path)
        self.freq = freq
        self.time_col = time_col
        self.value_col = value_col

    def fit(self, X: pd.DataFrame, h: int, n_windows: int, step_size: int):
        # Identify outliers with boxplot method
        import pandas as pd       

        X = (
            X
            .groupby("unique_id", group_keys=False)
            .apply(lambda x: impute_outliers_stl(x, "y", period=12, method="median"))
            .reset_index(drop=True)
        )
        
        ## cross validation
        sf = StatsForecast(
            models=self.__models,
            freq=self.freq,
            fallback_model=HistoricAverage(),
            n_jobs=-1,
        )
        # Check if we need to add small constant for non division by zero
        try:
            df_cv = sf.cross_validation(
                df=X, h=h, step_size=step_size, n_windows=n_windows, time_col=self.time_col, target_col=self.value_col
            )

        except ValueError as e:
            # check if error is due to short series
            series_sizes = np.diff(sf.ga.indptr)
            test_size = h + step_size * (n_windows - 1)
            short_series = series_sizes <= test_size
            if short_series.any():
                uids = pd.Series(sf.uids)
                short_ids = uids.loc[short_series].to_numpy().tolist()
                logger.error(f"Test size {test_size} is too large for some series: {short_ids}")
                return
            else:
                raise e

        # subtract constant
        # cross validation stats
        self.df_cv_stat = cross_validation_stats(pd.DataFrame(df_cv), time_col=self.time_col, value_col=self.value_col)
        if self.df_cv_stat.loc[:, "mape_L2"].isnull().all():
            raise ValueError(f"Impossible to compute MAPE L2 for dataset containing {X['unique_id'].unique()}")
        model_priority = [str(m) for m in self.__models]
        priority_map = {name: i for i, name in enumerate(model_priority)}
        self.df_best_model = (
            self.df_cv_stat
            .assign(priority=self.df_cv_stat["model"].map(priority_map))
            .sort_values(by=["unique_id", "mape_L2", "priority"], ascending=True)
            .drop_duplicates(subset=["unique_id"], keep="first")
            .drop(columns=["priority"])
            .reset_index(drop=True)
        )
        # self.df_best_cv = df_cv[df_cv["model"].isin(self.df_best_model["model"].unique())]

    @property
    def best_model_df(self) -> pd.DataFrame:
        if not hasattr(self, "df_best_model"):
            raise AttributeError("Model has not been fitted yet. Call fit() before accessing best_model_df.")
        return self.df_best_model

    @property
    def cv_stats_df(self) -> pd.DataFrame:
        if not hasattr(self, "df_cv_stat"):
            raise AttributeError("Model has not been fitted yet. Call fit() before accessing cv_stats_df.")
        return self.df_cv_stat

    def forecast_best(
        self,
        df_trn: pd.DataFrame,
        h: int,
        table: Literal["eco", "driver", "aggregate"],
        best_model_df: pd.DataFrame | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_trn = (
            df_trn
            .groupby("unique_id", group_keys=False)
            .apply(lambda x: impute_outliers_stl(x, "y", period=12, method="median"))
            .reset_index(drop=True)
        )
        forecasts = []
        fitted_values = []

        for uid in df_trn["unique_id"].unique():
            if table == "driver":
                result = best_model_df.query("DRV_COD == @uid")
                if len(result) == 0:
                    raise ValueError(f"No model found for {uid}")
                model_name = result.iloc[0]["model"]
            elif table == "eco":
                result = best_model_df.query("ECO_COD == @uid")
                if len(result) == 0:
                    # To DO: better fix
                    # raise ValueError(f"No model found for {uid}")
                    logger.warning(f"No model found for {uid}")
                    continue
                model_name = result.iloc[0]["model"]
            elif table == "aggregate":
                result = best_model_df.query("unique_id == @uid")
                if len(result) == 0:
                    raise ValueError(f"No model found for {uid}")
                model_name = result.iloc[0]["model"]                
            else:
                raise ValueError("Wrong table.")
            logger.info(f"Forecasting series {uid} using its best model {result['model'].values[0]}")
            # instantiate the correct model (object from model_dict)
            m = next(filter(lambda mod: mod.alias == model_name, self.__models))

            # subset this series for fitting/forecasting
            ser = df_trn.loc[df_trn["unique_id"] == uid, :]

            # fit & forecast on single-series with StatsForecast
            sf = StatsForecast(models=[m], freq=self.freq, n_jobs=1)

            fcst = sf.forecast(
                df=ser, h=h, fitted=True, time_col=self.time_col, target_col=self.value_col, X_df=X_df
            ).reset_index(drop=True)  # type: ignore
            fit = sf.forecast_fitted_values().reset_index(drop=True)  # type: ignore

            # find the forecast column name created by the model and normalize to 'yhat'
            pred_col = [c for c in fcst.columns if c not in {"unique_id", self.time_col}][0]
            fcst = fcst.rename(columns={pred_col: f"{self.value_col}_hat"})
            fcst["model"] = model_name

            forecasts.append(
                fcst[["unique_id", self.time_col, f"{self.value_col}_hat"]]  # only keep standardized columns
            )

            # normalize fitted column to 'yhat' as well
            fit_col = [c for c in fit.columns if c not in {"unique_id", self.time_col, self.value_col}][0]
            fit = fit.rename(columns={fit_col: f"{self.value_col}_hat"})

            fitted_values.append(fit[["unique_id", self.time_col, self.value_col, f"{self.value_col}_hat"]])

        # combine results after loop
        if len(forecasts) > 1:
            df_fct = pd.concat(forecasts, ignore_index=True)  # future forecasts
        else:
            df_fct = forecasts[0]
        if len(fitted_values) > 1:
            df_fit = pd.concat(fitted_values, ignore_index=True)  # in-sample actuals + fitted
        else:
            df_fit = fitted_values[0]

        return df_fit, df_fct


# init models
def init_model(model_dict, freq):
    """
    initialize a statsforecast instance from a mapping of model aliases to model instances.

    side-effects:
      - sets model.alias = alias for every model in model_dict so the model outputs
        are tagged with readable names.

    args:
      model_dict (dict): mapping alias (str) -> model instance (e.g. autoarima(...)).
      freq (str): frequency string expected by statsforecast (e.g. 'ms', 'd').

    returns:
      statsforecast: instantiated statsforecast configured with the provided models.
    """
    # ensure every model carries its alias for clear identification in cv / forecasts.

    # instantiate statsforecast with the provided models.
    # - fallback_model: used when a model fails for a series
    # - n_jobs: parallelism (-1 uses all available cores)
    sf = StatsForecast(
        models=list(model_dict.values()),
        freq=freq,
        fallback_model=HistoricAverage(),
        n_jobs=-1,
    )

    return sf


# best model
def best_model(df, sf, h, n_windows, step_size):
    """
    run cross-validation with a statsforecast instance and select the best model per series.

    steps:
      - runs sf.cross_validation on df with provided horizon and windows.
      - computes cross-validation statistics (mape-based).
      - picks per-unique_id the model with minimum mape_l2.

    args:
      df (pd.dataframe): training dataframe used for cross-validation.
      sf (statsforecast): configured statsforecast instance (models must have .alias).
      h (int): forecast horizon used in cv.
      n_windows (int): number of cv windows.
      step_size (int): cv step size (months/periods).

    returns:
      tuple: (df_best_model (pd.dataframe), df_cv_stat (pd.dataframe))
        - df_best_model: one row per unique_id with chosen model.
        - df_cv_stat: per-model cv statistics used for selection.
    """
    ## cross validation
    df_cv = sf.cross_validation(df=df, h=h, step_size=step_size, n_windows=n_windows)

    # cross validation stats
    df_cv_stat = cross_validation_stats(df_cv)

    # best model per series by mape_l2
    df_best_model = df_cv_stat.loc[df_cv_stat.groupby("unique_id")["mape_L2"].idxmin()].reset_index(drop=True)

    return df_best_model, df_cv_stat


# make model
def make_model(name: str, model_dict):
    """
    Return a model instance from a model dictionary by name.

    Args:
      name (str): alias/key for model in model_dict.
      model_dict (dict): mapping alias -> model instance.

    Returns:
      model instance: the model object (not fitted).
    """
    best_model = model_dict[name]
    return best_model


# Forecast Best Model
def forecast_best(df_trn, df_best_model, model_dict, h, freq):
    """
    Fit and forecast each series with its selected best model.

    Steps per series:
      - Instantiate the selected model.
      - Fit using StatsForecast on the single-series dataframe.
      - Retrieve h-step forecasts and in-sample fitted values.
      - Normalize forecast/fitted column names to 'yhat'.

    Args:
      df_trn (pd.DataFrame): training dataframe with columns ['unique_id','ds','y'].
      df_best_model (pd.DataFrame): dataframe with columns ['unique_id','model'] selecting model per series.
      model_dict (dict): mapping model alias -> model instance.
      h (int): forecast horizon.
      freq (str): frequency string for StatsForecast.

    Returns:
      tuple: (df_fit (pd.DataFrame), df_fct (pd.DataFrame))
        - df_fit: in-sample actuals + fitted values with columns ['unique_id','ds','y','yhat'].
        - df_fct: future forecasts with columns ['unique_id','ds','yhat'].
    """
    # Identifica e pialla outliers con boxplot method
    df_trn = (
        df_trn
        .groupby("unique_id", group_keys=False)
        .apply(lambda x: impute_outliers_stl(x, "y", period=12, method="median"))
        .reset_index(drop=True)
    )
    forecasts = []
    fitted_values = []

    for _, row in df_best_model.iterrows():
        uid = row["unique_id"]
        model_name = row["model"]

        # instantiate the correct model (object from model_dict)
        m = make_model(model_name, model_dict)

        # subset this series for fitting/forecasting
        ser = df_trn.loc[df_trn["unique_id"] == uid, ["unique_id", "ds", "y"]]

        # fit & forecast on single-series with StatsForecast
        sf = StatsForecast(models=[m], freq=freq, n_jobs=1)
        fcst = sf.forecast(df=ser, h=h, fitted=True).reset_index(drop=True)  # type: ignore
        fit = sf.forecast_fitted_values().reset_index(drop=True)  # type: ignore

        # find the forecast column name created by the model and normalize to 'yhat'
        pred_col = [c for c in fcst.columns if c not in {"unique_id", "ds"}][0]
        fcst = fcst.rename(columns={pred_col: "yhat"})
        fcst["model"] = model_name

        forecasts.append(
            fcst[["unique_id", "ds", "yhat"]]  # only keep standardized columns
        )

        # normalize fitted column to 'yhat' as well
        fit_col = [c for c in fit.columns if c not in {"unique_id", "ds", "y"}][0]
        fit = fit.rename(columns={fit_col: "yhat"})

        fitted_values.append(fit[["unique_id", "ds", "y", "yhat"]])

    # combine results after loop
    df_fct = pd.concat(forecasts, ignore_index=True)  # future forecasts
    df_fit = pd.concat(fitted_values, ignore_index=True)  # in-sample actuals + fitted

    return df_fit, df_fct

def impute_outliers_iqr(df, value_col, method="median"):
        """
        Sostituisce gli outlier identificati con IQR.
        method: "median" oppure "interpolate"
        """
        Q1 = df[value_col].quantile(0.05)
        Q3 = df[value_col].quantile(0.95)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df[value_col] < lower) | (df[value_col] > upper)      
        logger.info(f"{df['unique_id'][0]}: identified {outliers.sum()} outliers on {len(df)} obs. ({outliers.mean():.2%})")
         
        if method == "median":
            df.loc[outliers, value_col] = df[value_col].median()
        elif method == "interpolate":
            # mette NaN agli outlier e poi interpola
            df.loc[outliers, value_col] = pd.NA
            df[value_col] = df[value_col].interpolate(method="linear")
        else:
            raise ValueError("method deve essere 'median' o 'interpolate'")        
        return df

def impute_outliers_stl(df, value_col, period, method="median"):
    from statsmodels.tsa.seasonal import STL
    
    df = df.copy()
    # Esegui la decomposizione
    res = STL(df[value_col], period=period, robust=True).fit()
    
    # Calcola IQR solo sui residui (rimanenze)
    resid = res.resid
    q1 = resid.quantile(0.05)
    q3 = resid.quantile(0.95)
    iqr = q3 - q1
    
    lower = q1 - 3 * iqr # Più permissivo (3 invece di 1.5)
    upper = q3 + 3 * iqr
    
    outliers = (resid < lower) | (resid > upper)
    logger.info(f"{df['unique_id'].unique().item()}: identified {outliers.sum()} outliers on {len(df)} obs. ({outliers.mean():.2%})")
    
    if method == "median":
        # Sostituiamo il valore originale con: (Trend + Stagionalità + Mediana Residuo)
        # Questo preserva il picco stagionale atteso!
        df.loc[outliers, value_col] = res.trend[outliers] + res.seasonal[outliers] + resid.median()
    
    return df
