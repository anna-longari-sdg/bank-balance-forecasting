from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from hierarchicalforecast.utils import aggregate
from loguru import logger

from src import SRC_DIR
from src.back.preprocess import analize_zero_freq
from src.db.db import DB

DATA_DIR = SRC_DIR / "data"

_FILL_STRATEGY_TYPE = Optional[Literal["zero"]]

_TABLES_TO_BE_SAVED = [
    ("driver_best_model_df", "DRV_BEST_MODEL"),
    ("driver_selected_df", "DRV_SEL"),
    ("drivers_fit_df", "DRV_FIT"),
    ("drivers_forecast_df", "DRV_FORECAST"),
    ("eco_best_model_df", "ECO_BEST_MODEL"),
    ("eco_fit_df", "ECO_FIT"),
    ("eco_forecast_df", "ECO_FORECAST"),
    ("rec_forecast_df", "ECO_FORECAST_REC"),
    ("eco_group_df", "ECO_GROUP_VAL"),
    ("eco_group_fit_df", "ECO_GROUP_FIT"),
    ("cause_effect_df", "CAUSE_EFFECT"),
]


class Anagraphics:
    def __init__(self, data_folder: str | Path, data_filename: str, anag_sheet_name: str) -> None:
        if isinstance(data_folder, str):
            data_folder = Path(data_folder)
        self.xlsl = pd.ExcelFile(data_folder / "tracciato_record.xlsx")
        self.df = pd.read_csv(data_folder / data_filename, delimiter=";")
        self.anag_sheet_name = anag_sheet_name

    @property
    def desc(self) -> pd.DataFrame:
        return pd.read_excel(self.xlsl, sheet_name=self.anag_sheet_name)


def remove_corr_by_target(X, Y, corr_threshold=0.95):
    Y = pd.Series(Y, index=X.index, name="Target")
    Y = Y.fillna(0.0)
    X = X.fillna(0.0)
    corr_with_target = X.corrwith(Y).abs()
    corr_matrix_X = X.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix_X.shape), k=1).astype(bool)
    high_corr_pairs = corr_matrix_X.where(upper_triangle)
    to_drop = set()
    for col_i in high_corr_pairs.columns:
        correlated_cols_j = high_corr_pairs[col_i][high_corr_pairs[col_i] > corr_threshold].index.tolist()
        for col_j in correlated_cols_j:
            if col_i not in to_drop and col_j not in to_drop:
                corr_i_Y = corr_with_target.get(col_i, 0)
                corr_j_Y = corr_with_target.get(col_j, 0)
                if corr_i_Y < corr_j_Y:
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)
    return to_drop


def split_dataset(
    df: pd.DataFrame, 
    h: int | None = None, 
    cutoff_month: str | None = None, 
    time_col: str = "ds",  
    val_col = "y", 
    group_col="unique_id", 
    fillna_method: str = "zero"
    ):
    if h is None and cutoff_month is None:
        raise ValueError("One of 'h' and 'cutoff_month' must be set.")
    if h is not None:
        ds_split = df.loc[:, time_col].max() - pd.DateOffset(months=h)
    else:
        ds_split = pd.to_datetime(f"{cutoff_month}", format="%m%Y")
    df_trn = df.loc[df[time_col] <= ds_split, :].copy()
    df_tst = df.loc[df[time_col] > ds_split, :].copy()
    # Linear interpolation
    if fillna_method == "interpolate":
        df_trn[val_col] = df_trn.groupby(group_col)[val_col].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )
    elif fillna_method == "zero":
        df_trn[val_col] = df_trn[val_col].fillna(0.0)
    return df_trn, df_tst


class DataLoader:
    def __init__(
        self, db_path: str | Path, cut_missing: float | None = 0.8
    ) -> None:
        if isinstance(db_path, str):
            db_path = Path(db_path)
        self.db_path = db_path
        db = DB(path=self.db_path)
        self.eco_df = db.predefined_query("all_val_economics", fill_gaps="ECO_COD")
        self.driver_df = db.predefined_query("all_val_drivers", fill_gaps="DRV_COD")
        self.mapped_df = db.predefined_query("all_map_economics_drivers")
        self.driver_anag = db.predefined_query("all_anag_drivers")
        self.eco_anag = db.predefined_query("all_anag_economics")
        self.eco_budget = db.predefined_query("all_budget_economics")
        self.cut_missing = cut_missing
        self.driver_best_model_df = self._try_load_table(db, "all_driver_best_model")
        self.driver_best_stat_model_df = pd.DataFrame()
        self.driver_selected_df = self._try_load_table(db, "all_driver_selected")
        self.drivers_fit_df = self._try_load_table(db, "all_driver_fit")
        self.drivers_forecast_df = self._try_load_table(db, "all_driver_forecast")
        self.eco_best_model_df = self._try_load_table(db, "all_eco_best_model")
        self.eco_best_stat_model_df = pd.DataFrame()
        self.eco_fit_df = self._try_load_table(db, "all_eco_fit")
        self.eco_forecast_df = self._try_load_table(db, "all_eco_forecast")
        self.rec_forecast_df = self._try_load_table(db, "all_forecast_rec")        
        self.eco_group_df = self._try_load_table(db, "all_val_group")
        self.eco_group_fit_df = self._try_load_table(db, "all_group_fit")
        if self.cut_missing:
            self._cut_missing_ecocodes(self.cut_missing)
        self.cause_effect_df = self._try_load_table(db, "all_cause_effect")

    def _try_load_table(self, db: DB, query_name: str) -> pd.DataFrame | None:
        try:
            df = db.predefined_query(query_name)
            return df
        except Exception:
            logger.warning(f"Could not load table {query_name}: setting to None.")
            return None

    def set_lang(self, lang: str = 'it'):       
        dsc_cols_drv_anag = [col for col in self.driver_anag.columns if col.endswith("_DSC")]
        dsc_cols_eco_anag = [col for col in self.eco_anag.columns if col.endswith("_DSC")]

        if any(col.endswith(f"DRV_DSC_{lang.upper()}") for col in self.driver_anag.columns):
            for col in dsc_cols_drv_anag:
                self.driver_anag[col] = self.driver_anag[f"{col}_{lang.upper()}"]
            for col in dsc_cols_eco_anag:
                self.eco_anag[col] = self.eco_anag[f"{col}_{lang.upper()}"]
        else:
            raise ValueError("Unsupported language") 

    def save_table(self, data_attribute: str, table_name: str | None = None):
        if not hasattr(self, data_attribute):
            raise ValueError(f"No attribute named {data_attribute}")

        data = getattr(self, data_attribute)
        # if not isinstance(data, pd.DataFrame):
        #     raise ValueError(f"Selected data attribute {data_attribute} is not a dataframe")
        if isinstance(data, pd.DataFrame):
            db = DB()
            db.write_df(data, table_name=table_name if table_name is not None else data_attribute.upper())

    def save_all(self):
        for data_attribute, table_name in _TABLES_TO_BE_SAVED:
            if hasattr(self, data_attribute):
                self.save_table(data_attribute, table_name)
                

    def set_data_attribute(self, df: pd.DataFrame, name: str):
        setattr(self, name, df)

    def _cut_missing_ecocodes(self, threshold: float) -> None:
        df_drivers_zeros = analize_zero_freq(self.driver_df, "DRV_COD", "DATE_RIF", "VALUE")
        self.drivers_to_cut = df_drivers_zeros[
            (df_drivers_zeros["perc_zeros"] >= threshold) | (df_drivers_zeros["perc_zeros_last_year"] >= threshold)
        ]["DRV_COD"].tolist()
        logger.info(f"Cutting {len(self.drivers_to_cut)} drivers due to high missing values")
        df_eco_zeros = analize_zero_freq(self.eco_df, "ECO_COD", "DATE_RIF", "VALUE")
        # ANNA provo a togliere il filtro sui valori dell'ultimo anno         
        # self.eco_to_cut = df_eco_zeros[  
            #  (df_eco_zeros["perc_zeros"] >= threshold) | (df_eco_zeros["perc_zeros_last_year"] >= threshold)]["ECO_COD"].tolist()
        self.eco_to_cut = df_eco_zeros[(df_eco_zeros["perc_zeros"] >= threshold)]["ECO_COD"].tolist()
        logger.info(f"Cutting {len(self.eco_to_cut)} eco due to high missing values")
        self.driver_anag = self.driver_anag.loc[~self.driver_anag["DRV_COD"].isin(self.drivers_to_cut), :]
        self.eco_anag = self.eco_anag.loc[~self.eco_anag["ECO_COD"].isin(self.eco_to_cut), :]
        self.driver_df = self.driver_df.loc[~self.driver_df["DRV_COD"].isin(self.drivers_to_cut), :]
        self.eco_df = self.eco_df.loc[~self.eco_df["ECO_COD"].isin(self.eco_to_cut), :]
        self.mapped_df = self.mapped_df.loc[~self.mapped_df["DRV_COD"].isin(self.drivers_to_cut), :]
        self.mapped_df = self.mapped_df.loc[~self.mapped_df["ECO_COD"].isin(self.eco_to_cut), :]

    def reload_mapping(self):
        db = DB(path=self.db_path)
        self.mapped_df = db.predefined_query("all_map_economics_drivers")
        if self.cut_missing:
            self.mapped_df = self.mapped_df.loc[~self.mapped_df["DRV_COD"].isin(self.drivers_to_cut), :]
            self.mapped_df = self.mapped_df.loc[~self.mapped_df["ECO_COD"].isin(self.eco_to_cut), :]

    def generate_driver_selection_dataset(
        self,
        eco_code: str,
        score_column: Literal["mape_L2", "mape_mean"],
        score_threshold: float,
        fillnaX_strategy: _FILL_STRATEGY_TYPE = "zero",
        fillnaY_strategy: _FILL_STRATEGY_TYPE = "zero",
        null_counts_threshold=0.5,
        corr_threshold=0.9,
        cutoff_month: str | None = None,
    ):
        if self.driver_best_model_df is None:
            raise ValueError("driver_best_model_df not set. Please run driver model selection first.")
        # db = DB(path=self.db_path)
        drivers_of_eco_code_df = self.mapped_df.query("ECO_COD == @eco_code").merge(
            self.driver_best_model_df,
            on="DRV_COD",
            how="left",
        )
        logger.info(
            f"Removed {len(drivers_of_eco_code_df.loc[drivers_of_eco_code_df[score_column] > score_threshold, 'DRV_COD'].tolist())} with bad forecast."
        )
        drivers_of_eco_code = drivers_of_eco_code_df.loc[
            drivers_of_eco_code_df[score_column] <= score_threshold, "DRV_COD"
        ].tolist()
        relevant_drivers = self.driver_df.loc[self.driver_df["DRV_COD"].isin(drivers_of_eco_code), :]
        if len(drivers_of_eco_code) > 0:
            logger.info(f"ECO_COD {eco_code} has {len(drivers_of_eco_code)} potential drivers.")
        X_df_raw = relevant_drivers.pivot_table(index="DATE_RIF", columns="DRV_COD", values="VALUE")
        # Select only time interval relevant for the eco_code
        relevant_timestamps_df = self.eco_df.query("ECO_COD == @eco_code")[["DATE_RIF", "VALUE"]].drop_duplicates()
        relevant_timestamps_df = (
            relevant_timestamps_df[relevant_timestamps_df["DATE_RIF"] <= cutoff_month]
            if cutoff_month is not None
            else relevant_timestamps_df
        )
        logger.info(f"ECO_COD {eco_code} has {len(relevant_timestamps_df)} relevant timestamps.")
        X_df = (
            relevant_timestamps_df.loc[:, ["DATE_RIF"]]
            .merge(
                X_df_raw,
                on="DATE_RIF",
                how="left",  # Keep all timestamps from the eco_df
            )
            .set_index("DATE_RIF")
        )
        # remove timestamps with all drivers missing
        empty_timestamps = X_df[X_df.isna().all(axis=1)].index.tolist()
        if len(empty_timestamps) > 0:
            logger.info(f"ECO_COD {eco_code} has {len(empty_timestamps)} timestamps with all drivers missing.")
            relevant_timestamps_df = relevant_timestamps_df.loc[
                ~relevant_timestamps_df.loc[:, "DATE_RIF"].isin(empty_timestamps)
            ]
        X_df = X_df.merge(relevant_timestamps_df[["DATE_RIF"]], on="DATE_RIF", how="inner")
        # remove drivers with too many nulls
        null_counts = X_df.filter(regex="^DRV_").isnull().sum() / X_df.shape[0]
        drv_many_nulls = null_counts[null_counts > null_counts_threshold].index
        X_df = X_df.drop(columns=drv_many_nulls)
        drivers_of_eco_code = sorted(list(set(drivers_of_eco_code) - set(drv_many_nulls)))
        if len(drv_many_nulls) > 0:
            logger.info(f"Removed {len(drv_many_nulls)} drivers with many nulls {drv_many_nulls.to_list()}.")
        # remove drivers correlated
        to_drop = remove_corr_by_target(
            X_df, relevant_timestamps_df.loc[:, "VALUE"].to_numpy(dtype=np.float64), corr_threshold=corr_threshold
        )
        if len(to_drop) > 0:
            drivers_of_eco_code = sorted(list(set(drivers_of_eco_code) - set(to_drop)))
            X_df = X_df.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} correlated drivers {to_drop}.")
        # fill na
        if "DATE_RIF" in X_df.columns:
            X_df = X_df.drop(columns="DATE_RIF")
        if fillnaX_strategy == "zero":
            X = X_df.fillna(0.0).to_numpy(dtype=np.float64)
        elif fillnaX_strategy == "interpolate":
            X_interp = X_df.apply(lambda x: x.interpolate(method="linear").ffill().bfill(), axis=0)
            X_interp = X_interp.combine_first(X_df)
            X = X_interp.to_numpy(dtype=np.float64)
        else:
            X = X_df.to_numpy(dtype=np.float64)
        if fillnaY_strategy == "zero":
            Y = relevant_timestamps_df.loc[:, "VALUE"].fillna(0.0)
        elif fillnaY_strategy == "interpolate":
            Y = relevant_timestamps_df.loc[:, "VALUE"].transform(
                lambda x: x.interpolate(method="linear").ffill().bfill()
            )
        else:
            Y = relevant_timestamps_df.loc[:, "VALUE"]
        return X, Y, list(drivers_of_eco_code)

    def generate_economics_forecast_dataset(self, eco_code: str, driver_model_selection):
        eco_values_df = self.eco_df.query("ECO_COD == @eco_code")
        if len(eco_values_df) == 0:
            raise ValueError(f"No values found for {eco_code}")
        ## TO DO: get selection from db

        model_selection_df = self.get_model_selection_dataset(driver_model_selection)
        model_selection_df = model_selection_df[model_selection_df["ECO_COD"] == eco_code].reset_index(drop=True)
        drivers_for_eco = self.mapped_df.query("ECO_COD == @eco_code")
        selected_drivers_for_eco = drivers_for_eco[drivers_for_eco["DRV_COD"].isin(model_selection_df["DRV_COD"])].loc[
            :, ["ECO_COD", "DRV_COD"]
        ]
        if len(selected_drivers_for_eco) == 0:
            eco_values_df = eco_values_df.rename(columns={"ECO_COD": "unique_id", "DATE_RIF": "ds", "VALUE": "y"})
            return eco_values_df.fillna(0.0).reset_index(drop=True), None
        logger.info(f"Eco {eco_code} has selected drivers {selected_drivers_for_eco['DRV_COD'].values}")
        driver_value_df = self.driver_df.loc[self.driver_df["DRV_COD"].isin(selected_drivers_for_eco["DRV_COD"]), :]
        eco_driver_df = eco_values_df.merge(selected_drivers_for_eco, on="ECO_COD", how="left")
        eco_driver_df = eco_driver_df.merge(
            driver_value_df, on=["DRV_COD", "DATE_RIF"], how="left", suffixes=("_ECO", "_DRV")
        )
        eco_driver_df.loc[:, ["VALUE_ECO", "VALUE_DRV"]] = (
            eco_driver_df.loc[:, ["VALUE_ECO", "VALUE_DRV"]].astype(np.float64).fillna(0.0)
        )
        eco_driver_df = (
            eco_driver_df.pivot_table(index=["ECO_COD", "DATE_RIF", "VALUE_ECO"], columns="DRV_COD", values="VALUE_DRV")
            .reset_index()
            .rename(columns={"ECO_COD": "unique_id", "DATE_RIF": "ds", "VALUE_ECO": "y"})
            .rename_axis(index=None, columns=None)
        )
        if self.drivers_forecast_df is None:
            raise ValueError("driver_forecast is not set.")
        drivers_future_df = self.drivers_forecast_df[
            self.drivers_forecast_df["DRV_COD"].isin(selected_drivers_for_eco["DRV_COD"])
        ]
        drivers_future_df.loc[:, "FORECAST"] = drivers_future_df.loc[:, "FORECAST"].astype(np.float64).fillna(0.0)
        drivers_future_df = (
            drivers_future_df.pivot_table(index="DATE_RIF", columns="DRV_COD", values="FORECAST")
            .reset_index()
            .rename(columns={"DATE_RIF": "ds"})
            .rename_axis(index=None, columns=None)
        )
        drivers_future_df["unique_id"] = eco_code
        return eco_driver_df, drivers_future_df

    def get_model_selection_dataset(self, model_selection: str) -> pd.DataFrame:
        df = self.driver_selected_df[self.driver_selected_df["MODEL"] == model_selection].reset_index(drop=True)
        match model_selection:
            case "lasso":
                if df.empty:
                    raise ValueError("eco_lasso_df not set. Please run eco lasso model selection first.")
            case "lasso_lars":
                if df.empty:
                    raise ValueError("eco_lasso_lars_df not set. Please run eco lasso model selection first.")
            case "xgboost":
                if df.empty:
                    raise ValueError("eco_xgboost_df not set. Please run eco lasso model selection first.")
            case _:
                raise ValueError("Wrong model_selection param")
        return df

    def __get_dataset(
        self,
        data_source: Literal["eco", "driver", "driver_selected", "eco_aggregated"],
        query_args: dict | None = None,
    ) -> pd.DataFrame:
        match data_source:
            case "eco":
                df = self.eco_df
            case "driver":
                df = self.driver_df
            case "driver_selected":
                if query_args is None:
                    raise ValueError("query_args must be provided for 'driver_selected' data_source")
                if self.driver_best_model_df is None:
                    raise ValueError("driver_best_model_df not set. Please run driver model selection first.")
                model_selection_df = self.get_model_selection_dataset(query_args["model_selection"])
                drivers_above_threshoold = self.driver_best_model_df[
                    self.driver_best_model_df[query_args["score_column"]] <= query_args["score_threshold"]
                ]["DRV_COD"]
                selected_drivers = model_selection_df[model_selection_df["DRV_COD"].isin(drivers_above_threshoold)]
                df = self.driver_df.merge(
                    selected_drivers.loc[:, ["DRV_COD"]].drop_duplicates().reset_index(drop=True),
                    on="DRV_COD",
                    how="inner",
                )
            case "eco_aggregated":
                df = self.eco_df_aggregate
            case _:
                raise ValueError("data_source must be either 'eco' or 'driver'")
        df = df.rename(columns={"ECO_COD": "unique_id", "DRV_COD": "unique_id", "DATE_RIF": "ds", "VALUE": "y"})
        return df

    def eco_dataset(self) -> pd.DataFrame:
        return self.__get_dataset(data_source="eco")

    def driver_dataset(self, selected=True, query_args: dict | None = None) -> pd.DataFrame:
        return self.__get_dataset(
            data_source="driver_selected" if selected else "driver",
            query_args=query_args,
        )

    def eco_aggregated_dataset(self) -> pd.DataFrame:
        return self.__get_dataset(data_source="eco_aggregated")


# Prepare data
# need to be arranged for each dataset
def custom_aggregate(df):
    """
    Prepare input dataframe for hierarchical forecasting.

    Actions:
      - Ensures 'ds' is datetime.
      - Adds a top-level 'Total' column with value 'TOTAL'.
      - Builds hierarchical aggregation spec and returns aggregated frames.

    Args:
      df (pd.DataFrame): input with at least columns ['unique_id','ds','y','id'].

    Returns:
      tuple: (df_aggregated (pd.DataFrame), S_df (pd.DataFrame), tags (dict))
    """
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%d")

    # Build the hierarchy at runtime: a single TOTAL top level and bottoms by id
    df["Total"] = "TOTAL"
    spec = [
        ["Total"],  # top level (TOTAL)
        ["Total", "id"],  # bottom level (each series)
    ]

    # Aggregate series according to spec -> returns aggregated df, S matrix, and tags
    df, S_df, tags = hf_aggregate(df=df, spec=spec)

    return df, S_df, tags


# split trn & tst
# split
def split(df, h):
    """
    split dataframe into training and test sets by time.

    args:
      df (pd.dataframe): dataframe with 'ds' datetime column.
      h (int): number of periods (months) to keep as test.

    returns:
      tuple: (df_trn (pd.DataFrame), df_tst (pd.DataFrame))
    """
    # compute cutoff timestamp by subtracting h months from the max date
    ds_split = df["ds"].max() - pd.DateOffset(months=h)
    df_trn = df[df["ds"] <= ds_split]
    df_tst = df[df["ds"] > ds_split]

    return df_trn, df_tst
