# ┌                                                                              ┐
# │ Example of forecast pipeline for bank FCS data                               │
# └                                                                              ┘
# ━━ imports ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from hierarchicalforecast.utils import aggregate
from joblib import Parallel, delayed
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.back.configuration import PIPELINE_DEFAULT_CONFIGURATION as config
from src.back.data import DataLoader, split_dataset
from src.back.driver_selection import DriverSelector
from src.back.models import ForecastingModel
from src.back.plots import plot_generic_result
from src.back.reconciliation import Reconciliation

# ━━ constants ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT_DIR = Path(__file__).resolve().parents[2]
MODE = os.environ.get("MODE", "dev")
if MODE not in ["dev", "prod"]:
    raise ValueError("Invalid MODE. Expected 'dev' or 'prod'.")
TP_SOURCE = os.environ["TP_SOURCE"]
DATA_DIR = ROOT_DIR / "src" / "data" / TP_SOURCE
DB_DIR = ROOT_DIR / "src" / "db" / TP_SOURCE
__EXCEL_SHEETS = ["ANAG Economics", "VAL Economics", "ANAG Drivers", "VAL Drivers", "MAP Economics-Drivers"]
DB_PATH = DB_DIR / "bankfcs_anonymized.db"
# ANNA 20250119 added CONFIG_META_PATH
CONFIG_META_PATH = ROOT_DIR / "src" / "config_store"
CONFIG_STORE_PATH = ROOT_DIR / "src" / "config_store" / TP_SOURCE
DRIVERS_MODEL_CONFIG_PATH = CONFIG_STORE_PATH / "model_drivers.yaml"
ECONOMICS_MODEL_CONFIG_PATH = CONFIG_STORE_PATH / "model_eco.yaml"
DRIVER_SELECTION_CONFIG_PATH = CONFIG_STORE_PATH / "driver_selection.yaml"
RECONCILIATION_CONFIG_PATH = CONFIG_STORE_PATH / "reconciliation.yaml"
RECONCILIATION_APP_CONFIG_PATH = CONFIG_STORE_PATH / "reconciliation_app.yaml"
today = datetime.today().strftime("%Y-%m-%d")
JSON_FILE_PATH_LARS = Path(f"./{today}/results_lasso_lars.json").resolve()
JSON_FILE_PATH_LARS.parent.mkdir(parents=True, exist_ok=True)
JSON_FILE_PATH = Path(f"./{today}/results_lasso.json").resolve()
JSON_FILE_PATH_XGBOOST = Path(f"./{today}/results_xgboost.json").resolve()
FIT_PATH = Path(f"./{today}/fit_{config.driver_selection.score_column}.csv").resolve()
FORECAST_PATH = Path(f"./{today}/forecast_{config.driver_selection.score_column}.csv").resolve()
FIT_PATH_ECO = Path(f"./{today}/fit_eco.csv").resolve()
FORECAST_PATH_ECO = Path(f"./{today}/forecast_eco.csv").resolve()

logger_format = "[<level>{level: ^12}</level>] <level>{message}</level>"
logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=""), format=logger_format, colorize=True)])

if config.driver_selection.model_selection == "lasso":
    JSON_FILE_PATH = JSON_FILE_PATH
elif config.driver_selection.model_selection == "lasso_lars":
    JSON_FILE_PATH = JSON_FILE_PATH_LARS
elif config.driver_selection.model_selection == "xgboost":
    JSON_FILE_PATH = (
        JSON_FILE_PATH_XGBOOST  # ┌                                                                              ┐
    )


# │ 1. Driver model selection (D3)                                                     │
# └                                                                              ┘
def driver_best_model_search(data: DataLoader, model: ForecastingModel):
    df = data.driver_dataset(selected=False)
    df_train, _ = split_dataset(df, cutoff_month=config.dataset.cutoff_month, fillna_method="zero")
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.simplefilter("ignore")
    model.fit(X=df_train, h=config.cv.h, n_windows=config.cv.n_windows, step_size=config.cv.step_size)
    # db = DB(DB_PATH)
    best_model_df = model.best_model_df.rename(columns={"unique_id": "DRV_COD"})
    data.set_data_attribute(best_model_df, "driver_best_model_df")
    best_stat_model_df = model.cv_stats_df.rename(columns={"unique_id": "DRV_COD"})
    data.set_data_attribute(best_stat_model_df, "driver_best_stat_model_df")
    # db.write_df(best_model_df, table_name="DRV_BEST_MODEL")


def driver_selection(data: DataLoader, method: str = "lasso", json_file: Path = ""):
    results = {}
    eco_codes = data.mapped_df["ECO_COD"].unique()
    driver_sel_df = pd.DataFrame()
    for eco_code in (pbar := tqdm(eco_codes, disable=MODE == "prod")):
        pbar.set_description(f"ECO_COD: {eco_code}")
        X, Y, drivers_list = data.generate_driver_selection_dataset(
            eco_code,
            config.driver_selection.score_column,
            config.driver_selection.score_threshold,
            fillnaX_strategy="zero",
            corr_threshold=0.95,
            cutoff_month=config.dataset.cutoff_month,
        )
        cv_len = 5
        if len(Y) <= cv_len:
            logger.warning(f"Not enought data to perform selction on {eco_code}. skipping")
            results[eco_code] = "NOT ENOUGH DATA"
            continue
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        driver_selector = DriverSelector(configuration_path=DRIVER_SELECTION_CONFIG_PATH)

        if method == "lasso":
            driver_selector.set_model("LassoCV")
            driver_selector.fit(X_scaled, Y)
            setattr(driver_selector.selector_, "X", X_scaled)
            setattr(driver_selector.selector_, "Y", Y)
            try:
                selected_drivers, alpha = driver_selector.generate_results(
                    drivers_list, min_statistical_importance=config.driver_selection.min_statistical_importance
                )
            except ValueError:
                results[eco_code] = "NOT ENOUGH DATA"
                continue
            results[eco_code] = {"mapped_drivers": drivers_list, "lasso_coeffs": selected_drivers, "lasso_alpha": alpha}
            df = pd.DataFrame(
                list(selected_drivers.items()) if isinstance(selected_drivers, dict) else [],
                columns=["DRV_COD", "COEFF"],
            )
            # logger.info(f"Driver selezionati: {list(selected_drivers.keys())}")
        elif method == "lasso_lars":
            driver_selector.set_model("LassoLarsCV")
            driver_selector.fit(X_scaled, Y)
            try:
                selected_drivers, alpha = driver_selector.generate_results(drivers_list)
            except ValueError:
                results[eco_code] = "NOT ENOUGH DATA"
                continue
            results[eco_code] = {
                "mapped_drivers": drivers_list,
                "lasso_coeffs": selected_drivers,
                "lasso_alpha": alpha,
            }
            df = pd.DataFrame(
                list(selected_drivers.items()) if isinstance(selected_drivers, dict) else [],
                columns=["DRV_COD", "COEFF"],
            )
            # logger.info(f"Driver selezionati: {list(selected_drivers.keys())}")
        elif method == "xgboost":
            Xgb_df = pd.DataFrame(X, columns=drivers_list)
            driver_selector.set_model("XGBRegressor")
            driver_selector.fit(Xgb_df, Y)
            selected_drivers = driver_selector.generate_results(
                drivers_list,
                columns=Xgb_df.columns,
                min_statistical_importance=config.driver_selection.min_statistical_importance,
                cumulative_threshold=config.driver_selection.cumulative_threshold,
            )
            results[eco_code] = selected_drivers.to_dict(orient="records")
            df = selected_drivers[["Feature", "Importance"]]
            df.columns = ["DRV_COD", "COEFF"]
            logger.info(f"Driver selezionati: {selected_drivers['Feature'].tolist()}")
        # Preprare table db structure
        df["ECO_COD"] = eco_code
        df["MODEL"] = method
        df = df[["ECO_COD", "DRV_COD", "MODEL", "COEFF"]]
        driver_sel_df = pd.concat([driver_sel_df, df])

        # with open(JSON_FILE_PATH, "w") as f:
        #     json.dump(results, f, indent="\t")
    if data.driver_selected_df is not None:
        df_appo = data.driver_selected_df.copy()
        if not df_appo.empty:
            df_appo = df_appo[df_appo["MODEL"] != method]
            df_appo = pd.concat([df_appo, driver_sel_df], ignore_index=True)
    else:
        df_appo = driver_sel_df
    data.set_data_attribute(df_appo, "driver_selected_df")
    return df_appo


# ┌                                                                              ┐
# │ 2. Driver Forecasting Model Training (D4)                                    │
# └                                                                              ┘


def forcast_drivers(data: DataLoader, model: ForecastingModel, make_plots: bool = True):
    df = data.driver_dataset(
        selected=True,
        query_args={
            "score_column": config.driver_selection.score_column,
            "score_threshold": config.driver_selection.score_threshold,
            "model_selection": config.driver_selection.model_selection,
        },
    )
    df_train, df_test = split_dataset(df, cutoff_month=config.dataset.cutoff_month, fillna_method="zero")
    if data.driver_best_model_df is None:
        raise ValueError("No best model for drivers")
    df_fit, df_forecast = model.forecast_best(
        df_train, config.forecast.h, table="driver", best_model_df=data.driver_best_model_df
    )
    for drv_code in df_forecast["unique_id"].unique():
        drv_sign = data.driver_anag.loc[data.driver_anag["DRV_COD"] == drv_code, "SIGN"].values[0]
        # Control drv sign and adjust forecast if necessary
        if drv_sign == "negative":
            df_forecast.loc[df_forecast["unique_id"] == drv_code, "y_hat"] = df_forecast.loc[
                df_forecast["unique_id"] == drv_code, "y_hat"
            ].clip(upper=0)
        elif drv_sign == "positive":
            df_forecast.loc[df_forecast["unique_id"] == drv_code, "y_hat"] = df_forecast.loc[
                df_forecast["unique_id"] == drv_code, "y_hat"
            ].clip(lower=0)

    # df_fit.to_csv(FIT_PATH)
    df_forecast.to_csv(FORECAST_PATH)
    database_column_names = {
        "unique_id": "DRV_COD",
        "ds": "DATE_RIF",
        "y": "ACTUAL",
        "y_hat": "FORECAST",
    }
    df_fit.rename(columns=database_column_names, inplace=True)
    df_forecast.rename(columns=database_column_names, inplace=True)
    data.set_data_attribute(df_fit, "drivers_fit_df")
    data.set_data_attribute(df_forecast, "drivers_forecast_df")

    if make_plots:
        df_train, df_test = split_dataset(df, cutoff_month=config.dataset.cutoff_month, fillna_method="")
        for drv_code in df["unique_id"].unique():
            nm_model = data.driver_best_model_df.loc[data.driver_best_model_df["DRV_COD"] == drv_code, "model"].tolist()
            plot = plot_generic_result(
                df=pd.concat([df_train, df_test]),
                df_fct=df_forecast,
                df_tst=df_test,
                n_tail=36,
                total=False,
                eco_code=drv_code,
            )
            plot.save("./plot/forecast_" + drv_code + ".png")


def cross_validate_economics(data: DataLoader, model: ForecastingModel):
    best_models = []
    best_models_stat = []
    eco_codes = data.eco_df["ECO_COD"].unique()
    for eco_code in (pbar := tqdm(eco_codes, disable=MODE == "prod")):
        pbar.set_description(f"ECO_COD: {eco_code}")
        df, _ = data.generate_economics_forecast_dataset(eco_code, config.driver_selection.model_selection)
        df_train, df_test = split_dataset(df, cutoff_month=config.dataset.cutoff_month, fillna_method="zero")
        try:
            model.fit(
                X=df_train,
                h=config.cv.h,
                n_windows=config.cv.n_windows,
                step_size=config.cv.step_size,
            )
            best_models.append(model.best_model_df)
            best_models_stat.append(model.cv_stats_df)
        except ValueError as e:
            logger.warning(e)
    df_cv_forecast = pd.concat(best_models, ignore_index=True)
    df_cv_stat = pd.concat(best_models_stat, ignore_index=True)
    df_cv_forecast = df_cv_forecast.rename(columns={"unique_id": "ECO_COD"})
    df_cv_stat = df_cv_stat.rename(columns={"unique_id": "ECO_COD"})
    data.set_data_attribute(df_cv_forecast, "eco_best_model_df")
    data.set_data_attribute(df_cv_stat, "eco_best_stat_model_df")


def process_single_eco(eco_code, data, model_template, config):
    """Function to manage cross validation of a single economics code."""
    try:
        df, _ = data.generate_economics_forecast_dataset(eco_code, config.driver_selection.model_selection)
        df_train, df_test = split_dataset(df, cutoff_month=config.dataset.cutoff_month, fillna_method="zero")

        import copy

        local_model = copy.deepcopy(model_template)

        local_model.fit(
            X=df_train,
            h=config.cv.h,
            n_windows=config.cv.n_windows,
            step_size=config.cv.step_size,
        )
        return local_model.best_model_df, local_model.cv_stats_df

    except Exception as e:
        # Importante: logger deve essere thread-safe o gestire i processi
        print(f"Errore su {eco_code}: {e}")
        return None, None


def cross_validate_economics_parallel(data: DataLoader, model: ForecastingModel, n_jobs: int = -1):
    eco_codes = data.eco_df["ECO_COD"].unique()

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_eco)(code, data, model, config)
        for code in tqdm(eco_codes, desc="Parallel CV", disable=MODE == "prod")
    )

    # Filter results removing exeptions
    best_models = [r[0] for r in results if r[0] is not None]
    best_models_stat = [r[1] for r in results if r[1] is not None]

    df_cv_forecast = pd.concat(best_models, ignore_index=True)
    df_cv_stat = pd.concat(best_models_stat, ignore_index=True)
    df_cv_forecast = df_cv_forecast.rename(columns={"unique_id": "ECO_COD"})
    df_cv_stat = df_cv_stat.rename(columns={"unique_id": "ECO_COD"})

    data.set_data_attribute(df_cv_forecast, "eco_best_model_df")
    data.set_data_attribute(df_cv_stat, "eco_best_stat_model_df")


def _forecast_one_eco(eco_code, data, model, make_plots, config):
    try:
        df, X_df = data.generate_economics_forecast_dataset(eco_code, config.driver_selection.model_selection)
        df_train, _ = split_dataset(df, cutoff_month=config.dataset.cutoff_month, fillna_method="zero")

        eco_sign = data.eco_anag.loc[data.eco_anag["ECO_COD"] == eco_code, "SIGN"].values[0]

        df_fit, df_forecast = model.forecast_best(
            df_train, config.forecast.h, table="eco", best_model_df=data.eco_best_model_df, X_df=X_df
        )

        if eco_sign == "negative":
            df_forecast["y_hat"] = df_forecast["y_hat"].clip(upper=0)
        elif eco_sign == "positive":
            df_forecast["y_hat"] = df_forecast["y_hat"].clip(lower=0)

        if make_plots:
            df_train_plt, df_test_plt = split_dataset(df, cutoff_month=config.dataset.cutoff_month, fillna_method="")
            plot = plot_generic_result(
                df=pd.concat([df_train_plt, df_test_plt]),
                df_fct=df_forecast,
                df_tst=df_test_plt,
                n_tail=36,
                total=False,
                eco_code=eco_code,
            )
            plot.save(f"./plot/forecast_{eco_code}.png")

        return df_fit, df_forecast
    except Exception as e:
        # Usiamo un logger standard, non legato alla UI
        logger.warning(f"Exception in {eco_code}: {e}")
        return None, None


def forecast_economics(
    data: DataLoader, model: ForecastingModel, make_plots: bool = True, progress_callback=None, n_jobs: int = 1
):
    eco_codes = data.eco_df["ECO_COD"].unique()
    total_codes = len(eco_codes)

    if data.eco_best_model_df is None:
        raise ValueError("No best model for economics")

    # --- ESECUZIONE ---
    if n_jobs == 1:
        results = []
        for idx, eco_code in enumerate(tqdm(eco_codes, desc="Forecasting", disable=MODE == "prod")):
            res = _forecast_one_eco(eco_code, data, model, make_plots, config)
            results.append(res)
            # Chiamata al callback nel thread chiamante
            if progress_callback:
                progress_callback((idx + 1) / total_codes)
    else:
        # Parallelizzazione pulita: il generatore restituisce i risultati
        # al thread chiamante (main thread), che gestisce il callback.
        with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
            output_generator = parallel(
                delayed(_forecast_one_eco)(code, data, model, make_plots, config) for code in eco_codes
            )

            results = []
            for idx, res in enumerate(
                tqdm(output_generator, total=total_codes, desc="Forecasting (Parallel)", disable=MODE == "prod")
            ):
                results.append(res)
                # Il callback viene eseguito qui, "fuori" dal motore joblib
                if progress_callback:
                    progress_callback((idx + 1) / total_codes)

    # --- ASSEMBLAGGIO RISULTATI ---
    fit_list = [r[0] for r in results if r[0] is not None]
    fct_list = [r[1] for r in results if r[1] is not None]

    if not fit_list or not fct_list:
        logger.error("No results generated.")
        return

    df_fit = pd.concat(fit_list, ignore_index=True)
    df_forecast = pd.concat(fct_list, ignore_index=True)

    df_fit.rename(columns={"unique_id": "ECO_COD", "ds": "DATE_RIF"}, inplace=True)
    data.set_data_attribute(df_fit, "eco_fit_df")
    df_forecast.rename(columns={"unique_id": "ECO_COD", "ds": "DATE_RIF"}, inplace=True)
    data.set_data_attribute(df_forecast, "eco_forecast_df")

    # df_forecast.to_csv(FORECAST_PATH_ECO)


def fit_reconciliation(
    data: DataLoader, model: ForecastingModel, aggregation_spec=List[List[str]], make_plots: bool = True
):
    lst_eco_forecasted = data.eco_forecast_df["ECO_COD"].unique()
    eco_value_anag_df = data.eco_df[data.eco_df["ECO_COD"].isin(lst_eco_forecasted)].reset_index(drop=True)
    eco_value_anag_df = (
        eco_value_anag_df.merge(data.eco_anag.loc[:, aggregation_spec[-1]], on="ECO_COD", how="left")
        .dropna()
        .reset_index(drop=True)
    )

    df_aggregated, S_eco_df, tags_eco = aggregate(
        df=eco_value_anag_df,
        spec=aggregation_spec,
        time_col="DATE_RIF",
        target_cols=("VALUE",),
    )
    cols = config.forecast.spec[-1]
    df_aggregated[cols] = df_aggregated["unique_id"].str.split("/", expand=True)
    df_aggregated.rename(columns={"VALUE": "y", "DATE_RIF": "ds"}, inplace=True)
    df_aggregated_full = df_aggregated.copy()
    df_aggregated_full.drop(columns=cols, inplace=True)
    if "ECO_COD" in cols:
        df_aggregated = df_aggregated[df_aggregated["ECO_COD"].isna()].reset_index(drop=True)
    df_aggregated.drop(columns=cols, inplace=True)
    df_train, df_test = split_dataset(df_aggregated, cutoff_month=config.dataset.cutoff_month, fillna_method="zero")
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.simplefilter("ignore")
    model.fit(X=df_train, h=config.cv.h, n_windows=config.cv.h, step_size=config.cv.step_size)
    df_fit_aggr, df_forecast_aggr = model.forecast_best(
        df_train, config.forecast.h, table="aggregate", best_model_df=model.best_model_df
    )
    df_fit_eco = data.eco_fit_df
    df_forecast_eco = data.eco_forecast_df
    df_fit_eco = df_fit_eco.merge(data.eco_anag, on="ECO_COD", how="left")
    df_forecast_eco = df_forecast_eco.merge(data.eco_anag, on="ECO_COD", how="left")
    df_fit_eco = df_fit_eco.rename(columns={"DATE_RIF": "ds"})
    df_forecast_eco = df_forecast_eco.rename(columns={"DATE_RIF": "ds"})
    hierarchy = config.forecast.spec[-1]
    df_fit_eco["unique_id"] = df_fit_eco[hierarchy].astype(str).agg("/".join, axis=1)
    df_forecast_eco["unique_id"] = df_forecast_eco[hierarchy].astype(str).agg("/".join, axis=1)
    df_fit_eco = df_fit_eco[["unique_id", "ds", "y", "y_hat"]]
    df_forecast_eco = df_forecast_eco[["unique_id", "ds", "y_hat"]]

    df_fit = pd.concat([df_fit_eco, df_fit_aggr]).reset_index(drop=True)
    df_forecast = pd.concat([df_forecast_eco, df_forecast_aggr]).reset_index(drop=True)

    reconciler = Reconciliation(configuration_path=RECONCILIATION_CONFIG_PATH)
    df_rec = reconciler.reconcile_best(df_fct=df_forecast, df_fit=df_fit, S_df=S_eco_df, tags=tags_eco)
    df_group = df_aggregated.copy()
    df_group_fit = df_fit_aggr.copy()

    # Da rivedere perché poi non tornano più i totali
    # for uid in df_rec["unique_id"].unique():
    #     eco_code = uid.split('/')[-1]
    #     filtered = data.eco_anag.loc[data.eco_anag["ECO_COD"] == eco_code, "SIGN"]
    #     if not filtered.empty:
    #         eco_sign = filtered.values[0]
    #         # Control eco sign and adjust forecast if necessary
    #         if eco_sign == "negative":
    #             df_rec.loc[df_rec['unique_id'] == uid, 'y_hat_rec'] = df_rec.loc[df_rec['unique_id'] == uid, 'y_hat_rec'].clip(upper=0)
    #         elif eco_sign == "positive":
    #             df_rec.loc[df_rec['unique_id'] == uid, 'y_hat_rec'] = df_rec.loc[df_rec['unique_id'] == uid, 'y_hat_rec'].clip(lower=0)

    if make_plots:
        df_train, df_test = split_dataset(
            df_aggregated_full, cutoff_month=config.dataset.cutoff_month, fillna_method=""
        )
        for eco_code in df_rec["unique_id"].unique():
            # nm_model = data.driver_best_model_df.loc[data.driver_best_model_df["DRV_COD"] == drv_code, "model"].tolist()
            plot = plot_generic_result(
                df=pd.concat([df_train, df_test]),
                df_fct=df_rec,
                df_tst=df_test,
                n_tail=36,
                total=False,
                eco_code=eco_code,
            )
            plot.save("./plot/forecast_rec_" + eco_code.replace("/", "_") + ".png")

    df_rec[cols] = df_rec["unique_id"].str.split("/", expand=True)
    df_rec.rename(columns={"ds": "DATE_RIF"}, inplace=True)
    df_rec = df_rec[cols + ["DATE_RIF", "y_hat", "y_hat_rec"]]

    cols2 = config.forecast.spec[-2]
    df_group[cols2] = df_group["unique_id"].str.split("/", expand=True)
    df_group.rename(columns={"ds": "DATE_RIF"}, inplace=True)
    df_group = df_group[cols2 + ["DATE_RIF", "y"]]
    df_group_fit[cols2] = df_group_fit["unique_id"].str.split("/", expand=True)
    df_group_fit.rename(columns={"ds": "DATE_RIF"}, inplace=True)
    df_group_fit = df_group_fit[cols2 + ["DATE_RIF", "y", "y_hat"]]

    data.set_data_attribute(df_rec, "rec_forecast_df")
    data.set_data_attribute(df_group, "eco_group_df")
    data.set_data_attribute(df_group_fit, "eco_group_fit_df")

    return df_rec, df_group

