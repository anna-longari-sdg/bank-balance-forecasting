# Start timer
import time
start_time = time.time()

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.back.data import split_dataset
from src.back.pipeline import *
from src.back.plots import plot_result
from src.back.preprocess import *
from src.back.whatif import *

if __name__ == "__main__":
# ┌                                                                              ┐
# │ 0. Read and clean data (D1, D2)                                              │
# └                                                                              ┘

    # Step 1: Convert CSV and Excel files to DuckDB
    logger.info("Converting CSV and Excel files to DuckDB...")
    meta_input = load_metadata(meta_dir=CONFIG_META_PATH)
    get_data_all(meta_input, db_dir=DB_DIR, data_dir=DATA_DIR)
    create_mapping(db_dir=DB_DIR)

    # Step 2: Anonymize data
    logger.info("Anonymizing data...")

    anonymize_data(db_dir=DB_DIR)
    # reduce_db(db_path = DB_PATH, lst_eco_cod=["ECO_089", "ECO_239", "ECO_044", "ECO_090", "ECO_091", "ECO_132", "ECO_240", "ECO_241", "ECO_043", "ECO_045", "ECO_275"]) 
    # reduce_db(db_path = DB_PATH, lst_eco_cod=["ECO_089", "ECO_239", "ECO_044"]) 

    # Step 3: Initialize DataLoader
    data_loader = DataLoader(DB_PATH)
    logger.info(f"Economics loaded: {data_loader.eco_anag['ECO_COD'].unique()}")
    logger.info(f"Drivers loaded: {data_loader.driver_anag['DRV_COD'].unique()}")

    # Step 4: Driver best model search
    forecasting_model_drv = ForecastingModel(configuration_path=DRIVERS_MODEL_CONFIG_PATH, freq=config.forecast.freq)
    logger.info("Performing driver best model search...")
    driver_best_model_search(data_loader, forecasting_model_drv)
    
    # Step 5: Lasso driver selection
    logger.info("Performing " + config.driver_selection.model_selection + " driver selection...")
    driver_selection(data_loader, config.driver_selection.model_selection, JSON_FILE_PATH)

    # Step 7: Forecast drivers using the best model
    logger.info("Forecasting drivers using the best model...")
    forcast_drivers(data_loader, forecasting_model_drv)

    ##################################################################################################################
    # SOLO PER DEMO BPER
    import numpy as np
    drv_fore = data_loader.drivers_forecast_df
    drv_fore = drv_fore.merge(data_loader.driver_anag, on="DRV_COD", how="left")
    drv_fore_ex = drv_fore[(drv_fore["DRV_TYP_0"] == "EXO") | (drv_fore["DRV_TYP_0"] == "ENDO")].reset_index(drop=True)
    drv_df = data_loader.driver_df
    drv_fore_ex = drv_fore_ex.merge(drv_df, on=["DRV_COD", "DATE_RIF"], how="left")
    drv_fore_ex = drv_fore_ex[drv_fore_ex["VALUE"].notna()].reset_index(drop=True)
    drv_fore_ex["FORECAST_REAL"] = drv_fore_ex["VALUE"]
    drv_fore_ex = drv_fore_ex[["DRV_COD", "DATE_RIF", "FORECAST_REAL"]]
    drv_fore = drv_fore.merge(drv_fore_ex, on=["DRV_COD", "DATE_RIF"], how="left")
    drv_fore["FORECAST"] = drv_fore.apply(lambda x: x["FORECAST_REAL"] if pd.notna(x["FORECAST_REAL"]) else x["FORECAST"], axis=1)
    # mask = drv_fore["DRV_TYP_0"] == "ENDO"
    # # Esempio: aggiungi rumore normale con media 0 e deviazione standard 0.01 * valore
    # drv_fore.loc[mask, "FORECAST"] += np.random.normal(
    #     loc=0,
    #     scale=0.01 * drv_fore.loc[mask, "FORECAST"].abs()
    # )
    drv_fore = drv_fore[["DRV_COD", "DATE_RIF", "FORECAST"]]
    data_loader.drivers_forecast_df = drv_fore.copy()
    ##################################################################################################################

    data_loader.save_all()

    logger.info("Cross-validating economics forecasting models...")
    forecasting_model_eco = ForecastingModel(configuration_path=ECONOMICS_MODEL_CONFIG_PATH, freq=config.forecast.freq)
    # cross_validate_economics(data_loader, forecasting_model_eco)
    cross_validate_economics_parallel(data_loader, forecasting_model_eco, n_jobs=12)

    data_loader.save_all()

    logger.info("Forecasting economics using the best model...")
    forecast_economics(data_loader, forecasting_model_eco)

    data_loader.save_all()

    logger.info("Reconciliate economics...")
    forecasting_model_rec = ForecastingModel(configuration_path=DRIVERS_MODEL_CONFIG_PATH, freq=config.forecast.freq)
    fit_reconciliation(data_loader, forecasting_model_rec, aggregation_spec=config.forecast.spec)

    logger.info("Calculate cause-effect matrix for what-if...")
    calculate_cause_effect(data_loader)

    data_loader.save_all()

# End timer
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
