# Start timer
import time

start_time = time.time()

from dotenv import load_dotenv

load_dotenv()

from src.back.pipeline import *
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
    # reduce_db(db_path = DB_PATH, lst_eco_cod=[
    #     'ECO_321', 'ECO_326', 'ECO_324', 'ECO_328', 'ECO_319', 'ECO_325',
    #    'ECO_318', 'ECO_240', 'ECO_246', 'ECO_252', 'ECO_248', 'ECO_253',
    #    'ECO_257', 'ECO_250', 'ECO_249', 'ECO_191', 'ECO_241', 'ECO_244',
    #    'ECO_279', 'ECO_117', 'ECO_171', 'ECO_170', 'ECO_137', 'ECO_160',
    #    'ECO_110', 'ECO_104', 'ECO_109', 'ECO_334', 'ECO_163', 'ECO_333',
    #    'ECO_149', 'ECO_129', 'ECO_162', 'ECO_308', 'ECO_084', 'ECO_106',
    #    'ECO_277', 'ECO_111', 'ECO_212', 'ECO_209', 'ECO_261', 'ECO_289',
    #    'ECO_066', 'ECO_258', 'ECO_290', 'ECO_071', 'ECO_151', 'ECO_142',
    #    'ECO_157', 'ECO_158', 'ECO_195', 'ECO_268', 'ECO_267', 'ECO_042',
    #    'ECO_123', 'ECO_029', 'ECO_145', 'ECO_223', 'ECO_188', 'ECO_144',
    #    'ECO_045', 'ECO_130', 'ECO_043', 'ECO_238', 'ECO_089', 'ECO_239',
    #    'ECO_090', 'ECO_044', 'ECO_237', 'ECO_088', 'ECO_201', 'ECO_135',
    #    'ECO_332', 'ECO_138', 'ECO_198', 'ECO_064'])

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
    forcast_drivers(data_loader, forecasting_model_drv, make_plots=False)

    data_loader.save_all()

    logger.info("Cross-validating economics forecasting models...")
    forecasting_model_eco = ForecastingModel(configuration_path=ECONOMICS_MODEL_CONFIG_PATH, freq=config.forecast.freq)
    # cross_validate_economics(data_loader, forecasting_model_eco)
    cross_validate_economics_parallel(data_loader, forecasting_model_eco, n_jobs=8)

    data_loader.save_all()

    logger.info("Forecasting economics using the best model...")
    forecast_economics(data_loader, forecasting_model_eco, make_plots=False)

    data_loader.save_all()

    logger.info("Reconciliate economics...")
    forecasting_model_rec = ForecastingModel(configuration_path=DRIVERS_MODEL_CONFIG_PATH, freq=config.forecast.freq)
    fit_reconciliation(data_loader, forecasting_model_rec, aggregation_spec=config.forecast.spec, make_plots=False)

    logger.info("Calculate cause-effect matrix for what-if...")
    calculate_cause_effect(data_loader)

    data_loader.save_all()

# End timer
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
