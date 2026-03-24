import warnings
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsCV, Lasso
from sklearn.utils.validation import check_is_fitted

from src.back.configuration import get_instantiated_driver_selection_model

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


def __generate_results_lasso(selector_, drivers_list: list[str], method: str = "1se", min_statistical_importance: float = 0.20):
    check_is_fitted(selector_)
    # Rule 1SE (One Standard Error Rule) and alpha min
    mean_mse = np.mean(selector_.mse_path_, axis=1)
    std_mse = np.std(selector_.mse_path_, axis=1) / np.sqrt(selector_.mse_path_.shape[1])
    idx_min = np.argmin(mean_mse)
    target_mse = mean_mse[idx_min] + std_mse[idx_min]
    candidate_alphas = selector_.alphas_[mean_mse <= target_mse]
    alpha_1se = np.max(candidate_alphas)
    final_lasso = Lasso(alpha=alpha_1se)
    final_lasso.fit(selector_.X, selector_.Y)    
    if hasattr(selector_, "coef_") and selector_.coef_ is not None:
        if method == "1se":
            selector_.coef_ = final_lasso.coef_    
        coefficients = pd.Series(selector_.coef_, index=drivers_list)
        selected_drivers = coefficients.loc[coefficients.abs() > 1e-4].sort_values(ascending=False)
    else:
        selected_drivers = pd.Series(dtype=float)
    # Imposta una soglia relativa (o Relative Magnitude Thresholding) per separare i driver che contribuiscono realmente al segnale 
    # da quelli che il Lasso ha tenuto "in vita" solo per limare qualche decimale di errore.    
    relative_importance = coefficients / coefficients.abs().max()    
    selected_drivers = relative_importance.loc[relative_importance.abs() > min_statistical_importance].sort_values(ascending=False)
    if hasattr(selector_, "alpha_") and selector_.alpha_ is not None:
        if method == "1se":
            selector_.alpha_ = alpha_1se
        alpha = selector_.alpha_
    else:
        alpha = "NO RESULTS"
    return {driver: coeff for driver, coeff in selected_drivers.items()} if len(
        selected_drivers
    ) > 0 else "NO_RESULTS", alpha


def __generate_results_xgboost(
    selector_,
    drivers_list: list[str],
    columns: List[str] | None = None,
    verbose=False,
    min_statistical_importance: float = 0.05,
    cumulative_threshold: float = 0.95,
):
    check_is_fitted(selector_)
    feature_importances = pd.Series(selector_.feature_importances_)
    if columns is not None:
        feature_importances.index = columns
    feature_importances = feature_importances.sort_values(ascending=False).reset_index()
    feature_importances.columns = ["Feature", "Importance"]
    feature_importances["Cumulative"] = feature_importances["Importance"].cumsum()
    # Si inserisce una soglia fissa min_statistical_importance al posto della statistica per evitare di selezionare troppe features    
    # Taglio netto del rumore: Anche se la somma cumulativa non ha ancora raggiunto il 95%, tutte le variabili che contribuiscono per meno del 5% verranno scartate.
    # In uno scenario di "variabili deboli" (dove tutte pesano l'1-2%), la restituisce un DataFrame vuoto o con pochissimi driver. 
    # Questo è un segnale diagnostico importante: ti sta dicendo che nessun driver è abbastanza forte.
    if len(drivers_list) > 1:
        if min_statistical_importance > 0:
            selected_features = feature_importances[
                (feature_importances["Cumulative"] <= cumulative_threshold)
                & (feature_importances["Importance"] >= min_statistical_importance)
            ]
        else:
            selected_features = feature_importances[(feature_importances["Cumulative"] <= cumulative_threshold)]
    else:
        selected_features = feature_importances
  

    if verbose:
        logger.info(f"Total possible drivers: {feature_importances.shape[0]}")
        logger.info(f"Total selected drivers: {selected_features.shape[0]}")
        logger.info(f"Selected drivers:\n{selected_features}")
        logger.info(f"Minimal statistical importance: {min_statistical_importance}")

    return selected_features


RESULTS_GENERATORS = {
    "LassoCV": __generate_results_lasso,
    "LassoLarsCV": __generate_results_lasso,
    "XGBRegressor": __generate_results_xgboost,
}


class DriverSelector:
    def __init__(self, configuration_path: str | Path) -> None:
        self.configuration_path = configuration_path
        self.model_name = None

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.selector_ = get_instantiated_driver_selection_model(
            configuration_path=self.configuration_path, model_name=self.model_name
        )

    def fit(self, X, y) -> BaseEstimator | None:
        if self.model_name is None or self.selector_ is None:
            raise ValueError("Model name must be set before fitting.")
        try:
            self.selector_.fit(X, y)
            return self.selector_
        except Exception as e:
            if isinstance(self.selector_, LassoLarsCV):
                logger.error("LassoLarsCV failed to converge. Consider using 'lasso' model instead.")
            else:
                logger.error(f"An error occurred during fitting: {e}")
            return None

    def __check_selector_exists(self):
        if self.selector_ is None:
            raise ValueError("The model has not been fitted yet.")

    def __check_selector_has_attribute(self, attribute: str):
        self.__check_selector_exists()
        if not hasattr(self.selector_, attribute):
            raise ValueError(f"The fitted model does not have {attribute} attribute.")

    @property
    def support(self) -> list[bool]:
        self.__check_selector_has_attribute("coef_")
        return self.selector_.coef_ != 0

    @property
    def coef_(self):
        self.__check_selector_has_attribute("coef_")
        return self.selector_.coef_

    @property
    def alpha_(self):
        self.__check_selector_has_attribute("alpha_")
        return self.selector_.alpha_

    def generate_results(self, drivers_list: list[str], **kwargs):
        self.__check_selector_exists()
        model_type = type(self.selector_).__name__
        if model_type not in RESULTS_GENERATORS:
            raise ValueError(f"No results generator defined for model type: {model_type}")
        return RESULTS_GENERATORS[model_type](self.selector_, drivers_list, **kwargs)
