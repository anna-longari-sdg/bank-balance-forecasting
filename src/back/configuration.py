import inspect
import os
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from hierarchicalforecast.methods import BottomUp, MinTrace, OptimalCombination
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.model_selection import LeaveOneOut, TimeSeriesSplit
from statsforecast.models import (
    MSTL,
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoMFLES,
    AutoTheta,
    HistoricAverage,
    HoltWinters,
    SeasonalNaive,
)
from xgboost import XGBRegressor

from src import SRC_DIR
from src.back.utils import load_yml_config

SAFE_TYPES = {int, float, str, bool, list, dict, tuple, type(None)}


def _sanitize_type(annotation: Any) -> Any:
    """
    Recursively simplifies complex type annotations.
    If a type is too complex (custom Protocols, parameterized Callables),
    it falls back to typing.Any to prevent Pydantic crashes.
    """
    # 1. Handle Unresolved Strings (Forward Refs)
    if isinstance(annotation, str):
        return Any

    # 2. Check against Safe Primitives
    if annotation in SAFE_TYPES:
        return annotation

    # 3. Analyze Generics (Union, List, etc.)
    origin = get_origin(annotation)
    args = get_args(annotation)

    # If it has no origin (not a generic) and wasn't a safe type,
    # it's likely a custom class (like TimeSeriesSplit). Keep it.
    if origin is None:
        return annotation

    # 4. Handle Unions (e.g., Union[str, ComplexThing])
    if origin is Union:
        # Recursively sanitize arguments
        sanitized_args = tuple(_sanitize_type(arg) for arg in args)

        # Optimization: Union[Any, int] is logically just Any
        if Any in sanitized_args:
            return Any

        return Union[sanitized_args]

    # 5. Handle Containers (List, Dict)
    # We try to keep List[int], but if it's List[ComplexThing], we might accept List[Any]
    if origin in (list, dict, tuple, List, Dict, Tuple):
        sanitized_args = tuple(_sanitize_type(arg) for arg in args)
        return origin[sanitized_args]

    # 6. Fallback for Callables and other complex Protocols
    # This catches the XGBoost 'Callable[[numpy...], ...]' issue
    if origin in (Callable, Sequence, Iterable):
        return Any

    # 7. Default Fallback
    return Any


def _generate_model_from_init(target_class: Type[Any]) -> Type[BaseModel]:
    init_method = target_class.__init__
    signature = inspect.signature(init_method)
    parameters = signature.parameters

    # 1. Try to resolve types
    try:
        resolved_hints = get_type_hints(init_method)
    except Exception:
        # If imports fail (e.g. missing numpy inside the lib), ignore hints
        resolved_hints = {}

    fields = {}

    for param_name, param in parameters.items():
        if param_name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # 2. Get Raw Annotation
        if param_name in resolved_hints:
            raw_annotation = resolved_hints[param_name]
        elif param.annotation != inspect.Parameter.empty:
            raw_annotation = param.annotation
        else:
            raw_annotation = Any

        # 3. SANITIZE THE ANNOTATION (The Fix)
        # This converts the complex XGBoost Union into simply 'Any' (or 'Union[str, Any]')
        # allowing validation to pass.
        annotation = _sanitize_type(raw_annotation)

        # 4. Get Default
        if param.default == inspect.Parameter.empty:
            default = ...
        else:
            default = param.default

        fields[param_name] = (annotation, default)

    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    extra_behavior = "allow" if has_kwargs else "forbid"

    return create_model(
        f"{target_class.__name__}Config",
        __config__=ConfigDict(extra=extra_behavior, arbitrary_types_allowed=True),
        **fields,
    )


class ModelFactory:
    def __init__(self):
        self._registry: Dict[str, Type[Any]] = {}
        self._validator_cache: Dict[str, Type[BaseModel]] = {}

    def register(self, *classes: Type[Any]):
        for cls in classes:
            self._registry[cls.__name__] = cls

    def _hydrate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scans parameters for nested model definitions.
        If it finds a dict like {'TimeSeriesSplit': {...}} where 'TimeSeriesSplit' is registered,
        it recursively instantiates it.
        """
        hydrated = params.copy()

        for key, value in params.items():
            # Check if the value is a dictionary that represents a Class
            # Rule: It must be a dict, have exactly 1 key, and that key must be in our registry.
            if isinstance(value, dict) and len(value) == 1:
                candidate_model_name = next(iter(value.keys()))

                if candidate_model_name in self._registry:
                    # RECURSION: Found a nested model! Create it.
                    inner_config = value[candidate_model_name] or {}

                    # We pass alias=None because usually nested objects don't need global unique aliases
                    hydrated[key] = self.create(candidate_model_name, inner_config, alias=None)

            # Check if the value is a LIST of potential models (e.g. sklearn Pipeline steps)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    # Same check for list items
                    if isinstance(item, dict) and len(item) == 1:
                        candidate_model_name = next(iter(item.keys()))
                        if candidate_model_name in self._registry:
                            inner_config = item[candidate_model_name] or {}
                            new_list.append(self.create(candidate_model_name, inner_config, alias=None))
                            continue
                    new_list.append(item)
                hydrated[key] = new_list

        return hydrated

    def create(self, model_name: str, config: Dict[str, Any], alias: Optional[str] = None) -> Any:
        # 1. Lookup Class
        target_class = self._registry.get(model_name)
        if not target_class:
            raise ValueError(f"Unknown model: '{model_name}'.")

        # 2. HYDRATE: Resolve nested objects BEFORE validation
        # This turns {'cv': {'TimeSeriesSplit': ...}} into {'cv': TimeSeriesSplitObject}
        hydrated_config = self._hydrate_params(config)

        # 3. Get Validator
        if model_name not in self._validator_cache:
            self._validator_cache[model_name] = _generate_model_from_init(target_class)
        Validator = self._validator_cache[model_name]

        # 4. Validate
        try:
            # Pydantic will accept the object instances because we set arbitrary_types_allowed=True
            validated_data = Validator(**hydrated_config)
        except ValidationError as e:
            raise ValueError(f"Validation failed for '{alias or model_name}':\n{e}")

        # 5. Instantiate
        instance = target_class(**validated_data.model_dump())

        if alias:
            instance.alias = alias

        return instance

    def create_from_list(self, config_list: List[Dict[str, Any]]) -> List[Any]:
        # (Same logic as previous step: handle aliases, handle counters, call self.create)
        instantiated_models = []
        final_configs = []
        all_aliases = []
        type_counters = Counter()

        for entry in config_list:
            model_name, params = next(iter(entry.items()))
            params = params or {}

            if "alias" in params:
                current_alias = params.pop("alias")
            else:
                idx = type_counters[model_name]
                current_alias = f"{model_name}_{idx}"
                type_counters[model_name] += 1

            final_configs.append({"model_name": model_name, "params": params, "alias": current_alias})
            all_aliases.append(current_alias)

        if len(all_aliases) != len(set(all_aliases)):
            raise ValueError(f"Duplicate aliases found: {[k for k, v in Counter(all_aliases).items() if v > 1]}")

        for config in final_configs:
            model = self.create(config["model_name"], config["params"], config["alias"])
            instantiated_models.append(model)

        return instantiated_models


nixtla_model_factory = ModelFactory()
nixtla_model_factory.register(
    AutoARIMA, AutoETS, HistoricAverage, AutoTheta, AutoCES, AutoMFLES, SeasonalNaive, HoltWinters, MSTL
)

driver_selection_model_factory = ModelFactory()
driver_selection_model_factory.register(LassoCV, LassoLarsCV, LeaveOneOut, TimeSeriesSplit, XGBRegressor)

reconciliation_model_factory = ModelFactory()
reconciliation_model_factory.register(MinTrace, OptimalCombination, BottomUp)


class CvParams(BaseModel):
    freq: str = "MS"
    h: int = 6
    n_windows: int = 4
    step_size: int = 1


class ForecastingParams(CvParams):
    spec: Optional[List[List[str]]] = None


class DatasetParams(BaseModel):
    cutoff_month: Optional[str] = None
    last_date: Optional[str] = None


class DriverSelectionParams(BaseModel):
    score_column: Literal["mape_L2", "mape_mean"] = "mape_L2"
    score_threshold: float = 0.1
    min_statistical_importance: float = 0.05
    cumulative_threshold: float = 0.95
    model_selection: Literal["lasso", "lasso_lars", "xgboost"] = "lasso"


class PipelineConfiguration(BaseModel):
    dataset: DatasetParams = DatasetParams()
    cv: CvParams = CvParams()
    forecast: ForecastingParams = ForecastingParams()
    driver_selection: DriverSelectionParams = DriverSelectionParams()


TP_SOURCE = os.environ["TP_SOURCE"]
PIPELINE_CONFUGIRATION_PATH = SRC_DIR / "config_store" / TP_SOURCE / "pipeline.yaml"
PIPELINE_DEFAULT_CONFIGURATION = PipelineConfiguration.model_validate(load_yml_config(PIPELINE_CONFUGIRATION_PATH))


def check_configuration_type(configuration: Any):
    if type(configuration) is not dict:
        raise ValueError("Parsed configuration file is not a dictionary.")
    if "models" not in configuration:
        raise ValueError("Configuration file must contain a 'models' key.")
    return configuration["models"]


def get_instantiated_nixtla_models(configuration_path: str | Path):
    configuration = check_configuration_type(load_yml_config(configuration_path))
    models_list = nixtla_model_factory.create_from_list(configuration)
    logger.info(f"Successfully created {len(models_list)} models:")
    for m in models_list:
        # Check if alias exists, otherwise use class name
        name_tag = getattr(m, "alias", m.__class__.__name__)
        logger.info(f"• Model: {name_tag:<15} | Type: {type(m).__name__}")
        if hasattr(m, "season_length"):
            logger.info(f"  └─ Season Length: {m.season_length}")
    return models_list


def get_instantiated_driver_selection_model(configuration_path: str | Path, model_name: str, verbose=False):
    configuration = check_configuration_type(load_yml_config(configuration_path))
    if model_name not in configuration:
        raise ValueError(f"Model '{model_name}' not found in configuration.")
    params = configuration[model_name]
    if "alphas" in params:
        alphas = params["alphas"]
        if isinstance(alphas, dict) and alphas.get("type") == "logspace":
            params["alphas"] = np.logspace(alphas["start"], alphas["stop"], alphas["num"])
    model_instance = driver_selection_model_factory.create(model_name, params or {})
    if verbose:
        logger.info(f"Successfully created driver selection model: {model_name}")
    return model_instance


def get_instantiated_reconciliation_model(configuration_path: str | Path):
    configuration = check_configuration_type(load_yml_config(configuration_path))
    model_instance = reconciliation_model_factory.create_from_list(configuration)
    logger.info("Successfully created reconciliation model: MinTrace")
    return model_instance
