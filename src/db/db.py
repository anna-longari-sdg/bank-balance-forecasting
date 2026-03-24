import os
from string import Template
from typing import cast

import duckdb
import pandas as pd
from dotenv import load_dotenv

from src import SRC_DIR
from src.back.configuration import PIPELINE_DEFAULT_CONFIGURATION as config
from src.back.utils import fill_order_gaps
from src.db.queries import QUERY_MAP

load_dotenv()

TP_SOURCE = os.environ["TP_SOURCE"]
DB_DIR = SRC_DIR / "db" / TP_SOURCE

DB_PATH = DB_DIR / "bankfcs_anonymized.db"

fill_gaps_rows = set(["ECO_COD", "DRV_COD"])


class DB:
    def __init__(self, path=DB_PATH):
        self.path = path

    def query(self, query_string, as_df=True) -> pd.DataFrame | duckdb.DuckDBPyRelation:
        with duckdb.connect(self.path) as conn:
            result = conn.sql(query_string).df() if as_df else conn.sql(query_string)
        if not as_df:
            return result
        result = cast(pd.DataFrame, result)
        if "VALUE" in result.columns:
            result["VALUE"] = pd.to_numeric(result["VALUE"])
        if "DATE_RIF" in result.columns:
            result["DATE_RIF"] = pd.to_datetime(result["DATE_RIF"], format="%Y%m")
        return result

    def write_df(self, df: pd.DataFrame, table_name: str, drop_if_exists=True) -> None:
        with duckdb.connect(self.path) as conn:
            if drop_if_exists:
                conn.sql(f"DROP TABLE IF EXISTS {table_name}")
            conn.from_df(df).create(table_name)

    def predefined_query(
        self, query_name, fill_gaps: str | None = None, query_args: dict | None = None, as_df=True
    ) -> pd.DataFrame:
        query_string = QUERY_MAP.get(query_name)
        if query_string is None:
            raise ValueError(f"Query '{query_name}' is not defined.")
        if isinstance(query_string, Template):
            if query_args is None:
                raise ValueError(f"Query '{query_name}' requires arguments, but none were provided.")
            query_string = query_string.substitute(**query_args)
        df = self.query(query_string, as_df=as_df)
        if fill_gaps is not None:
            if fill_gaps not in fill_gaps_rows:
                raise ValueError(f"fill_gaps must be either 'ECO_COD' or 'DRV_COD', got '{fill_gaps}'")
            df = fill_order_gaps(
                df, nm_unique_id=fill_gaps, nm_ds="DATE_RIF", nm_y="VALUE", last_date=config.dataset.last_date
            )
        if as_df:
            return cast(pd.DataFrame, df)
        return df
