import os
from pathlib import Path
import base64

from dotenv import load_dotenv

load_dotenv()
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from hierarchicalforecast.utils import aggregate
from loguru import logger

# logger.add("app.log")


TP_SOURCE = os.environ.get("TP_SOURCE")

# --- IMPORT LIBRARIES ---
from src.back.configuration import PIPELINE_DEFAULT_CONFIGURATION as config
from src.back.data import DataLoader
from src.back.pipeline import *
from src.back.reconciliation import Reconciliation
from src.back.whatif import get_drivers_wave, propagate_shock

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "src" / "db" / TP_SOURCE / "bankfcs_anonymized.db"
VAR_FORECAST = "y_hat_rec"
MODELS_WITH_EXO_VAR = ["arima_1", "arima_12", "mfles_12"]


st.set_page_config(
    page_title="iLabs BankFCS",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    section[data-testid="stSidebar"] {
        background-color: #00123b !important;
        color: white !important;
    }

    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
        transition: none !important;
    }

    section[data-testid="stSidebar"][aria-expanded="true"] > div {
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
        transition: none !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] > div {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
        transition: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- CACHE & DATA LOADING ---
@st.cache_resource
def get_data_loader(path):
    logger.info(f"Inizializzazione DataLoader presso {path}")
    return DataLoader(path)


@st.cache_data
def load_and_prepare_budget(path):
    logger.info(f"Caricamento dati {path} \n")
    loader = DataLoader(path)
    ele_eco_forecasted = loader.rec_forecast_df["ECO_COD"].unique()
    loader.eco_anag = loader.eco_anag[loader.eco_anag["ECO_COD"].isin(ele_eco_forecasted)].reset_index(drop=True)
    logger.info(f"Sistemazione driver selezionati in base al modello scelto {path} \n")
    # loader.driver_selected_df = loader.driver_selected_df[loader.driver_selected_df["ECO_COD"].isin(ele_eco_forecasted)].reset_index(drop=True)
    loader.driver_selected_df = loader.driver_selected_df[
        loader.driver_selected_df["ECO_COD"].isin(
            loader.eco_best_model_df[loader.eco_best_model_df["model"].isin(MODELS_WITH_EXO_VAR)]["ECO_COD"].unique()
        )
    ].reset_index(drop=True)

    logger.info(f"Sistemazione dati riconciliati {path} \n")
    # Separa i dati salvati a livello di dettaglio (con ECO_COD) da quelli aggregati (senza ECO_COD)
    rec_forecast_df_details = loader.rec_forecast_df[loader.rec_forecast_df["ECO_COD"].notna()].reset_index(drop=True)
    rec_forecast_df_aggr = loader.rec_forecast_df[loader.rec_forecast_df["ECO_COD"].isna()].reset_index(drop=True)[
        ["DATE_RIF"] + config.forecast.spec[-1] + ["y_hat", "y_hat_rec"]
    ]
    # A partire dai dati di dettaglio ricalcolo la somma dei forecast
    rec_forecast_df_sum, _, _ = aggregate(
        df=rec_forecast_df_details,
        spec=config.forecast.spec,
        time_col="DATE_RIF",
        target_cols=("y_hat",),
    )
    rec_forecast_df_sum[config.forecast.spec[-1]] = rec_forecast_df_sum["unique_id"].str.split("/", expand=True)
    rec_forecast_df_sum = rec_forecast_df_sum[rec_forecast_df_sum["ECO_COD"].isna()].reset_index(drop=True)
    rec_forecast_df_sum = rec_forecast_df_sum.rename(columns={"y_hat": "y_hat_sum"}).drop(columns=["unique_id"])[
        ["DATE_RIF"] + config.forecast.spec[-1] + ["y_hat_sum"]
    ]

    rec_forecast_df_sum = rec_forecast_df_aggr.merge(
        rec_forecast_df_sum, on=(config.forecast.spec[-1] + ["DATE_RIF"]), how="left"
    )
    rec_forecast_df_details["y_hat_sum"] = rec_forecast_df_details["y_hat"]
    rec_forecast_df = pd.concat([rec_forecast_df_sum, rec_forecast_df_details], ignore_index=True)
    loader.rec_forecast_df = rec_forecast_df
    loader.rec_forecast_df["forecast_adj"] = loader.rec_forecast_df[VAR_FORECAST]
    loader.drivers_forecast_df["forecast_adj"] = loader.drivers_forecast_df["FORECAST"]
    logger.info(f"Caricamento Budget {path} \n")
    df_eco_budget = loader.eco_budget[loader.eco_budget["ECO_COD"].notna()].reset_index(drop=True)
    df_eco_budget = df_eco_budget.merge(loader.eco_anag, on="ECO_COD", how="inner")
    df_eco_budget["VALUE_BUDGET"] = df_eco_budget["VALUE_BUDGET"].fillna(0)
    eco_grp_budget, _, _ = aggregate(
        df=df_eco_budget,
        spec=config.forecast.spec,
        time_col="DATE_RIF",
        target_cols=("VALUE_BUDGET",),
    )
    eco_grp_budget[config.forecast.spec[-1]] = eco_grp_budget["unique_id"].str.split("/", expand=True)
    eco_grp_budget = eco_grp_budget.drop(columns=["unique_id"]).rename(columns={"VALUE_BUDGET": "budget"})
    eco_grp_budget = eco_grp_budget[eco_grp_budget["DATE_RIF"].isin(rec_forecast_df["DATE_RIF"].unique())].reset_index(
        drop=True
    )
    return loader, eco_grp_budget


# Chiamata unica
data_loader, eco_grp_budget = load_and_prepare_budget(DB_PATH)
if "drivers_forecast_df" not in st.session_state:
    st.session_state.drivers_forecast_df = data_loader.drivers_forecast_df.copy()
    st.session_state.rec_forecast_df = data_loader.rec_forecast_df.copy()
    st.session_state.rec_fit_df = data_loader.eco_fit_df.copy()

if "modified_economics" not in st.session_state:
    st.session_state.modified_economics = set()

if "modified_driver_cells" not in st.session_state:
    st.session_state.modified_driver_cells = set()

if "modified_economic_cells" not in st.session_state:
    st.session_state.modified_economic_cells = set()

if "modified_driver_changes" not in st.session_state:
    st.session_state.modified_driver_changes = {}

if "modified_economic_changes" not in st.session_state:
    st.session_state.modified_economic_changes = {}


@st.cache_data
def get_eco_anagrafica():
    eco_anag = data_loader.eco_anag.copy()
    eco_fcst = data_loader.eco_forecast_df["ECO_COD"].drop_duplicates()
    eco_df = data_loader.eco_df.copy()
    eco_df["VALUE"] = eco_df["VALUE"].abs().fillna(0)
    eco_df = eco_df.merge(eco_anag, on="ECO_COD", how="inner").merge(eco_fcst, on="ECO_COD", how="inner")
    eco_grp, _, _ = aggregate(
        df=eco_df,
        spec=config.forecast.spec,
        time_col="DATE_RIF",
        target_cols=("VALUE",),
    )
    cols = config.forecast.spec[-1]
    eco_grp = eco_grp.groupby("unique_id")["VALUE"].sum().reset_index().rename(columns={"VALUE": "WEIGHT"})
    eco_grp[cols] = eco_grp["unique_id"].str.split("/", expand=True)
    eco_df = eco_grp[eco_grp["ECO_COD"].notna()].reset_index(drop=True)[["ECO_COD", "WEIGHT"]]
    eco_grp = (
        eco_grp[(eco_grp["ECO_COD"].isna()) & (eco_grp["ECO_GRP_2"].isna()) & (eco_grp["ECO_GRP_1"].isna())]
        .reset_index(drop=True)[["ECO_GRP_0", "WEIGHT"]]
        .rename(columns={"WEIGHT": "WEIGHT_AGGR"})
    )
    eco_anag["ECO_GRP_0"] = eco_anag["ECO_GRP_0"].astype(str)
    eco_anag["ECO_GRP_1"] = eco_anag["ECO_GRP_1"].astype(str)
    eco_anag["ECO_GRP_2"] = eco_anag["ECO_GRP_2"].astype(str)
    eco_anag = eco_anag.merge(eco_df, on="ECO_COD", how="inner").merge(eco_grp, on=["ECO_GRP_0"], how="left")
    eco_anag["PERC"] = np.round(eco_anag["WEIGHT"] / eco_anag["WEIGHT_AGGR"] * 100, 2)
    eco_anag["ECO_GRP_2_DSC"] = eco_anag["ECO_GRP_2_DSC"].str.replace("Segmento", "", case=False).str.strip()
    eco_anag.loc[eco_anag["ECO_GRP_2_DSC"] == "_", "ECO_GRP_2_DSC"] = np.nan
    eco_anag.loc[eco_anag["ECO_GRP_2"] == "_", "ECO_GRP_2"] = np.nan
    return eco_anag.sort_values(
        by=["ECO_GRP_0_DSC", "PERC", "ECO_GRP_1_DSC", "ECO_GRP_2_DSC"], ascending=[True, False, True, True]
    ).reset_index(drop=True)


if "eco_anagrafica" not in st.session_state:
    st.session_state.eco_anagrafica = get_eco_anagrafica()
    prime_aree = st.session_state.eco_anagrafica["ECO_GRP_0_DSC"].unique()
    st.session_state.selected_area = prime_aree[0] if len(prime_aree) > 0 else None

# --- BUSINESS & PLOTTING ---


# Plotting function with plotly
def generate_plotly_plot(df_plot, title="", height=350):
    """Genera un grafico Plotly interattivo."""
    if df_plot.empty:
        return None
    mask_pred = df_plot["forecast"].notnull()
    cutoff_date = df_plot[mask_pred]["ds"].min() if mask_pred.any() else None
    df_plot_fore = df_plot[df_plot["forecast"].notna()].reset_index(drop=True)
    count_y = df_plot_fore["y"].notna().sum()
    count_bdg = df_plot_fore["budget"].notna().sum()
    if count_y > 0:
        df_plot_agg = (
            df_plot_fore[df_plot_fore["y"].notna()]
            .groupby("unique_id")
            .agg({"forecast_adj": "sum", "y": "sum"})
            .reset_index()
        )
        df_plot_agg["delta"] = df_plot_agg["y"] - df_plot_agg["forecast_adj"]
        df_plot_agg["mape"] = np.abs(df_plot_agg["delta"] / df_plot_agg["y"] * 100)
    else:
        df_plot_agg = (
            df_plot_fore[df_plot_fore["budget"].notna()]
            .groupby("unique_id")
            .agg({"forecast_adj": "sum", "budget": "sum"})
            .reset_index()
        )
        df_plot_agg["delta"] = df_plot_agg["budget"] - df_plot_agg["forecast_adj"]
        df_plot_agg["mape"] = np.abs(df_plot_agg["delta"] / df_plot_agg["budget"] * 100)
    fig = go.Figure()
    # Actual
    fig.add_trace(
        go.Scatter(x=df_plot["ds"], y=df_plot["y"], mode="lines+markers", name="Actual", line=dict(color="black", width=2), marker=dict(size=2))
    )
    # Forecast Adjusted
    fig.add_trace(
        go.Scatter(
            x=df_plot["ds"], y=df_plot["forecast_adj"], mode="lines", name="AI Adj", line=dict(color="#FF007F", width=2)
        )
    )
    # Forecast
    fig.add_trace(
        go.Scatter(x=df_plot["ds"], y=df_plot["forecast"], mode="lines", name="AI", line=dict(color="orange", width=2))
    )
    if count_y <= 0:
        # Budget
        fig.add_trace(
            go.Scatter(
                x=df_plot["ds"], y=df_plot["budget"], mode="lines", name="Budget", line=dict(color="green", width=2)
            )
        )
    if df_plot_agg.shape[0] > 0:
        if count_y > 0:
            annotext = (
                "AI Adj="
                + f"{df_plot_agg['forecast_adj'][0]:,.0f}"
                + "| Actual="
                + f"{df_plot_agg['y'][0]:,.0f}"
                + "| Delta="
                + f"{df_plot_agg['delta'][0]:,.0f}"
                + " ("
                + f"{df_plot_agg['mape'][0]:.2f}"
                + "%)"
            )
        else:
            annotext = (
                "AI Adj="
                + f"{df_plot_agg['forecast_adj'][0]:,.0f}"
                + "| Budget="
                + f"{df_plot_agg['budget'][0]:,.0f}"
                + "| Delta="
                + f"{df_plot_agg['delta'][0]:,.0f}"
                + " ("
                + f"{df_plot_agg['mape'][0]:.2f}"
                + "%)"
            )
    else:
        annotext = ""

    fig.update_layout(
        title=title,
        title_y=0.999,
        # Add an annotation as a subtitle
        annotations=[
            dict(
                text=annotext,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.07,  # Positions it just below the title
                xanchor="center",
                font=dict(size=14, color="gray"),
            )
        ],
        height=height,
        template="simple_white",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


# Fuction to retrieve data at level 0, 1, 2 or detail (3)
def get_data_at_level(row, level):
    df_fcst = st.session_state.rec_forecast_df.merge(
        eco_grp_budget, on=["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "ECO_COD", "DATE_RIF"], how="left"
    )
    if level < 3:
        df_hist = data_loader.eco_group_df
    else:
        df_hist = data_loader.eco_df.merge(
            data_loader.eco_anag[["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "ECO_COD"]].astype(str),
            on="ECO_COD",
            how="left",
        ).rename(columns={"VALUE": "y"})

    f_fcst = df_fcst["ECO_GRP_0"] == str(row["ECO_GRP_0"])
    f_hist = df_hist["ECO_GRP_0"] == str(row["ECO_GRP_0"])
    if level >= 1:
        f_fcst &= df_fcst["ECO_GRP_1"] == str(row["ECO_GRP_1"])
        f_hist &= df_hist["ECO_GRP_1"] == str(row["ECO_GRP_1"])
    if level >= 2:
        f_fcst &= df_fcst["ECO_GRP_2"] == str(row["ECO_GRP_2"])
        f_hist &= df_hist["ECO_GRP_2"] == str(row["ECO_GRP_2"])
    if level >= 3:
        f_fcst &= df_fcst["ECO_COD"] == str(row["ECO_COD"])
        f_hist &= df_hist["ECO_COD"] == str(row["ECO_COD"])
    # Totals
    if level == 0:
        f_fcst &= df_fcst["ECO_GRP_1"].isnull()
        f_fcst &= df_fcst["ECO_GRP_2"].isnull()
        f_fcst &= df_fcst["ECO_COD"].isnull()
        f_hist &= df_hist["ECO_GRP_1"].isnull()
        f_hist &= df_hist["ECO_GRP_2"].isnull()
    elif level == 1:
        f_fcst &= df_fcst["ECO_GRP_2"].isnull()
        f_fcst &= df_fcst["ECO_COD"].isnull()
        f_hist &= df_hist["ECO_GRP_2"].isnull()
    elif level == 2:
        f_fcst &= df_fcst["ECO_COD"].isnull()
    elif level == 3:
        f_fcst = df_fcst["ECO_COD"] == row["ECO_COD"]
        f_hist = df_hist["ECO_COD"] == row["ECO_COD"]

    # Apply filters and prepare dataframes
    df_fcst = df_fcst[f_fcst].copy().rename(columns={"DATE_RIF": "ds", VAR_FORECAST: "forecast"})
    df_hist = df_hist[f_hist].copy().rename(columns={"DATE_RIF": "ds"})
    if level < 3:
        df_fcst["unique_id"] = (
            df_fcst[["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2"]].fillna("").astype(str).agg("|".join, axis=1)
        )
        df_hist["unique_id"] = (
            df_hist[["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2"]].fillna("").astype(str).agg("|".join, axis=1)
        )
    else:
        df_fcst["unique_id"] = (
            df_fcst[["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "ECO_COD"]].fillna("").astype(str).agg("|".join, axis=1)
        )
        df_hist["unique_id"] = (
            df_hist[["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "ECO_COD"]].fillna("").astype(str).agg("|".join, axis=1)
        )
    df_hist = df_hist[["unique_id", "ds", "y"]]
    df_fcst = df_fcst[["unique_id", "ds", "forecast", "forecast_adj", "budget"]]
    df_fcst = df_fcst.merge(df_hist, on=["unique_id", "ds"], how="left")
    df_full = pd.concat([df_hist[df_hist["ds"] < df_fcst["ds"].min()], df_fcst], ignore_index=True)
    # df_full = pd.concat([df_hist, df_fcst[df_fcst["ds"] > df_hist["ds"].max()]], ignore_index=True)

    return df_full


def get_summary_averages(row):
    avg_last_year = "N/A"
    avg_forecast_12m = "N/A"
    avg_forecast_adj_12m = "N/A"
    avg_budget_12m = "N/A"

    detail_df = get_data_at_level(row, 3).sort_values("ds")

    if "y" in detail_df.columns:
        actual_values = detail_df["y"].dropna().tail(12)
        if len(actual_values) > 0:
            avg_last_year = f"{actual_values.mean():,.0f}"

    if "forecast" in detail_df.columns:
        forecast_values = detail_df["forecast"].dropna().head(12)
        if len(forecast_values) > 0:
            avg_forecast_12m = f"{forecast_values.mean():,.0f}"

    if "forecast_adj" in detail_df.columns:
        forecast_adj_values = detail_df["forecast_adj"].dropna().head(12)
        if len(forecast_adj_values) > 0:
            avg_forecast_adj_12m = f"{forecast_adj_values.mean():,.0f}"

    if "budget" in detail_df.columns:
        budget_values = detail_df["budget"].dropna().head(12)
        if len(budget_values) > 0:
            avg_budget_12m = f"{budget_values.mean():,.0f}"

    return avg_last_year, avg_forecast_12m, avg_forecast_adj_12m, avg_budget_12m


def get_forecast_values(selected_code=None):
    # Usa il Code selezionato dalla tabella filtrata se fornito
    if selected_code is None:
        if not st.session_state.eco_anag_df.selection.rows:
            return
        filtered_df = st.session_state.filtered_eco_anag_df
        selected_idx = st.session_state.eco_anag_df.selection.rows[0]
        selected_code = filtered_df.iloc[selected_idx]["Code"]
    df_eco_forecast = st.session_state.rec_forecast_df[st.session_state.rec_forecast_df["ECO_COD"] == selected_code]
    df_eco_forecast = df_eco_forecast.rename(
        columns={"ECO_COD": "unique_id", "DATE_RIF": "ds", "forecast_adj": "forecast_adj"}
    )
    df_eco_forecast = df_eco_forecast[["unique_id", "ds", "forecast_adj"]]
    df_eco_anag = data_loader.eco_anag[data_loader.eco_anag["ECO_COD"] == selected_code][
        ["ECO_COD", "ECO_DSC"]
    ].rename(columns={"ECO_COD": "unique_id", "ECO_DSC": "description"})
    df_eco_anag["is_drv"] = False
    df_eco_anag["order"] = 999
    drivers_selected_eco = data_loader.driver_selected_df[
        data_loader.driver_selected_df["ECO_COD"] == selected_code
    ][["DRV_COD", "COEFF"]]
    df_drv_forecast = st.session_state.drivers_forecast_df[
        st.session_state.drivers_forecast_df["DRV_COD"].isin(drivers_selected_eco["DRV_COD"])
    ]
    df_drv_forecast = df_drv_forecast.rename(
        columns={"DRV_COD": "unique_id", "DATE_RIF": "ds", "forecast_adj": "forecast_adj"}
    )
    df_drv_forecast = df_drv_forecast[["unique_id", "ds", "forecast_adj"]]
    df_drv_anag = data_loader.driver_anag.merge(drivers_selected_eco, on="DRV_COD", how="inner")[
        ["DRV_COD", "DRV_DSC", "COEFF"]
    ].rename(columns={"DRV_COD": "unique_id", "DRV_DSC": "description"})
    df_drv_anag["is_drv"] = True
    df_drv_anag["order"] = df_drv_anag["COEFF"].abs().rank(ascending=True, method="first").astype(int)

    cutoff_date = df_eco_forecast["ds"].min()
    first_date = cutoff_date - pd.DateOffset(months=2)
    df_eco_full = df_eco_forecast.copy()
    df_drv_full = df_drv_forecast.copy()
    df_anag = pd.concat([df_eco_anag, df_drv_anag], ignore_index=True)
    df_values = pd.concat([df_eco_full, df_drv_full]).reset_index(drop=True)
    df_values["ds"] = df_values["ds"].dt.strftime("%Y-%m")
    df_values = df_values.pivot_table(
        index=["unique_id"], columns="ds", values="forecast_adj", dropna=False
    ).reset_index()
    df_values = (
        df_values.merge(df_anag, on="unique_id", how="left")
        .sort_values(by=["order"], ascending=[False])
        .reset_index(drop=True)
    )

    cols = ["description", "unique_id", "COEFF", "is_drv", "order"] + [
        c for c in df_values.columns if c not in ["description", "unique_id", "COEFF", "is_drv", "order"]
    ]
    df_values = df_values[cols]
    driver_selection_df = data_loader.get_model_selection_dataset(
        model_selection=config.driver_selection.model_selection
    ).query("ECO_COD == @selected_code")
    driver_selection_df["COEFF"] = driver_selection_df["COEFF"].abs()
    driver_selection_df.sort_values(by="COEFF", ascending=False, inplace=True)
    selected_drivers = driver_selection_df["DRV_COD"].tolist()
    st.session_state.selected_drivers_ordered = selected_drivers
    st.session_state.selected_values_df = df_values


def get_row_select():
    if "eco_anag_df" not in st.session_state:
        return
    if not st.session_state.eco_anag_df.selection.rows:
        return
    if "filtered_eco_anag_df" not in st.session_state:
        return
    filtered_df = st.session_state.filtered_eco_anag_df
    selected_idx = st.session_state.eco_anag_df.selection.rows[0]
    if selected_idx >= len(filtered_df):
        return
    selected_code = filtered_df.iloc[selected_idx]["Code"]

    apply_selection_by_code(selected_code, force_refresh=True)


def apply_selection_by_code(selected_code, force_refresh=False):
    if selected_code is None:
        return

    if (not force_refresh) and st.session_state.get("last_selected_code") == selected_code:
        return

    row_df = st.session_state.current_eco_df[st.session_state.current_eco_df["ECO_COD"] == selected_code]
    if row_df.empty:
        return
    row = row_df.iloc[0]

    for k in ["g0_fig", "g1_fig", "g2_fig", "det_fig"]:
        st.session_state[k] = None

    st.session_state.selected_labels = {
        "g0": row["ECO_GRP_0_DSC"],
        "g1": row["ECO_GRP_1_DSC"],
        "g2": row["ECO_GRP_2_DSC"] if pd.notnull(row["ECO_GRP_2"]) else None,
        # "det": f"{row['ECO_COD']} - {row['ECO_DSC']}",
        "det": f"{row['ECO_DSC']} [{row['ECO_COD']}]",
    }
    st.session_state.g0_fig = generate_plotly_plot(
        # get_data_at_level(row, 0), title=f"TREND   ►   Total Area {st.session_state.selected_labels['g0']}", height=300
        get_data_at_level(row, 0), title=f"Total Area {st.session_state.selected_labels['g0']}", height=300
    )
    st.session_state.g1_fig = generate_plotly_plot(
        get_data_at_level(row, 1), title=f"Total Sector {st.session_state.selected_labels['g1']}", height=300
    )
    if pd.notnull(row["ECO_GRP_2"]):
        st.session_state.g2_fig = generate_plotly_plot(
            get_data_at_level(row, 2), title=f"Total Segment {st.session_state.selected_labels['g2']}", height=300
        )
    st.session_state.det_fig = generate_plotly_plot(
        get_data_at_level(row, 3), title=f"{st.session_state.selected_labels['det']}", height=300
    )
    get_forecast_values(selected_code)
    st.session_state.last_selected_code = selected_code


def select_area(area):
    # Aggiorna l'area selezionata
    st.session_state.selected_area = area
    st.session_state.last_selected_code = None

    area_df = st.session_state.eco_anagrafica[
        st.session_state.eco_anagrafica["ECO_GRP_0_DSC"] == st.session_state.selected_area
    ].reset_index(drop=True)
    st.session_state.current_eco_df = area_df

    if area_df.empty:
        st.session_state.selected_values_df = None
        for k in ["g0_fig", "g1_fig", "g2_fig", "det_fig"]:
            st.session_state[k] = None
        st.session_state.selected_labels = None
        return

    st.session_state.selected_values_df = None
    for k in ["g0_fig", "g1_fig", "g2_fig", "det_fig"]:
        st.session_state[k] = None
    st.session_state.selected_labels = None


def sync_area_selector():
    select_area(st.session_state.area_selector)


def add_edited_drivers():
    if "edited_drivers" not in st.session_state:
        st.session_state.edited_drivers = set()
    for row_index in st.session_state.value_df["edited_rows"].keys():
        # +1 because eco row (index 0 in full df) is shown separately above the editor
        st.session_state.edited_drivers.add(st.session_state.selected_values_df.iloc[row_index + 1]["unique_id"])

def plot_variable_importance(selected_code, selected_desc, fixed_height=None):
    # Recupera i driver e i loro COEFF per il codice selezionato
    drivers_df = data_loader.driver_selected_df[data_loader.driver_selected_df["ECO_COD"] == selected_code]
    if drivers_df.empty:
        st.info("No drivers available for the selected code.")
        return
    # Merge per ottenere le descrizioni
    drivers_info = data_loader.driver_anag.merge(drivers_df, on="DRV_COD", how="inner")
    drivers_info = drivers_info[["DRV_COD", "DRV_DSC", "COEFF"]].drop_duplicates()
    # Ordina per valore assoluto decrescente
    drivers_info = drivers_info.sort_values(by="COEFF", key=abs, ascending=False).reset_index(drop=True)
    import plotly.graph_objects as go
    # Tonalità di blu
    stick_color = "#1f77b4"  # blu principale plotly
    marker_color = "#3399ff"  # blu più chiaro
    n_drivers = len(drivers_info)
    # Altezza dinamica: 32px per driver, min 220, max 500 (oppure altezza forzata)
    plot_height = fixed_height if fixed_height is not None else min(max(32 * n_drivers, 220), 500)
    fig = go.Figure()
    # Stick: linee orizzontali da 0 a coeff
    for idx, row in drivers_info.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, row["COEFF"]],
            y=[row["DRV_DSC"], row["DRV_DSC"]],
            mode="lines",
            line=dict(color=stick_color, width=3),
            showlegend=False,
            hoverinfo='skip',
        ))
    # Punti (candy) e etichette
    fig.add_trace(go.Scatter(
        x=drivers_info["COEFF"],
        y=drivers_info["DRV_DSC"],
        mode="markers+text",
        marker=dict(color=marker_color, size=8),
        text=drivers_info["COEFF"].round(2).astype(str),
        textposition=["middle right" if c >= 0 else "middle left" for c in drivers_info["COEFF"]],
        textfont=dict(color="black", size=11),
        showlegend=False,
        hovertext=drivers_info["DRV_COD"],
    ))
    fig.update_layout(
        # title=f"DRIVER IMPORTANCE   ►   {selected_code} - {selected_desc}",
        title="Number of drivers: " + str(n_drivers),
        xaxis_title="Importance",
        yaxis_title= None,
        yaxis=dict(autorange="reversed"),
        height=plot_height,
        margin=dict(l=10, r=10, t=40, b=10),
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True, key="plot_importance")


def render_weight_gauge(label, value, key):
    gauge_color = "#2f69a1"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(np.clip(value, 0.0, 100.0)),
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 14, "color": gauge_color}},
            title={"text": label, "font": {"size": 10, "color": "#4b5563"}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickwidth": 0, "showticklabels": False},
                "bar": {"color": gauge_color, "thickness": 0.30},
                "bgcolor": "#e5e7eb",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        margin=dict(l=4, r=4, t=22, b=2),
        height=122,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def format_display_number(value, decimals=3):
    if pd.isna(value):
        return ""
    return f"{float(value):,.{decimals}f}"


def parse_display_number(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    value_str = str(value).strip()
    if value_str == "":
        return np.nan

    return float(value_str.replace(",", ""))
    
# --- UI ---
def render_sidebar_logo(path, width_px=180, margin_bottom="0.25rem", align="center"):
    if not path.exists():
        return
    encoded_logo = base64.b64encode(path.read_bytes()).decode("utf-8")
    text_align = "left" if align == "left" else "center"
    img_margin = "0" if align == "left" else "0 auto"
    st.sidebar.markdown(
        (
            f"<div style='text-align:{text_align}; margin:0 0 {margin_bottom} 0;'>"
            f"<img src='data:image/png;base64,{encoded_logo}' "
            f"style='width:{width_px}px; max-width:100%; height:auto; display:block; margin:{img_margin};' />"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


sdg_logo_path = ROOT_DIR / "images" / "front" / "sdg_logo.png"
if sdg_logo_path.exists():
    render_sidebar_logo(sdg_logo_path, width_px=80, margin_bottom="2rem", align="left")

bbf_logo_path = ROOT_DIR / "images" / "front" / "logoBBF.png"
render_sidebar_logo(bbf_logo_path, width_px=400, margin_bottom="0.35rem")

st.sidebar.header("Area", divider="blue")

area_options = st.session_state.eco_anagrafica["ECO_GRP_0_DSC"].unique().tolist()
current_area = st.session_state.get("selected_area")
default_area_idx = area_options.index(current_area) if current_area in area_options else 0

st.sidebar.radio(
    "Area Selector",
    options=area_options,
    index=default_area_idx,
    key="area_selector",
    label_visibility="collapsed",
    on_change=sync_area_selector,
)
st.sidebar.markdown(
    "<hr style='border: 0; border-top: 1px solid #60a5fa; margin: 0.35rem 0 0.55rem 0;'>",
    unsafe_allow_html=True,
)
if "edited_drivers" in st.session_state and len(st.session_state.edited_drivers) > 0:
    badges = [f":violet-badge[{driver}]" for driver in st.session_state.edited_drivers]
    st.sidebar.info(f"Edited drivers: {' '.join(badges)}")

if "propagated_drivers" in st.session_state and len(st.session_state.propagated_drivers) > 0:
    sorted_drivers = sorted(list(st.session_state.propagated_drivers))
    badges = [f":blue-badge[{driver}]" for driver in sorted_drivers]
    with st.sidebar.expander("Propagated Drivers"):
        st.info(" ".join(badges))

if "propagated_economics" in st.session_state and len(st.session_state.propagated_economics) > 0:
    sorted_ecos = sorted(list(st.session_state.propagated_economics))
    badges = [f":green-badge[{eco}]" for eco in sorted_ecos]
    with st.sidebar.expander("Propagated Economics"):
        st.info(" ".join(badges))


# --- Top section: Table 1 (50% width) ---
if st.session_state.get("selected_area") is not None:
    st.session_state.current_eco_df = st.session_state.eco_anagrafica[
        st.session_state.eco_anagrafica["ECO_GRP_0_DSC"] == st.session_state.selected_area
    ].reset_index(drop=True)
else:
    st.session_state.current_eco_df = st.session_state.eco_anagrafica.copy().reset_index(drop=True)
eco_df = st.session_state.current_eco_df

area_title = st.session_state.selected_area if st.session_state.get("selected_area") is not None else "Economic Voices"

total_settori = eco_df["ECO_GRP_1_DSC"].dropna().nunique()
total_segmenti = eco_df["ECO_GRP_2_DSC"].dropna().nunique()
totale_economici = eco_df["ECO_COD"].dropna().nunique()

st.markdown(
    (
        "<div style='display:flex; gap:0.45rem; margin:1.1rem 0 0.25rem 0; flex-wrap:wrap;'>"
        "<div style='min-width:110px; border:1px solid #e5e7eb; border-radius:6px; padding:0.18rem 0.45rem; background:#ffffff;'>"
        "<div style='font-size:0.78rem; color:#64748b; line-height:1.05;'>Totale Settori</div>"
        f"<div style='font-size:1rem; color:#0f172a; font-weight:700; line-height:1.1;'>{total_settori}</div>"
        "</div>"
        "<div style='min-width:110px; border:1px solid #e5e7eb; border-radius:6px; padding:0.18rem 0.45rem; background:#ffffff;'>"
        "<div style='font-size:0.78rem; color:#64748b; line-height:1.05;'>Totale Segmenti</div>"
        f"<div style='font-size:1rem; color:#0f172a; font-weight:700; line-height:1.1;'>{total_segmenti}</div>"
        "</div>"
        "<div style='min-width:110px; border:1px solid #e5e7eb; border-radius:6px; padding:0.18rem 0.45rem; background:#ffffff;'>"
        "<div style='font-size:0.78rem; color:#64748b; line-height:1.05;'>Totale Voci</div>"
        f"<div style='font-size:1rem; color:#0f172a; font-weight:700; line-height:1.1;'>{totale_economici}</div>"
        "</div>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([0.55, 0.45])

with top_left:
    st.markdown(f"<h2 style='color: #00123b; margin-bottom: 0.1rem;'>{area_title}</h2>", unsafe_allow_html=True)

    table_df = eco_df.copy()
    sector_totals = table_df.groupby("ECO_GRP_1")["WEIGHT"].transform("sum")
    segment_totals = table_df.groupby(["ECO_GRP_1", "ECO_GRP_2"], dropna=False)["WEIGHT"].transform("sum")
    table_df["PCT_AREA"] = table_df["PERC"].round(2)
    table_df["PCT_SECTOR"] = np.where(
        sector_totals.notna() & sector_totals.ne(0),
        np.round(table_df["WEIGHT"] / sector_totals * 100, 2),
        np.nan,
    )
    table_df["PCT_SEGMENT"] = np.where(
        table_df["ECO_GRP_2"].notna() & segment_totals.notna() & segment_totals.ne(0),
        np.round(table_df["WEIGHT"] / segment_totals * 100, 2),
        np.nan,
    )

    filter_cols = [
        "ECO_GRP_0_DSC",
        "ECO_GRP_1_DSC",
        "ECO_GRP_2_DSC",
        "ECO_COD",
        "ECO_DSC",
        "PCT_AREA",
        "PCT_SECTOR",
        "PCT_SEGMENT",
    ]
    filter_df = table_df[filter_cols].rename(
        columns={
            "ECO_GRP_0_DSC": "Area",
            "ECO_GRP_1_DSC": "Sector",
            "ECO_GRP_2_DSC": "Segment",
            "ECO_COD": "Code",
            "ECO_DSC": "Description",
            "PCT_AREA": "% Area",
            "PCT_SECTOR": "% Sector",
            "PCT_SEGMENT": "% Segment",
        }
    )
    filter_df["% Area"] = filter_df["% Area"].round(2)
    filter_df["% Sector"] = filter_df["% Sector"].round(2)
    filter_df["% Segment"] = filter_df["% Segment"].round(2)

    filter_input = filter_df[["Sector", "Segment", "Code", "Description", "% Area", "% Sector", "% Segment"]].copy()
    for col in filter_input.select_dtypes(include="object").columns:
        filter_input[col] = filter_input[col].fillna("")
    from streamlit_extras.dataframe_explorer import dataframe_explorer

    filtered_input = dataframe_explorer(filter_input, case=False)
    filtered_df = filter_df.loc[filtered_input.index].reset_index(drop=True)
    st.session_state.filtered_eco_anag_df = filtered_df

    # Streamlit non consente di impostare programmaticamente la selezione di st.dataframe.
    # La continuità della voce selezionata è gestita tramite last_selected_code.

    # Evidenzia in azzurro la cella Code per gli economici modificati in run precedenti
    _modified_set = st.session_state.get("modified_economics", set())

    def _style_modified_codes(df_style):
        styles = pd.DataFrame("", index=df_style.index, columns=df_style.columns)
        for row_idx, row in df_style.iterrows():
            if row["Code"] in _modified_set:
                styles.at[row_idx, "Code"] = "background-color: #dbeafe; color: #1e3a8a; font-weight: 600"
        return styles

    styled_filtered_df = filtered_df.style.apply(_style_modified_codes, axis=None)

    st.dataframe(
        data=styled_filtered_df,
        key="eco_anag_df",
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
        height=400,
        hide_index=True,
        column_config={
            "Area": None,
            "Sector": st.column_config.TextColumn("Sector", width="small"),
            "Segment": st.column_config.TextColumn("Segment", width="small"),
            "Code": st.column_config.TextColumn("Code", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "% Area": st.column_config.NumberColumn("% Area", format="%.2f%%", width="small"),
            "% Sector": st.column_config.NumberColumn("% Sector", format="%.2f%%", width="small"),
            "% Segment": st.column_config.NumberColumn("% Segment", format="%.2f%%", width="small"),
        },
    )

    total_rows = len(filter_df)
    filtered_rows = len(filtered_df)
    st.caption(f"Rows: {filtered_rows} / {total_rows}")

with top_right:
    # Show selection info card when row is selected
    selected_filtered_df = st.session_state.get("filtered_eco_anag_df")
    if (
        selected_filtered_df is not None
        and "eco_anag_df" in st.session_state
        and st.session_state.eco_anag_df.selection.rows
    ):
        selected_idx = st.session_state.eco_anag_df.selection.rows[0]
        if selected_idx < len(selected_filtered_df):
            selected_row = selected_filtered_df.iloc[selected_idx]
            sel_code = selected_row["Code"]
            sel_desc = selected_row["Description"]
            sel_weight = selected_row["% Area"]
            sel_sector = selected_row["Sector"]
            sel_segment = selected_row["Segment"]
            apply_selection_by_code(sel_code)

            st.markdown(
                f"<p style='color: #00123b; font-size: 1.4rem; font-weight: 600; margin-top: 1.35rem; margin-bottom: 0.6rem;'>{sel_desc} [{sel_code}]</p>",
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

                avg_last_year = "N/A"
                avg_forecast_12m = "N/A"
                avg_forecast_adj_12m = "N/A"
                avg_budget_12m = "N/A"
                weight_area_val = float(sel_weight) if pd.notna(sel_weight) else np.nan
                weight_sector_val = np.nan
                weight_segment_val = np.nan
                area_label_name = str(st.session_state.get("selected_area") or "")
                sector_label_name = str(sel_sector) if pd.notna(sel_sector) else ""
                segment_label_name = (
                    str(sel_segment)
                    if pd.notna(sel_segment) and str(sel_segment).strip() not in {"", "None", "nan"}
                    else None
                )

                row_df = st.session_state.current_eco_df[st.session_state.current_eco_df["ECO_COD"] == sel_code]
                if not row_df.empty:
                    row = row_df.iloc[0]
                    area_label_name = str(row.get("ECO_GRP_0_DSC", area_label_name))
                    sector_label_name = str(row.get("ECO_GRP_1_DSC", sector_label_name))
                    if pd.notna(row.get("ECO_GRP_2", np.nan)):
                        segment_label_name = str(row.get("ECO_GRP_2_DSC", segment_label_name))
                    if pd.notna(row.get("PERC", np.nan)):
                        weight_area_val = float(row["PERC"])

                    row_weight = row.get("WEIGHT", np.nan)
                    if pd.notna(row_weight):
                        sector_total = st.session_state.current_eco_df.loc[
                            st.session_state.current_eco_df["ECO_GRP_1"] == row["ECO_GRP_1"], "WEIGHT"
                        ].sum()
                        if pd.notna(sector_total) and sector_total != 0:
                            weight_sector_val = float((row_weight / sector_total) * 100)

                        if pd.notna(row["ECO_GRP_2"]):
                            segment_total = st.session_state.current_eco_df.loc[
                                (st.session_state.current_eco_df["ECO_GRP_1"] == row["ECO_GRP_1"])
                                & (st.session_state.current_eco_df["ECO_GRP_2"] == row["ECO_GRP_2"]),
                                "WEIGHT",
                            ].sum()
                            if pd.notna(segment_total) and segment_total != 0:
                                weight_segment_val = float((row_weight / segment_total) * 100)

                    avg_last_year, avg_forecast_12m, avg_forecast_adj_12m, avg_budget_12m = get_summary_averages(row)

                weight_items = []
                if pd.notna(weight_area_val):
                    weight_items.append((f"Weight on Area {area_label_name}", weight_area_val, f"weight_area_{sel_code}"))
                if pd.notna(weight_sector_val):
                    weight_items.append((f"Weight on Sector {sector_label_name}", weight_sector_val, f"weight_sector_{sel_code}"))
                if segment_label_name is not None and pd.notna(weight_segment_val):
                    weight_items.append(
                        (f"Weight on Segment {segment_label_name}", weight_segment_val, f"weight_segment_{sel_code}")
                    )

                if len(weight_items) > 0:
                    weight_cols = st.columns(len(weight_items))
                    for weight_col, (weight_label, weight_value, weight_key) in zip(weight_cols, weight_items):
                        with weight_col:
                            render_weight_gauge(weight_label, weight_value, weight_key)
                else:
                    st.markdown(
                        "<p style='margin: 0.3rem 0; font-size: 0.85rem; color: #666;'><strong>Weights:</strong> N/A</p>",
                        unsafe_allow_html=True,
                    )

                avg_cards = [
                    {
                        "label": "Avg Actual (last 12m)",
                        "value": avg_last_year,
                        "color": "#111111",
                        "bg": "#f9fafb",
                    },
                ]
                if avg_budget_12m != "N/A":
                    avg_cards.append(
                        {
                            "label": "Avg Budget (next 12m)",
                            "value": avg_budget_12m,
                            "color": "#16a34a",
                            "bg": "#ecfdf5",
                        }
                    )
                avg_cards.extend(
                    [
                        {
                            "label": "Avg Forecast (next 12m)",
                            "value": avg_forecast_12m,
                            "color": "#ff8c00",
                            "bg": "#fff7ed",
                        },
                        {
                            "label": "Avg Forecast Adj (next 12m)",
                            "value": avg_forecast_adj_12m,
                            "color": "#FF007F",
                            "bg": "#fff1f8",
                        },
                    ]
                )

                cards_html = "".join(
                    [
                        (
                            "<div style='border:1px solid #e5e7eb; border-radius:8px; padding:0.35rem 0.5rem; "
                            f"background:{card['bg']}; text-align:center;'>"
                            f"<div style='font-size:0.70rem; color:#6b7280;'>{card['label']}</div>"
                            f"<div style='font-size:1.05rem; font-weight:700; color:{card['color']}; line-height:1.2;'>{card['value']}</div>"
                            "</div>"
                        )
                        for card in avg_cards
                    ]
                )

                st.markdown(
                    f"""
                    <div style='display:grid; grid-template-columns:repeat({len(avg_cards)}, minmax(0, 1fr)); gap:0.45rem; margin:0.35rem 0 0.5rem 0;'>
                        {cards_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                num_drivers = 0
                if hasattr(data_loader, "driver_selected_df"):
                    num_drivers = len(
                        data_loader.driver_selected_df[
                            data_loader.driver_selected_df["ECO_COD"] == sel_code
                        ]
                    )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                plot_variable_importance(sel_code, sel_desc, fixed_height=240)
    elif selected_filtered_df is not None and st.session_state.get("last_selected_code") is not None:
        _sel_code = st.session_state.get("last_selected_code")
        _sel_row_df = selected_filtered_df[selected_filtered_df["Code"] == _sel_code]
        if not _sel_row_df.empty:
            _sel_row = _sel_row_df.iloc[0]
            apply_selection_by_code(_sel_code)
            st.markdown(
                f"<p style='color: #00123b; font-size: 1.5rem; font-weight: 600; margin-top: 1.35rem; margin-bottom: 0.6rem;'>{_sel_row['Description']} [{_sel_row['Code']}]</p>",
                unsafe_allow_html=True,
            )
    else:
        st.empty()

# --- Selection card: detail trend + variable importance (side by side) ---
selected_filtered_df = st.session_state.get("filtered_eco_anag_df")
selected_code = st.session_state.get("last_selected_code")
selected_desc = None

if (
    selected_filtered_df is not None
    and "eco_anag_df" in st.session_state
    and st.session_state.eco_anag_df.selection.rows
):
    selected_idx = st.session_state.eco_anag_df.selection.rows[0]
    if selected_idx < len(selected_filtered_df):
        selected_row = selected_filtered_df.iloc[selected_idx]
        selected_code = selected_row["Code"]
        selected_desc = selected_row["Description"]
        apply_selection_by_code(selected_code)

if selected_desc is None and selected_code is not None:
    selected_row_df = st.session_state.current_eco_df[st.session_state.current_eco_df["ECO_COD"] == selected_code]
    if not selected_row_df.empty:
        selected_desc = selected_row_df.iloc[0]["ECO_DSC"]

if selected_desc is None:
    selected_desc = ""

has_selection = selected_code is not None and st.session_state.get("det_fig") is not None

if has_selection:
    selection_chart_height = 340

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    det_left, det_center, det_right = st.columns([0.12, 0.76, 0.12])

    with det_center:
        with st.container(border=True):
            if st.session_state.get("det_fig"):
                det_fig_same_height = go.Figure(st.session_state.det_fig)
                det_fig_same_height.update_layout(height=selection_chart_height)
                st.plotly_chart(det_fig_same_height, width="stretch", key="plot_det")
            else:
                st.info("No detail trend available for the selected economic voice.")

# --- Middle section: totals charts (2/3 aligned) ---
# st.markdown("### Totals")
totals_figs = [
    ("g2_fig", "plot_g2"),
    ("g1_fig", "plot_g1"),
    ("g0_fig", "plot_g0"),
]
available_totals = [(fig_key, chart_key) for fig_key, chart_key in totals_figs if st.session_state.get(fig_key)]

if available_totals:
    total_cols = st.columns(len(available_totals))
    for col, (fig_key, chart_key) in zip(total_cols, available_totals):
        with col:
            with st.container(border=True):
                st.plotly_chart(st.session_state[fig_key], width="stretch", key=chart_key)

# --- Bottom section: table 2 and actions ---
if st.session_state.get("selected_values_df") is not None:
    st.markdown('<div class="driver-section-spacer"></div>', unsafe_allow_html=True)

    df = st.session_state.selected_values_df
    # st.markdown(f"#### FORECAST   ►  {df.loc[0, 'unique_id']} and Drivers")
    st.markdown(f"### 🖊️ What-if scenario [{df.loc[0, 'unique_id']}]")

    cols_forecast = (
        st.session_state.drivers_forecast_df["DATE_RIF"].dt.strftime("%Y-%m").astype(str).unique().tolist()
    )
    cols_to_freeze = df.columns.difference(cols_forecast).tolist()
    modified_driver_cells = st.session_state.get("modified_driver_cells", set())
    modified_economic_cells = st.session_state.get("modified_economic_cells", set())

    def _style_driver_table(df_style):
        styles = pd.DataFrame("", index=df_style.index, columns=df_style.columns)
        for col in cols_to_freeze:
            if col in styles.columns:
                styles.loc[:, col] = "color: gray; background-color: #f9f9f9"

        for row_idx, row in df_style.iterrows():
            if not bool(row.get("is_drv", False)):
                continue
            driver_code = row.get("unique_id")
            for month_col in cols_forecast:
                if (driver_code, month_col) in modified_driver_cells and month_col in styles.columns:
                    styles.at[row_idx, month_col] = "background-color: #dbeafe; color: #1e3a8a; font-weight: 600"

        return styles

    df_display = df.style.apply(_style_driver_table, axis=None)
    logger.info(f"Colonne da freezare: {cols_to_freeze}")

    def _style_editor_table(df_style):
        styles = pd.DataFrame("", index=df_style.index, columns=df_style.columns)
        for col in cols_to_freeze:
            if col in styles.columns:
                styles.loc[:, col] = "color: gray; background-color: #f8f9fb"
        for row_idx, row in df_style.iterrows():
            if not bool(row.get("is_drv", False)):
                for col in df_style.columns:
                    styles.at[row_idx, col] = "color: gray; background-color: #f8f9fb"
            else:
                for col in cols_forecast:
                    if (row.get("unique_id"), col) in modified_driver_cells and col in styles.columns:
                        styles.at[row_idx, col] = "background-color: #dbeafe; color: #1e3a8a; font-weight: 600"
        return styles

    df_editor = df.copy()
    if "COEFF" in df_editor.columns:
        df_editor["COEFF"] = df_editor["COEFF"].map(format_display_number)
    for col in cols_forecast:
        if col in df_editor.columns:
            df_editor[col] = df_editor[col].map(format_display_number)

    # Split eco row (read-only) from driver rows (editable)
    df_eco_editor = df_editor[df_editor["is_drv"] == False].reset_index(drop=True)
    df_drivers_editor = df_editor[df_editor["is_drv"] == True].reset_index(drop=True)

    def _style_eco_row(df_style):
        return pd.DataFrame(
            "color: gray; background-color: #f8f9fb",
            index=df_style.index,
            columns=df_style.columns,
        )

    table2_column_config = {
        "unique_id": st.column_config.TextColumn("Code"),
        "COEFF": st.column_config.TextColumn("Importance", width="small"),
        "description": st.column_config.TextColumn("Description", width="medium"),
        "is_drv": None,
        "order": None,
    } | {
        c: st.column_config.TextColumn(c, width="small")
        for c in df.columns
        if c not in ["description", "unique_id", "COEFF", "is_drv", "order"]
    }

    driver_modified_values_df = pd.DataFrame(columns=["Code", "Description", "Month", "Original Value", "New Value"])
    modified_driver_changes = st.session_state.get("modified_driver_changes", {})
    if len(modified_driver_changes) > 0:
        driver_modified_values_df = pd.DataFrame(
            [
                {
                    "Code": code,
                    "Month": month,
                    "Original Value": values["old_value"],
                    "New Value": values["new_value"],
                }
                for (code, month), values in sorted(modified_driver_changes.items())
            ]
        )
        driver_modified_values_df = (
            driver_modified_values_df.merge(
                data_loader.driver_anag[["DRV_COD", "DRV_DSC"]].rename(
                    columns={"DRV_COD": "Code", "DRV_DSC": "Description"}
                ),
                on="Code",
                how="left",
            )
            [["Code", "Description", "Month", "Original Value", "New Value"]]
            .sort_values(by=["Code", "Month"])
            .reset_index(drop=True)
        )
        driver_modified_values_df["Original Value"] = driver_modified_values_df["Original Value"].map(format_display_number)
        driver_modified_values_df["New Value"] = driver_modified_values_df["New Value"].map(format_display_number)

    economic_modified_values_df = pd.DataFrame(columns=["Code", "Description", "Month", "Original Value", "New Value"])
    modified_economic_changes = st.session_state.get("modified_economic_changes", {})
    if len(modified_economic_changes) > 0:
        economic_modified_values_df = pd.DataFrame(
            [
                {
                    "Code": code,
                    "Month": month,
                    "Original Value": values["old_value"],
                    "New Value": values["new_value"],
                }
                for (code, month), values in sorted(modified_economic_changes.items())
            ]
        )
        economic_modified_values_df = (
            economic_modified_values_df.merge(
                data_loader.eco_anag[["ECO_COD", "ECO_DSC"]].rename(
                    columns={"ECO_COD": "Code", "ECO_DSC": "Description"}
                ),
                on="Code",
                how="left",
            )
            [["Code", "Description", "Month", "Original Value", "New Value"]]
            .sort_values(by=["Code", "Month"])
            .reset_index(drop=True)
        )
        economic_modified_values_df["Original Value"] = economic_modified_values_df["Original Value"].map(format_display_number)
        economic_modified_values_df["New Value"] = economic_modified_values_df["New Value"].map(format_display_number)

    if len(modified_driver_cells) > 0 or len(modified_economic_cells) > 0:
        edit_tab, highlight_tab = st.tabs(["Edit", "Modified cells"])

        with edit_tab:
            st.dataframe(
                df_eco_editor.style.apply(_style_eco_row, axis=None),
                width="stretch",
                hide_index=True,
                column_config=table2_column_config,
            )
            st.data_editor(
                df_drivers_editor.style.apply(_style_editor_table, axis=None),
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                disabled=cols_to_freeze,
                key="value_df",
                column_config=table2_column_config,
                on_change=add_edited_drivers,
            )

        with highlight_tab:
            if not driver_modified_values_df.empty:
                st.caption(f"Modified driver values: {len(driver_modified_values_df)}")
                st.dataframe(
                    driver_modified_values_df,
                    width="stretch",
                    hide_index=True,
                    height=220,
                    column_config={
                        "Code": st.column_config.TextColumn("Code", width="small"),
                        "Description": st.column_config.TextColumn("Description", width="medium"),
                        "Month": st.column_config.TextColumn("Month", width="small"),
                        "Original Value": st.column_config.TextColumn("Original Value", width="small"),
                        "New Value": st.column_config.TextColumn("New Value", width="small"),
                    },
                )

            if not economic_modified_values_df.empty:
                st.caption(f"Modified economic values: {len(economic_modified_values_df)}")
                st.dataframe(
                    economic_modified_values_df,
                    width="stretch",
                    hide_index=True,
                    height=220,
                    column_config={
                        "Code": st.column_config.TextColumn("Code", width="small"),
                        "Description": st.column_config.TextColumn("Description", width="medium"),
                        "Month": st.column_config.TextColumn("Month", width="small"),
                        "Original Value": st.column_config.TextColumn("Original Value", width="small"),
                        "New Value": st.column_config.TextColumn("New Value", width="small"),
                    },
                )
    else:
        st.dataframe(
            df_eco_editor.style.apply(_style_eco_row, axis=None),
            width="stretch",
            hide_index=True,
            column_config=table2_column_config,
        )
        st.data_editor(
            df_drivers_editor.style.apply(_style_editor_table, axis=None),
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=cols_to_freeze,
            key="value_df",
            column_config=table2_column_config,
            on_change=add_edited_drivers,
        )

    text_on_drv_click = ""
    preview_drivers_count = None
    preview_economics_count = None
    if not (
        "value_df" not in st.session_state
        or "edited_rows" not in st.session_state.value_df
        or len(st.session_state.value_df["edited_rows"]) == 0
    ):
        drv_click = []
        for row_index, row in st.session_state.value_df["edited_rows"].items():
            drv_click.append(df.iloc[row_index + 1]["unique_id"])
        drv_click_wave = get_drivers_wave(data_loader.cause_effect_df, drv_click)
        drv_click_wave_descr = data_loader.driver_anag[data_loader.driver_anag["DRV_COD"].isin(drv_click_wave)][
            ["DRV_COD", "DRV_DSC"]
        ]
        drv_click_wave_descr = (drv_click_wave_descr["DRV_COD"] + ": " + drv_click_wave_descr["DRV_DSC"]).tolist()
        eco_click_wave = data_loader.driver_selected_df[
            data_loader.driver_selected_df["DRV_COD"].isin(drv_click_wave)
        ]["ECO_COD"].unique()
        preview_drivers_count = len(drv_click_wave)
        preview_economics_count = len(eco_click_wave)
        eco_click_wave_descr = data_loader.eco_anag[data_loader.eco_anag["ECO_COD"].isin(eco_click_wave)][
            ["ECO_COD", "ECO_DSC"]
        ]
        eco_click_wave_descr = (eco_click_wave_descr["ECO_COD"] + ": " + eco_click_wave_descr["ECO_DSC"]).tolist()
        if len(drv_click_wave_descr) > 0:
            drv_list = "<ul>" + "".join(f"<li>{item}</li>" for item in drv_click_wave_descr) + "</ul>"
            eco_list = "<ul>" + "".join(f"<li>{item}</li>" for item in eco_click_wave_descr) + "</ul>"
            text_on_drv_click = (
                "**Execution time will be proportional to the number of economics affected, but generally should not exceed 1-2 minutes!**<br><br>"
                + "Running this will propagate current edits to the following "
                + str(len(drv_click_wave_descr))
                + " drivers:"
                + drv_list
                + "and to the following "
                + str(len(eco_click_wave_descr))
                + " economics:"
                + eco_list
            )
        else:
            text_on_drv_click = "Edited drivers have no propagation effect"

    if st.button(
        "Run",
        disabled="value_df" not in st.session_state
        or "edited_rows" not in st.session_state.value_df
        or len(st.session_state.value_df["edited_rows"]) == 0,
    ):
        start = time.time()
        with st.spinner("Processing..."):
            previous_drivers_forecast_df = st.session_state.drivers_forecast_df[["DRV_COD", "DATE_RIF", "forecast_adj"]].copy()
            previous_rec_forecast_df = st.session_state.rec_forecast_df[
                (st.session_state.rec_forecast_df["ECO_COD"].notna()) & (st.session_state.rec_forecast_df["ECO_COD"] != "")
            ][["ECO_COD", "DATE_RIF", "forecast_adj"]].copy()
            edits = []
            for row_index, row in st.session_state.value_df["edited_rows"].items():
                for col, new_value in row.items():
                    if col not in cols_to_freeze:
                        edits.append((df.iloc[row_index + 1]["unique_id"], col, parse_display_number(new_value)))
            edits_df = pd.DataFrame(edits, columns=["Code", "Month", "New_Value"])
            edits_df["Code"] = pd.Categorical(edits_df["Code"], categories=st.session_state.selected_drivers_ordered, ordered=True)
            edits_df.sort_values(by=["Code"], inplace=True)
            edits_df.sort_values(by=["Month"], inplace=True)
            min_month = str(edits_df["Month"].min())
            forecast_columns = df.columns.difference(["Code", "Desc"] + cols_to_freeze).tolist()
            df_driver_forecast_updated = st.session_state.drivers_forecast_df.copy()
            for _, row in edits_df.iterrows():
                code = row["Code"]
                col = row["Month"]
                new_value = row["New_Value"]
                col_index = forecast_columns.index(col)
                df_driver_forecast_updated = propagate_shock(
                    drivers_forecast_df=df_driver_forecast_updated,
                    cause_effect_df=data_loader.cause_effect_df,
                    shock_driver=code,
                    shock_month_index=col_index,
                    shock_value=new_value,
                    col_forecast="forecast_adj",
                )
                df_driver_forecast_updated.rename(columns={"FORECAST_WHATIF": "forecast_adj"}, inplace=True)

            edited_driver_codes = edits_df["Code"].dropna().unique().tolist()
            drivers_wave = get_drivers_wave(data_loader.cause_effect_df, edited_driver_codes)
            st.session_state.last_run_drivers_wave_count = len(drivers_wave)

            if "propagated_drivers" not in st.session_state:
                st.session_state.propagated_drivers = set(drivers_wave)
            else:
                current_drivers = st.session_state.propagated_drivers
                if isinstance(current_drivers, list):
                    current_drivers = set(current_drivers)
                current_drivers.update(drivers_wave)
                st.session_state.propagated_drivers = current_drivers

            st.session_state.drivers_forecast_df.drop(columns=["forecast_adj"], inplace=True)
            st.session_state.drivers_forecast_df = st.session_state.drivers_forecast_df.merge(
                df_driver_forecast_updated, on=["DRV_COD", "DATE_RIF"], how="left"
            ).copy()

            current_driver_compare_df = st.session_state.drivers_forecast_df[["DRV_COD", "DATE_RIF", "forecast_adj"]].copy()
            previous_driver_compare_df = previous_drivers_forecast_df.copy()
            previous_driver_compare_df["Month"] = previous_driver_compare_df["DATE_RIF"].dt.strftime("%Y-%m")
            current_driver_compare_df["Month"] = current_driver_compare_df["DATE_RIF"].dt.strftime("%Y-%m")
            previous_driver_compare_df = previous_driver_compare_df[
                previous_driver_compare_df["DRV_COD"].isin(drivers_wave)
                & (previous_driver_compare_df["Month"] >= min_month)
            ][["DRV_COD", "Month", "forecast_adj"]].rename(columns={"forecast_adj": "old_value"})
            current_driver_compare_df = current_driver_compare_df[
                current_driver_compare_df["DRV_COD"].isin(drivers_wave)
                & (current_driver_compare_df["Month"] >= min_month)
            ][["DRV_COD", "Month", "forecast_adj"]].rename(columns={"forecast_adj": "new_value"})
            driver_compare_df = previous_driver_compare_df.merge(
                current_driver_compare_df, on=["DRV_COD", "Month"], how="outer"
            )
            driver_changed_mask = (
                driver_compare_df["old_value"].isna() != driver_compare_df["new_value"].isna()
            ) | (
                ~np.isclose(
                    driver_compare_df["old_value"].fillna(0),
                    driver_compare_df["new_value"].fillna(0),
                    equal_nan=True,
                )
            )
            current_modified_driver_cells = st.session_state.modified_driver_cells
            if isinstance(current_modified_driver_cells, list):
                current_modified_driver_cells = set(current_modified_driver_cells)
            current_modified_driver_changes = st.session_state.modified_driver_changes
            if not isinstance(current_modified_driver_changes, dict):
                current_modified_driver_changes = dict(current_modified_driver_changes)
            for _, row in driver_compare_df.loc[driver_changed_mask, ["DRV_COD", "Month"]].iterrows():
                current_modified_driver_cells.add((row["DRV_COD"], row["Month"]))
            for _, row in driver_compare_df.loc[driver_changed_mask, ["DRV_COD", "Month", "old_value", "new_value"]].iterrows():
                current_modified_driver_changes[(row["DRV_COD"], row["Month"])] = {
                    "old_value": row["old_value"],
                    "new_value": row["new_value"],
                }
            st.session_state.modified_driver_cells = current_modified_driver_cells
            st.session_state.modified_driver_changes = current_modified_driver_changes

            df_driver_forecast_updated["DATE_RIF"] = df_driver_forecast_updated["DATE_RIF"].dt.strftime("%Y-%m")
            df_driver_forecast_updated = df_driver_forecast_updated.pivot(index="DRV_COD", columns="DATE_RIF", values="forecast_adj")
            common_drivers = st.session_state.selected_values_df["unique_id"]
            common_drivers = common_drivers[common_drivers.isin(df_driver_forecast_updated.index)]
            selected_df = st.session_state.selected_values_df.set_index("unique_id")
            common_months = [col for col in df_driver_forecast_updated.columns if col in selected_df.columns]

            selected_df.loc[common_drivers, df_driver_forecast_updated.columns] = df_driver_forecast_updated.loc[common_drivers, :]
            st.session_state.selected_values_df.update(selected_df.reset_index())
            eco_involved = data_loader.driver_selected_df[data_loader.driver_selected_df["DRV_COD"].isin(drivers_wave)]["ECO_COD"].unique()
            st.session_state.last_run_economics_wave_count = len(eco_involved)
            eco_involved_df = data_loader.eco_df[data_loader.eco_df["ECO_COD"].isin(eco_involved)].reset_index(drop=True)
            data_loader.drivers_forecast_df = st.session_state.drivers_forecast_df.copy()
            data_loader.drivers_forecast_df["FORECAST"] = data_loader.drivers_forecast_df["forecast_adj"]
            data_loader.drivers_forecast_df.drop(columns=["forecast_adj"], inplace=True)
            data_loader.eco_df = eco_involved_df.copy()

            if "propagated_economics" not in st.session_state:
                st.session_state.propagated_economics = set(eco_involved)
            else:
                current_eco = st.session_state.propagated_economics
                if isinstance(current_eco, list):
                    current_eco = set(current_eco)
                current_eco.update(eco_involved)
                st.session_state.propagated_economics = current_eco

            # Traccia gli economici modificati da run precedenti
            _modified = st.session_state.modified_economics
            if isinstance(_modified, list):
                _modified = set(_modified)
            _modified.update(eco_involved)
            st.session_state.modified_economics = _modified
            progress = st.progress(0)

            def update_progress(pct):
                progress.progress(pct)

            forecasting_model_eco = ForecastingModel(configuration_path=ECONOMICS_MODEL_CONFIG_PATH, freq=config.forecast.freq)
            forecast_economics(data_loader, forecasting_model_eco, make_plots=False, progress_callback=update_progress, n_jobs=-1)
            data_loader.eco_forecast_df.rename(columns={"y_hat": "y_hat_adj"}, inplace=True)
            df_rec_forecast_df = st.session_state.rec_forecast_df.copy()
            df_rec_forecast_df = df_rec_forecast_df[(df_rec_forecast_df["ECO_COD"].notna()) & (df_rec_forecast_df["ECO_COD"] != "")].reset_index(drop=True)
            df_rec_forecast_df = df_rec_forecast_df.merge(data_loader.eco_forecast_df, on=["ECO_COD", "DATE_RIF"], how="left")

            mask = (df_rec_forecast_df["ECO_COD"].isin(eco_involved)) & (df_rec_forecast_df["DATE_RIF"].dt.strftime("%Y-%m") >= min_month)
            df_rec_forecast_df.loc[mask, "forecast_adj"] = df_rec_forecast_df.loc[mask, "y_hat_adj"]
            df_rec_forecast_df.drop(columns=["y_hat_adj"], inplace=True)

            df_rec_forecast_df["forecast_adj"] = df_rec_forecast_df["forecast_adj"].fillna(0)
            df_aggr, S_eco_df, tags_eco = aggregate(df=df_rec_forecast_df, spec=config.forecast.spec, time_col="DATE_RIF", target_cols=("forecast_adj",))

            if VAR_FORECAST == "y_hat_rec":
                ele_eco_full = df_rec_forecast_df["ECO_COD"].unique()
                df_fit_eco = st.session_state.rec_fit_df[st.session_state.rec_fit_df["ECO_COD"].isin(ele_eco_full)].merge(data_loader.eco_anag, on="ECO_COD", how="left").copy()
                df_fit_aggr = data_loader.eco_group_fit_df.copy()
                df_forecast_eco = df_rec_forecast_df[config.forecast.spec[-1] + ["DATE_RIF", "forecast_adj"]].copy().rename(columns={"DATE_RIF": "ds", "forecast_adj": "y_hat"})
                df_forecast_aggr = data_loader.rec_forecast_df[data_loader.rec_forecast_df["ECO_COD"].isna()][config.forecast.spec[-1] + ["DATE_RIF", "y_hat"]].copy()
                df_fit_eco["unique_id"] = df_fit_eco[config.forecast.spec[-1]].fillna("").apply(lambda row: "/".join([str(x) for x in row if pd.notnull(x) and str(x) != ""]), axis=1)
                df_forecast_eco["unique_id"] = df_forecast_eco[config.forecast.spec[-1]].fillna("").apply(lambda row: "/".join([str(x) for x in row if pd.notnull(x) and str(x) != ""]), axis=1)
                df_fit_eco = df_fit_eco.rename(columns={"DATE_RIF": "ds"})[["unique_id", "ds", "y", "y_hat"]]
                df_forecast_eco = df_forecast_eco[["unique_id", "ds", "y_hat"]]
                df_fit_aggr["ECO_COD"] = None
                df_fit_aggr["unique_id"] = df_fit_aggr[config.forecast.spec[-1]].fillna("").apply(lambda row: "/".join([str(x) for x in row if pd.notnull(x) and str(x) != ""]), axis=1)
                df_fit_aggr = df_fit_aggr.rename(columns={"DATE_RIF": "ds"})[["unique_id", "ds", "y", "y_hat"]]
                df_forecast_aggr["unique_id"] = df_forecast_aggr[config.forecast.spec[-1]].fillna("").apply(lambda row: "/".join([str(x) for x in row if pd.notnull(x) and str(x) != ""]), axis=1)
                df_forecast_aggr = df_forecast_aggr.rename(columns={"DATE_RIF": "ds"})[["unique_id", "ds", "y_hat"]]
                df_fit = pd.concat([df_fit_eco, df_fit_aggr]).reset_index(drop=True)
                df_forecast = pd.concat([df_forecast_eco, df_forecast_aggr]).reset_index(drop=True)

                reconciler = Reconciliation(configuration_path=RECONCILIATION_APP_CONFIG_PATH)
                df_aggr = reconciler.reconcile_best(df_fct=df_forecast, df_fit=df_fit, S_df=S_eco_df, tags=tags_eco)
                df_aggr.rename(columns={"y_hat_rec": "forecast_adj", "ds": "DATE_RIF"}, inplace=True)
                df_aggr.drop(columns=["y_hat"], inplace=True)

            df_aggr[config.forecast.spec[-1]] = df_aggr["unique_id"].str.split("/", expand=True)
            df_aggr = df_aggr.drop(columns=["unique_id"])
            df_aggr = df_aggr.rename(columns={"forecast_adj": "forecast_adj_new"})
            st.session_state.rec_forecast_df = st.session_state.rec_forecast_df.merge(
                df_aggr, on=["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "ECO_COD", "DATE_RIF"], how="left"
            )

            st.session_state.rec_forecast_df["forecast_adj"] = st.session_state.rec_forecast_df["forecast_adj_new"]
            st.session_state.rec_forecast_df.drop(columns=["forecast_adj_new"], inplace=True)

            current_economic_compare_df = st.session_state.rec_forecast_df[
                (st.session_state.rec_forecast_df["ECO_COD"].notna()) & (st.session_state.rec_forecast_df["ECO_COD"] != "")
            ][["ECO_COD", "DATE_RIF", "forecast_adj"]].copy()
            previous_economic_compare_df = previous_rec_forecast_df.copy()
            previous_economic_compare_df["Month"] = previous_economic_compare_df["DATE_RIF"].dt.strftime("%Y-%m")
            current_economic_compare_df["Month"] = current_economic_compare_df["DATE_RIF"].dt.strftime("%Y-%m")
            previous_economic_compare_df = previous_economic_compare_df[
                previous_economic_compare_df["ECO_COD"].isin(eco_involved)
                & (previous_economic_compare_df["Month"] >= min_month)
            ][["ECO_COD", "Month", "forecast_adj"]].rename(columns={"forecast_adj": "old_value"})
            current_economic_compare_df = current_economic_compare_df[
                current_economic_compare_df["ECO_COD"].isin(eco_involved)
                & (current_economic_compare_df["Month"] >= min_month)
            ][["ECO_COD", "Month", "forecast_adj"]].rename(columns={"forecast_adj": "new_value"})
            economic_compare_df = previous_economic_compare_df.merge(
                current_economic_compare_df, on=["ECO_COD", "Month"], how="outer"
            )
            economic_changed_mask = (
                economic_compare_df["old_value"].isna() != economic_compare_df["new_value"].isna()
            ) | (
                ~np.isclose(
                    economic_compare_df["old_value"].fillna(0),
                    economic_compare_df["new_value"].fillna(0),
                    equal_nan=True,
                )
            )
            current_modified_economic_cells = st.session_state.modified_economic_cells
            if isinstance(current_modified_economic_cells, list):
                current_modified_economic_cells = set(current_modified_economic_cells)
            current_modified_economic_changes = st.session_state.modified_economic_changes
            if not isinstance(current_modified_economic_changes, dict):
                current_modified_economic_changes = dict(current_modified_economic_changes)
            for _, row in economic_compare_df.loc[economic_changed_mask, ["ECO_COD", "Month"]].iterrows():
                current_modified_economic_cells.add((row["ECO_COD"], row["Month"]))
            for _, row in economic_compare_df.loc[economic_changed_mask, ["ECO_COD", "Month", "old_value", "new_value"]].iterrows():
                current_modified_economic_changes[(row["ECO_COD"], row["Month"])] = {
                    "old_value": row["old_value"],
                    "new_value": row["new_value"],
                }
            st.session_state.modified_economic_cells = current_modified_economic_cells
            st.session_state.modified_economic_changes = current_modified_economic_changes

            # Mantieni la riga attiva selezionata anche dopo la run
            _selected_code_before_run = st.session_state.get("last_selected_code")
            if (
                "eco_anag_df" in st.session_state
                and "filtered_eco_anag_df" in st.session_state
                and st.session_state.eco_anag_df.selection.rows
            ):
                _idx_selected = st.session_state.eco_anag_df.selection.rows[0]
                _filtered_df_sel = st.session_state.filtered_eco_anag_df
                if _idx_selected < len(_filtered_df_sel):
                    _selected_code_before_run = _filtered_df_sel.iloc[_idx_selected]["Code"]

            st.session_state.last_selected_code = _selected_code_before_run
            if _selected_code_before_run is not None:
                apply_selection_by_code(_selected_code_before_run, force_refresh=True)
            else:
                get_row_select()

            st.rerun()
            time.sleep(2)
        elapsed = time.time() - start

    st.markdown(text_on_drv_click, unsafe_allow_html=True)
    if preview_drivers_count is not None and preview_economics_count is not None:
        post_drivers_count = st.session_state.get("last_run_drivers_wave_count")
        post_economics_count = st.session_state.get("last_run_economics_wave_count")
        post_drivers_label = post_drivers_count if post_drivers_count is not None else "—"
        post_economics_label = post_economics_count if post_economics_count is not None else "—"
        st.caption(
            f"Preview wave — Drivers: {preview_drivers_count}, Economics: {preview_economics_count} | "
            f"Last Run — Drivers: {post_drivers_label}, Economics: {post_economics_label}"
        )

# CSS Custom
st.markdown(
    """
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stPlotlyChart { border: 1px solid #f0f2f6; border-radius: 5px; }
    .st-key-plot_det .stPlotlyChart,
    .st-key-plot_importance .stPlotlyChart,
    .st-key-plot_g2 .stPlotlyChart,
    .st-key-plot_g1 .stPlotlyChart,
    .st-key-plot_g0 .stPlotlyChart,
    .st-key-plot_det.stPlotlyChart,
    .st-key-plot_importance.stPlotlyChart,
    .st-key-plot_g2.stPlotlyChart,
    .st-key-plot_g1.stPlotlyChart,
    .st-key-plot_g0.stPlotlyChart {
        border: none !important;
        border-radius: 0 !important;
    }
    
    .stMainBlockContainer {
        padding-top: 2.2rem !important; 
        padding-left: max(0px, 0px + 1.25rem) !important; 
        padding-right: max(0px, 0px + 1.25rem) !important;
    }

    header { visibility: visible !important; height: 2.2rem !important; }
    [data-testid="stHeader"] {
        display: block !important;
        background: transparent !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #00123b !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stVerticalBlock button {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        color: white !important;
        box-shadow: none !important;
        outline: none !important;
        transition: none !important;
    }

    section[data-testid="stSidebar"] button[kind="tertiary"] {
        background-color: transparent !important;
        border: none !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] button[kind="tertiary"]:hover,
    section[data-testid="stSidebar"] button[kind="tertiary"]:active,
    section[data-testid="stSidebar"] button[kind="tertiary"]:focus,
    section[data-testid="stSidebar"] button[kind="tertiary"]:focus-visible {
        background-color: transparent !important;
        border: none !important;
        color: white !important;
        box-shadow: none !important;
        outline: none !important;
    }

    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="primary"]:hover,
    section[data-testid="stSidebar"] button[kind="primary"]:active,
    section[data-testid="stSidebar"] button[kind="primary"]:focus,
    section[data-testid="stSidebar"] button[kind="primary"]:focus-visible {
        background-color: orange !important;
        border: 1px solid orange !important;
        color: white !important;
        box-shadow: none !important;
        outline: none !important;
    }

    section[data-testid="stSidebar"] button[aria-pressed="true"],
    section[data-testid="stSidebar"] button[aria-selected="true"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
    section[data-testid="stSidebar"] button[data-baseweb="button"][data-kind="primary"],
    section[data-testid="stSidebar"] button[aria-pressed="true"]:hover,
    section[data-testid="stSidebar"] button[aria-pressed="true"]:active,
    section[data-testid="stSidebar"] button[aria-pressed="true"]:focus,
    section[data-testid="stSidebar"] button[aria-pressed="true"]:focus-visible {
        background-color: orange !important;
        border-color: orange !important;
        color: white !important;
        box-shadow: none !important;
        outline: none !important;
    }

    section[data-testid="stSidebar"] .stVerticalBlock button > div,
    section[data-testid="stSidebar"] .stVerticalBlock button span,
    section[data-testid="stSidebar"] .stVerticalBlock button span div,
    section[data-testid="stSidebar"] .stVerticalBlock button p {
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        color: white !important;
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #00123b !important;
        border: none !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] details,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
    section[data-testid="stSidebar"] [data-testid="stExpander"] div {
        background: #00123b !important;
        border: none !important;
        box-shadow: none !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-visible,
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary,
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > div {
        background: #00123b !important;
        color: white !important;
        box-shadow: none !important;
        outline: none !important;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] details > div {
        margin-top: 0 !important;
        padding-top: 0.15rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stAlert"] {
        margin-top: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #00123b !important;
        border: none !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] [data-testid="stAlert"] > div,
    section[data-testid="stSidebar"] [data-testid="stAlert"] [role="alert"],
    section[data-testid="stSidebar"] [data-testid="stAlert"] [data-baseweb="notification"] {
        background-color: #00123b !important;
        border: none !important;
        box-shadow: none !important;
        color: white !important;
    }

    /* Compattiamo i titoli*/
    h1, h2, h3, h4 {
        margin-top: 1rem !important;
        padding-top: 0px !important;
    }

    /* ELEMENTO CHIAVE: Spazio fisico tra la tabella superiore e quella dei driver */
    .driver-section-spacer {
        height: 40px;
        display: block;
        clear: both;
    }

    /* Spinge giù l'intera colonna dei grafici */
    .plots-container {
        margin-top: 60px !important; 
    }

    .driver-section-spacer {
    height: 40px; /* Spazio verticale garantito tra le tabelle */
    display: block;
    clear: both; /* Impedisce la sovrapposizione con elementi precedenti */
}

</style>
""",
    unsafe_allow_html=True,
)
