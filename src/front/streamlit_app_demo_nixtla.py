import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from hierarchicalforecast.utils import aggregate
from loguru import logger
import time

# logger.add("app.log")


TP_SOURCE = os.environ.get("TP_SOURCE")

# --- IMPORT LIBRARIES ---
from src.back.configuration import PIPELINE_DEFAULT_CONFIGURATION as config
from src.back.data import DataLoader
from src.back.pipeline import *
from src.back.whatif import propagate_shock, get_drivers_wave
from src.back.reconciliation import Reconciliation

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "src" / "db" / TP_SOURCE / "bankfcs_anonymized.db"
VAR_FORECAST = "y_hat_rec"
MODELS_WITH_EXO_VAR = ['arima_1', 'arima_12', 'mfles_12']


st.set_page_config(
    page_title="iLabs BankFCS",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
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
    loader.driver_selected_df = loader.driver_selected_df[loader.driver_selected_df['ECO_COD'].isin(loader.eco_best_model_df[loader.eco_best_model_df['model'].isin(MODELS_WITH_EXO_VAR)]['ECO_COD'].unique())].reset_index(drop=True)

    logger.info(f"Sistemazione dati riconciliati {path} \n")
    # Separa i dati salvati a livello di dettaglio (con ECO_COD) da quelli aggregati (senza ECO_COD)
    rec_forecast_df_details = loader.rec_forecast_df[loader.rec_forecast_df["ECO_COD"].notna()].reset_index(drop=True)
    rec_forecast_df_aggr = loader.rec_forecast_df[loader.rec_forecast_df["ECO_COD"].isna()].reset_index(drop=True)[["DATE_RIF"] + config.forecast.spec[-1] + ["y_hat", "y_hat_rec"]]
    # A partire dai dati di dettaglio ricalcolo la somma dei forecast
    rec_forecast_df_sum, _, _ = aggregate(
        df=rec_forecast_df_details,
        spec=config.forecast.spec,
        time_col="DATE_RIF",
        target_cols=("y_hat",),
    )
    rec_forecast_df_sum[config.forecast.spec[-1]] = rec_forecast_df_sum["unique_id"].str.split("/", expand=True)
    rec_forecast_df_sum = rec_forecast_df_sum[rec_forecast_df_sum["ECO_COD"].isna()].reset_index(drop=True)
    rec_forecast_df_sum = rec_forecast_df_sum.rename(columns={"y_hat": "y_hat_sum"}).drop(columns=["unique_id"])[["DATE_RIF"] + config.forecast.spec[-1] + ["y_hat_sum"]]

    rec_forecast_df_sum = rec_forecast_df_aggr.merge(rec_forecast_df_sum, on=(config.forecast.spec[-1] + ["DATE_RIF"]), how="left")
    rec_forecast_df_details['y_hat_sum'] = rec_forecast_df_details['y_hat']
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
    eco_grp_budget = eco_grp_budget[eco_grp_budget['DATE_RIF'].isin(rec_forecast_df['DATE_RIF'].unique())].reset_index(drop=True)
    return loader, eco_grp_budget


# Chiamata unica
data_loader, eco_grp_budget = load_and_prepare_budget(DB_PATH)
if "drivers_forecast_df" not in st.session_state:
    st.session_state.drivers_forecast_df = data_loader.drivers_forecast_df.copy()
    st.session_state.rec_forecast_df = data_loader.rec_forecast_df.copy()
    st.session_state.rec_fit_df = data_loader.eco_fit_df.copy()


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
    # Preselezione automatica della prima area
    # prime_aree = st.session_state.eco_anagrafica["ECO_GRP_0_DSC"].unique()
    # if len(prime_aree) > 0:
    #     st.session_state.selected_area = prime_aree[0]

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
    if (count_y>0):
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
        go.Scatter(x=df_plot["ds"], y=df_plot["y"], mode="lines", name="Actual", line=dict(color="black", width=2))
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
    if count_y<=0:
        # Budget
        fig.add_trace(
            go.Scatter(x=df_plot["ds"], y=df_plot["budget"], mode="lines", name="Budget", line=dict(color="green", width=2))
        )
    if cutoff_date:
        fig.add_vline(x=cutoff_date, line_width=1.5, line_dash="dash", line_color="grey")
    # Layout
    if df_plot_agg.shape[0] > 0:
        if count_y>0:
            annotext = (
                "AI Adj="
                + str(round(df_plot_agg["forecast_adj"][0]))
                + "| Actual="
                + str(round(df_plot_agg["y"][0]))
                + "| Delta="
                + str(round(df_plot_agg["delta"][0]))
                + " ("
                + str(round(df_plot_agg["mape"][0], 2))
                + "%)"
            )
        else:
            annotext = (
                "AI Adj="
                + str(round(df_plot_agg["forecast_adj"][0]))
                + "| Budget="
                + str(round(df_plot_agg["budget"][0]))
                + "| Delta="
                + str(round(df_plot_agg["delta"][0]))
                + " ("
                + str(round(df_plot_agg["mape"][0], 2))
                + "%)"
            )
    else:
        annotext = ""

    fig.update_layout(
        title=title,
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
        df_hist = data_loader.eco_df.merge(data_loader.eco_anag[["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "ECO_COD"]].astype(str), on="ECO_COD", how="left").rename(
            columns={"VALUE": "y"}
        )

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


def get_forecast_values():
    selected_row_index = st.session_state.eco_anag_df.selection.rows[0]
    selected_eco_code = st.session_state.current_eco_df.iloc[selected_row_index]["ECO_COD"]
    df_eco_forecast = st.session_state.rec_forecast_df[st.session_state.rec_forecast_df["ECO_COD"] == selected_eco_code]
    df_eco_forecast = df_eco_forecast.rename(
        columns={"ECO_COD": "unique_id", "DATE_RIF": "ds", "forecast_adj": "forecast_adj"}
    )
    df_eco_forecast = df_eco_forecast[["unique_id", "ds", "forecast_adj"]]
    df_eco_anag = data_loader.eco_anag[data_loader.eco_anag["ECO_COD"] == selected_eco_code][
        ["ECO_COD", "ECO_DSC"]
    ].rename(columns={"ECO_COD": "unique_id", "ECO_DSC": "description"})
    df_eco_anag["is_drv"] = False
    df_eco_anag["order"] = 999
    drivers_selected_eco = data_loader.driver_selected_df[
        data_loader.driver_selected_df["ECO_COD"] == selected_eco_code
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
    ).query("ECO_COD == @selected_eco_code")
    driver_selection_df["COEFF"] = driver_selection_df["COEFF"].abs()
    driver_selection_df.sort_values(by="COEFF", ascending=False, inplace=True)
    selected_drivers = driver_selection_df["DRV_COD"].tolist()
    st.session_state.selected_drivers_ordered = selected_drivers
    st.session_state.selected_values_df = df_values


def get_row_select():
    if not st.session_state.eco_anag_df.selection.rows:
        return
    idx = st.session_state.eco_anag_df.selection.rows[0]
    row = st.session_state.current_eco_df.iloc[idx]

    for k in ["g0_fig", "g1_fig", "g2_fig", "det_fig"]:
        st.session_state[k] = None

    # Plot different aggregations
    st.session_state.selected_labels = {
        "g0": row["ECO_GRP_0_DSC"],
        "g1": row["ECO_GRP_1_DSC"],
        "g2": row["ECO_GRP_2_DSC"] if pd.notnull(row["ECO_GRP_2"]) else None,
        "det": f"{row['ECO_COD']} - {row['ECO_DSC']}",
    }
    st.session_state.g0_fig = generate_plotly_plot(
        get_data_at_level(row, 0), title=f"Total {st.session_state.selected_labels['g0']}", height=300
    )
    st.session_state.g1_fig = generate_plotly_plot(
        get_data_at_level(row, 1), title=f"Total {st.session_state.selected_labels['g1']}", height=300
    )
    if pd.notnull(row["ECO_GRP_2"]):
        st.session_state.g2_fig = generate_plotly_plot(
            get_data_at_level(row, 2), title=f"Total {st.session_state.selected_labels['g2']}", height=300
        )
    st.session_state.det_fig = generate_plotly_plot(
        get_data_at_level(row, 3), title=f"{st.session_state.selected_labels['det']}", height=300
    )
    get_forecast_values()


def select_area(area):
    # Aggiorna l'area selezionata
    st.session_state.selected_area = area

    # Resetta i dati dei driver (tabella 2)
    st.session_state.selected_values_df = None

    # Resetta i riferimenti ai grafici (colonna t2)
    for k in ["g0_fig", "g1_fig", "g2_fig", "det_fig"]:
        st.session_state[k] = None

    # Opzionale: pulisce anche le etichette selezionate
    if "selected_labels" in st.session_state:
        st.session_state.selected_labels = None


def add_edited_drivers():
    if "edited_drivers" not in st.session_state:
        st.session_state.edited_drivers = set()
    for row_index in st.session_state.value_df["edited_rows"].keys():
        st.session_state.edited_drivers.add(st.session_state.selected_values_df.iloc[row_index]["unique_id"])

keys_to_init = ["g0_fig", "g1_fig", "g2_fig", "det_fig", "selected_values_df"]
for key in keys_to_init:
    if key not in st.session_state:
        st.session_state[key] = None

# --- UI ---
# st.sidebar.header("Area", divider="blue")
st.sidebar.title("iLabs BankFCS")
st.sidebar.header("Area", divider="blue")

# Inizializza selected_area come None se non esiste
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None

for area in st.session_state.eco_anagrafica["ECO_GRP_0_DSC"].unique():
    # Ora il confronto funzionerà correttamente (False al primo avvio)
    is_selected = st.session_state.selected_area == area
    st.sidebar.button(
        f"- {area}",
        type="primary" if is_selected else "tertiary",
        on_click=select_area,
        args=(area,),
        key=f"btn_{area}",
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


t1, t2 = st.columns([0.6, 0.4])
# Anagrafic Economics table
with t1:
    if st.session_state.selected_area is None:
        st.info("Select an area from the sidebar to view related economics and forecasts.")
    else:
        # --- Tabella Superiore (Voices) ---
        with st.container():
            st.markdown(f"### {st.session_state.selected_area}: voices")

            # Logica di caricamento dati (rimane invariata)
            if st.session_state.get("selected_area") is not None:
                st.session_state.current_eco_df = st.session_state.eco_anagrafica[
                    st.session_state.eco_anagrafica["ECO_GRP_0_DSC"] == st.session_state.selected_area
                ].reset_index(drop=True)
            else:
                st.session_state.current_eco_df = st.session_state.eco_anagrafica.copy().reset_index(drop=True)
            eco_df = st.session_state.current_eco_df

            # RIMOSSO 'height': ora la tabella occupa lo spazio necessario
            # e sposta dinamicamente tutto ciò che segue.
            st.dataframe(
                data=eco_df[["ECO_GRP_0_DSC", "ECO_GRP_1_DSC", "ECO_GRP_2_DSC", "ECO_COD", "ECO_DSC", "PERC"]],
                key="eco_anag_df",
                on_select=get_row_select,
                selection_mode="single-row",
                width="stretch",
                column_config={
                    "ECO_GRP_0_DSC": None,
                    "ECO_GRP_1_DSC": "Sector",
                    "ECO_GRP_2_DSC": "Segment",
                    "ECO_COD": "Code",
                    "ECO_DSC": "Description",
                    "PERC": "Weight %",
                },
            )

# --- Table Values Section (Dinamica) ---
if st.session_state.get("selected_values_df") is not None:
    with t1:
        # Spacer per garantire distacco dinamico
        st.markdown('<div class="driver-section-spacer"></div>', unsafe_allow_html=True)

        df = st.session_state.selected_values_df
        st.markdown(f"#### Forecast of {df.loc[0, 'unique_id']} and related drivers")

        cols_forecast = (
            st.session_state.drivers_forecast_df["DATE_RIF"].dt.strftime("%Y-%m").astype(str).unique().tolist()
        )
        cols_to_freeze = df.columns.difference(cols_forecast).tolist()
        df_display = df.style.set_properties(
            subset=cols_to_freeze,
            **{
                "color": "gray",
                "background-color": "#f9f9f9",  # Optional: adds a slight shading to the background
            },
        )
        logger.info(f"Colonne da freezare: {cols_to_freeze}")

        st.data_editor(
            df_display,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=cols_to_freeze,
            key="value_df",
            column_config={
                "unique_id": st.column_config.TextColumn("Code"),
                "COEFF": st.column_config.NumberColumn("Importance", format="%.3f"),
                "description": st.column_config.TextColumn("Description", width="medium"),
                "is_drv": None,
                "order": None,
            }
            | {
                c: st.column_config.NumberColumn(c, format="%.3f")
                for c in df.columns
                if c not in ["description", "unique_id", "COEFF", "is_drv", "order"]
            },
            on_change=add_edited_drivers,
        )

        text_on_drv_click = ""
        if not (
                "value_df" not in st.session_state
                or "edited_rows" not in st.session_state.value_df
                or len(st.session_state.value_df["edited_rows"]) == 0
            ):
            drv_click = []
            for row_index, row in st.session_state.value_df["edited_rows"].items():
                if row_index > 0:
                    drv_click.append(df.iloc[row_index]["unique_id"])
            drv_click_wave = get_drivers_wave(data_loader.cause_effect_df, drv_click)                
            drv_click_wave_descr = data_loader.driver_anag[data_loader.driver_anag["DRV_COD"].isin(drv_click_wave)][['DRV_COD', 'DRV_DSC']]
            drv_click_wave_descr = (drv_click_wave_descr['DRV_COD'] + ": " + drv_click_wave_descr['DRV_DSC']).tolist()
            eco_click_wave = data_loader.driver_selected_df[data_loader.driver_selected_df["DRV_COD"].isin(drv_click_wave)]["ECO_COD"].unique()
            eco_click_wave_descr = data_loader.eco_anag[data_loader.eco_anag["ECO_COD"].isin(eco_click_wave)][['ECO_COD', 'ECO_DSC']]
            eco_click_wave_descr = (eco_click_wave_descr['ECO_COD'] + ": " + eco_click_wave_descr['ECO_DSC']).tolist()
            if len(drv_click_wave_descr) > 0:
                drv_list = "<ul>" + "".join(f"<li>{item}</li>" for item in drv_click_wave_descr) + "</ul>"
                eco_list = "<ul>" + "".join(f"<li>{item}</li>" for item in eco_click_wave_descr) + "</ul>"
                text_on_drv_click = (
                "**Execution time will be proportional to the number of economics affected, but generally should not exceed 1-2 minutes!**<br><br>" +
                "Running this will propagate current edits to the following " + str(len(drv_click_wave_descr)) + " drivers:" + drv_list +
                "and to the following " + str(len(eco_click_wave_descr)) + " economics:" + eco_list                
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
            
                edits = []
                for row_index, row in st.session_state.value_df["edited_rows"].items():
                    for col, new_value in row.items():
                        edits.append((df.iloc[row_index]["unique_id"], col, new_value))
                edits_df = pd.DataFrame(edits, columns=["Code", "Month", "New_Value"])
                edits_df["Code"] = pd.Categorical(
                    edits_df["Code"], categories=st.session_state.selected_drivers_ordered, ordered=True
                )
                edits_df.sort_values(by=["Code"], inplace=True)
                edits_df.sort_values(by=["Month"], inplace=True)
                # drivers_wave = get_drivers_wave(data_loader.cause_effect_df, edits_df["Code"].unique().tolist())
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

                # Trova tutti i drivers diversi                
                merged = st.session_state.drivers_forecast_df[["DRV_COD", "DATE_RIF", "forecast_adj"]].merge(
                    df_driver_forecast_updated[["DRV_COD", "DATE_RIF", "forecast_adj"]],
                    on=["DRV_COD", "DATE_RIF"],
                    how="outer",
                    suffixes=("_pre", "_post"),
                    indicator=True,
                )
                merged["diff"] =  round(merged["forecast_adj_pre"] - merged["forecast_adj_post"], 8)
                merged["diff"] = merged["diff"]!=0
                drivers_wave  = merged.loc[merged["diff"], "DRV_COD"].unique()
                
                if "propagated_drivers" not in st.session_state:
                    st.session_state.propagated_drivers = set(drivers_wave)
                else:
                    # Assicurati che sia un set prima di aggiornare
                    current_drivers = st.session_state.propagated_drivers
                    if isinstance(current_drivers, list):
                        current_drivers = set(current_drivers)
                    current_drivers.update(drivers_wave)
                    st.session_state.propagated_drivers = current_drivers
                # st.session_state.propagated_drivers = sorted(st.session_state.propagated_drivers)

                # Aggiorno la tabella dei driver con i nuovi forecast propagati
                st.session_state.drivers_forecast_df.drop(columns=["forecast_adj"], inplace=True)
                st.session_state.drivers_forecast_df = st.session_state.drivers_forecast_df.merge(
                    df_driver_forecast_updated, on=["DRV_COD", "DATE_RIF"], how="left"
                ).copy()
                df_driver_forecast_updated["DATE_RIF"] = df_driver_forecast_updated["DATE_RIF"].dt.strftime("%Y-%m")
                df_driver_forecast_updated = df_driver_forecast_updated.pivot(
                    index="DRV_COD", columns="DATE_RIF", values="forecast_adj"
                )
                common_drivers = st.session_state.selected_values_df["unique_id"]
                common_drivers = common_drivers[common_drivers.isin(df_driver_forecast_updated.index)]
                selected_df = st.session_state.selected_values_df.set_index("unique_id")
                selected_df.loc[common_drivers, df_driver_forecast_updated.columns] = df_driver_forecast_updated.loc[
                    common_drivers, :
                ]
                st.session_state.selected_values_df.update(selected_df.reset_index())
                # Identify involved economics based on the edited drivers
                eco_involved = data_loader.driver_selected_df[
                    data_loader.driver_selected_df["DRV_COD"].isin(drivers_wave)
                ]["ECO_COD"].unique()
                # Retrieve data related to the involved economics
                eco_involved_df = data_loader.eco_df[data_loader.eco_df["ECO_COD"].isin(eco_involved)].reset_index(drop=True)
                # Create a local copy of the data loader to avoid modifying the cached version
                data_loader.drivers_forecast_df = st.session_state.drivers_forecast_df.copy()
                data_loader.drivers_forecast_df["FORECAST"] = data_loader.drivers_forecast_df["forecast_adj"]
                data_loader.drivers_forecast_df.drop(columns=["forecast_adj"], inplace=True)
                data_loader.eco_df = eco_involved_df.copy()                

                if "propagated_economics" not in st.session_state:
                    st.session_state.propagated_economics = set(eco_involved)
                else:
                    # Assicurati che sia un set prima di aggiornare
                    current_eco = st.session_state.propagated_economics
                    if isinstance(current_eco, list):
                        current_eco = set(current_eco)
                    current_eco.update(eco_involved)
                    st.session_state.propagated_economics = current_eco
                # Re-forecast all economics involved
                progress = st.progress(0)
                def update_progress(pct):
                    progress.progress(pct)      
                forecasting_model_eco = ForecastingModel(
                    configuration_path=ECONOMICS_MODEL_CONFIG_PATH, freq=config.forecast.freq
                )
                forecast_economics(data_loader, forecasting_model_eco, make_plots=False, progress_callback=update_progress, n_jobs=-1)
                # Save results in session state and update the economics table with new forecasts
                data_loader.eco_forecast_df.rename(columns={"y_hat": "y_hat_adj"}, inplace=True)
                df_rec_forecast_df = st.session_state.rec_forecast_df.copy()
                df_rec_forecast_df = df_rec_forecast_df[
                    (df_rec_forecast_df["ECO_COD"].notna()) & (df_rec_forecast_df["ECO_COD"] != "")
                ].reset_index(drop=True)
                df_rec_forecast_df = df_rec_forecast_df.merge(
                    data_loader.eco_forecast_df, on=["ECO_COD", "DATE_RIF"], how="left"
                )
                mask = df_rec_forecast_df["ECO_COD"].isin(eco_involved)
                df_rec_forecast_df.loc[mask, "forecast_adj"] = df_rec_forecast_df.loc[mask, "y_hat_adj"]
                df_rec_forecast_df.drop(columns=["y_hat_adj"], inplace=True)  

                # Recalculate sum
                df_rec_forecast_df["forecast_adj"] = df_rec_forecast_df["forecast_adj"].fillna(0)                
                df_aggr, S_eco_df, tags_eco = aggregate(
                        df=df_rec_forecast_df,
                        spec=config.forecast.spec,
                        time_col="DATE_RIF",
                        target_cols=("forecast_adj",),
                    )                  
                
                # If the forecast variable is the reconciled one, we need to run reconciliation again to adjust the forecasts at all levels based on the new economic forecasts
                if VAR_FORECAST == 'y_hat_rec':
                    ele_eco_full = df_rec_forecast_df['ECO_COD'].unique()
                    df_fit_eco =  st.session_state.rec_fit_df[st.session_state.rec_fit_df['ECO_COD'].isin(ele_eco_full)].merge(data_loader.eco_anag, on="ECO_COD", how="left").copy()                 
                    df_fit_aggr = data_loader.eco_group_fit_df.copy()                    
                    df_forecast_eco = df_rec_forecast_df[config.forecast.spec[-1] + ['DATE_RIF', 'forecast_adj']].copy().rename(columns={'DATE_RIF': 'ds','forecast_adj': 'y_hat'})                    
                    df_forecast_aggr = data_loader.rec_forecast_df[data_loader.rec_forecast_df["ECO_COD"].isna()][config.forecast.spec[-1] + ['DATE_RIF','y_hat']].copy()
                    df_fit_eco['unique_id'] = df_fit_eco[config.forecast.spec[-1]].fillna('').apply(lambda row: '/'.join([str(x) for x in row if pd.notnull(x) and str(x) != '']),axis=1)
                    df_forecast_eco['unique_id'] = df_forecast_eco[config.forecast.spec[-1]].fillna('').apply(lambda row: '/'.join([str(x) for x in row if pd.notnull(x) and str(x) != '']),axis=1)
                    df_fit_eco = df_fit_eco.rename(columns={'DATE_RIF': 'ds'})[['unique_id', 'ds', 'y', 'y_hat']]
                    df_forecast_eco = df_forecast_eco[['unique_id', 'ds', 'y_hat']]
                    df_fit_aggr['ECO_COD'] = None
                    df_fit_aggr['unique_id'] = df_fit_aggr[config.forecast.spec[-1]].fillna('').apply(lambda row: '/'.join([str(x) for x in row if pd.notnull(x) and str(x) != '']),axis=1)
                    df_fit_aggr = df_fit_aggr.rename(columns={'DATE_RIF': 'ds'})[['unique_id', 'ds', 'y', 'y_hat']]
                    df_forecast_aggr['unique_id'] = df_forecast_aggr[config.forecast.spec[-1]].fillna('').apply(lambda row: '/'.join([str(x) for x in row if pd.notnull(x) and str(x) != '']),axis=1)
                    df_forecast_aggr = df_forecast_aggr.rename(columns={'DATE_RIF': 'ds'})[['unique_id', 'ds', 'y_hat']]
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
                get_row_select()
                st.rerun()
                time.sleep(2)  # Simula elaborazione
            elapsed = time.time() - start

        st.markdown(text_on_drv_click, unsafe_allow_html=True)
        

with t2:
    # Creiamo un div che sposta tutto il contenuto di t2 verso il basso
    st.markdown('<div class="plots-container">', unsafe_allow_html=True)

    if st.session_state.get("det_fig"):
        st.plotly_chart(st.session_state.det_fig, width="stretch", key="plot_det")

    if st.session_state.get("g2_fig"):
        st.plotly_chart(st.session_state.g2_fig, width="stretch", key="plot_g2")

    if st.session_state.get("g1_fig"):
        st.plotly_chart(st.session_state.g1_fig, width="stretch", key="plot_g1")

    if st.session_state.get("g0_fig"):
        st.plotly_chart(st.session_state.g0_fig, width="stretch", key="plot_g0")

    st.markdown("</div>", unsafe_allow_html=True)

# CSS Custom
st.markdown(
    """
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stPlotlyChart { border: 1px solid #f0f2f6; border-radius: 5px; }
    
    .stMainBlockContainer {
        padding-top: 1.5rem !important; 
        padding-left: max(0px, 0px + 1.25rem) !important; 
        padding-right: max(0px, 0px + 1.25rem) !important;
    }

    header { visibility: hidden; height: 0px; }
    
    .stSidebar { width: 100px !important; }
    .stSidebar .stVerticalBlock button span div { font-size: 0.9rem; }

    button[kind="primary"] {
        background-color: Orange;
        color: white !important;
        border: none !important;
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

