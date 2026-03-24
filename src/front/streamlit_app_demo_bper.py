# --- TP_SOURCE ---
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from loguru import logger
from hierarchicalforecast.utils import aggregate

TP_SOURCE = os.environ.get("TP_SOURCE")

# --- IMPORT LIBRARIES ---
from src.back.configuration import PIPELINE_DEFAULT_CONFIGURATION as config
from src.back.data import DataLoader

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "src" / "db" / TP_SOURCE / "bankfcs_anonymized.db"
VAR_FORECAST = 'y_hat_rec'


st.set_page_config(
    page_title="BankLab TS",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CACHE & DATA LOADING ---
@st.cache_resource
def get_data_loader(path):
    logger.info(f"Inizializzazione DataLoader presso {path}")
    return DataLoader(path)

data_loader = get_data_loader(DB_PATH)

@st.cache_data
def get_eco_anagrafica():
    eco_anag = data_loader.eco_anag.copy()
    eco_fcst = data_loader.eco_forecast_df['ECO_COD'].drop_duplicates()
    eco_df = data_loader.eco_df.copy()
    eco_df['VALUE'] = eco_df['VALUE'].abs()
    eco_df = eco_df.merge(eco_anag, on="ECO_COD", how="inner").merge(eco_fcst, on="ECO_COD", how="inner")
    eco_grp, _, _ = aggregate(df=eco_df, spec=config.forecast.spec, time_col="DATE_RIF", target_cols=("VALUE",),)
    cols = config.forecast.spec[-1]
    eco_grp = eco_grp.groupby("unique_id")["VALUE"].sum().reset_index().rename(columns={"VALUE": "WEIGHT"})
    eco_grp[cols] = eco_grp['unique_id'].str.split('/', expand=True)
    eco_df = eco_grp[eco_grp['ECO_COD'].notna()].reset_index(drop=True)[["ECO_COD", "WEIGHT"]]
    eco_grp = eco_grp[(eco_grp['ECO_COD'].isna()) & (eco_grp["ECO_GRP_2"].isna()) & (eco_grp["ECO_GRP_1"].isna())].reset_index(drop=True)[["ECO_GRP_0", "WEIGHT"]].rename(columns={"WEIGHT": "WEIGHT_AGGR"})
    eco_anag = eco_anag.merge(eco_df, on="ECO_COD", how="inner").merge(eco_grp, on=["ECO_GRP_0"], how="left")
    eco_anag['PERC'] = np.round(eco_anag["WEIGHT"] / eco_anag["WEIGHT_AGGR"] * 100, 2)
    eco_anag["ECO_GRP_2_DSC"] = eco_anag["ECO_GRP_2_DSC"].str.replace("Segmento", "", case=False).str.strip()
    eco_anag.loc[eco_anag['ECO_GRP_2_DSC'] == '_', 'ECO_GRP_2_DSC'] = np.nan
    eco_anag.loc[eco_anag['ECO_GRP_2'] == '_', 'ECO_GRP_2'] = np.nan
    return eco_anag.sort_values(by=["ECO_GRP_0_DSC", "PERC", "ECO_GRP_1_DSC", "ECO_GRP_2_DSC"], ascending=[True, False, True, True]).reset_index(drop=True)

if "eco_anagrafica" not in st.session_state:
    st.session_state.eco_anagrafica = get_eco_anagrafica()

# --- BUSINESS & PLOTTING ---

def generate_plotly_plot(df_plot, title="", height=350):
    if df_plot.empty:
        return None
    mask_pred = df_plot["forecast"].notnull()
    cutoff_date = df_plot[mask_pred]["ds"].min() if mask_pred.any() else None  
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["y"], mode='lines', name='Actual', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=df_plot["ds"], y=df_plot["forecast"], mode='lines', name='Forecast', line=dict(color='orange', width=2)))
    if cutoff_date:
        fig.add_vline(x=cutoff_date, line_width=1.5, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=title, # Rimarrà vuoto come da chiamata sotto
        height=height, template="simple_white",
        margin=dict(l=10, r=10, t=10, b=10), # Margine superiore ridotto perché il titolo è fuori
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def get_data_at_level(row, level):
    df_fcst = data_loader.rec_forecast_df
    if level < 3:
        df_hist = data_loader.eco_group_df
    else:
        df_hist = data_loader.eco_df.merge(data_loader.eco_anag, on="ECO_COD", how="left").rename(columns={"VALUE": "y"})    
    f_fcst = (df_fcst["ECO_GRP_0"] == str(row["ECO_GRP_0"]))
    f_hist = (df_hist["ECO_GRP_0"] == str(row["ECO_GRP_0"]))    
    if level >= 1:
        f_fcst &= (df_fcst["ECO_GRP_1"] == str(row["ECO_GRP_1"]))
        f_hist &= (df_hist["ECO_GRP_1"] == str(row["ECO_GRP_1"]))
    if level >= 2:
        f_fcst &= (df_fcst["ECO_GRP_2"] == str(row["ECO_GRP_2"]))
        f_hist &= (df_hist["ECO_GRP_2"] == str(row["ECO_GRP_2"]))
    if level >= 3:
        f_fcst &= (df_fcst["ECO_COD"] == str(row["ECO_COD"]))
        f_hist &= (df_hist["ECO_COD"] == str(row["ECO_COD"]))
    if level == 0:
        f_fcst &= df_fcst["ECO_GRP_1"].isnull(); f_fcst &= df_fcst["ECO_GRP_2"].isnull(); f_fcst &= df_fcst["ECO_COD"].isnull()
        f_hist &= df_hist["ECO_GRP_1"].isnull(); f_hist &= df_hist["ECO_GRP_2"].isnull()
    elif level == 1:
        f_fcst &= df_fcst["ECO_GRP_2"].isnull(); f_fcst &= df_fcst["ECO_COD"].isnull()
        f_hist &= df_hist["ECO_GRP_2"].isnull()
    elif level == 2:
        f_fcst &= df_fcst["ECO_COD"].isnull()
    elif level == 3:
        f_fcst = (df_fcst["ECO_COD"] == row["ECO_COD"])
        f_hist = (df_hist["ECO_COD"] == row["ECO_COD"])
    df_fcst = df_fcst[f_fcst].copy()
    df_hist = df_hist[f_hist].copy()
    # if level == 3:
    #     ########################
    #     # TAROCCO DA TOGLIERE
    #     if row["ECO_COD"] == 'ECO_021':
    #         df_fcst['y_hat_rec'] = df_fcst['y_hat']
    #     ########################   
    df_fcst = df_fcst.rename(columns={"DATE_RIF": "ds", VAR_FORECAST: "forecast"})
    df_hist = df_hist.rename(columns={"DATE_RIF": "ds"})
    id_cols = ["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2"] if level < 3 else ["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "ECO_COD"]
    df_fcst["unique_id"] = df_fcst[id_cols].fillna('').astype(str).agg('|'.join, axis=1)
    df_hist["unique_id"] = df_hist[id_cols].fillna('').astype(str).agg('|'.join, axis=1)
    df_hist = df_hist[["unique_id", "ds", "y"]]
    df_fcst = df_fcst[["unique_id", "ds", "forecast"]]
    df_hist = df_hist.merge(df_fcst, on=["unique_id", "ds"], how="left")    
    df_ret = pd.concat([df_hist, df_fcst[df_fcst['ds'] > df_hist['ds'].max()]], ignore_index=True)
    df_mape = df_ret[(df_ret['y'].notna()) & (df_ret['forecast'].notna())].reset_index(drop=True)
    df_mape['APE'] = (df_mape['y'] - df_mape['forecast']).abs() / df_mape['y'].abs()
    mape = df_mape.groupby('unique_id')['APE'].mean().reset_index().rename(columns={'APE': 'MAPE'})
    medianape = df_mape.groupby('unique_id')['APE'].median().reset_index().rename(columns={'APE': 'WMAPE'})
    df_ret = df_ret.merge(mape, on="unique_id", how="left").merge(medianape, on="unique_id", how="left")

    return df_ret

def get_forecast_values():
    selected_row_index = st.session_state.eco_anag_df.selection.rows[0]
    selected_eco_code = st.session_state.eco_anagrafica.iloc[selected_row_index]["ECO_COD"]
    df_eco_forecast = data_loader.rec_forecast_df[data_loader.rec_forecast_df["ECO_COD"] == selected_eco_code]
    # ########################
    # # TAROCCO DA TOGLIERE
    # if selected_eco_code == 'ECO_021':
    #     df_eco_forecast['y_hat_rec'] = df_eco_forecast['y_hat']
    # ########################
    df_eco_forecast = df_eco_forecast.rename(columns={"ECO_COD": "unique_id", "DATE_RIF": "ds", VAR_FORECAST: "y"})[["unique_id", "ds", "y"]]
    df_eco_history = data_loader.eco_df[data_loader.eco_df["ECO_COD"] == selected_eco_code][["ECO_COD", "DATE_RIF", "VALUE"]]
    df_eco_history = df_eco_history.rename(columns={"ECO_COD": "unique_id", "DATE_RIF": "ds", "VALUE": "y"})[["unique_id", "ds", "y"]]
    df_eco_anag = data_loader.eco_anag[data_loader.eco_anag["ECO_COD"] == selected_eco_code][["ECO_COD", "ECO_DSC"]].rename(columns={"ECO_COD": "unique_id", "ECO_DSC": "description"})
    df_eco_anag['is_drv'], df_eco_anag['order'] = False, 999
    drivers_selected_eco = data_loader.driver_selected_df[data_loader.driver_selected_df["ECO_COD"] == selected_eco_code][['DRV_COD', 'COEFF']]
    df_drv_forecast = data_loader.drivers_forecast_df[data_loader.drivers_forecast_df["DRV_COD"].isin(drivers_selected_eco['DRV_COD'])]
    df_drv_forecast = df_drv_forecast.rename(columns={"DRV_COD": "unique_id", "DATE_RIF": "ds", "FORECAST": "y"})[["unique_id", "ds", "y"]]
    df_drv_history = data_loader.driver_df[data_loader.driver_df["DRV_COD"].isin(drivers_selected_eco['DRV_COD'])]
    df_drv_history = df_drv_history.rename(columns={"DRV_COD": "unique_id", "DATE_RIF": "ds" , "VALUE": "y"})[["unique_id", "ds", "y"]]
    df_drv_anag = data_loader.driver_anag.merge(drivers_selected_eco, on="DRV_COD", how="inner")[["DRV_COD", "DRV_DSC", "COEFF"]].rename(columns={"DRV_COD": "unique_id", "DRV_DSC": "description"})    
    df_drv_anag['is_drv'] = True
    df_drv_anag['order'] = df_drv_anag['COEFF'].abs().rank(ascending=True, method='first').astype(int)
    cutoff_date = df_eco_forecast["ds"].min()        
    first_date = cutoff_date - pd.DateOffset(months=6)        
    df_eco_full = pd.concat([df_eco_history[(df_eco_history['ds'] < cutoff_date) & (df_eco_history['ds'] >= first_date)], df_eco_forecast], ignore_index=True)
    df_drv_full = pd.concat([df_drv_history[(df_drv_history['ds'] < cutoff_date) & (df_drv_history['ds'] >= first_date)], df_drv_forecast], ignore_index=True)
    df_values = pd.concat([df_eco_full, df_drv_full]).reset_index(drop=True)    
    df_values["ds"] = df_values["ds"].dt.strftime("%Y-%m")    
    df_values = df_values.pivot_table(index=["unique_id"], columns="ds", values="y", dropna=False).reset_index()
    df_values = df_values.merge(pd.concat([df_eco_anag, df_drv_anag]), on="unique_id", how="left").sort_values(by=["order"], ascending=[False]).reset_index(drop=True)
    #df_values['COEFF'] = pd.cut(df_values['COEFF'].abs(), bins=[0, 0.2, 0.5, 1.0], labels=['⚪', '⚪⚪', '⚪⚪⚪'], include_lowest=True).astype(str).replace('nan', "")

    cols = ['unique_id', 'description', 'COEFF', 'is_drv', 'order'] + [c for c in df_values.columns if c not in ['description', 'unique_id', 'COEFF', 'is_drv', 'order']]
    st.session_state.selected_values_df = df_values[cols]

def get_row_select():
    if not st.session_state.eco_anag_df.selection.rows:
        return  
    idx = st.session_state.eco_anag_df.selection.rows[0]
    row = st.session_state.eco_anagrafica.iloc[idx]
    for k in ["g0_fig", "g1_fig", "g2_fig", "det_fig"]:
        st.session_state[k] = None
    st.session_state.selected_labels = {
        "g0": row["ECO_GRP_0_DSC"], "g1": row["ECO_GRP_1_DSC"],
        "g2": row["ECO_GRP_2_DSC"] if pd.notnull(row["ECO_GRP_2"]) else None,
        "det": f"{row['ECO_COD']} - {row['ECO_DSC']}"
    }
    # Generazione Grafici senza titolo Plotly
    st.session_state.g0_fig = generate_plotly_plot(get_data_at_level(row, 0), title="", height=350)
    st.session_state.g1_fig = generate_plotly_plot(get_data_at_level(row, 1), title="", height=350)
    if pd.notnull(row["ECO_GRP_2"]):
        st.session_state.g2_fig = generate_plotly_plot(get_data_at_level(row, 2), title="", height=350)      
    st.session_state.det_fig = generate_plotly_plot(get_data_at_level(row, 3), title="", height=350)
    get_forecast_values()

# --- UI ---
st.title("BankLab TS")

with st.container():
    st.subheader("Selezionare una voce", divider="green")
    st.dataframe(
        st.session_state.eco_anagrafica[["ECO_GRP_0_DSC", "ECO_GRP_1_DSC", "ECO_GRP_2_DSC", "ECO_DSC",  "ECO_COD", "PERC"]],
        key="eco_anag_df",
        on_select=get_row_select,
        selection_mode="single-row",
        use_container_width=True,
        height=250,
        column_config={
            "ECO_GRP_0_DSC": "Area", 
            "ECO_GRP_1_DSC": "Settore", 
            "ECO_GRP_2_DSC": None, 
            "ECO_DSC": "Descrizione", 
            "ECO_COD": "Codice", 
            "PERC": "Peso % sul Totale Area"
        }
    )

# --- GRID LAYOUT CON TITOLI ESTERNI ---
if st.session_state.get("det_fig"):
    st.write("---")
    
    # Riga 1: Dettaglio e Segmento
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        st.markdown(f"##### Dettaglio: {st.session_state.selected_labels['det']}")
        st.plotly_chart(st.session_state.det_fig, use_container_width=True, key="plot_det")
    with r1_c2:
        if st.session_state.get("g2_fig"):
            st.markdown(f"##### Totale Segmento: {st.session_state.selected_labels['g2']}")
            st.plotly_chart(st.session_state.g2_fig, use_container_width=True, key="plot_g2")

    # Riga 2: Settore e Area
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        if st.session_state.get("g1_fig"):
            st.markdown(f"##### Totale Settore: {st.session_state.selected_labels['g1']}")
            st.plotly_chart(st.session_state.g1_fig, use_container_width=True, key="plot_g1")
    with r2_c2:
        if st.session_state.get("g0_fig"):
            st.markdown(f"##### Totale Area: {st.session_state.selected_labels['g0']}")
            st.plotly_chart(st.session_state.g0_fig, use_container_width=True, key="plot_g0")

# --- TABLE VALUES ---
# if st.session_state.get("selected_values_df") is not None:
#     st.write("---")
#     st.header("Driver Analysis", divider="orange")
#     df = st.session_state.selected_values_df
#     cols_forecast = data_loader.drivers_forecast_df['DATE_RIF'].dt.strftime("%Y-%m").unique().tolist()
#     cols_to_freeze = df.columns.difference(cols_forecast).tolist()
#     # Esempio usando barre colorate o formati specifici
#     st.data_editor(
#         df,
#         use_container_width=True,
#         hide_index=True,
#         disabled=cols_to_freeze,
#         column_config={
#             "unique_id": "Codice",
#             "COEFF": "Importanza",
#             "description": "Descrizione",
#             "is_drv": None, 
#             "order": None,
#             **{col: st.column_config.NumberColumn(col, format="%.2f", help="Previsione editabile") 
#             for col in cols_forecast}
#         }
#     )

# --- TABLE VALUES ---
if st.session_state.get("selected_values_df") is not None:
    st.write("---")
    st.subheader(f"Analisi driver: {st.session_state.selected_labels['det']}", divider="orange")
    
    df = st.session_state.selected_values_df.copy()
    cols_forecast = data_loader.drivers_forecast_df['DATE_RIF'].dt.strftime("%Y-%m").unique().tolist()
    
    # Definiamo le colonne da bloccare (non forecast)
    cols_to_freeze = df.columns.difference(cols_forecast).tolist()

    # 1. Applichiamo lo stile: prima il testo blu alle colonne forecast
    # 2. Poi lo sfondo rosa alla prima riga (sovrascrive o si aggiunge)
    styled_df = df.style.set_properties(
        subset=cols_forecast, 
        **{'color': '#1f77b4', 'font-weight': 'bold'} # Blu stile Plotly
    ).apply(
        lambda s: ['background-color: #ffe4e1' if s.name == 0 else '' for _ in s], 
        axis=1
    )

    cols_2dec = cols_to_freeze + cols_forecast + ['COEFF']
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "unique_id": "Codice",
            "COEFF": "Importanza",
            "description": "Descrizione",
            "is_drv": None, 
            "order": None,
            **{col: st.column_config.NumberColumn(col, format="%.2f") for col in cols_2dec if col not in ['unique_id', 'description', 'is_drv', 'order']}
        }
    )

# CSS Custom 
st.markdown("""
<style>
    /* Metriche */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    /* Plotly */
    .stPlotlyChart {
        border: 1px solid #f0f2f6;
        border-radius: 5px;
    }
    /* Titoli markdown sopra i grafici */
    h5 {
        margin-bottom: -15px !important;
        color: #4a4a4a;
        font-weight: 600;
    }
    /* Data editor */
    div[data-testid="stDataEditor"] input {
        color: #1f77b4;  /* blu stile Plotly */
        font-weight: 600;  /* opzionale */
    }
</style>
""", unsafe_allow_html=True)

