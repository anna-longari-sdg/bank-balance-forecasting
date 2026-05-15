import os
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger


def load_metadata(meta_dir: Path) -> pd.DataFrame:
    meta_table = pd.read_csv(meta_dir / "meta_table.csv")
    meta_column = pd.read_csv(meta_dir / "meta_column.csv")
    meta_input = meta_table.merge(meta_column, on=["id_table", "nm_table"], how="left")
    meta_input = meta_input.loc[meta_input["tp_source"] == os.environ["TP_SOURCE"], :]
    return meta_input


def get_data_all(meta_input: pd.DataFrame, db_dir: Path, data_dir: Path):
    lst_tables = meta_input["nm_table"].unique().tolist()
    tp_source = meta_input["tp_source"].iloc[0]
    dataframes = {}
    for nm_table in lst_tables:
        logger.info(f"Loading data {nm_table}")
        meta_input_curr = meta_input[meta_input["nm_table"] == nm_table]
        folder = str(meta_input_curr.iat[0, meta_input_curr.columns.get_loc("nm_folder")])
        if folder == "nan":
            folder = ""
        file = str(meta_input_curr.iat[0, meta_input_curr.columns.get_loc("nm_file")])
        sep = str(meta_input_curr.iat[0, meta_input_curr.columns.get_loc("sy_separator")])
        nm_table = str(meta_input_curr.iat[0, meta_input_curr.columns.get_loc("nm_table")])
        tp_table = str(meta_input_curr.iat[0, meta_input_curr.columns.get_loc("tp_table")])
        # Read the data file
        df = pd.read_csv(data_dir / file, sep=sep)
        # Keep only the columns that have nm_column valorized
        to_keep_columns = meta_input_curr[meta_input_curr.loc[:, "nm_column"].notna()]
        df = df[to_keep_columns["nm_column_csv"].tolist()]
        df.columns = to_keep_columns["nm_column"].tolist()
        # Force to numeric the columns that are numeric
        for col in to_keep_columns["nm_column"]:
            tp_col = to_keep_columns[to_keep_columns["nm_column"] == col]["tp_column"].iloc[0]
            fmt_col = to_keep_columns[to_keep_columns["nm_column"] == col]["fmt_date"].iloc[0]
            if tp_col == "float64":
                cols_numeric = to_keep_columns[to_keep_columns["tp_column"] == tp_col]["nm_column"].tolist()
                is_string = pd.api.types.is_string_dtype(df[col])                
                if is_string:
                    # df.loc[:, cols_numeric] = df.loc[:, cols_numeric].map(lambda v: v.replace(".", "").replace(",", "."))
                    df.loc[:, cols_numeric] = df.loc[:, cols_numeric].map(lambda v: v.replace(",", "."))
                df.loc[:, cols_numeric] = df.loc[:, cols_numeric].apply(pd.to_numeric, errors="coerce")
            if (tp_col == 'string')  & (fmt_col == '%Y%m'):
                 df[col] = df[col].astype(str)
            if (tp_col == 'string') & (fmt_col == '%b.%y'):
                months = {
                    "Gen": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr", "Mag": "May", "Giu": "Jun",
                    "Lug": "Jul", "Ago": "Aug", "Set": "Sep", "Ott": "Oct", "Nov": "Nov", "Dic": "Dec"
                }
                df[col] = df[col].replace(months, regex=True)
                df[col] = pd.to_datetime(df[col], format="%b.%y").dt.strftime("%Y%m").astype(str)                
        # Force not existent variables to be created with NaN
        if (tp_table == "EA") | (tp_table == "EDM"):
            if "ECO_GRP_0" not in df.columns:
                df["ECO_GRP_0"] = "_"
            else: 
                df["ECO_GRP_0"] = df["ECO_GRP_0"].fillna("_")
            if "ECO_GRP_1" not in df.columns:
                df["ECO_GRP_1"] = "_"
            else: 
                df["ECO_GRP_1"] = df["ECO_GRP_1"].fillna("_")
            if "ECO_GRP_2" not in df.columns:
                df["ECO_GRP_2"] = "_"
            else: 
                df["ECO_GRP_2"] = df["ECO_GRP_2"].fillna("_")
            if tp_table == "EA":
                if "ECO_DSC" not in df.columns:
                    df["ECO_DSC"] = df["ECO_COD"]    
                if "ECO_GRP_0_DSC" not in df.columns:
                    df["ECO_GRP_0_DSC"] = df["ECO_GRP_0"]
                if "ECO_GRP_1_DSC" not in df.columns:
                    df["ECO_GRP_1_DSC"] = df["ECO_GRP_1"]
                if "ECO_GRP_2_DSC" not in df.columns:
                    df["ECO_GRP_2_DSC"] = df["ECO_GRP_2"]
        if (tp_table == "DA_EN") | (tp_table == "DA_EX"):
            if "DRV_TYP_0" not in df.columns:
                df["DRV_TYP_0"] = "_"
            else: 
                df["DRV_TYP_0"] = df["DRV_TYP_0"].fillna("_")
            if "DRV_TYP_1" not in df.columns:
                df["DRV_TYP_1"] = "_"
            else: 
                df["DRV_TYP_1"] = df["DRV_TYP_1"].fillna("_")
            if "DRV_TYP_2" not in df.columns:
                df["DRV_TYP_2"] = "_"
            else: 
                df["DRV_TYP_2"] = df["DRV_TYP_2"].fillna("_")
            if "DRV_DSC" not in df.columns:
                df["DRV_DSC"] = df["DRV_COD"]
            if "DRV_TYP_0_DSC" not in df.columns:
                df["DRV_TYP_0_DSC"] = df["DRV_TYP_0"]
            if "DRV_TYP_1_DSC" not in df.columns:             
                df["DRV_TYP_1_DSC"] = df["DRV_TYP_1"]
            if "DRV_TYP_2_DSC" not in df.columns:
                df["DRV_TYP_2_DSC"] = df["DRV_TYP_2"]
        
        # Force variable for endogenous or exogenous drivers
        if tp_table == "DA_EN":
            df["DRV_TYP_0"] = "ENDO"
            df["DRV_TYP_0_DSC"] = "Endogenous"
        elif tp_table == "DA_EX":
            df["DRV_TYP_0"] = "EXO"
            df["DRV_TYP_0_DSC"] = "Exogenous"
        if tp_table == "DA_EN" or tp_table == "DA_EX":
            # df["DRV_TYP_1_DSC"] = ""
            # df["DRV_TYP_2_DSC"] = ""
            df = df[
                [
                    "DRV_COD",
                    "DRV_TYP_0",
                    "DRV_TYP_1",
                    "DRV_TYP_2",
                    "DRV_DSC",
                    "DRV_TYP_0_DSC",
                    "DRV_TYP_1_DSC",
                    "DRV_TYP_2_DSC",
                ]
            ]
        if tp_table == "EA":
            # df["ECO_GRP_0_DSC"] = ""
            # df["ECO_GRP_1_DSC"] = ""
            # df["ECO_GRP_2_DSC"] = ""
            df = df[
                [
                    "ECO_COD",
                    "ECO_GRP_0",
                    "ECO_GRP_1",
                    "ECO_GRP_2",
                    "ECO_DSC",
                    "ECO_GRP_0_DSC",
                    "ECO_GRP_1_DSC",
                    "ECO_GRP_2_DSC",
                ]
            ]
        if tp_table == "EDM":
            df = df[["ECO_GRP_0", "ECO_GRP_1", "ECO_GRP_2", "DRV_COD"]]

        if tp_source == "CE":
            if tp_table == "DA_EN":
                drv_typ_1_mapping = {
                    "BANCA": "Bancari",
                    "CONS": "Consulenza",
                    "DONOTFORECAST": "DoNotForecast",
                    "EL": "Elettronici",
                    "FON": "Fondi",
                    "GP": "Gestioni Patrimoniali",
                    "NEGO": "Negoziazione",
                    "TRAD": "Tradizionali",
                    "VITA": "Assicurativo Vita",
                }
                df["DRV_TYP_1_DSC"] = df["DRV_TYP_1"].map(drv_typ_1_mapping)
                drv_typ_2_mapping = {
                    "A": "Altri",
                    "F": "Flussi",
                    "G": "Gestione",
                    "M": "Mercati",
                    "R": "Risk",
                    "S": "Stock",
                }
                df["DRV_TYP_2_DSC"] = df["DRV_TYP_2"].map(drv_typ_2_mapping)
            elif tp_table == "EA":
                eco_grp_0_mapping = {
                    20: "Servizi di investimento",
                    40: "Sistemi di pagamento",
                }
                df["ECO_GRP_0_DSC"] = df["ECO_GRP_0"].map(eco_grp_0_mapping)
                eco_grp_1_mapping = {
                    40: "Negoziazioni online",
                    43: "Negoziazioni tradizionali",
                    45: "Commissioni accessorie",
                    50: "Fondi",
                    55: "Gestioni Patrimoniali",
                    60: "Assicurativo vita",
                    65: "Consulenza",
                    70: "Operatività titoli",
                    110: "Tradizionali",
                    115: "Elettronici",
                    120: "Estero",
                }
                df["ECO_GRP_1_DSC"] = df["ECO_GRP_1"].map(eco_grp_1_mapping)
                df["ECO_GRP_2_DSC"] = df["ECO_GRP_2"].map(lambda v: f"Segmento {v}")

        dataframes[nm_table] = df
        logger.info(f"Loaded data shape for {nm_table}: {df.shape}")

    # Se esistono le tabelle EX, le unisce a quelle EN
    if ("DA_EX" in meta_input["tp_table"].unique()):
        name_anag_en = meta_input[meta_input["tp_table"] == "DA_EN"]["nm_table"].iloc[0]
        name_anag_ex = meta_input[meta_input["tp_table"] == "DA_EX"]["nm_table"].iloc[0]
        name_val_en = meta_input[meta_input["tp_table"] == "DV_EN"]["nm_table"].iloc[0]
        name_val_ex = meta_input[meta_input["tp_table"] == "DV_EX"]["nm_table"].iloc[0]
        dataframes[name_anag_en] = pd.concat([dataframes[name_anag_en], dataframes[name_anag_ex]], axis=0).reset_index(drop=True)
        dataframes[name_val_en] = pd.concat([dataframes[name_val_en], dataframes[name_val_ex]], axis=0).reset_index(drop=True)
        del dataframes[name_anag_ex]
        del dataframes[name_val_ex]     

    # Se non esiste la tabella EB la crea vuota
    if not ("EB" in meta_input["tp_table"].unique()):
        name_eb = 'ECO_BUDGET'
        dataframes[name_eb] = pd.DataFrame(columns=["DATE_RIF", "ECO_COD", "VALUE_BUDGET"])

    def sign_type(x):
        if (x > 0).all():
            return "positive"
        elif (x < 0).all():
            return "negative"
        else:
            return "mixed"

    dv_table = meta_input[meta_input["tp_table"] == "DV_EN"]["nm_table"].iloc[0]
    da_table = meta_input[meta_input["tp_table"] == "DA_EN"]["nm_table"].iloc[0]
    sign = dataframes[dv_table].groupby("DRV_COD")["VALUE"].agg(q1=lambda x: x.quantile(0.05),q3=lambda x: x.quantile(0.95)).reset_index()
    sign['SIGN'] = dataframes[dv_table].groupby("DRV_COD")["VALUE"].apply(sign_type).values
    sign = sign[['DRV_COD', 'SIGN']].reset_index(drop=True)
    dataframes[da_table] = dataframes[da_table].merge(sign, on="DRV_COD", how="left")

    ev_table = meta_input[meta_input["tp_table"] == "EV"]["nm_table"].iloc[0]
    ea_table = meta_input[meta_input["tp_table"] == "EA"]["nm_table"].iloc[0]
    sign = dataframes[ev_table].groupby("ECO_COD")["VALUE"].agg(q1=lambda x: x.quantile(0.05),q3=lambda x: x.quantile(0.95)).reset_index()
    sign['SIGN'] = dataframes[ev_table].groupby("ECO_COD")["VALUE"].apply(sign_type).values
    sign = sign[['ECO_COD', 'SIGN']].reset_index(drop=True)
    dataframes[ea_table] = dataframes[ea_table].merge(sign, on="ECO_COD", how="left")
    
    # Percorso del database
    db_path = db_dir / "bankfcs_cleaned.db"
    
    # Controlla se il file esiste e cancellalo
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Database {db_path} esistente cancellato.")
    
    with duckdb.connect(db_path) as conn:
        for nm_table, df in dataframes.items():
            logger.info(f"Writing table {nm_table} to DuckDB")
            conn.execute(f"CREATE OR REPLACE TABLE {nm_table} AS SELECT * FROM df")
            logger.info(f"Table {nm_table} written to DuckDB successfully")


def analize_zero_freq(data, nm_unique_id, nm_ds, nm_y, interval_to_sub=12):
    interval_to_sub_dt = pd.DateOffset(months=interval_to_sub)

    # counts the number of zeros in the data
    number_of_zeros = (
        data.groupby(nm_unique_id)[nm_y]
        .agg(n_zeros=lambda s: (s.isna()).sum(), n_tot_points="size")
        .reset_index()  # ANNA
    )

    # define time trashold
    time_treshold = data[nm_ds].max() - interval_to_sub_dt

    # conta quanti zeri nell'ultimo periodo
    zeros_last_year = (
        data[data[nm_ds] > time_treshold]
        .groupby(nm_unique_id)[nm_y]
        .agg(count_zeros_last_year=lambda u: (u.isna()).sum())  # ANNA
    )
    df_zeros = number_of_zeros.merge(zeros_last_year, how="left", on=nm_unique_id)
    df_zeros["perc_zeros"] = df_zeros["n_zeros"] / df_zeros["n_tot_points"]
    df_zeros["perc_zeros_last_year"] = df_zeros["count_zeros_last_year"] / interval_to_sub

    return df_zeros


def create_mapping(db_dir: Path):
    con = duckdb.connect(db_dir / "bankfcs_cleaned.db")

    # Leggi una tabella in un DataFrame pandas
    drv_anag = con.execute("SELECT * FROM DRV_ANAG").df()
    eco_anag = con.execute("SELECT * FROM ECO_ANAG").df()
    eco_drv_rules = con.execute("SELECT * FROM ECO_DRV_RULES").df()

    eco_drv_map = eco_anag.merge(drv_anag, how="cross")
    eco_drv_map = eco_drv_map.merge(eco_drv_rules, on=["DRV_COD", "ECO_GRP_0", "ECO_GRP_1"], how="inner")[
        ["ECO_COD", "DRV_COD"]
    ]
    eco_drv_map = eco_drv_map.drop_duplicates().reset_index(drop=True)
    table_name = "ECO_DRV_MAP"
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS
                SELECT * 
                FROM eco_drv_map;
                """
    )
    # Chiudi la connessione quando finisci
    con.close()


def anonymize_data(db_dir: Path):
    con = duckdb.connect(db_dir / "bankfcs_cleaned.db")

    # Leggi una tabella in un DataFrame pandas
    drv_anag = con.execute("SELECT * FROM DRV_ANAG").df()
    drv_val = con.execute("SELECT * FROM DRV_VAL").df()
    eco_anag = con.execute("SELECT * FROM ECO_ANAG").df()
    eco_val = con.execute("SELECT * FROM ECO_VAL").df()
    eco_drv_map = con.execute("SELECT * FROM ECO_DRV_MAP").df()
    eco_budget = con.execute("SELECT * FROM ECO_BUDGET").df()

    # Chiudi la connessione quando finisci
    con.close()

    # %%
    # Create anonymized mapping for drivers
    lst_drv_en = drv_anag[drv_anag["DRV_TYP_0"] == "ENDO"]["DRV_COD"].unique()
    lst_drv_ex = drv_anag[drv_anag["DRV_TYP_0"] == "EXO"]["DRV_COD"].unique()
    mapping_drv_en = {code: f"DRV_{i:03d}" for i, code in enumerate(lst_drv_en, 1)}
    mapping_drv_ex = {code: f"DRV_EX_{i:03d}" for i, code in enumerate(lst_drv_ex, 1)}

    # Apply to a drv_anag
    drv_anag["DRV_COD_REAL"] = drv_anag["DRV_COD"]
    drv_anag.loc[drv_anag["DRV_COD"].isin(lst_drv_en), "DRV_COD"] = drv_anag.loc[drv_anag["DRV_COD"].isin(lst_drv_en), "DRV_COD"].map(mapping_drv_en)
    drv_anag.loc[drv_anag["DRV_COD"].isin(lst_drv_ex), "DRV_COD"] = drv_anag.loc[drv_anag["DRV_COD"].isin(lst_drv_ex), "DRV_COD"].map(mapping_drv_ex)

    # Apply to other tables containing drv_cod
    drv_val.loc[drv_val["DRV_COD"].isin(lst_drv_en), "DRV_COD"] = drv_val.loc[drv_val["DRV_COD"].isin(lst_drv_en), "DRV_COD"].map(mapping_drv_en)
    drv_val.loc[drv_val["DRV_COD"].isin(lst_drv_ex), "DRV_COD"] = drv_val.loc[drv_val["DRV_COD"].isin(lst_drv_ex), "DRV_COD"].map(mapping_drv_ex)
    eco_drv_map.loc[eco_drv_map["DRV_COD"].isin(lst_drv_en), "DRV_COD"] = eco_drv_map.loc[eco_drv_map["DRV_COD"].isin(lst_drv_en), "DRV_COD"].map(mapping_drv_en)
    eco_drv_map.loc[eco_drv_map["DRV_COD"].isin(lst_drv_ex), "DRV_COD"] = eco_drv_map.loc[eco_drv_map["DRV_COD"].isin(lst_drv_ex), "DRV_COD"].map(mapping_drv_ex)

    # %%
    # Create anonymized mapping for economics
    mapping_eco = {code: f"ECO_{i:03d}" for i, code in enumerate(eco_anag["ECO_COD"].unique(), 1)}

    # Apply to a eco_anag
    eco_anag["ECO_COD_REAL"] = eco_anag["ECO_COD"]
    eco_anag.loc[:, "ECO_COD"] = eco_anag.loc[:, "ECO_COD"].map(mapping_eco)

    # Apply to other tables containing eco_cod
    eco_val.loc[:, "ECO_COD"] = eco_val.loc[:, "ECO_COD"].map(mapping_eco)
    eco_drv_map.loc[:, "ECO_COD"] = eco_drv_map.loc[:, "ECO_COD"].map(mapping_eco)
    eco_budget.loc[:, "ECO_COD"] = eco_budget.loc[:, "ECO_COD"].map(mapping_eco)

    # %%
    # Add flag column to eco_drv_map
    # eco_drv_map["IS_SELECTED"] = False removed to allow pipeline to set is selected according to model

    # Force character on val tables
    drv_val["DATE_RIF"] = drv_val["DATE_RIF"].astype(str)
    eco_val["DATE_RIF"] = eco_val["DATE_RIF"].astype(str)

    # %%
    dsc_cols_drv_anag = [col for col in drv_anag.columns if col.endswith("_DSC")]
    dsc_cols_eco_anag = [col for col in eco_anag.columns if col.endswith("_DSC")]

    # print(dsc_cols_drv_anag)
    # print(dsc_cols_eco_anag)

    # Create a copy_real of all dsc columns
    for col in dsc_cols_drv_anag:
        drv_anag[col + "_REAL"] = drv_anag[col]
    for col in dsc_cols_eco_anag:
        eco_anag[col + "_REAL"] = eco_anag[col]

    # %%
    import re

    def anonymize_description(text):
        """Anonimizza descrizioni mantenendo significato generico"""
        if pd.isna(text):
            return text

        # Rimuovi underscore PRIMA
        result = str(text).replace("_", " ")

        # Converte tutto in minuscolo
        result = result.lower()

        # Pattern di sostituzione generici
        replacements = {
            # Termini bancari specifici
            # r"comm(?:issioni?)?\b": "commissioni",
            # r"proventi": "ricavi",
            # r"brokeraggio": "intermediazione",
            # r"canone": "quota periodica",
            # r"bonifici?": "trasferimenti",
            # r"carte?\s+(?:di\s+)?(?:credito|debito)": "strumenti di pagamento",
            # r"pos\b": "terminali pagamento",            
            # r"atm\b": "sportelli automatici"
            r"atm\b": "ATM",
            r"pos\b": "POS",
            r"car\b": "CAR",
            r"com\b": "COM",
            # r"acquiring": "gestione transazioni",
            # r"issuing": "emissione strumenti",
            # r"interchange\s+fee": "quota interbancaria",
            # r"merchant\s+fee": "quota esercente",
            # r"transato": "volume transazioni",
            # # Prodotti finanziari
            # r"fondi?\b": "prodotti gestiti",
            # r"titoli?\b": "strumenti finanziari",
            # r"polizze?": "prodotti assicurativi",
            # r"obbligazioni?": "strumenti di debito",
            # r"\bazion\b": "strumenti azionari",
            # # Operazioni
            # r"incasso": "riscossione",
            # r"pagamento": "versamento",
            # r"prelievo": "ritiro contante",
            # # Circuiti
            # r"bancomat": "circuito nazionale",
            # r"visa|mastercard": "circuito internazionale",
            # Altri termini generici
            r"racc ": "raccolta ",
            r"\bbanca sella\b": "banca",
            r"\bcartalis\b": "carta",
            r"\bsella\b": " ",
            r"\bfarad\b": "fd",
            r"\blombard\b": "ld",
            r"\bamissima\b": "aa",
            r"\bzurich\b": "zh",
            r"\bhdi\b": "hd",
            r"\balleanza\b": "az",
            r"\bconto tuo valore\b": "ctv",
            r"\bgbs\b": "ggg",
            r"\bholding\b": "gruppo",
            r"\bfabrick\b": "fb",
            r"\baxerve\b": "ae",
            r"\bhype\b": "online",
            r"\bsellaextreme\b": "ee",
            r"\bbps\b": "patrimoni",
            r"\bspc\b": "personal credit",
            r"\bappago\b": "dilazioni",
            r"\bcba\b": "spec",
            r"\bbsh\b": "uno",
            r"\bbse\b": "due",
            r"\bg life\b": "",
            r"\b   \b": ""
        
        }

        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Capitalizza la prima lettera
        result = result.capitalize()

        return result

    # Applica anonimizzazione a tutte le colonne _DSC
    for col in dsc_cols_drv_anag:
        # col = dsc_cols_drv_anag[0]
        drv_anag[col] = drv_anag[col + "_REAL"].apply(anonymize_description)
        # drv_anag[col] = drv_anag[col + "_REAL"]
        # drv_anag[col] = drv_anag[col + "_REAL"].str.lower().str.capitalize()

    for col in dsc_cols_eco_anag:
        # col = dsc_cols_eco_anag[0]
        eco_anag[col] = eco_anag[col + "_REAL"].apply(anonymize_description)
        # eco_anag[col] = eco_anag[col + "_REAL"]
        # eco_anag[col] = eco_anag[col + "_REAL"].str.lower().str.capitalize()

    logger.info("Anonimizzazione descrizioni completata!")

    # Percorso del database
    db_path = db_dir / "bankfcs_anonymized.db"
    
    # Controlla se il file esiste e cancellalo
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Database {db_path} esistente cancellato.")

    # %%
    # Create and connect to a new db
    con_new = duckdb.connect(db_path)

    # Create and populate tables in the new database
    con_new.register("drv_anag_temp", drv_anag)
    con_new.execute("CREATE OR REPLACE TABLE DRV_ANAG AS SELECT * FROM drv_anag_temp")

    con_new.register("drv_val_temp", drv_val)
    con_new.execute("CREATE OR REPLACE TABLE DRV_VAL AS SELECT * FROM drv_val_temp")

    con_new.register("eco_anag_temp", eco_anag)
    con_new.execute("CREATE OR REPLACE TABLE ECO_ANAG AS SELECT * FROM eco_anag_temp")

    con_new.register("eco_val_temp", eco_val)
    con_new.execute("CREATE OR REPLACE TABLE ECO_VAL AS SELECT * FROM eco_val_temp")

    con_new.register("eco_drv_map_temp", eco_drv_map)
    con_new.execute("CREATE OR REPLACE TABLE ECO_DRV_MAP AS SELECT * FROM eco_drv_map_temp")

    con_new.register("eco_budget_temp", eco_budget)
    con_new.execute("CREATE OR REPLACE TABLE ECO_BUDGET AS SELECT * FROM eco_budget_temp")

    # Close connection to new db
    con_new.close()   

def reduce_db(db_path: Path, lst_eco_cod: list = [], num_eco: int = 100) -> None:
    con = duckdb.connect(db_path)
    # Leggi una tabella in un DataFrame pandas
    drv_anag = con.execute("SELECT * FROM DRV_ANAG").df()
    drv_val = con.execute("SELECT * FROM DRV_VAL").df()
    eco_anag = con.execute("SELECT * FROM ECO_ANAG").df()
    eco_val = con.execute("SELECT * FROM ECO_VAL").df()
    eco_drv_map = con.execute("SELECT * FROM ECO_DRV_MAP").df()
    # Chiudi la connessione quando finisci
    con.close()

    if not lst_eco_cod:
        # Se lst_eco_cod è vuota, seleziona un campione casuale
        eco_anag = eco_anag.sample(n=num_eco, random_state=42).reset_index(drop=True)
        eco_codes_to_keep = eco_anag["ECO_COD"].unique()
    else:
        # Se lst_eco_cod non è vuota, usa i codici forniti
        eco_anag = eco_anag[eco_anag["ECO_COD"].isin(lst_eco_cod)].reset_index(drop=True)
        eco_codes_to_keep = lst_eco_cod
    eco_val = eco_val[eco_val["ECO_COD"].isin(eco_codes_to_keep)].reset_index(drop=True)
    eco_drv_map = eco_drv_map[eco_drv_map["ECO_COD"].isin(eco_codes_to_keep)].reset_index(drop=True)
    drv_codes_to_keep = eco_drv_map["DRV_COD"].unique()
    drv_anag = drv_anag[drv_anag["DRV_COD"].isin(drv_codes_to_keep)].reset_index(drop=True)
    drv_val = drv_val[drv_val["DRV_COD"].isin(drv_codes_to_keep)].reset_index(drop=True)

    # Create and connect to a new db
    con_new = duckdb.connect(db_path)

    # Create and populate tables in the new database
    con_new.register("drv_anag_temp", drv_anag)
    con_new.execute("CREATE OR REPLACE TABLE DRV_ANAG AS SELECT * FROM drv_anag_temp")

    con_new.register("drv_val_temp", drv_val)
    con_new.execute("CREATE OR REPLACE TABLE DRV_VAL AS SELECT * FROM drv_val_temp")

    con_new.register("eco_anag_temp", eco_anag)
    con_new.execute("CREATE OR REPLACE TABLE ECO_ANAG AS SELECT * FROM eco_anag_temp")

    con_new.register("eco_val_temp", eco_val)
    con_new.execute("CREATE OR REPLACE TABLE ECO_VAL AS SELECT * FROM eco_val_temp")

    con_new.register("eco_drv_map_temp", eco_drv_map)
    con_new.execute("CREATE OR REPLACE TABLE ECO_DRV_MAP AS SELECT * FROM eco_drv_map_temp")

    # Close connection to new db
    con_new.close()