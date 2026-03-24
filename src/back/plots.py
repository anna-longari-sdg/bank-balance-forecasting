import pandas as pd
import plotnine as p9
from mizani.formatters import percent_format

from src.db.db import DB


class __custom_theme(p9.theme_538):
    def __init__(self, base_size=11, base_family="DejaVu Sans", figure_size=(16, 8)):
        super().__init__(base_size=base_size, base_family=base_family)
        self += p9.theme(figure_size=figure_size)


def plot_economics(
    economics_df: pd.DataFrame,
    group: str,
    cutoff: dict | None = None,
    scale_x_datetime_kwargs: dict | None = None,
    figure_size=(16, 8),
    anonymize: bool = False,
) -> p9.ggplot:
    agg_group = group if anonymize else f"{group}_DSC"
    grouped_df = economics_df.groupby(by=["DATE_RIF", agg_group])["VALUE"].sum().reset_index()
    grouped_df.loc[:, agg_group] = grouped_df.loc[:, agg_group].astype(str)
    if cutoff is not None:
        latest_date = grouped_df["DATE_RIF"].max()
        cutoff_date = latest_date - pd.DateOffset(**cutoff)
        grouped_df = grouped_df.loc[grouped_df["DATE_RIF"] >= cutoff_date, :]

    plot = (
        p9.ggplot(grouped_df)
        + p9.aes(x="DATE_RIF", y="VALUE", color=agg_group)
        + p9.geom_line()
        + p9.geom_point()
        + __custom_theme(figure_size=figure_size)
        + p9.labs(
            title=f"Economic Indicator over Time by {group}",
            x="Date",
            y="Value",
            color=agg_group,
        )
    )
    if scale_x_datetime_kwargs is not None:
        plot += p9.scale_x_datetime(**scale_x_datetime_kwargs)
    return plot


# Plot results
def plot_result(
    df: pd.DataFrame, df_fct: pd.DataFrame, df_tst: pd.DataFrame, n_tail: int, total=False, eco_code: str | None = None
) -> p9.ggplot:
    """
    Create a faceted time-series plot with actuals and forecasts.

    Behavior:
      - Merges test and forecast frames to get forecasted values per timestamp.
      - Optionally excludes 'TOTAL' series from plotting.
      - Keeps the last n_tail observations per series for visualization.
      - Returns a plotnine (ggplot) object.

    Args:
      df (pd.DataFrame): original (possibly aggregated) dataframe with 'y'.
      df_fct (pd.DataFrame): forecast dataframe with 'yhat' (and possibly 'yhat_rec').
      df_tst (pd.DataFrame): test dataframe used to align forecasts by ds.
      n_tail (int): number of latest observations per series to show.
      total (bool): if True include TOTAL in plotting; otherwise exclude it.

    Returns:
      plotnine.ggplot: faceted plot object.
    """
    # Merge test and forecasts on unique_id & ds to obtain forecast trace
    df_merged = pd.merge(df_fct, df_tst, on=["ECO_COD", "DATE_RIF"], how="left")
    df_merged = df_merged[
        df_merged.columns.difference([drv_col for drv_col in df_merged.columns if drv_col.startswith("DRV")])
    ]

    # Filter: Exclude TOTAL series unless requested
    if total:
        df_plot = df.copy()
    else:
        df_plot = df.loc[df["ECO_COD"] != "TOTAL", :].copy()
        df_merged = df_merged.loc[df_merged["ECO_COD"] != "TOTAL", :]

    # Filter: Specific eco_code
    if eco_code is not None:
        df_plot = df_plot.loc[df_plot["ECO_COD"] == eco_code, :]
        df_merged = df_merged.loc[df_merged["ECO_COD"] == eco_code, :]

    id_vars = ["ECO_COD", "DATE_RIF"]
    if "y" in df_merged.columns:
        id_vars.append("y")

    forecast_cols = [c for c in df_merged.columns if c not in id_vars]

    # 2. Melt (Unpivot) to Long Format for Plotnine
    # This creates a 'model' column and a 'y_hat' column
    df_fct_long = pd.melt(df_merged, id_vars=["ECO_COD", "DATE_RIF"], value_vars=forecast_cols, var_name="model")
    cutoff_date = df_fct["DATE_RIF"].min()

    # --- NEW LOGIC END ---

    # Build plot
    plt = (
        p9.ggplot()
        # 1. Plot Actuals (Black line)
        + p9.geom_line(data=df_plot, mapping=p9.aes(x="DATE_RIF", y="y"), color="black", size=0.8)
        # 2. Plot Forecasts (Colored by model)
        + p9.geom_line(data=df_fct_long, mapping=p9.aes(x="DATE_RIF", y="value", color="model"))
        + p9.geom_vline(xintercept=cutoff_date, linetype="dashed")
        # 3. Faceting and Styling
        + p9.facet_wrap("ECO_COD", scales="free_y")
        + p9.theme_minimal()
        + p9.labs(y="Values", x="Date", color="Forecast Method")
        + p9.theme(
            legend_position="top",
            plot_caption=p9.element_text(hjust=0, size=10, margin={"t": 15}, family="monospace", linespacing=1.2),
        )
    )

    return plt

# Plot results
def plot_generic_result(
    df: pd.DataFrame, df_fct: pd.DataFrame, df_tst: pd.DataFrame, n_tail: int, total=False, eco_code: str | None = None
) -> p9.ggplot:
    """
    Create a faceted time-series plot with actuals and forecasts, both for DRV and ECO

    Behavior:
      - Merges test and forecast frames to get forecasted values per timestamp.
      - Optionally excludes 'TOTAL' series from plotting.
      - Keeps the last n_tail observations per series for visualization.
      - Returns a plotnine (ggplot) object.

    Args:
      df (pd.DataFrame): original (possibly aggregated) dataframe with 'y'.
      df_fct (pd.DataFrame): forecast dataframe with 'yhat' (and possibly 'yhat_rec').
      df_tst (pd.DataFrame): test dataframe used to align forecasts by ds.
      n_tail (int): number of latest observations per series to show.
      total (bool): if True include TOTAL in plotting; otherwise exclude it.

    Returns:
      plotnine.ggplot: faceted plot object.
    """
    # Merge test and forecasts on unique_id & ds to obtain forecast trace
    if "ECO_COD" in df_fct.columns:
        df_fct = df_fct.rename(columns={"ECO_COD": "unique_id", "DATE_RIF": "ds"})
    if "DRV_COD" in df_fct.columns:
        df_fct = df_fct.rename(columns={"DRV_COD": "unique_id", "DATE_RIF": "ds"})        
    if "ECO_COD" in df_tst.columns:
        df_tst = df_tst.rename(columns={"ECO_COD": "unique_id", "DATE_RIF": "ds"})
    if "DRV_COD" in df_tst.columns:
        df_tst = df_tst.rename(columns={"DRV_COD": "unique_id", "DATE_RIF": "ds"})

    df_merged = pd.merge(df_fct, df_tst, on=["unique_id", "ds"], how="left")
    df_merged = df_merged[
        df_merged.columns.difference([drv_col for drv_col in df_merged.columns if drv_col.startswith("DRV")])
    ]

    df_plot = df.copy()
    
    # Filter: Exclude TOTAL series unless requested
    # if total:
    #     df_plot = df.copy()
    # else:
    #     df_plot = df.loc[df["unique_id"] != "TOTAL", :].copy()
    #     df_merged = df_merged.loc[df_merged["unique_id"] != "TOTAL", :]

    # Filter: Specific eco_code
    if eco_code is not None:
        df_plot = df_plot.loc[df_plot["unique_id"] == eco_code, :]
        df_merged = df_merged.loc[df_merged["unique_id"] == eco_code, :]

    id_vars = ["unique_id", "ds"]
    if "y" in df_merged.columns:
        id_vars.append("y")

    forecast_cols = [c for c in df_merged.columns if c not in id_vars]

    # 2. Melt (Unpivot) to Long Format for Plotnine
    # This creates a 'model' column and a 'y_hat' column
    df_fct_long = pd.melt(df_merged, id_vars=["unique_id", "ds"], value_vars=forecast_cols, var_name="model")
    cutoff_date = df_fct["ds"].min()

    # --- NEW LOGIC END ---

    # Build plot
    plt = (
        p9.ggplot()
        # 1. Plot Actuals (Black line)
        + p9.geom_line(data=df_plot, mapping=p9.aes(x="ds", y="y"), color="black", size=0.8)
        # 2. Plot Forecasts (Colored by model)
        + p9.geom_line(data=df_fct_long, mapping=p9.aes(x="ds", y="value", color="model"))
        + p9.geom_vline(xintercept=cutoff_date, linetype="dashed")
        # 3. Faceting and Styling
        # + p9.facet_wrap("unique_id", scales="free_y")
        + p9.theme_minimal()
        + p9.labs(y="Values", x="Date", color="Forecast Method")
        + p9.theme(
            legend_position="top",
            plot_caption=p9.element_text(hjust=0, size=10, margin={"t": 15}, family="monospace", linespacing=1.2),
        )
    )

    return plt


def __build_result_df(coeffs_dict, columns):
    return (
        pd.DataFrame.from_records([(drv_code, coeff) for drv_code, coeff in coeffs_dict.items()], columns=columns)
        if coeffs_dict != "NO_RESULTS"
        else pd.DataFrame({col_name: pd.Series() for col_name in columns})
    )


def plot_lasso(
    lasso_coeff: dict,
    lasso_lars_coeff: dict,
    eco_name: str | None = None,
    lasso_alpha: float | None = None,
    lasso_lars_alpha: float | None = None,
    plot_desc=False,
) -> p9.ggplot:
    df_lasso = __build_result_df(lasso_coeff, columns=["DRV_COD", "LASSO"])
    df_lasso_lars = __build_result_df(lasso_lars_coeff, columns=["DRV_COD", "LASSO_LARS"])
    df_coeffs = pd.merge(df_lasso, df_lasso_lars, on="DRV_COD", how="outer")
    if len(df_coeffs) == 0:
        raise RuntimeError("No coeffs present for this economic")
    df_long = df_coeffs.melt(id_vars="DRV_COD", var_name="Model", value_name="Coefficient").fillna(0.0)
    order_idx = df_coeffs.sort_values(by="LASSO", key=abs, ascending=True).index
    sorted_features = df_coeffs.loc[order_idx, "DRV_COD"].tolist()
    # Convert 'Feature' to a categorical type with specific order
    df_long["DRV_COD"] = pd.Categorical(df_long["DRV_COD"], categories=sorted_features, ordered=True)
    df_long["Label"] = df_long["Coefficient"].round(3).astype(str)
    data_range = df_long["Coefficient"].max() - df_long["Coefficient"].min()
    offset = data_range * 0.03  # 3% of the total range
    # Create a position column: push text slightly away from 0 based on sign
    df_long["Label_Pos"] = df_long["Coefficient"].apply(lambda x: x + offset if x >= 0 else x - offset)

    alpha_text = ""

    if lasso_alpha is not None:
        alpha_text += f"LassoCV α: {lasso_alpha:.4f}\n"
    if lasso_lars_alpha is not None:
        alpha_text += f"LassoLarsCV α: {lasso_lars_alpha:.4f}\n"

    dodge_width = 0.6
    plot_title = "LassoCV vs LassoLarsCV Coefficient Comparison"
    if eco_name is not None:
        plot_title += f" for {eco_name}"

    if plot_desc:
        db = DB()
        query = f"""
            SELECT dva.DRV_COD, dva.DRV_DSC
            FROM DRV_ANAG dva
            WHERE list_contains({df_long["DRV_COD"].unique().tolist()}, dva.DRV_COD)
        """
        drv_names = db.query(query).set_index("DRV_COD")  # type: ignore
        driver_names_list = [f"{drv}: {drv_names.loc[drv, 'DRV_DSC']}" for drv in df_long["DRV_COD"].unique()]
        alpha_text += f"\nDriver Names:\n{'\n'.join(driver_names_list)}"

    plot = (
        p9.ggplot(df_long, p9.aes(x="DRV_COD", y="Coefficient", color="Model"))
        # The "Stick": geom_linerange is preferred over geom_segment for grouped/dodged data.
        # It draws a line from ymin to ymax at the specific x position.
        + p9.geom_linerange(
            p9.aes(ymin=0, ymax="Coefficient"), position=p9.position_dodge(width=dodge_width), size=1.2, alpha=0.7
        )
        # The "Candy": The point at the end of the stick
        + p9.geom_point(position=p9.position_dodge(width=dodge_width), size=3)
        # Layer for Positive/Zero Coefficients (align left, sits to the right)
        + p9.geom_text(
            data=df_long.loc[df_long["Coefficient"] >= 0, :],
            mapping=p9.aes(y="Label_Pos", label="Label", color="Model"),
            position=p9.position_dodge(width=dodge_width),
            ha="left",  # Text starts at pos and goes right
            va="center",
            size=8,
        )
        # Layer for Negative Coefficients (align right, sits to the left)
        + p9.geom_text(
            data=df_long.loc[df_long["Coefficient"] < 0, :],
            mapping=p9.aes(y="Label_Pos", label="Label", color="Model"),
            position=p9.position_dodge(width=dodge_width),
            ha="right",  # Text ends at pos and goes left
            va="center",
            size=8,
        )
        # Visual Polish
        + p9.geom_hline(yintercept=0, color="gray", linetype="dashed")  # Zero reference line
        + p9.coord_flip()  # Flip coordinates to make names readable on Y-axis
        + p9.theme_minimal()
        + p9.labs(title=plot_title, x="Features (Sorted by Magnitude)", y="Coefficient Value", caption=alpha_text)
        + p9.theme(
            figure_size=(14, 8 if len(df_long["DRV_COD"].unique()) < 20 else 16),
            legend_position="top",
            plot_caption=p9.element_text(hjust=0, size=10, margin={"t": 15}, family="monospace", linespacing=1.2),
        )
    )
    return plot


def plot_xgboost(df: pd.DataFrame, threshold: float, eco_code: str) -> p9.ggplot:
    plot_df = df.copy()

    # We sort by importance for the Pareto logic
    plot_df = plot_df.sort_values("Importance", ascending=False)

    # Colors
    color_individual = "#0072B2"  # Blue for single features
    color_cumulative = "#D55E00"  # Red/Orange for running total

    plot = (
        p9.ggplot(plot_df, p9.aes(x="reorder(Feature, -Importance)"))
        # --- Layer 1: The Lollipop (Individual Importance) ---
        + p9.geom_segment(p9.aes(xend="Feature", y=0, yend="Importance"), color="gray", size=0.5)
        + p9.geom_point(p9.aes(y="Importance"), fill=color_individual, color="white", size=3, stroke=0.5)
        # --- Layer 2: The Pareto Line (Cumulative Importance) ---
        + p9.geom_line(p9.aes(y="Cumulative", group=1), color=color_cumulative, size=1.2)
        + p9.geom_point(p9.aes(y="Cumulative"), color=color_cumulative, size=1.5)
        # --- Layer 3: The Threshold ---
        + p9.geom_hline(yintercept=threshold, linetype="dashed", color="#333333", alpha=0.5)
        + p9.annotate(
            "text", x=len(plot_df) / 2, y=threshold - 0.05, label=f"Cutoff: {threshold:.0%}", size=9, color="#333333"
        )
        # --- Formatting ---
        + p9.scale_y_continuous(labels=percent_format(), limits=(0, 1.05), expand=(0, 0))
        + p9.labs(
            title=f"Feature Importance: Hybrid Pareto ({eco_code})",
            subtitle="Blue Dots: Individual Contribution | Orange Line: Cumulative",
            x="Drivers (Sorted by Importance)",
            y="Importance / Cumulative %",
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(12, 6),
            axis_text_x=p9.element_text(rotation=45, hjust=1),  # Rotate x-labels for readability
            panel_grid_major_x=p9.element_blank(),  # Remove vertical grid lines for cleaner look
        )
    )

    return plot
