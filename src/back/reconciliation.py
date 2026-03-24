from pathlib import Path

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace, OptimalCombination, BottomUp

from src.back.configuration import get_instantiated_reconciliation_model


class Reconciliation:
    def __init__(self, configuration_path: str | Path) -> None:
        self.__reconcilers = get_instantiated_reconciliation_model(configuration_path)

    # Reconcile
    def reconcile_best(self, df_fct, df_fit, S_df, tags):
        """
        Reconcile bottom-level forecasts to the hierarchy using hierarchicalforecast.

        Args:
        df_fct (pd.DataFrame): base forecasts with columns ['unique_id','ds','yhat'].
        df_fit (pd.DataFrame): in-sample actuals + fitted values used to compute S matrix if needed.
        S_df (pd.DataFrame): summing matrix returned by hf_aggregate.
        tags (dict): tags describing aggregation mapping returned by hf_aggregate.

        Returns:
        pd.DataFrame: reconciled forecasts; column 'yhat_rec' contains reconciled values.
        """
        # choose reconciler (mint_shrink chosen as more stable than OLS)
        hrec = HierarchicalReconciliation(reconcilers=self.__reconcilers)
        df_fct = hrec.reconcile(
            Y_hat_df=df_fct,
            Y_df=df_fit,
            S_df=S_df,
            tags=tags,
        )
        # rename reconciled column
        col = df_fct.columns.difference(["unique_id", "ds", "y_hat"])[0]
        df_fct = df_fct.rename(columns={col: "y_hat_rec"})
        return df_fct

