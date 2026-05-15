# Modify app

## Plan

This project starts from the interactive Streamlit app src/front/streamlit_app.py to modify the first table, adding the possibility to filter data using user-friendly labels.

**Implementation plan:**

1. Use `from streamlit_extras.dataframe_explorer import dataframe_explorer` to insert filters above the first table.
2. Do not use the technical field names in the filters. Instead, rename the columns before passing them to `dataframe_explorer` so that only the following labels appear:
	- "Sector" (was: ECO_GRP_1_DSC)
	- "Segment" (was: ECO_GRP_2_DSC)
	- "Code" (was: ECO_COD)
	- "Description" (was: ECO_DSC)
	- "Weight %" (was: PERC)
3. Do **not** include "ECO_GRP_0_DSC" (Area) among the filterable columns, but keep it visible in the table if needed.
4. The filters and the table must always show the user-friendly labels, not the technical field names.
5. When you apply a filter, the table must not show the index (set `hide_index=True` or equivalent in Streamlit).
6. Attention!!! When you apply a filter, second table must update accordingly to reflect the filtered data.
7. The rest of the app logic (row selection, downstream features) must remain unchanged.

## Add variable importance plot

To further enhance the interpretability of the model, add a variable importance plot for the selected economic code (row) in the second table. This plot should visually display the relative importance (COEFF) of each driver associated with the selected economic code.

**Implementation plan:**

1. When a row is selected in the first table, retrieve the list of drivers and their COEFF values for the corresponding economic code.
2. Create a horizontal bar plot (e.g., with Plotly or matplotlib) showing driver codes/descriptions on the y-axis and their importance (COEFF) on the x-axis, sorted descending by importance.
3. Display the plot below or next to the second table (driver values) in the UI.
4. If no drivers are available for the selected code, show a message or an empty plot.
5. The plot should update automatically when the selection in the first table changes or when filters are applied.
6. Use a clear title, axis labels, and a consistent color (e.g., orange) for the bars.

---


