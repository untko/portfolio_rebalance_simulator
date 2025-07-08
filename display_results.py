import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def display_simulation_results(all_results, results_df, results_df_normalized, enable_debugging=False):
    if not all_results:
        st.write("Please select at least one rebalancing strategy.")
    else:
        st.subheader("Simulation Results")

        fig = px.line(results_df_normalized, 
                      title="Portfolio Growth Over Time",
                      labels={"index": "Date", "value": "Portfolio Value (%)", "variable": "Strategy"})
        
        # Update legend names to include final percentage return
        for trace in fig.data:
            strategy_name = trace.name
            final_value = results_df_normalized[strategy_name].iloc[-1]
            trace.name = f"{strategy_name} ({final_value:.2f}%)"

            # Set distinct line style for individual Buy and Hold assets
            if strategy_name.startswith("Buy and Hold (") and strategy_name.endswith(")"):
                trace.line.dash = 'dash'

        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Rebalancing Strategy',
            yaxis_title="Portfolio Value (% of Initial)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sort columns by their final value (highest to lowest)
        final_values_sorted_columns = results_df.iloc[-1].sort_values(ascending=False).index
        results_df_normalized = results_df_normalized[final_values_sorted_columns]

        if enable_debugging:
            st.write("Debug: results_df tail:")
            st.dataframe(results_df.tail())
            st.write("Debug: results_df_normalized tail:")
            st.dataframe(results_df_normalized.tail())

        st.subheader("Final Portfolio Values")

        final_values = results_df.iloc[-1]
        percentage_change = results_df_normalized.iloc[-1] - 100

        # Calculate Volatility (Annualized Standard Deviation of Daily Returns)
        daily_returns_normalized = results_df_normalized.pct_change().fillna(0)
        volatility = daily_returns_normalized.std() * np.sqrt(252) * 100 # Annualized volatility in %

        # Calculate % Difference from Max Value
        max_final_value = final_values.max()
        percentage_diff_from_max = ((final_values - max_final_value) / max_final_value) * 100

        final_summary_df = pd.DataFrame({
            "Final Value": final_values,
            "% Change": percentage_change,
            "Volatility (%)": volatility,
            "% Diff from Max": percentage_diff_from_max
        })

        # Sort by Final Value (highest to lowest) before formatting
        final_summary_df = final_summary_df.sort_values(by="Final Value", ascending=False)

        # Apply formatting after sorting
        final_summary_df["Final Value"] = final_summary_df["Final Value"].map(lambda x: f"{x:.2f}")
        final_summary_df["% Change"] = final_summary_df["% Change"].map(lambda x: f"{x:.2f}%")
        final_summary_df["Volatility (%)"] = final_summary_df["Volatility (%)"].map(lambda x: f"{x:.2f}%")
        final_summary_df["% Diff from Max"] = final_summary_df["% Diff from Max"].map(lambda x: f"{x:.2f}%")

        st.dataframe(final_summary_df, use_container_width=True, height=300)