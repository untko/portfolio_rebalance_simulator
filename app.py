import streamlit as st
import pandas as pd
import numpy as np

from data_handler import get_data
from simulation_logic import run_simulation, run_individual_buy_and_hold
from ui_components import render_simulation_parameters
from display_results import display_simulation_results

st.title("Portfolio Rebalancing Simulator")

# Render UI components and get parameters
amount, assets, ratios, start_date, end_date, selected_periodic, selected_threshold, selected_individual_assets, enable_debugging, simulate_button = render_simulation_parameters()

# --- Main App Logic ---
if simulate_button:
    if not assets:
        st.warning("Please enter at least one asset.")
    elif sum(ratios) != 100 and len(assets) > 1:
        st.error("Ratios must sum to 100% to run the simulation.")
    else:
        data, assets = get_data(assets, start_date, end_date, enable_debugging)

        if data.empty:
            # Error message already displayed by get_data
            pass
        else:
            all_results = {}

            # Run Periodic Simulations
            for days in selected_periodic:
                result = run_simulation(data, amount, ratios, "Periodic", days, enable_debugging, debug_interval=100)
                if result is not None:
                    all_results[f"Periodic ({days} days)"] = result

            # Run Threshold Simulations
            for percentage in selected_threshold:
                result = run_simulation(data, amount, ratios, "Threshold", percentage, enable_debugging, debug_interval=100)
                if result is not None:
                    all_results[f"Threshold ({percentage*100:.1f}%)"] = result

            # Run Individual Asset Buy and Hold Simulations
            for asset_ticker in selected_individual_assets:
                # Only run if the asset_ticker is in the actual_assets list
                if asset_ticker in assets:
                    result = run_individual_buy_and_hold(data, amount, asset_ticker, enable_debugging)
                    if result is not None:
                        all_results[f"Buy and Hold ({asset_ticker})"] = result

            # Add a "Buy and Hold" strategy for comparison
            buy_and_hold_values = run_simulation(data, amount, ratios, 'Periodic', float('inf'), enable_debugging, debug_interval=100)
            all_results['Buy and Hold (Portfolio)'] = buy_and_hold_values

            # Combine results into a single DataFrame
            results_df = pd.DataFrame(all_results)

            # Normalize results to show percentage growth
            results_df_normalized = (results_df / results_df.iloc[0]) * 100

            # Sort columns by their final value (highest to lowest)
            final_values_sorted_columns = results_df.iloc[-1].sort_values(ascending=False).index
            results_df_normalized = results_df_normalized[final_values_sorted_columns]

            display_simulation_results(all_results, results_df, results_df_normalized, enable_debugging)