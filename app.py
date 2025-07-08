
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.title("Portfolio Rebalancing Simulator")

# --- Sidebar ---
st.sidebar.header("Simulation Parameters")

# Amount
amount = st.sidebar.number_input("Amount", min_value=1, value=1000)

# Assets
assets_str = st.sidebar.text_input("Assets (comma-separated tickers)", "BTC-USD,GLD")
assets = [s.strip().upper() for s in assets_str.split(',')]

# Ratios
st.sidebar.subheader("Ratio")
ratios = []

if len(assets) == 2:
    ratio_asset1 = st.sidebar.slider(f"Ratio between {assets[0]} and {assets[1]}", 0, 100, 50, 1)
    ratios = [ratio_asset1, 100 - ratio_asset1]
    st.sidebar.subheader("Manual Input (%)")
    cols = st.sidebar.columns(2)
    with cols[0]:
        st.metric(label=assets[0], value=f"{ratios[0]}%")
    with cols[1]:
        st.metric(label=assets[1], value=f"{ratios[1]}%")
elif len(assets) > 2:
    st.sidebar.subheader("Manual Input (%)")
    ratios = []
    cols = st.sidebar.columns(len(assets))
    for i, asset in enumerate(assets):
        with cols[i]:
            ratios.append(st.number_input(f"{asset}", min_value=0, max_value=100, value=int(100/len(assets)), key=f"ratio_{asset}"))
    if sum(ratios) != 100:
        st.sidebar.warning("Ratios must sum to 100.")
elif len(assets) == 1:
    ratios = [100]
    st.sidebar.write(f"100% {assets[0]}")
# Range
st.sidebar.subheader("Range")

predefined_ranges = {
    "5 Years": 5,
    "1 Year": 1,
    "6 Months": 0.5,
    "1 Month": 1/12,
}

range_option = st.sidebar.selectbox("Select Range", list(predefined_ranges.keys()) + ["Custom"])

from datetime import datetime, timedelta

end_date = datetime.now().date()

if range_option != "Custom":
    years_to_subtract = predefined_ranges[range_option]
    if years_to_subtract < 1:
        start_date = end_date - timedelta(days=int(365 * years_to_subtract))
    else:
        start_date = end_date - timedelta(days=int(365 * years_to_subtract))
    st.sidebar.write(f"Earliest available data: {start_date.strftime('%d-%m-%y')}")
# Rebalancing Strategy
st.sidebar.subheader("Rebalancing Strategy")

# Periodic Rebalancing
st.sidebar.write("Periodic")
periodic_options = {
    "daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365
}
selected_periodic = []
cols = st.sidebar.columns(len(periodic_options))
for i, (label, days) in enumerate(periodic_options.items()):
    if cols[i].checkbox(label, key=f"periodic_{label}"):
        selected_periodic.append(days)

custom_periodic = st.sidebar.number_input("Custom (in days)", min_value=1, step=1)
if custom_periodic:
    selected_periodic.append(custom_periodic)

# Threshold Rebalancing
st.sidebar.write("Threshold")
threshold_options = {
    "0.1%": 0.001, "1%": 0.01, "5%": 0.05, "10%": 0.10, "50%": 0.50
}
selected_threshold = []
cols = st.sidebar.columns(len(threshold_options))
for i, (label, percentage) in enumerate(threshold_options.items()):
    if cols[i].checkbox(label, key=f"threshold_{label}"):
        selected_threshold.append(percentage)

custom_threshold = st.sidebar.number_input("Custom (in %)", min_value=0.1, step=0.1)
if custom_threshold:
    selected_threshold.append(custom_threshold / 100.0)

# Individual Asset Buy and Hold
st.sidebar.subheader("Individual Asset Buy and Hold")
st.sidebar.info("For individual asset Buy and Hold, ensure the asset is also listed in the 'Assets (comma-separated tickers)' input above.")
selected_individual_assets = []
for asset in assets:
    if st.sidebar.checkbox(f"Buy and Hold {asset}", value=True, key=f"bh_{asset}"):
        selected_individual_assets.append(asset)

# Combined Debug Checkbox
enable_debugging = st.sidebar.checkbox("Enable Debugging")

# Simulation Button
simulate_button = st.sidebar.button("Simulate")
automatic_checkbox = st.sidebar.checkbox("automatic")

# --- Data Fetching ---
import time

@st.cache_data
def get_data(assets, start, end):
    data = pd.DataFrame()
    retries = 3
    for i in range(retries):
        try:
            data = yf.download(assets, start=start, end=end, progress=False)
            if not data.empty:
                break # Success
            else:
                st.warning(f"yfinance returned empty data for {assets}. Retrying... ({i+1}/{retries})")
        except Exception as e:
            st.error(f"Error fetching data with yfinance: {e}. Retrying... ({i+1}/{retries})")
        time.sleep(2) # Wait before retrying

    if data.empty:
        st.error("Failed to fetch data after multiple retries. Please check asset tickers and date range.")
        return pd.DataFrame() # Return empty DataFrame on persistent failure

    if isinstance(data.columns, pd.MultiIndex):
        # Multiple assets, select 'Close' and ensure column names are tickers
        data = data['Close']
    elif 'Adj Close' in data.columns:
        # Single asset, select 'Adj Close' and rename column to ticker
        data = data[['Adj Close']]
        data.columns = assets # Rename the single column to the asset ticker
    elif 'Close' in data.columns:
        # Fallback for single asset if 'Adj Close' is not available
        data = data[['Close']]
        data.columns = assets # Rename the single column to the asset ticker
    else:
        # Generic fallback if neither 'Adj Close' nor 'Close' is found
        # This might happen for very obscure data, but we should handle it.
        # Take the first column and rename it.
        data = data.iloc[:, :1]
        data.columns = assets # Rename the single column to the asset ticker

    # Ensure the index is datetime and sorted
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if enable_debugging:
        st.write(f"Debug: Final data columns from get_data: {data.columns.tolist()}")

    return data


if st.checkbox("Debug Data Fetching"):
    st.subheader("Data Debugging")
    debug_assets = ["BTC-USD", "GLD"] # Using sample assets for debugging
    debug_start_date = datetime(2023, 1, 1).date()
    debug_end_date = datetime(2024, 1, 1).date()
    
    st.write(f"Fetching data for: {debug_assets} from {debug_start_date} to {debug_end_date}")
    debug_data = get_data(debug_assets, debug_start_date, debug_end_date)
    
    if not debug_data.empty:
        st.write("Data Head:")
        st.dataframe(debug_data.head())
        
        st.write("Data Info:")
        # Streamlit doesn't directly display df.info() well, so convert to string
        import io
        buffer = io.StringIO()
        debug_data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("Data Columns:")
        st.write(debug_data.columns.tolist())
    else:
        st.warning("Debug data could not be fetched or is empty.")

# --- Simulation Logic ---
def run_simulation(data, initial_amount, ratios, strategy_type, param, enable_debugging=False):

    ratios_arr = np.array(ratios) / 100.0

    # Handle single asset case (no rebalancing)
    if len(data.columns) == 1:
        initial_shares = initial_amount / data.iloc[0, 0]
        portfolio_values = data.iloc[:, 0] * initial_shares
        return portfolio_values

    portfolio_values_history = pd.Series(index=data.index)
    portfolio_values_history.iloc[0] = initial_amount

    # Calculate daily returns for all assets
    daily_returns = data.pct_change().fillna(0) # Fill NaN (first row) with 0

    # Initialize asset holdings (monetary value) based on initial amount and ratios
    asset_holdings = initial_amount * ratios_arr

    last_rebalance_date = data.index[0]

    for i in range(1, len(data)):
        current_date = data.index[i]
        
        # Update asset holdings based on daily returns
        asset_holdings = asset_holdings * (1 + daily_returns.iloc[i])

        # Calculate current total portfolio value
        current_portfolio_value = asset_holdings.sum()

        # Rebalancing logic
        rebalanced = False
        if strategy_type == 'Periodic':
            if (current_date - last_rebalance_date).days >= param:
                rebalanced = True
        elif strategy_type == 'Threshold':
            if current_portfolio_value > 0:
                current_weights = asset_holdings / current_portfolio_value
                if np.any(np.abs(current_weights - ratios_arr) > param):
                    rebalanced = True
            else:
                rebalanced = False

        if enable_debugging:
            st.write(f"--- Date: {current_date} ---")
            st.write(f"  Daily Returns: {daily_returns.iloc[i].to_dict()}")
            st.write(f"  Asset Holdings (before rebalance): {asset_holdings.to_dict()}")
            st.write(f"  Current Portfolio Value (before rebalance): {current_portfolio_value:.2f}")
            st.write(f"  Rebalanced: {rebalanced}")

        if rebalanced:
            # Rebalance asset holdings to target allocations based on current_portfolio_value
            asset_holdings = current_portfolio_value * ratios_arr
            last_rebalance_date = current_date
            # Recalculate current_portfolio_value after rebalancing for accurate recording
            current_portfolio_value = asset_holdings.sum()
            if enable_debugging:
                st.write(f"  Asset Holdings (after rebalance): {asset_holdings.to_dict()}")
                st.write(f"  Current Portfolio Value (after rebalance): {current_portfolio_value:.2f}")

        # Record the total portfolio value for the day (after potential rebalancing)
        portfolio_values_history[current_date] = current_portfolio_value

    return portfolio_values_history

def run_individual_buy_and_hold(data, initial_amount, asset_ticker, enable_debugging=False):
    if enable_debugging:
        st.write(f"Debug BH: Processing asset_ticker: {asset_ticker}")
        st.write(f"Debug BH: Data columns: {data.columns.tolist()}")

    if asset_ticker not in data.columns:
        st.warning(f"Asset {asset_ticker} not found in data.")
        return None

    asset_prices = data[asset_ticker]
    if asset_prices.empty:
        return None

    # Calculate initial shares based on the first day's price
    initial_price = asset_prices.iloc[0]
    if initial_price == 0:
        st.warning(f"Initial price for {asset_ticker} is zero. Cannot calculate Buy and Hold.")
        return None

    initial_shares = initial_amount / initial_price

    # Calculate portfolio value over time
    portfolio_values = asset_prices * initial_shares
    return portfolio_values

# --- Main App ---
if simulate_button:
    if not assets:
        st.warning("Please enter at least one asset.")
    elif sum(ratios) != 100 and len(assets) > 1:
        st.error("Ratios must sum to 100% to run the simulation.")
    else:
        st.write(f"Attempting to fetch data for assets: {assets}, from {start_date} to {end_date}")
        data = get_data(assets, start_date, end_date, enable_debugging)

        if data.empty:
            st.error("Could not fetch data for the selected assets and date range. Please try different tickers or a wider date range.")
        else:
            all_results = {}

            # Run Periodic Simulations
            for days in selected_periodic:
                result = run_simulation(data, amount, ratios, "Periodic", days, enable_debugging)
                if result is not None:
                    all_results[f"Periodic ({days} days)"] = result

            # Run Threshold Simulations
            for percentage in selected_threshold:
                result = run_simulation(data, amount, ratios, "Threshold", percentage, enable_debugging)
                if result is not None:
                    all_results[f"Threshold ({percentage*100:.1f}%)"] = result

            # Run Individual Asset Buy and Hold Simulations
            for asset_ticker in selected_individual_assets:
                result = run_individual_buy_and_hold(data, amount, asset_ticker, enable_debugging)
                if result is not None:
                    all_results[f"Buy and Hold ({asset_ticker})"] = result

            if not all_results:
                st.write("Please select at least one rebalancing strategy.")
            else:
                # Combine results into a single DataFrame
                results_df = pd.DataFrame(all_results)
                
                # Add a "Buy and Hold" strategy for comparison
                buy_and_hold_values = run_simulation(data, amount, ratios, 'Periodic', float('inf'), enable_debugging)
                results_df['Buy and Hold (Portfolio)'] = buy_and_hold_values

                # Normalize results to show percentage growth
                results_df_normalized = (results_df / results_df.iloc[0]) * 100

                # Sort columns by their final value (highest to lowest)
                final_values_sorted_columns = results_df.iloc[-1].sort_values(ascending=False).index
                results_df_normalized = results_df_normalized[final_values_sorted_columns]

                st.subheader("Simulation Results")

                # Plot Size Adjustment
                st.sidebar.subheader("Plot Settings")
                plot_width = st.sidebar.number_input("Plot Width (pixels)", min_value=300, value=800, step=50)
                plot_height = st.sidebar.number_input("Plot Height (pixels)", min_value=300, value=500, step=50)

                fig = px.line(results_df_normalized, 
                              title="Portfolio Growth Over Time",
                              labels={"index": "Date", "value": "Portfolio Value (%)", "variable": "Strategy"})
                
                fig.update_layout(
                    hovermode='x unified',
                    legend_title_text='Rebalancing Strategy',
                    yaxis_title="Portfolio Value (% of Initial)",
                    xaxis_title="Date"
                )
                st.plotly_chart(fig, use_container_width=False, width=plot_width, height=plot_height)

                st.subheader("Final Portfolio Values")
                st.dataframe(results_df.iloc[-1].sort_values(ascending=False).to_frame(name="Final Value"), use_container_width=True, height=300)

