import streamlit as st
import pandas as pd
import numpy as np

def run_simulation(data, initial_amount, ratios, strategy_type, param, enable_debugging=False, debug_interval=100):

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
    asset_holdings = pd.Series(initial_amount * ratios_arr, index=data.columns)

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

        if enable_debugging and (i % debug_interval == 0 or rebalanced):
            st.write(f"--- Date: {current_date} ---")
            st.write(f"  Daily Returns: {daily_returns.iloc[i].to_dict()}")
            st.write(f"  Asset Holdings (before rebalance): {asset_holdings.to_dict()}")
            st.write(f"  Current Portfolio Value (before rebalance): {current_portfolio_value:.2f}")
            st.write(f"  Rebalanced: {rebalanced}")

        if rebalanced:
            # Rebalance asset holdings to target allocations based on current_portfolio_value
            asset_holdings = pd.Series(current_portfolio_value * ratios_arr, index=data.columns)
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
    
    if enable_debugging:
        st.write(f"Debug BH: {asset_ticker} asset_prices head:\n{asset_prices.head()}")
        st.write(f"Debug BH: {asset_ticker} asset_prices tail:\n{asset_prices.tail()}")

    if asset_prices.empty:
        if enable_debugging:
            st.write(f"Debug BH: {asset_ticker} asset_prices is empty.")
        return None

    # Calculate initial shares based on the first day's price
    initial_price = asset_prices.iloc[0]
    if enable_debugging:
        st.write(f"Debug BH: {asset_ticker} initial_price: {initial_price}")

    if initial_price == 0 or np.isnan(initial_price):
        st.warning(f"Initial price for {asset_ticker} is zero or NaN. Cannot calculate Buy and Hold.")
        return None

    initial_shares = initial_amount / initial_price

    # Calculate portfolio value over time
    portfolio_values = asset_prices * initial_shares
    
    # Fill any NaN values with the last valid observation
    portfolio_values = portfolio_values.ffill()

    return portfolio_values