import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import numpy as np

def get_earliest_date(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch the full history
        hist = ticker.history(period="max")
        if not hist.empty:
            return hist.index[0].date()
    except Exception:
        return None
    return None

def render_simulation_parameters():
    st.header("Simulation Parameters")

    # Amount and Fee
    col1, col2, col3, col4 = st.columns([0.15, 0.35, 0.15, 0.35])
    with col1:
        st.write("Amount ($)")
    with col2:
        amount = st.number_input("", min_value=1, value=1000, label_visibility="collapsed", key="amount_input")
    with col3:
        st.write("Fee (%)")
    with col4:
        fee = st.number_input("", min_value=0.0, value=0.1, step=0.01, format="%.2f", label_visibility="collapsed", key="fee_input")


    # Assets
    assets_str = st.text_input("Assets (comma-separated tickers)", "BTC-USD,GLD")
    st.info("You can use any tickers available on Yahoo Finance (e.g., AAPL, GOOG, GLD, ETH-USD, ^GSPC). Find them at https://finance.yahoo.com/")
    assets = [s.strip().upper() for s in assets_str.split(',')]
    
    earliest_dates = {}
    # Display current prices of selected assets
    if assets:
        st.subheader("Current Asset Prices")
        cols = st.columns(len(assets))
        for i, asset in enumerate(assets):
            with cols[i]:
                current_price = "N/A"
                asset_long_name = ""
                earliest_date_str = "N/A"
                try:
                    ticker = yf.Ticker(asset)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current_price = f"{hist['Close'].iloc[-1]:.2f}$"
                    
                    # Fetch long name
                    info = ticker.info
                    if 'longName' in info:
                        asset_long_name = info['longName']
                    elif 'shortName' in info:
                        asset_long_name = info['shortName']
                    
                    # Get earliest date
                    earliest_date = get_earliest_date(asset)
                    if earliest_date:
                        earliest_dates[asset] = earliest_date
                        earliest_date_str = earliest_date.strftime('%d-%m-%Y')

                except Exception:
                    pass # Ignore errors if price or info cannot be fetched
                
                display_name = f"{asset} - {asset_long_name}" if asset_long_name else asset
                st.markdown(f"<h4 style='text-align: center; margin-bottom: 0px;'>{display_name}</h4><p style='text-align: center; font-size: 1.2em; margin-top: 0px;'>{current_price}</p><p style='text-align: center; font-size: 0.9em; margin-top: 0px;'>Since: {earliest_date_str}</p>", unsafe_allow_html=True)

    # Ratios
    st.subheader("Ratio")
    ratios = []

    if len(assets) == 2:
        ratio_asset1 = st.slider(f"Ratio between {assets[0]} and {assets[1]}", 0, 100, 50, 1)
        ratios = [ratio_asset1, 100 - ratio_asset1]
        st.write("Current Allocation (%)")
        cols = st.columns(2)
        with cols[0]:
            st.metric(label=assets[0], value=f"{ratios[0]}%")
        with cols[1]:
            st.metric(label=assets[1], value=f"{ratios[1]}%")
    elif len(assets) > 2:
        st.subheader("Enter Ratios (comma-separated percentages)")
        ratios_str = st.text_input("e.g., 40,30,30", value=",".join([str(int(100/len(assets))) for _ in assets]))
        try:
            ratios = [float(r.strip()) for r in ratios_str.split(',')]
            if len(ratios) != len(assets):
                st.warning(f"Please enter {len(assets)} ratios, one for each asset.")
                ratios = [] # Clear ratios to prevent simulation with incorrect count
            elif sum(ratios) != 100:
                st.warning("Ratios must sum to 100.")
            else:
                st.subheader("Current Allocation (%)")
                cols = st.columns(len(assets))
                for i, asset in enumerate(assets):
                    with cols[i]:
                        st.metric(label=asset, value=f"{ratios[i]}%")
        except ValueError:
            st.warning("Please enter valid numbers for ratios.")
            ratios = [] # Clear ratios to prevent simulation with invalid input
    elif len(assets) == 1:
        ratios = [100]
        st.write(f"100% {assets[0]}")

    # Range
    st.subheader("Range")

    predefined_ranges = {
        "10 Years": 10,
        "5 Years": 5,
        "3 Years": 3,
        "1 Year": 1,
        "6 Months": 0.5,
        "3 Months": 3/12,
        "1 Month": 1/12,
    }

    # Get the index of '5 Years' to set it as default
    range_keys = list(predefined_ranges.keys())
    # Add "Automatic" to the beginning of the list
    all_range_options = ["Automatic"] + range_keys + ["Custom"]
    default_range_index = all_range_options.index("5 Years")

    range_option = st.selectbox("Select Range (predefined or custom)", all_range_options, index=default_range_index)

    end_date = datetime.now().date()

    if range_option == "Automatic":
        if earliest_dates:
            # Find the latest of the earliest dates
            start_date = max(earliest_dates.values())
            st.write(f"Simulation will start from {start_date.strftime('%d-%m-%Y')} (based on the asset with the latest start date).")
        else:
            # Fallback if no earliest dates could be found
            start_date = end_date - timedelta(days=365*5)
            st.warning("Could not determine earliest date automatically. Defaulting to 5 years.")

    elif range_option != "Custom":
        years_to_subtract = predefined_ranges[range_option]
        if years_to_subtract < 1:
            start_date = end_date - timedelta(days=int(365 * years_to_subtract))
        else:
            start_date = end_date - timedelta(days=int(365 * years_to_subtract))
        st.write(f"Selected range: {range_option}. Start date: {start_date.strftime('%d-%m-%y')}")
    else:
        st.subheader("Custom Date Range")
        start_date = st.date_input("Start Date", value=end_date - timedelta(days=365))
        end_date = st.date_input("End Date", value=end_date)

    # Rebalancing Strategy
    st.subheader("Rebalancing Strategy")

    # Periodic Rebalancing
    st.write("Periodic")
    periodic_options = {
        "daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90, "6monthly": 180, "yearly": 365
    }
    selected_periodic = []
    cols = st.columns(len(periodic_options))
    for i, (label, days) in enumerate(periodic_options.items()):
        if cols[i].checkbox(label, value=True, key=f"periodic_{label}"):
            selected_periodic.append(days)

    custom_periodic_enabled = st.checkbox("Custom (days)", value=False, key="enable_custom_periodic")
    if custom_periodic_enabled:
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.write("Custom (in days)")
        with col2:
            custom_periodic = st.number_input("", min_value=1, value=1, label_visibility="collapsed", key="custom_periodic_input")
        if custom_periodic:
            selected_periodic.append(custom_periodic)

    # Threshold Rebalancing
    st.write("Threshold")
    threshold_options = {
        "1%": 0.01, "5%": 0.05, "10%": 0.10, "50%": 0.50
    }
    selected_threshold = []
    cols = st.columns(len(threshold_options))
    for i, (label, percentage) in enumerate(threshold_options.items()):
        # Set 50% threshold to be off by default
        default_value = False if label == "50%" else True
        if cols[i].checkbox(label, value=default_value, key=f"threshold_{label}"):
            selected_threshold.append(percentage)

    custom_threshold_enabled = st.checkbox("Custom (%)", value=False, key="enable_custom_threshold")
    if custom_threshold_enabled:
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.write("Custom (in %)")
        with col2:
            custom_threshold = st.number_input("", min_value=0.1, value=0.1, label_visibility="collapsed", key="custom_threshold_input")
        if custom_threshold:
            selected_threshold.append(custom_threshold / 100.0)

    # Individual Asset Buy and Hold
    st.subheader("Individual Asset Buy and Hold")
    
    selected_individual_assets = []
    for asset in assets:
        if st.checkbox(f"Buy and Hold {asset}", value=True, key=f"bh_{asset}"):
            selected_individual_assets.append(asset)

    # Combined Debug Checkbox
    enable_debugging = st.checkbox("Enable Debugging", key="enable_debugging_checkbox")

    # Simulation Button
    simulate_button = st.button("Simulate", key="simulate_button")

    return amount, assets, ratios, start_date, end_date, selected_periodic, selected_threshold, selected_individual_assets, enable_debugging, simulate_button, fee