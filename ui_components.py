import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

def render_simulation_parameters():
    st.header("Simulation Parameters")

    # Amount
    amount = st.number_input("Amount", min_value=1, value=1000)

    # Assets
    assets_str = st.text_input("Assets (comma-separated tickers)", "BTC-USD,GLD")
    assets = [s.strip().upper() for s in assets_str.split(',')]

    # Ratios
    st.subheader("Ratio")
    ratios = []

    if len(assets) == 2:
        ratio_asset1 = st.slider(f"Ratio between {assets[0]} and {assets[1]}", 0, 100, 50, 1)
        ratios = [ratio_asset1, 100 - ratio_asset1]
        st.subheader("Current Allocation (%)")
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
        "5 Years": 5,
        "3 Years": 3,
        "1 Year": 1,
        "6 Months": 0.5,
        "1 Month": 1/12,
    }

    range_option = st.selectbox("Select Range", list(predefined_ranges.keys()) + ["Custom"])

    end_date = datetime.now().date()

    if range_option != "Custom":
        years_to_subtract = predefined_ranges[range_option]
        if years_to_subtract < 1:
            start_date = end_date - timedelta(days=int(365 * years_to_subtract))
        else:
            start_date = end_date - timedelta(days=int(365 * years_to_subtract))
        st.write(f"Earliest available data: {start_date.strftime('%d-%m-%y')}")
    else:
        st.subheader("Custom Date Range")
        start_date = st.date_input("Start Date", value=end_date - timedelta(days=365))
        end_date = st.date_input("End Date", value=end_date)

    # Rebalancing Strategy
    st.subheader("Rebalancing Strategy")

    # Periodic Rebalancing
    st.write("Periodic")
    periodic_options = {
        "daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365
    }
    selected_periodic = []
    cols = st.columns(len(periodic_options))
    for i, (label, days) in enumerate(periodic_options.items()):
        if cols[i].checkbox(label, value=True, key=f"periodic_{label}"):
            selected_periodic.append(days)

    custom_periodic_enabled = st.checkbox("Enable Custom Periodic (days)", value=False)
    if custom_periodic_enabled:
        custom_periodic = st.number_input("Custom (in days)", min_value=1, step=1)
        if custom_periodic:
            selected_periodic.append(custom_periodic)

    # Threshold Rebalancing
    st.write("Threshold")
    threshold_options = {
        "0.1%": 0.001, "1%": 0.01, "5%": 0.05, "10%": 0.10, "50%": 0.50
    }
    selected_threshold = []
    cols = st.columns(len(threshold_options))
    for i, (label, percentage) in enumerate(threshold_options.items()):
        if cols[i].checkbox(label, value=True, key=f"threshold_{label}"):
            selected_threshold.append(percentage)

    custom_threshold_enabled = st.checkbox("Enable Custom Threshold (%)", value=False)
    if custom_threshold_enabled:
        custom_threshold = st.number_input("Custom (in %)", min_value=0.1, step=0.1)
        if custom_threshold:
            selected_threshold.append(custom_threshold / 100.0)

    # Individual Asset Buy and Hold
    st.subheader("Individual Asset Buy and Hold")
    st.info("For individual asset Buy and Hold, ensure the asset is also listed in the 'Assets (comma-separated tickers)' input above.")
    selected_individual_assets = []
    for asset in assets:
        if st.checkbox(f"Buy and Hold {asset}", value=True, key=f"bh_{asset}"):
            selected_individual_assets.append(asset)

    # Combined Debug Checkbox
    enable_debugging = st.checkbox("Enable Debugging")

    # Simulation Button
    simulate_button = st.button("Simulate")

    return amount, assets, ratios, start_date, end_date, selected_periodic, selected_threshold, selected_individual_assets, enable_debugging, simulate_button