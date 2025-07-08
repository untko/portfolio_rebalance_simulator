import streamlit as st
import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime, timedelta

CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _get_cache_filepath(asset):
    # Sanitize asset name for filename (replace problematic characters)
    safe_asset_name = asset.replace('^', '_').replace('-', '_') # Example: ^GSPC -> _GSPC, BTC-USD -> BTC_USD
    return os.path.join(CACHE_DIR, f"{safe_asset_name}.csv")

def _load_from_cache(asset, enable_debugging):
    filepath = _get_cache_filepath(asset)
    if os.path.exists(filepath):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        if datetime.now() - file_mod_time < timedelta(days=1): # Cache valid for 1 day
            if enable_debugging:
                st.write(f"Debug: Loading full history for {asset} from cache: {filepath}")
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            if enable_debugging:
                st.write(f"Debug: Cache for {asset} is stale. Fetching new data.")
    return None

def _save_to_cache(df, asset, enable_debugging):
    filepath = _get_cache_filepath(asset)
    if enable_debugging:
        st.write(f"Debug: Saving full history for {asset} to cache: {filepath}")
    df.to_csv(filepath)

@st.cache_data
def get_data(assets, start, end, enable_debugging=False):
    full_data = pd.DataFrame()
    
    for asset in assets:
        asset_data = _load_from_cache(asset, enable_debugging)
        if asset_data is None or asset_data.empty:
            # Fetch from yfinance if not in cache or stale
            if enable_debugging:
                st.write(f"Debug: Fetching {asset} from yfinance...")
            retries = 3
            for i in range(retries):
                try:
                    # Fetch full history from yfinance
                    temp_data = yf.download(asset, start="1900-01-01", end=datetime.now().date(), progress=False)
                    if not temp_data.empty:
                        # Standardize data format from yfinance
                        if isinstance(temp_data.columns, pd.MultiIndex):
                            # yfinance returns (Attribute, Ticker) for MultiIndex
                            # We need to find the 'Close' price for the current asset
                            if ('Close', asset) in temp_data.columns:
                                asset_data = temp_data['Close'][asset].to_frame(name=asset)
                            elif ('Adj Close', asset) in temp_data.columns:
                                asset_data = temp_data['Adj Close'][asset].to_frame(name=asset)
                            else:
                                # Fallback if 'Close' or 'Adj Close' not found in MultiIndex
                                st.warning(f"Could not find 'Close' or 'Adj Close' for {asset} in MultiIndex. Using first column.")
                                asset_data = temp_data.iloc[:, 0].to_frame(name=asset)
                        elif 'Close' in temp_data.columns:
                            asset_data = temp_data[['Close']].rename(columns={'Close': asset})
                        elif 'Adj Close' in temp_data.columns:
                            asset_data = temp_data[['Adj Close']].rename(columns={'Adj Close': asset})
                        else:
                            # Fallback if neither 'Close' nor 'Adj Close' is found in single index
                            st.warning(f"Could not find 'Close' or 'Adj Close' for {asset}. Using first column.")
                            asset_data = temp_data.iloc[:, :1]
                            asset_data.columns = [asset]

                        asset_data.index = pd.to_datetime(asset_data.index)
                        asset_data = asset_data.sort_index()
                        _save_to_cache(asset_data, asset, enable_debugging)
                        break # Success
                    else:
                        st.warning(f"yfinance returned empty data for {asset}. Retrying... ({i+1}/{retries})")
                except Exception as e:
                    st.error(f"Error fetching data for {asset} with yfinance: {e}. Retrying... ({i+1}/{retries})")
                time.sleep(2) # Wait before retrying

        if asset_data is not None and not asset_data.empty:
            # Ensure the asset_data has the correct column name before concat
            if asset_data.columns[0] != asset:
                asset_data = asset_data.rename(columns={asset_data.columns[0]: asset})
            full_data = pd.concat([full_data, asset_data], axis=1)
        else:
            st.warning(f"Could not retrieve data for {asset}. It will be excluded from the simulation.")

    if full_data.empty:
        st.error("No valid data available for simulation. Please check asset tickers and date range.")
        return pd.DataFrame()

    # Filter the full data to the requested date range
    data = full_data.loc[start:end]

    if data.empty:
        st.warning(f"No data available for the selected date range ({start} to {end}). Please try a wider range.")
        return pd.DataFrame()

    if enable_debugging:
        st.write(f"Debug: Final data columns from get_data: {data.columns.tolist()}")

    # Final check to ensure all requested assets are in the DataFrame
    actual_assets = data.columns.tolist()
    if len(actual_assets) < len(assets):
        missing_assets = [asset for asset in assets if asset not in actual_assets]
        st.error(f"Error: Data for the following assets could not be retrieved: {', '.join(missing_assets)}. Please check the ticker symbols.")
        return pd.DataFrame(), [] # Return empty DataFrame and empty assets list to stop simulation

    return data, actual_assets