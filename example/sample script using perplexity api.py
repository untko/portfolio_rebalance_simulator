# %%
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Download Bitcoin price data
btc_response = requests.get('https://perplexity.ai/rest/finance/history/BTCUSD/csv?start_date=2020-07-07&end_date=2025-07-07')
with open('btc_data.csv', 'wb') as f:
    f.write(btc_response.content)

# Download PAX Gold price data
paxg_response = requests.get('https://perplexity.ai/rest/finance/history/PAXGUSD/csv?start_date=2020-07-07&end_date=2025-07-07')
with open('paxg_data.csv', 'wb') as f:
    f.write(paxg_response.content)

# Load the data
btc_df = pd.read_csv('btc_data.csv')
paxg_df = pd.read_csv('paxg_data.csv')

print("Bitcoin data shape:", btc_df.shape)
print("PAX Gold data shape:", paxg_df.shape)
print("\nBitcoin data sample:")
print(btc_df.head())
print("\nPAX Gold data sample:")
print(paxg_df.head())

# Check date ranges
print("\nBitcoin date range:", btc_df['date'].min(), "to", btc_df['date'].max())
print("PAX Gold date range:", paxg_df['date'].min(), "to", paxg_df['date'].max())

# %%
# Clean and prepare data
btc_df['date'] = pd.to_datetime(btc_df['date'])
paxg_df['date'] = pd.to_datetime(paxg_df['date'])

# Sort by date ascending
btc_df = btc_df.sort_values('date')
paxg_df = paxg_df.sort_values('date')

# Merge the datasets
merged_df = pd.merge(btc_df[['date', 'close']], paxg_df[['date', 'close']], on='date', suffixes=('_btc', '_paxg'))

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
print(f"Total days: {len(merged_df)}")

# Check for any missing data
print(f"\nMissing BTC data: {merged_df['close_btc'].isna().sum()}")
print(f"Missing PAXG data: {merged_df['close_paxg'].isna().sum()}")

# Display first and last few rows
print("\nFirst 5 rows:")
print(merged_df.head())
print("\nLast 5 rows:")
print(merged_df.tail())

# %%
# Function to calculate rebalancing strategy performance
def calculate_rebalancing_performance(df, btc_weight=0.5, paxg_weight=0.5, rebalance_frequency='weekly', threshold=0.0):
    """
    Calculate performance of a rebalancing strategy
    
    Args:
        df: DataFrame with date, close_btc, close_paxg columns
        btc_weight: Target BTC allocation (0.5 for 50%)
        paxg_weight: Target PAXG allocation (0.5 for 50%)
        rebalance_frequency: 'daily', 'weekly', 'monthly', or 'threshold'
        threshold: Only rebalance if allocation drifts by this percentage (e.g., 0.05 for 5%)
    
    Returns:
        DataFrame with performance metrics
    """
    
    # Initialize portfolio
    initial_value = 10000  # Starting with $10,000
    
    # Calculate daily returns
    df = df.copy()
    df['btc_return'] = df['close_btc'].pct_change()
    df['paxg_return'] = df['close_paxg'].pct_change()
    
    # Initialize tracking variables
    results = []
    btc_value = initial_value * btc_weight
    paxg_value = initial_value * paxg_weight
    total_value = initial_value
    rebalance_count = 0
    
    # Determine rebalancing days
    df['rebalance_day'] = False
    
    if rebalance_frequency == 'daily':
        df['rebalance_day'] = True
    elif rebalance_frequency == 'weekly':
        # Rebalance every Monday (weekday 0)
        df['weekday'] = df['date'].dt.weekday
        df['rebalance_day'] = (df['weekday'] == 0)
    elif rebalance_frequency == 'monthly':
        # Rebalance on first day of each month
        df['month'] = df['date'].dt.to_period('M')
        df['rebalance_day'] = df['month'] != df['month'].shift(1)
    
    for i, row in df.iterrows():
        if i == 0:  # Skip first row (no previous data for returns)
            results.append({
                'date': row['date'],
                'btc_value': btc_value,
                'paxg_value': paxg_value,
                'total_value': total_value,
                'btc_allocation': btc_weight,
                'paxg_allocation': paxg_weight,
                'rebalanced': False
            })
            continue
        
        # Update values based on daily returns
        if not pd.isna(row['btc_return']):
            btc_value *= (1 + row['btc_return'])
        if not pd.isna(row['paxg_return']):
            paxg_value *= (1 + row['paxg_return'])
        
        total_value = btc_value + paxg_value
        current_btc_allocation = btc_value / total_value
        current_paxg_allocation = paxg_value / total_value
        
        # Check if rebalancing is needed
        rebalance_needed = False
        
        if rebalance_frequency == 'threshold':
            # Rebalance if allocation drifts beyond threshold
            if abs(current_btc_allocation - btc_weight) > threshold:
                rebalance_needed = True
        else:
            # Rebalance on scheduled days
            rebalance_needed = row['rebalance_day']
        
        if rebalance_needed:
            # Rebalance to target allocation
            btc_value = total_value * btc_weight
            paxg_value = total_value * paxg_weight
            rebalance_count += 1
            rebalanced = True
        else:
            rebalanced = False
        
        results.append({
            'date': row['date'],
            'btc_value': btc_value,
            'paxg_value': paxg_value,
            'total_value': total_value,
            'btc_allocation': btc_value / total_value,
            'paxg_allocation': paxg_value / total_value,
            'rebalanced': rebalanced
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    total_return = (results_df['total_value'].iloc[-1] / initial_value - 1) * 100
    
    # Calculate daily returns for volatility
    results_df['daily_return'] = results_df['total_value'].pct_change()
    volatility = results_df['daily_return'].std() * np.sqrt(252) * 100  # Annualized volatility
    
    # Calculate maximum drawdown
    running_max = results_df['total_value'].expanding().max()
    drawdown = (results_df['total_value'] / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_return = (results_df['daily_return'].mean() * 252) - risk_free_rate
    sharpe_ratio = excess_return / (results_df['daily_return'].std() * np.sqrt(252))
    
    # Calculate annualized return
    days = len(results_df)
    years = days / 365.25
    annualized_return = ((results_df['total_value'].iloc[-1] / initial_value) ** (1/years) - 1) * 100
    
    return {
        'results_df': results_df,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'rebalance_count': rebalance_count,
        'final_value': results_df['total_value'].iloc[-1]
    }

# Test the function with a simple weekly rebalancing
print("Testing rebalancing calculation function...")
test_result = calculate_rebalancing_performance(merged_df, rebalance_frequency='weekly')
print(f"Total return: {test_result['total_return']:.2f}%")
print(f"Annualized return: {test_result['annualized_return']:.2f}%")
print(f"Volatility: {test_result['volatility']:.2f}%")
print(f"Max drawdown: {test_result['max_drawdown']:.2f}%")
print(f"Sharpe ratio: {test_result['sharpe_ratio']:.2f}")
print(f"Rebalance count: {test_result['rebalance_count']}")
print(f"Final portfolio value: ${test_result['final_value']:,.2f}")

# %%
# Now let's run comprehensive analysis for different time periods and thresholds
# Time periods to analyze: 1 year, 2 years, 3 years, 4 years, 5 years

def analyze_multiple_periods_and_thresholds(df):
    """
    Analyze 50-50 rebalancing strategy across different time periods and thresholds
    """
    
    # Define time periods (ending at 2025-07-07)
    end_date = pd.Timestamp('2025-07-07')
    periods = {
        '1 Year': end_date - pd.DateOffset(years=1),
        '2 Years': end_date - pd.DateOffset(years=2),
        '3 Years': end_date - pd.DateOffset(years=3),
        '4 Years': end_date - pd.DateOffset(years=4),
        '5 Years': end_date - pd.DateOffset(years=5)
    }
    
    # Define different rebalancing strategies
    strategies = {
        'Weekly': {'frequency': 'weekly', 'threshold': 0.0},
        'Monthly': {'frequency': 'monthly', 'threshold': 0.0},
        'Threshold 1%': {'frequency': 'threshold', 'threshold': 0.01},
        'Threshold 2%': {'frequency': 'threshold', 'threshold': 0.02},
        'Threshold 5%': {'frequency': 'threshold', 'threshold': 0.05},
        'Threshold 10%': {'frequency': 'threshold', 'threshold': 0.10},
        'Buy & Hold': {'frequency': 'monthly', 'threshold': 0.0}  # Monthly to minimize rebalancing
    }
    
    results = []
    
    for period_name, start_date in periods.items():
        print(f"\nAnalyzing {period_name} period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
        
        # Filter data for this period
        period_df = df[df['date'] >= start_date].copy()
        
        if len(period_df) < 30:  # Skip if too few data points
            continue
        
        for strategy_name, strategy_config in strategies.items():
            try:
                if strategy_name == 'Buy & Hold':
                    # Special case for buy & hold - no rebalancing
                    result = calculate_buy_and_hold_performance(period_df)
                else:
                    result = calculate_rebalancing_performance(
                        period_df, 
                        btc_weight=0.5, 
                        paxg_weight=0.5,
                        rebalance_frequency=strategy_config['frequency'],
                        threshold=strategy_config['threshold']
                    )
                
                results.append({
                    'Period': period_name,
                    'Strategy': strategy_name,
                    'Start Date': start_date.strftime('%Y-%m-%d'),
                    'End Date': end_date.strftime('%Y-%m-%d'),
                    'Days': len(period_df),
                    'Total Return (%)': result['total_return'],
                    'Annualized Return (%)': result['annualized_return'],
                    'Volatility (%)': result['volatility'],
                    'Max Drawdown (%)': result['max_drawdown'],
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Rebalance Count': result['rebalance_count'],
                    'Final Value ($)': result['final_value']
                })
                
            except Exception as e:
                print(f"Error calculating {strategy_name} for {period_name}: {e}")
                continue
    
    return pd.DataFrame(results)

def calculate_buy_and_hold_performance(df):
    """Calculate buy and hold performance for 50-50 allocation"""
    initial_value = 10000
    btc_weight = 0.5
    paxg_weight = 0.5
    
    # Initial allocation
    btc_initial = initial_value * btc_weight
    paxg_initial = initial_value * paxg_weight
    
    # Final values
    btc_final = btc_initial * (df['close_btc'].iloc[-1] / df['close_btc'].iloc[0])
    paxg_final = paxg_initial * (df['close_paxg'].iloc[-1] / df['close_paxg'].iloc[0])
    
    final_value = btc_final + paxg_final
    
    # Calculate metrics
    total_return = (final_value / initial_value - 1) * 100
    
    # Calculate daily returns
    btc_returns = df['close_btc'].pct_change()
    paxg_returns = df['close_paxg'].pct_change()
    
    # Calculate portfolio returns (assuming no rebalancing)
    portfolio_values = []
    btc_value = btc_initial
    paxg_value = paxg_initial
    
    for i, row in df.iterrows():
        if i == 0:
            portfolio_values.append(initial_value)
            continue
        
        btc_return = btc_returns.iloc[i]
        paxg_return = paxg_returns.iloc[i]
        
        if not pd.isna(btc_return):
            btc_value *= (1 + btc_return)
        if not pd.isna(paxg_return):
            paxg_value *= (1 + paxg_return)
        
        portfolio_values.append(btc_value + paxg_value)
    
    portfolio_returns = pd.Series(portfolio_values).pct_change()
    volatility = portfolio_returns.std() * np.sqrt(252) * 100
    
    # Maximum drawdown
    running_max = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio
    risk_free_rate = 0.02
    excess_return = (portfolio_returns.mean() * 252) - risk_free_rate
    sharpe_ratio = excess_return / (portfolio_returns.std() * np.sqrt(252))
    
    # Annualized return
    days = len(df)
    years = days / 365.25
    annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'rebalance_count': 0,
        'final_value': final_value
    }

# Run the comprehensive analysis
print("Running comprehensive analysis for different periods and thresholds...")
results_df = analyze_multiple_periods_and_thresholds(merged_df)
print(f"\nGenerated {len(results_df)} result combinations")

# Display results
print("\nResults summary:")
print(results_df.head(10))

# %%


# %%
# Fix the buy and hold calculation and rerun analysis
def calculate_buy_and_hold_performance_fixed(df):
    """Calculate buy and hold performance for 50-50 allocation - fixed version"""
    initial_value = 10000
    btc_weight = 0.5
    paxg_weight = 0.5
    
    # Check if we have enough data
    if len(df) < 2:
        return None
    
    # Initial allocation
    btc_initial = initial_value * btc_weight
    paxg_initial = initial_value * paxg_weight
    
    # Final values
    btc_final = btc_initial * (df['close_btc'].iloc[-1] / df['close_btc'].iloc[0])
    paxg_final = paxg_initial * (df['close_paxg'].iloc[-1] / df['close_paxg'].iloc[0])
    
    final_value = btc_final + paxg_final
    
    # Calculate metrics
    total_return = (final_value / initial_value - 1) * 100
    
    # Calculate daily portfolio values without rebalancing
    portfolio_values = []
    btc_values = []
    paxg_values = []
    
    for i, row in df.iterrows():
        if i == 0:
            btc_values.append(btc_initial)
            paxg_values.append(paxg_initial)
            portfolio_values.append(initial_value)
        else:
            # Calculate values based on price changes from initial
            btc_val = btc_initial * (row['close_btc'] / df['close_btc'].iloc[0])
            paxg_val = paxg_initial * (row['close_paxg'] / df['close_paxg'].iloc[0])
            
            btc_values.append(btc_val)
            paxg_values.append(paxg_val)
            portfolio_values.append(btc_val + paxg_val)
    
    portfolio_values = pd.Series(portfolio_values)
    portfolio_returns = portfolio_values.pct_change().dropna()
    
    volatility = portfolio_returns.std() * np.sqrt(252) * 100
    
    # Maximum drawdown
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio
    risk_free_rate = 0.02
    mean_return = portfolio_returns.mean() * 252
    excess_return = mean_return - risk_free_rate
    sharpe_ratio = excess_return / (portfolio_returns.std() * np.sqrt(252))
    
    # Annualized return
    days = len(df)
    years = days / 365.25
    annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'rebalance_count': 0,
        'final_value': final_value
    }

# Rerun analysis with fixed buy and hold
def analyze_multiple_periods_and_thresholds_fixed(df):
    """
    Analyze 50-50 rebalancing strategy across different time periods and thresholds - fixed version
    """
    
    # Define time periods (ending at 2025-07-07)
    end_date = pd.Timestamp('2025-07-07')
    periods = {
        '1 Year': end_date - pd.DateOffset(years=1),
        '2 Years': end_date - pd.DateOffset(years=2),
        '3 Years': end_date - pd.DateOffset(years=3),
        '4 Years': end_date - pd.DateOffset(years=4),
        '5 Years': end_date - pd.DateOffset(years=5)
    }
    
    # Define different rebalancing strategies
    strategies = {
        'Weekly': {'frequency': 'weekly', 'threshold': 0.0},
        'Monthly': {'frequency': 'monthly', 'threshold': 0.0},
        'Threshold 1%': {'frequency': 'threshold', 'threshold': 0.01},
        'Threshold 2%': {'frequency': 'threshold', 'threshold': 0.02},
        'Threshold 5%': {'frequency': 'threshold', 'threshold': 0.05},
        'Threshold 10%': {'frequency': 'threshold', 'threshold': 0.10},
        'Buy & Hold': {'frequency': 'none', 'threshold': 0.0}
    }
    
    results = []
    
    for period_name, start_date in periods.items():
        print(f"Analyzing {period_name} period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
        
        # Filter data for this period
        period_df = df[df['date'] >= start_date].copy().reset_index(drop=True)
        
        if len(period_df) < 30:  # Skip if too few data points
            continue
        
        for strategy_name, strategy_config in strategies.items():
            try:
                if strategy_name == 'Buy & Hold':
                    # Special case for buy & hold - no rebalancing
                    result = calculate_buy_and_hold_performance_fixed(period_df)
                else:
                    result = calculate_rebalancing_performance(
                        period_df, 
                        btc_weight=0.5, 
                        paxg_weight=0.5,
                        rebalance_frequency=strategy_config['frequency'],
                        threshold=strategy_config['threshold']
                    )
                
                if result is not None:
                    results.append({
                        'Period': period_name,
                        'Strategy': strategy_name,
                        'Start Date': start_date.strftime('%Y-%m-%d'),
                        'End Date': end_date.strftime('%Y-%m-%d'),
                        'Days': len(period_df),
                        'Total Return (%)': result['total_return'],
                        'Annualized Return (%)': result['annualized_return'],
                        'Volatility (%)': result['volatility'],
                        'Max Drawdown (%)': result['max_drawdown'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Rebalance Count': result['rebalance_count'],
                        'Final Value ($)': result['final_value']
                    })
                
            except Exception as e:
                print(f"Error calculating {strategy_name} for {period_name}: {e}")
                continue
    
    return pd.DataFrame(results)

# Run the fixed comprehensive analysis
print("Running fixed comprehensive analysis...")
results_df = analyze_multiple_periods_and_thresholds_fixed(merged_df)
print(f"\nGenerated {len(results_df)} result combinations")

# Sort by period and strategy for better readability
results_df = results_df.sort_values(['Period', 'Strategy'])
print("\nComplete results:")
print(results_df.to_string(index=False))

# %%
# Fix the buy and hold calculation and rerun analysis
def calculate_buy_and_hold_performance_fixed(df):
    """Calculate buy and hold performance for 50-50 allocation - fixed version"""
    initial_value = 10000
    btc_weight = 0.5
    paxg_weight = 0.5
    
    # Check if we have enough data
    if len(df) < 2:
        return None
    
    # Initial allocation
    btc_initial = initial_value * btc_weight
    paxg_initial = initial_value * paxg_weight
    
    # Final values
    btc_final = btc_initial * (df['close_btc'].iloc[-1] / df['close_btc'].iloc[0])
    paxg_final = paxg_initial * (df['close_paxg'].iloc[-1] / df['close_paxg'].iloc[0])
    
    final_value = btc_final + paxg_final
    
    # Calculate metrics
    total_return = (final_value / initial_value - 1) * 100
    
    # Calculate daily portfolio values without rebalancing
    portfolio_values = []
    btc_values = []
    paxg_values = []
    
    for i, row in df.iterrows():
        if i == 0:
            btc_values.append(btc_initial)
            paxg_values.append(paxg_initial)
            portfolio_values.append(initial_value)
        else:
            # Calculate values based on price changes from initial
            btc_val = btc_initial * (row['close_btc'] / df['close_btc'].iloc[0])
            paxg_val = paxg_initial * (row['close_paxg'] / df['close_paxg'].iloc[0])
            
            btc_values.append(btc_val)
            paxg_values.append(paxg_val)
            portfolio_values.append(btc_val + paxg_val)
    
    portfolio_values = pd.Series(portfolio_values)
    portfolio_returns = portfolio_values.pct_change().dropna()
    
    volatility = portfolio_returns.std() * np.sqrt(252) * 100
    
    # Maximum drawdown
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio
    risk_free_rate = 0.02
    mean_return = portfolio_returns.mean() * 252
    excess_return = mean_return - risk_free_rate
    sharpe_ratio = excess_return / (portfolio_returns.std() * np.sqrt(252))
    
    # Annualized return
    days = len(df)
    years = days / 365.25
    annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'rebalance_count': 0,
        'final_value': final_value
    }

# Rerun analysis with fixed buy and hold
def analyze_multiple_periods_and_thresholds_fixed(df):
    """
    Analyze 50-50 rebalancing strategy across different time periods and thresholds - fixed version
    """
    
    # Define time periods (ending at 2025-07-07)
    end_date = pd.Timestamp('2025-07-07')
    periods = {
        '1 Year': end_date - pd.DateOffset(years=1),
        '2 Years': end_date - pd.DateOffset(years=2),
        '3 Years': end_date - pd.DateOffset(years=3),
        '4 Years': end_date - pd.DateOffset(years=4),
        '5 Years': end_date - pd.DateOffset(years=5)
    }
    
    # Define different rebalancing strategies
    strategies = {
        'Weekly': {'frequency': 'weekly', 'threshold': 0.0},
        'Monthly': {'frequency': 'monthly', 'threshold': 0.0},
        'Threshold 1%': {'frequency': 'threshold', 'threshold': 0.01},
        'Threshold 2%': {'frequency': 'threshold', 'threshold': 0.02},
        'Threshold 5%': {'frequency': 'threshold', 'threshold': 0.05},
        'Threshold 10%': {'frequency': 'threshold', 'threshold': 0.10},
        'Buy & Hold': {'frequency': 'none', 'threshold': 0.0}
    }
    
    results = []
    
    for period_name, start_date in periods.items():
        print(f"Analyzing {period_name} period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
        
        # Filter data for this period
        period_df = df[df['date'] >= start_date].copy().reset_index(drop=True)
        
        if len(period_df) < 30:  # Skip if too few data points
            continue
        
        for strategy_name, strategy_config in strategies.items():
            try:
                if strategy_name == 'Buy & Hold':
                    # Special case for buy & hold - no rebalancing
                    result = calculate_buy_and_hold_performance_fixed(period_df)
                else:
                    result = calculate_rebalancing_performance(
                        period_df, 
                        btc_weight=0.5, 
                        paxg_weight=0.5,
                        rebalance_frequency=strategy_config['frequency'],
                        threshold=strategy_config['threshold']
                    )
                
                if result is not None:
                    results.append({
                        'Period': period_name,
                        'Strategy': strategy_name,
                        'Start Date': start_date.strftime('%Y-%m-%d'),
                        'End Date': end_date.strftime('%Y-%m-%d'),
                        'Days': len(period_df),
                        'Total Return (%)': result['total_return'],
                        'Annualized Return (%)': result['annualized_return'],
                        'Volatility (%)': result['volatility'],
                        'Max Drawdown (%)': result['max_drawdown'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Rebalance Count': result['rebalance_count'],
                        'Final Value ($)': result['final_value']
                    })
                
            except Exception as e:
                print(f"Error calculating {strategy_name} for {period_name}: {e}")
                continue
    
    return pd.DataFrame(results)

# Run the fixed comprehensive analysis
print("Running fixed comprehensive analysis...")
results_df = analyze_multiple_periods_and_thresholds_fixed(merged_df)
print(f"\nGenerated {len(results_df)} result combinations")

# Sort by period and strategy for better readability
results_df = results_df.sort_values(['Period', 'Strategy'])
print("\nComplete results:")
print(results_df.to_string(index=False))

# %%
# Create a summary table focusing on the key metrics for 50-50 rebalancing strategies
summary_table = results_df.pivot_table(
    index='Period', 
    columns='Strategy', 
    values=['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Rebalance Count'],
    aggfunc='first'
)

# Create separate tables for each metric
metrics = ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Rebalance Count']

print("=== COMPREHENSIVE 50-50 REBALANCING STRATEGY ANALYSIS ===")
print("Period: 2020-2025 (5 years of historical data)")
print("\n")

# Create individual metric tables
for metric in metrics:
    print(f"=== {metric} ===")
    metric_table = summary_table[metric].round(2)
    print(metric_table.to_string())
    print("\n")

# Calculate best performing strategies by period
print("=== BEST PERFORMING STRATEGIES BY PERIOD ===")
best_strategies = results_df.groupby('Period').apply(
    lambda x: x.loc[x['Sharpe Ratio'].idxmax()]
)[['Period', 'Strategy', 'Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Rebalance Count']]

print(best_strategies.to_string(index=False))
print("\n")

# Save the complete results to CSV
results_df.to_csv('btc_paxg_rebalancing_analysis_2020_2025.csv', index=False)
print("Complete results saved to 'btc_paxg_rebalancing_analysis_2020_2025.csv'")

# Calculate average performance across all periods
print("=== AVERAGE PERFORMANCE ACROSS ALL PERIODS ===")
avg_performance = results_df.groupby('Strategy').agg({
    'Total Return (%)': 'mean',
    'Annualized Return (%)': 'mean',
    'Volatility (%)': 'mean',
    'Max Drawdown (%)': 'mean',
    'Sharpe Ratio': 'mean',
    'Rebalance Count': 'mean'
}).round(2)

print(avg_performance.to_string())
print("\n")

# Rank strategies by average Sharpe ratio
print("=== STRATEGIES RANKED BY AVERAGE SHARPE RATIO ===")
sharpe_ranking = avg_performance.sort_values('Sharpe Ratio', ascending=False)
print(sharpe_ranking.to_string())
print("\n")

# %%
# Create detailed analysis by time period to show performance trends
print("=== DETAILED PERFORMANCE ANALYSIS BY TIME PERIOD ===\n")

# Analysis by period showing key insights
for period in ['1 Year', '2 Years', '3 Years', '4 Years', '5 Years']:
    period_data = results_df[results_df['Period'] == period].copy()
    
    print(f"### {period} Analysis (as of July 2025)")
    print(f"Data Period: {period_data['Start Date'].iloc[0]} to {period_data['End Date'].iloc[0]}")
    print(f"Trading Days: {period_data['Days'].iloc[0]}")
    
    # Find best strategy by Sharpe ratio
    best_strategy = period_data.loc[period_data['Sharpe Ratio'].idxmax()]
    
    print(f"\n**Best Strategy: {best_strategy['Strategy']}**")
    print(f"- Total Return: {best_strategy['Total Return (%)']:.2f}%")
    print(f"- Annualized Return: {best_strategy['Annualized Return (%)']:.2f}%")
    print(f"- Volatility: {best_strategy['Volatility (%)']:.2f}%")
    print(f"- Max Drawdown: {best_strategy['Max Drawdown (%)']:.2f}%")
    print(f"- Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
    print(f"- Rebalances: {best_strategy['Rebalance Count']}")
    
    # Compare with buy & hold
    buy_hold = period_data[period_data['Strategy'] == 'Buy & Hold'].iloc[0]
    outperformance = best_strategy['Total Return (%)'] - buy_hold['Total Return (%)']
    
    print(f"\n**Comparison with Buy & Hold:**")
    print(f"- Buy & Hold Return: {buy_hold['Total Return (%)']:.2f}%")
    print(f"- Outperformance: {outperformance:.2f} percentage points")
    print(f"- Volatility Reduction: {buy_hold['Volatility (%)'] - best_strategy['Volatility (%)']:.2f} percentage points")
    print(f"- Drawdown Improvement: {buy_hold['Max Drawdown (%)'] - best_strategy['Max Drawdown (%)']:.2f} percentage points")
    
    # Show all strategies for this period
    print(f"\n**All Strategies Performance:**")
    strategy_summary = period_data[['Strategy', 'Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Rebalance Count']].sort_values('Sharpe Ratio', ascending=False)
    print(strategy_summary.round(2).to_string(index=False))
    
    print("\n" + "="*80 + "\n")

# Create threshold analysis
print("=== THRESHOLD STRATEGY ANALYSIS ===\n")

threshold_strategies = ['Threshold 1%', 'Threshold 2%', 'Threshold 5%', 'Threshold 10%']
threshold_analysis = results_df[results_df['Strategy'].isin(threshold_strategies)].copy()

print("**Threshold Strategy Performance by Period:**")
threshold_summary = threshold_analysis.pivot_table(
    index='Period',
    columns='Strategy',
    values=['Total Return (%)', 'Sharpe Ratio', 'Rebalance Count'],
    aggfunc='first'
).round(2)

print("\nTotal Returns by Threshold:")
print(threshold_summary['Total Return (%)'].to_string())

print("\nSharpe Ratios by Threshold:")
print(threshold_summary['Sharpe Ratio'].to_string())

print("\nRebalance Counts by Threshold:")
print(threshold_summary['Rebalance Count'].to_string())

print("\n**Key Insights:**")
print("- Lower thresholds (1%, 2%) generate more rebalances but may provide better risk-adjusted returns")
print("- Higher thresholds (5%, 10%) reduce transaction costs but may miss optimization opportunities")
print("- Optimal threshold varies by market conditions and time period")
print("- Threshold strategies generally outperform buy & hold in most periods")

# %%
!pip install plotly
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv('btc_paxg_rebalancing_analysis_2020_2025.csv')

# Define color palette (distinct, alternating)
colors = ['#1FB8CD', '#FFC185', '#ECEBD5', '#5D878F', '#D2BA4C', '#B4413C', '#964325']

# Get unique strategies and periods
strategies = df['Strategy'].unique()
periods = df['Period'].unique()

# Assign a color to each strategy
color_map = {strategy: colors[i % len(colors)] for i, strategy in enumerate(strategies)}

# Create subplot figure
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        'Total Return by Period',
        'Sharpe Ratio by Period',
        'Rebalance Count by Period'
    )
)

# 1. Line chart: Total Return (%)
for i, strategy in enumerate(strategies):
    data = df[df['Strategy'] == strategy]
    fig.add_trace(
        go.Scatter(
            x=data['Period'],
            y=data['Total Return (%)'],
            mode='lines+markers',
            name=strategy,
            marker=dict(symbol=i, color=color_map[strategy], size=10),
            line=dict(color=color_map[strategy], dash=['solid','dash','dot','dashdot','longdash','longdashdot','solid'][i%7]),
            showlegend=True,
            cliponaxis=False,
            hovertemplate='Strat: %{text}<br>Per: %{x}<br>TotRet: %{y:.2f}%',
            text=[strategy]*len(data)
        ),
        row=1, col=1
    )

# 2. Bar chart: Sharpe Ratio (no text labels)
for i, strategy in enumerate(strategies):
    data = df[df['Strategy'] == strategy]
    fig.add_trace(
        go.Bar(
            x=data['Period'],
            y=data['Sharpe Ratio'],
            name=strategy,
            marker_color=color_map[strategy],
            showlegend=False,
            cliponaxis=False,
            hovertemplate='Strat: %{text}<br>Per: %{x}<br>Sharpe: %{y:.2f}',
            text=[strategy]*len(data)
        ),
        row=2, col=1
    )

# 3. Bar chart: Rebalance Count (no text labels)
for i, strategy in enumerate(strategies):
    data = df[df['Strategy'] == strategy]
    fig.add_trace(
        go.Bar(
            x=data['Period'],
            y=data['Rebalance Count'],
            name=strategy,
            marker_color=color_map[strategy],
            showlegend=False,
            cliponaxis=False,
            hovertemplate='Strat: %{text}<br>Per: %{x}<br>Rebal: %{y}',
            text=[strategy]*len(data)
        ),
        row=3, col=1
    )

# Update axes and layout
fig.update_xaxes(title_text='Period', row=3, col=1)
fig.update_yaxes(title_text='TotRet (%)', row=1, col=1)
fig.update_yaxes(title_text='Sharpe', row=2, col=1)
fig.update_yaxes(title_text='Rebal Cnt', row=3, col=1)

# Center legend under title (all strategies shown)
fig.update_layout(
    title_text='BTC-PAXG 50-50 Rebal Perf',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5, title='', traceorder='normal'),
    hovermode='x unified'
)

fig.write_image('btc_paxg_rebal_performance.png')


