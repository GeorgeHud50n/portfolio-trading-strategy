import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import calendar
import matplotlib.ticker as ticker

# Define the list of cryptocurrencies to include in the portfolio
crypto_list = ['BTC-GBP', 'ETH-GBP', 'SOL-GBP','DOT-GBP','LTC-GBP','ATOM-GBP','ADA-GBP', 'XRP-GBP', 'MATIC-GBP', 'HBAR-GBP', 'XLM-GBP', 'ICP-GBP']

# Fetch historical price data
start_date = '2021-12-30'
end_date = '2023-05-13'

prices = yf.download(crypto_list, start=start_date, end=end_date)['Adj Close']

# Calculate the negative Sharpe Ratio
def sharpe_ratio_neg(weights, mean_daily_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_daily_returns) * 365
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def find_optimal_portfolio(prices_month):
    daily_returns_month = prices_month.pct_change().dropna()
    mean_daily_returns_month = daily_returns_month.mean()

    # Select top 5 assets based on mean daily return
    top_assets = mean_daily_returns_month.nlargest(5).index

    daily_returns_month = daily_returns_month[top_assets]
    mean_daily_returns_month = mean_daily_returns_month[top_assets] # ensure only top 5 are considered
    cov_matrix_month = daily_returns_month.cov()

    result = minimize(sharpe_ratio_neg, [1/len(daily_returns_month.columns)]*len(daily_returns_month.columns),
                      args=(mean_daily_returns_month, cov_matrix_month, 0.02),
                      bounds=[(0, 1)]*len(daily_returns_month.columns),
                      constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}])
    weights = pd.Series(0, index=prices_month.columns)
    weights[top_assets] = result.x

    return weights


# Initialize portfolio value and weights
portfolio_value = 5000
weights = np.array([1/len(crypto_list)]*len(crypto_list))

# Set the date for the first day of the next month
current_date = datetime.strptime(start_date, '%Y-%m-%d')
next_month = current_date.replace(day=calendar.monthrange(current_date.year, current_date.month)[1]) + timedelta(days=1)

# Initialize lists to store portfolio values and weights over time
portfolio_values = [portfolio_value]
weights_over_time = [weights]

# Initialize lists to store returns and volatility over time
returns_over_time = []
volatility_over_time = []

# Calculate portfolio value for the first month
prices_month = prices[current_date.strftime('%Y-%m-%d'):next_month.strftime('%Y-%m-%d')]
returns_month = prices_month.pct_change().dropna()
portfolio_returns_month = returns_month.dot(weights)
portfolio_value *= (1 + portfolio_returns_month).product()

while next_month < datetime.strptime(end_date, '%Y-%m-%d'):
    # Calculate and store the returns and volatility for the month
    monthly_return = portfolio_value / portfolio_values[-1] - 1
    monthly_volatility = returns_month.dot(weights).std() * np.sqrt(30)  
    returns_over_time.append(monthly_return)
    volatility_over_time.append(monthly_volatility)

    # Print the opening and closing portfolio values, returns, and volatility for the month
    print(f"Opening portfolio value for {current_date.strftime('%Y-%m-%d')}: £{portfolio_values[-1]:,.2f}")
    print(f"Closing portfolio value for {next_month.strftime('%Y-%m-%d')}: £{portfolio_value:,.2f}")
    print(f"Monthly return: {monthly_return:.2%}")
    print(f"Monthly volatility: {monthly_volatility:.2%}")

    # Append portfolio value to the list
    portfolio_values.append(portfolio_value)

    # Update current date
    current_date = next_month
    next_month += pd.DateOffset(months=1)

    # Find optimal portfolio for the next month
    prices_next_month = prices[current_date.strftime('%Y-%m-%d'):next_month.strftime('%Y-%m-%d')]
    weights = find_optimal_portfolio(prices_next_month)
    weights_over_time.append(weights)

    # Calculate portfolio value for the next month
    prices_month = prices[current_date.strftime('%Y-%m-%d'):next_month.strftime('%Y-%m-%d')]
    returns_month = prices_month.pct_change().dropna()
    portfolio_returns_month = returns_month.dot(weights)
    portfolio_value *= (1 + portfolio_returns_month).product()


# Fetch historical price data for BTC
btc_prices = yf.download('BTC-GBP', start=start_date, end=end_date)['Adj Close']

# Calculate the buy and hold portfolio value for BTC
btc_investment = 5000  # Initial investment in BTC
btc_returns = btc_prices.pct_change().fillna(0)
btc_portfolio_values = (1 + btc_returns).cumprod() * btc_investment

# Calculate the Sharpe ratio over time
risk_free_rate = 0.02
sharpe_ratio_over_time = [(ret - risk_free_rate) / vol if vol else 0 for ret, vol in zip(returns_over_time, volatility_over_time)]

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Shift portfolio values one month backward
portfolio_values_dates = pd.date_range(start=start_date, periods=len(portfolio_values), freq='M')
portfolio_values_series = pd.Series(portfolio_values, index=portfolio_values_dates).shift(-1)
portfolio_values_series = portfolio_values_series[:-1]

# Plot portfolio value over time
axs[0].plot(portfolio_values_series.index, portfolio_values_series, label='Portfolio', linewidth=1.5, color='blue')
axs[0].plot(btc_prices.index, btc_portfolio_values, label='Buy and Hold (BTC)', linewidth=1.5, color='orange', linestyle='--') 
axs[0].set_ylabel('Value (£)')
axs[0].set_title('Portfolio Value Over Time')
axs[0].legend()
axs[0].grid(True)

# Plot monthly returns and volatility
returns_series = pd.Series(returns_over_time, index=pd.date_range(start=start_date, periods=len(returns_over_time), freq='M'))
axs[1].bar(returns_series.index, returns_series, label='Monthly Returns', color='black', alpha=0.75, width=20)
volatility_series = pd.Series(volatility_over_time, index=pd.date_range(start=start_date, periods=len(volatility_over_time), freq='M'))
axs[1].plot(volatility_series.index, volatility_series, label='Monthly Volatility', color='red', linestyle='--', marker='s', markersize=3)
axs[1].set_ylabel('Returns / Volatility')
axs[1].set_title('Monthly Returns / Volatility / Sharpe')
axs[1].legend(loc='upper left')
axs[1].grid(True)

# Plot Sharpe ratio on a secondary y-axis
ax2 = axs[1].twinx()
sharpe_ratio_series = pd.Series(sharpe_ratio_over_time, index=pd.date_range(start=start_date, periods=len(sharpe_ratio_over_time), freq='M'))
ax2.plot(sharpe_ratio_series.index, sharpe_ratio_series, label='Monthly Sharpe Ratio', color='orange', linestyle='-', marker='o', markersize=3)
ax2.set_ylabel('Sharpe Ratio')
ax2.legend(loc='upper right')

# Format y-axis tick labels with thousands separators and set decimal places
axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2%}'))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Use two decimal places for Sharpe ratio

# Set x-axis label for the last subplot
axs[1].set_xlabel('Time')

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0.5)

plt.show()

