import matplotlib.dates
import yfinance as yf
#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data
yf.pdr_override()

# input
symbols = ['VLO', 'MGY', 'LNG','SPY']
start = '2022-01-01'
end = '2023-01-01'
# Read data
dataset = yf.download(symbols, start, end)['Adj Close']

# View Columns
dataset.head()
# Calculate Daily Returns
returns = dataset.pct_change()
returns = returns.dropna()
returns.head()
# Calculate mean returns
meanDailyReturns = returns.mean()
print(meanDailyReturns)

# Calculate std returns
stdDailyReturns = returns.std()
print(stdDailyReturns)

# Define weights for the portfolio
weights = np.array([0.25, 0.25, 0.25, 0.25])
# Calculate the covariance matrix on daily returns
cov_matrix = (returns.cov())*252
print(cov_matrix)
# Calculate expected portfolio performance
portReturn = np.sum(meanDailyReturns*weights)
# Print the portfolio return
print(portReturn)
# Create portfolio returns column
returns['Portfolio'] = returns.dot(weights)
returns.head()
# Calculate cumulative returns
daily_cum_ret = (1+returns).cumprod()
print(daily_cum_ret.tail())

returns['Portfolio'].hist()
plt.show()


# Plot the portfolio cumulative returns only
fig, ax = plt.subplots()
ax.plot(daily_cum_ret.index, daily_cum_ret.Portfolio,
        color='purple', label="portfolio")
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
plt.legend()
plt.show()


# Print the mean
print("mean : ", returns['Portfolio'].mean()*100)

# Print the standard deviation
print("Std. dev: ", returns['Portfolio'].std()*100)

# Print the skewness
print("skew: ", returns['Portfolio'].skew())

# Print the kurtosis
print("kurt: ", returns['Portfolio'].kurtosis())

# Calculate the standard deviation by taking the square root
port_standard_dev = np.sqrt(np.dot(weights.T, np.dot(weights, cov_matrix)))

# Print the results
print(str(np.round(port_standard_dev, 4) * 100) + '%')

# Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

# Print the result
print(str(np.round(port_variance, 4) * 100) + '%')


# Calculate total return and annualized return from price data
total_return = (returns['Portfolio'][-1] -
                returns['Portfolio'][0]) / returns['Portfolio'][0]

# Annualize the total return over 6 year
annualized_return = ((total_return + 1)**(1/6))-1
# Calculate annualized volatility from the standard deviation
vol_port = returns['Portfolio'].std() * np.sqrt(250)
# Calculate the Sharpe ratio
rf = 0.01
sharpe_ratio = ((annualized_return - rf) / vol_port)
print(sharpe_ratio)
# Create a downside return column with the negative returns only
target = 0
downside_returns = returns.loc[returns['Portfolio'] < target]

# Calculate expected return and std dev of downside
expected_return = returns['Portfolio'].mean()
down_stdev = downside_returns.std()

# Calculate the sortino ratio
rf = 0.01
sortino_ratio = (expected_return - rf)/down_stdev

# Print the results
print("Expected return: ", expected_return*100)
print('-' * 50)
print("Downside risk:")
print(down_stdev*100)
print('-' * 50)
print("Sortino ratio:")
print(sortino_ratio)
# Calculate the max value
roll_max = returns['Portfolio'].rolling(
    center=False, min_periods=1, window=252).max()

# Calculate the daily draw-down relative to the max
daily_draw_down = returns['Portfolio']/roll_max - 1.0

# Calculate the minimum (negative) daily draw-down
max_daily_draw_down = daily_draw_down.rolling(
    center=False, min_periods=1, window=252).min()

# Plot the results
plt.figure(figsize=(15, 15))
plt.plot(returns.index, daily_draw_down, label='Daily drawdown')
plt.plot(returns.index, max_daily_draw_down,
         label='Maximum daily drawdown in time-window')
plt.legend()
plt.show()
