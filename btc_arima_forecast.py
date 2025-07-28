import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt
import os

# Step 1: Download or Load Data
csv_file = 'btc_data.csv'

if os.path.exists(csv_file):
    print("✅ Loading from CSV...")
    btc = pd.read_csv(csv_file)

    # Remove repeated headers if present
    btc = btc[btc['Date'] != 'Date']

    # Clean up and ensure proper types
    btc['Close'] = pd.to_numeric(btc['Close'], errors='coerce')
    btc['Date'] = pd.to_datetime(btc['Date'], errors='coerce')
    btc = btc.dropna(subset=['Date', 'Close'])
    btc.set_index('Date', inplace=True)
else:
    print("📡 Downloading from yfinance...")
    raw = yf.download('BTC-USD', start='2020-01-01', end='2025-01-01')

    # Fix MultiIndex (if exists)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    btc = raw[['Close']].dropna().copy()
    btc = btc.reset_index()  # 'Date' becomes column
    btc['Date'] = pd.to_datetime(btc['Date'])

    # ✅ Save only Date and Close columns cleanly
    btc[['Date', 'Close']].to_csv(csv_file, index=False)

    btc.set_index('Date', inplace=True)
    print("✅ Saved to CSV.")




# Step 2: Plot the raw closing price
plt.figure(figsize=(12, 6))
plt.plot(btc['Close'], label='BTC-USD Closing Price')
plt.title('Bitcoin Closing Price (2020 - 2025)')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.legend()
plt.tight_layout()
plt.show()

# Step 3: Check Stationarity
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series):
    # Rolling statistics
    roll_mean = series.rolling(window=30).mean()
    roll_std = series.rolling(window=30).std()

    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original')
    plt.plot(roll_mean, label='Rolling mean')
    plt.plot(roll_std, label='Rolling Std Dev')
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ADF Test
    result = adfuller(series.dropna())
    print("ADF Statistics :", result[0])
    print("p-value :", result[1])
    for key, value in result[4].items():
        print(f"Critical Value ({key}) : {value}")
    if result[1] < 0.05:
        print("✅ Data is stationary (reject null hypothesis)")
    else:
        print("❌ Data is non-stationary (fail to reject null)")

# Run stationarity test
test_stationarity(btc['Close'])

# Step 4: Differencing to make data stationary
btc_diff = btc['Close'].diff().dropna()

# Re-test stationarity after differencing
print("\n=== After First Differencing ===")
test_stationarity(btc_diff)

#ARIMA Model Building
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore') #To supress convergence warnings

# Step 5: Plot ACF and PACF to choose p and q

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_acf(btc_diff, ax=plt.gca(), lags=40)
plt.title('Autocorrelation (ACF)')

plt.subplot(1,2,2)
plot_pacf(btc_diff, ax=plt.gca(), lags=40, method='ywm')
plt.title('Partial Autocorrelation (PACF)')

plt.tight_layout()
plt.show()

# Plot the differenced series
plt.figure(figsize=(12,6))
plt.plot(btc_diff)
plt.title('Differenced series (1st Order)')
plt.show()

from pmdarima import auto_arima

auto_model = auto_arima(btc['Close'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
print(auto_model.summary())

n_periods = 30
forecast, conf_int = auto_model.predict(n_periods=n_periods, return_conf_int=True)

# Plot
forecast_index = pd.date_range(btc.index[-1] + pd.Timedelta(days=1), periods=n_periods)

plt.figure(figsize=(12,6))
plt.plot(btc['Close'], label='Historical')
plt.plot(forecast_index, forecast, label='Forecast (ARIMA)', linestyle='--')
plt.legend()
plt.title('BTC Forecast')
plt.show()

