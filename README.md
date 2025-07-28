# 📈 Bitcoin Price Forecasting with ARIMA

This project analyzes historical Bitcoin (BTC-USD) price data from 2020 to 2024 and applies ARIMA modeling for forecasting.

---

## 📂 Project Structure

- `btc_arima_forecast.py` – Main script for data loading, preprocessing, ARIMA modeling, and visualization.
- `btc_data.csv` – Cleaned historical data (Date + Close).
- `ACF_PACF.png` – Plot showing ACF and PACF for model selection.
- `README.md` – Project documentation.

---

## 🔍 Project Steps

### 1. Data Collection and Preprocessing
- Data source: Yahoo Finance (`BTC-USD`)
- Period: Jan 1, 2020 – Dec 31, 2024
- Missing values and duplicate headers handled
- Only `Date` and `Close` columns retained

### 2. Stationarity Testing
- ADF (Augmented Dickey-Fuller) test used
- Differencing applied to achieve stationarity
- Rolling mean & standard deviation visualized

### 3. ACF and PACF Analysis
- ACF and PACF plots generated
- Initial ARIMA orders estimated visually
- Auto ARIMA also used to determine optimal parameters

### 4. ARIMA Modeling
- Best model selected: `ARIMA(1, 1, 0)` based on AIC
- Model summary and diagnostics checked
- 30-day forecast generated

### 5. Forecast Visualization
- Forecast plotted with historical prices
- Interpreted model trend and forecast confidence

---

## 📊 Output Insights

- The model suggests a slow increasing trend in future prices.
- ACF/PACF plots helped eliminate noise and overfitting.
- Forecasts are visualized with confidence intervals.

---

## 🔧 Future Improvements

- Integrate seasonal components (SARIMA)
- Tune model with grid search or cross-validation
- Compare with LSTM or Prophet

---

## 📌 Dependencies

- `pandas`, `matplotlib`, `statsmodels`, `pmdarima`, `yfinance`

Install dependencies:
```bash
pip install -r requirements.txt

## 🧠 Author

**Saif Ullah**  
Data Science Graduate | Crypto Analyst  
[GitHub](https://github.com/saif696)  
[LinkedIn](https://www.linkedin.com/in/saif-ullah-831476135)
