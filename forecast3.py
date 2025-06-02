# Required Libraries
import pandas as pd
import numpy as np
import matplotlib as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import streamlit as st

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("aapl.csv")  # Rename your file to avoid spaces
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    data = df['Close'].asfreq('D')
    return data.interpolate(method='time')

# --- Forecast Function ---
def forecast_prices(data, steps=30):
    train = data[:-steps]
    test = data[-steps:]

    # Use auto_arima to select best SARIMA model
    model = auto_arima(train, seasonal=True, m=7, trace=False,
                       error_action='ignore', suppress_warnings=True,
                       stepwise=True)

    # Fit the best SARIMA model
    sarimax = SARIMAX(train, 
                      order=model.order, 
                      seasonal_order=model.seasonal_order,
                      enforce_stationarity=False, 
                      enforce_invertibility=False)
    result = sarimax.fit()
    forecast = result.forecast(steps=steps)
    return train, test, forecast, result

# --- Streamlit UI ---
st.title("📈 Apple Stock Price Forecast")
data = load_data()

# Sidebar Input
st.sidebar.header("Forecast Settings")
steps = st.sidebar.slider("Forecast Days", 7, 60, 30)

# Forecast Computation
with st.spinner("Training SARIMA model..."):
    train, test, forecast, result = forecast_prices(data, steps)

# --- Plot ---
st.subheader("Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10, 5))
train.plot(ax=ax, label="Train")
test.plot(ax=ax, label="Actual")
forecast.plot(ax=ax, label="Forecast", style='--')
plt.legend()
st.pyplot(fig)

# --- RMSE Metric ---
rmse = np.sqrt(mean_squared_error(test, forecast))
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

# --- Display Best Model Orders ---
st.write(f"**Best SARIMA Order:** {result.model_orders['order']}")
st.write(f"**Best Seasonal Order:** {result.model_orders['seasonal_order']}")
