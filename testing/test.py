import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet

# -------------------- Load and Clean Data --------------------
dataset = pd.read_csv(r"C:\Users\GANGOTRI\OneDrive\Desktop\Sales Performance Dashboard\car sales.csv")
dataset.columns = dataset.columns.str.strip()

dataset.rename(columns={'Price ($)': 'Price'}, inplace=True)

dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
dataset.dropna(subset=['Date', 'Price'], inplace=True)

# -------------------- Prepare Daily Sales Data --------------------
daily_sales = dataset.groupby('Date')['Price'].sum().reset_index()
daily_sales.columns = ['ds', 'y']
daily_sales = daily_sales.sort_values('ds')

# -------------------- Train NeuralProphet Model --------------------
model = NeuralProphet(
    n_lags=30,
    n_forecasts=30,
    yearly_seasonality=True,
    weekly_seasonality=True
)

metrics = model.fit(daily_sales, freq='D')

# -------------------- Forecast Next 30 Days --------------------
future = model.make_future_dataframe(daily_sales, periods=30)
forecast = model.predict(future)

# -------------------- Print Available Columns --------------------
print("\nAvailable Forecast Columns:\n")
print(forecast.columns)

# -------------------- Extract Forecast Output --------------------
forecast_output = forecast[['ds', 'yhat1']].tail(30)

# -------------------- Print Forecast Output --------------------
print("\n================= FORECAST OUTPUT (NEXT 30 DAYS) =================\n")
print(forecast_output)

# -------------------- Save Forecast Output CSV --------------------
forecast_output.to_csv(
    r"C:\Users\GANGOTRI\OneDrive\Desktop\Sales Performance Dashboard\forecastoutput.csv",
    index=False
)

print("\n✅ Forecast Output saved successfully as forecastoutput.csv")

# -------------------- Save Full Forecast CSV --------------------
forecast.to_csv(
    r"C:\Users\GANGOTRI\OneDrive\Desktop\Sales Performance Dashboard\full_forecast.csv",
    index=False
)

print("✅ Full Forecast saved successfully as full_forecast.csv")

# -------------------- Plot Forecast --------------------
fig = model.plot(forecast)
plt.show()
