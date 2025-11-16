import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("Oil and Gas 1932-2014.csv")


df = df.dropna(subset=["oil_prod32_14"])


# Lag features
lags = [1, 2, 3, 5, 10]
for lag in lags:
    df[f"oil_prod_lag{lag}"] = df["oil_prod32_14"].shift(lag)

# Rolling mean features
df["roll_3"] = df["oil_prod32_14"].rolling(3).mean()
df["roll_5"] = df["oil_prod32_14"].rolling(5).mean()
df["roll_10"] = df["oil_prod32_14"].rolling(10).mean()

features = [
    "oil_prod_lag1",
    "oil_prod_lag2",
    "oil_prod_lag3",
    "oil_prod_lag5",
    "oil_prod_lag10",
    "roll_3",
    "roll_5",
    "roll_10",
    "net_oil_exports",
    "population",
    "year"
]

df["log_oil"] = np.log1p(df["oil_prod32_14"])
target = "log_oil"

df_model = df[["cty_name"] + features + [target]].dropna()



test_pool = df_model.copy()

# 20% test
test = test_pool.sample(frac=0.2, random_state=42)

# Remaining 80% is training
train = test_pool.drop(test.index)


print("train dataset:")
print(len(train))
print("test dataset:")
print(len(test))

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)

df_plot = test.copy()
df_plot["predicted"] = y_pred

# Convert log scale to REAL barrels
df_plot["actual_barrels"] = np.expm1(df_plot["log_oil"])
df_plot["pred_barrels"] = np.expm1(df_plot["predicted"])

# Group by year (average per country)
df_year = df_plot.groupby("year")[["actual_barrels", "pred_barrels"]].mean()

plt.figure(figsize=(12,6))
plt.plot(df_year.index, df_year["actual_barrels"], label="Actual (Avg)", linewidth=3)
plt.plot(df_year.index, df_year["pred_barrels"], label="Predicted (Avg)", linewidth=3)

plt.legend()
plt.title("Overall Model Performance (Average Across All Countries)")
plt.xlabel("Year")
plt.ylabel("Oil Production (Barrels)")    # updated axis label
plt.ticklabel_format(style="plain", axis="y")  # remove scientific notation formatting
plt.grid(True, alpha=0.3)

plt.show()

idx = 6

year_value = int(test.iloc[idx]["year"])

actual_log = y_test.iloc[idx]
pred_log = y_pred[idx]

actual_real = np.expm1(actual_log)  # back-transform log
pred_real = np.expm1(pred_log)

# Convert barrels → gallons (if dataset uses barrels)
actual_gallons = actual_real * 42
pred_gallons = pred_real * 42

print("=== Global Oil Production Prediction Details ===")
print(f"Year: {year_value}")

print("\n--- Log Scale ---")
print(f"Actual (log):     {actual_log}")
print(f"Predicted (log):  {pred_log}")

print("\n--- Real Scale (Barrels) ---")
print(f"Actual:           {actual_real:,.2f} barrels")
print(f"Predicted:        {pred_real:,.2f} barrels")

print("\n--- Real Scale (Gallons) ---")
print(f"Actual:           {actual_gallons:,.2f} gallons")
print(f"Predicted:        {pred_gallons:,.2f} gallons")

