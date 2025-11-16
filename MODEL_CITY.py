import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("Oil and Gas 1932-2014.csv")
print(len(df))

df = df.dropna(subset=["oil_prod32_14"])
print(len(df))

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

df_model = df[features + [target]].dropna()

print("full dataset:")
print(len(df_model))

test_pool = df_model[df_model["year"] > 2000]
test = test_pool.sample(frac=0.2, random_state=42)
train = df_model

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

df_year = df_plot.groupby("year")[["log_oil", "predicted"]].mean()

plt.figure(figsize=(12,6))
plt.plot(df_year.index, df_year["log_oil"], label="Actual (Avg)", linewidth=3)
plt.plot(df_year.index, df_year["predicted"], label="Predicted (Avg)", linewidth=3)
plt.legend()
plt.title("Overall Model Performance (Average Across All Countries)")
plt.xlabel("Year")
plt.ylabel("Oil Production")
plt.show()

print(df["cty_name"].nunique())

importances = model.feature_importances_
plt.figure(figsize=(8,4))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()
