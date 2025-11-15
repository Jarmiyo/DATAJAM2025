#Albania

import pandas as pd


df = pd.read_csv("Oil and Gas 1932-2014.csv")

# --- Create lag features ---
df["oil_prod_lag1"] = df["oil_prod32_14"].shift(1)
df["oil_prod_lag2"] = df["oil_prod32_14"].shift(2)
df = df.loc[df['cty_name'] == "Bangladesh"]

features = [
    "oil_prod_lag1",
    "oil_prod_lag2",
    "oil_price_2000",
    "oil_price_nom",
    "net_oil_exports",
    "population",
    "year"
]

target = "oil_prod32_14"

df_model = df[features + [target]].dropna()

train = df_model[df_model["year"] <= 2000]
test = df_model[df_model["year"] > 2000]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(test["year"], y_test, label="Actual", linewidth=3)
plt.plot(test["year"], y_pred, label="Predicted", linewidth=3)
plt.legend()
plt.title("Oil Production: Actual vs Predicted")
plt.xlabel("Year")
plt.ylabel("Oil Production")
plt.show()

import numpy as np

importances = model.feature_importances_
plt.figure(figsize=(8,4))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()
