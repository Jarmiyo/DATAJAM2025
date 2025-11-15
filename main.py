import pandas as pd


df = pd.read_csv("Oil and Gas 1932-2014.csv")

# --- Create lag features ---
df["oil_prod_lag1"] = df["oil_prod32_14"].shift(1)
df["oil_prod_lag2"] = df["oil_prod32_14"].shift(2)

# Keep only the strongest 6 features
features = [
    "oil_prod_lag1",
    "oil_prod_lag2",
    "oil_price_2000",
    "oil_price_nom",
    "net_oil_exports",
    "population"
]

target = "oil_prod32_14"

df_model = df[features + [target, "year"]].dropna()
