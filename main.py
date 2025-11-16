import pandas as pd


df = pd.read_csv("Oil and Gas 1932-2014.csv")

df = df.dropna(subset=["oil_prod32_14"])
df = df[df["oil_prod32_14"] != 0]
