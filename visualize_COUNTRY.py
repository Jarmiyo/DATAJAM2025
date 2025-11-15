import pandas as pd
import matplotlib.pyplot as plt





def view_features(str):
    df = pd.read_csv("Oil and Gas 1932-2014.csv")

    # --- Create lag features ---
    df["oil_prod_lag1"] = df["oil_prod32_14"].shift(1)
    df["oil_prod_lag2"] = df["oil_prod32_14"].shift(2)

    df = df.loc[df['cty_name'] == str]

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

    df_model = df[features + [target]].dropna()

    #visualize year


    df_model["oil_prod32_14"].plot(kind='line', color='grey')
    df_model["oil_prod_lag1"].plot(kind='line', color='grey')
    df_model["oil_prod_lag2"].plot(kind='line', color='grey')
    df_model["oil_price_2000"].plot(kind='line', color='blue')
    df_model["net_oil_exports"].plot(kind='line', color='green')
    df_model["oil_price_nom"].plot(kind='line', color='yellow')
    df_model["population"].plot(kind='line', color='black')

    plt.legend()
    plt.title(str)
    plt.show()


view_features("Afghanistan")
view_features("Albania")
view_features("Algeria")
view_features("Angola")
view_features("Argentina")
view_features("Armenia")
view_features("Australia")
view_features("Austria")
view_features("Azerbaijan")
view_features("Bahrain")
view_features("Bahamas, The")
view_features("Bangladesh")
view_features("Barbados")
view_features("Belgium")
view_features("Belize")
view_features("Bhutan")
view_features("Benin")
view_features("Bolivia")
