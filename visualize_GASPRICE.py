
import pandas as pd
import matplotlib.pyplot as plt

DATASETPATH = "Oil and Gas 1932-2014.csv"


# Read the CSV file into a pandas DataFrame
df = pd.read_csv(DATASETPATH)

selectedRows = df.loc[df['cty_name'] == "Australia"]
print(selectedRows)

selectedRows.plot(kind='line', x='year', color='red', y='oil_prod32_14', linestyle='-')
plt.title('gas price VS year')
plt.xlabel('year')
plt.ylabel('prices')
plt.legend()
plt.show()

