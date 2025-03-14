import pandas as pd

df = pd.read_csv("raj-edges.csv")

print(df)

new_df = pd.DataFrame()
new_df["layer"] = df["type"]
new_df["sublayer"] = 0
new_df["vertex1"] = df["vertex1"]
new_df["vertex2"] = df["vertex2"]
new_df["probability"] = df["weight"]
new_df["intensity"] = 1.0

print(new_df)

new_df.to_csv("raj-full-edges.csv", index=None)
