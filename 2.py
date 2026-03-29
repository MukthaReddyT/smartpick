import pandas as pd
df1 = pd.read_csv("dataset3_cams.csv")
print(df1["brand"].value_counts().head(20))
print("\n")
print(df1["source"].value_counts())