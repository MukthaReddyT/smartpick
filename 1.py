import pandas as pd

df1 = pd.read_csv("mobile_reviews.csv")  # replace with actual filename
print(df1.columns.tolist())
print(df1.head(2))
print(df1.shape)