import pandas as pd

df2 = pd.read_csv("product_reviews.csv")  # replace with actual filename
print(df2.columns.tolist())
print(df2.head(2))
print(df2.shape)