import pandas as pd

df = pd.read_csv("train.tsv", sep="\t")

print("Total rows:", len(df))
print("Unique Records:", df["Record Number"].nunique())
print(df.head())
