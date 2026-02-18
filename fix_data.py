import pandas as pd

print("Loading data...")

df = pd.read_csv("train.tsv", sep="\t")

print("Filling empty tags with O...")

# Empty tags ko O se replace karo
df["Tag"] = df["Tag"].fillna("O")

# Save clean file
df.to_csv("train_clean.tsv", sep="\t", index=False)

print("Done ✅ train_clean.tsv created")
