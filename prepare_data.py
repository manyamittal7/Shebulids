import pandas as pd

print("Loading train.tsv...")

# TSV file load karo
df = pd.read_csv("train.tsv", sep="\t")

print("Fixing empty tags...")

# Blank tags ko O karo (NER rule)
df["Tag"] = df["Tag"].fillna("O")

# Agar blank strings ho toh unko bhi O karo
df["Tag"] = df["Tag"].replace("", "O")

# Sirf required columns
bio_df = df[["Record Number", "Token", "Tag"]]

# Save new file
bio_df.to_csv("train_bio.csv", index=False)

print("Done! train_bio.csv created successfully ✅")
