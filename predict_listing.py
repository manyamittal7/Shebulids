import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

# -------------------------
# Load Model
# -------------------------
model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

device = torch.device("cpu")
model.to(device)
model.eval()

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("train.tsv", sep="\t")

results = []

current_record = None
current_category = None
current_tag = None
current_value = ""

# -------------------------
# Process Tokens
# -------------------------
for _, row in tqdm(df.iterrows(), total=len(df)):

    record = row["Record Number"]
    category = row["Category"]
    token = str(row["Token"])
    tag = str(row["Tag"])

    # New record detected
    if current_record is not None and record != current_record:
        if current_tag is not None:
            results.append([
                current_record,
                current_category,
                current_tag,
                current_value.strip()
            ])
        current_tag = None
        current_value = ""

    # If token has a valid tag
    if tag != "O" and tag != "nan":
        if current_tag == tag:
            current_value += " " + token
        else:
            if current_tag is not None:
                results.append([
                    current_record,
                    current_category,
                    current_tag,
                    current_value.strip()
                ])
            current_tag = tag
            current_value = token
    else:
        if current_tag is not None:
            results.append([
                current_record,
                current_category,
                current_tag,
                current_value.strip()
            ])
            current_tag = None
            current_value = ""

    current_record = record
    current_category = category

# Save last value
if current_tag is not None:
    results.append([
        current_record,
        current_category,
        current_tag,
        current_value.strip()
    ])

# -------------------------
# Create DataFrame
# -------------------------
submission = pd.DataFrame(
    results,
    columns=[
        "Record Number",
        "Category",
        "Aspect Name",
        "Aspect Value"
    ]
)

# -------------------------
# Console Preview (Aligned)
# -------------------------
print("\n================ FINAL OUTPUT PREVIEW ================\n")
print(submission.head(20).to_string(index=False))
print("\n======================================================\n")

# -------------------------
# Save File (CSV Recommended)
# -------------------------
submission.to_csv("final_output.csv", index=False)

print("✅ DONE — final_output.csv created successfully!")
