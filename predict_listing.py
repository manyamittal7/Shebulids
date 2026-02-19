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

device = torch.device("cpu")  # change to "cuda" if GPU available
model.to(device)
model.eval()

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("train.tsv", sep="\t")
df = df.apply(lambda x: x.astype(str).str.strip())

# Select 5001–12000
df = df[(df["Record Number"].astype(int) >= 5001) &
        (df["Record Number"].astype(int) <= 12000)].copy()

print("Total records selected:", len(df))

# -------------------------
# Prediction Function
# -------------------------
def predict_sentence(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_tags = [model.config.id2label[p.item()] for p in predictions[0]]

    return tokens, predicted_tags

# -------------------------
# Extract Aspects (FIXED WORD JOINING)
# -------------------------
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    record = row["Record Number"]
    category = row["Category"]
    title = row["Title"]

    tokens, tags = predict_sentence(title)

    current_tag = None
    current_value = ""

    for token, tag in zip(tokens, tags):

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Handle subword tokens properly
        if token.startswith("##"):
            current_value += token[2:]
            continue

        if tag != "O":
            if current_tag == tag:
                current_value += " " + token
            else:
                if current_tag is not None:
                    results.append([
                        record,
                        category,
                        current_tag,
                        current_value.strip()
                    ])
                current_tag = tag
                current_value = token
        else:
            if current_tag is not None:
                results.append([
                    record,
                    category,
                    current_tag,
                    current_value.strip()
                ])
                current_tag = None
                current_value = ""

    # Save last pending tag
    if current_tag is not None:
        results.append([
            record,
            category,
            current_tag,
            current_value.strip()
        ])

# -------------------------
# Create Final Output
# -------------------------
submission = pd.DataFrame(
    results,
    columns=["Record Number", "Category", "Aspect Name", "Aspect Value"]
)

submission.to_csv("final_output_subset.csv", index=False, encoding="utf-8")

print("✅ DONE — final_output_subset.csv created successfully!")
