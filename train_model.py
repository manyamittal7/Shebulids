import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 1. LOAD CLEAN DATA
# ==============================
print("Loading train_clean.tsv...")

df_full = pd.read_csv("train_clean.tsv", sep="\t")

# Fix Token column
df_full["Token"] = df_full["Token"].fillna("").astype(str)

# Fix Tag column
df_full["Tag"] = df_full["Tag"].fillna("O").astype(str)

# -----------------------------
# SUBSET: 5001-12000
# -----------------------------
df = df_full[(df_full["Record Number"].astype(int) >= 5001) &
             (df_full["Record Number"].astype(int) <= 12000)].copy()

print("Total rows in subset:", len(df))
print("Unique Records in subset:", df["Record Number"].nunique())

# ==============================
# 2. GROUP INTO SENTENCES
# ==============================
sentences = []
labels = []

for record_id in df["Record Number"].unique():
    group = df[df["Record Number"] == record_id]
    sentences.append(group["Token"].tolist())
    labels.append(group["Tag"].tolist())

# ==============================
# 3. LABEL MAPPING
# ==============================
unique_labels = sorted(list(set(df["Tag"])))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print("Labels:", label2id)

# Convert labels to IDs
labels = [[label2id[l] for l in label_list] for label_list in labels]

# ==============================
# 4. CREATE DATASET
# ==============================
dataset = Dataset.from_dict({
    "tokens": sentences,
    "ner_tags": labels
})

# ==============================
# 5. TOKENIZER
# ==============================
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Shuffle dataset for training
tokenized_dataset = tokenized_dataset.shuffle(seed=42)

# ==============================
# 6. LOAD MODEL
# ==============================
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# ==============================
# 7. TRAINING ARGUMENTS
# ==============================
training_args = TrainingArguments(
    output_dir="./results_5001_12000",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    report_to="none"
)

# ==============================
# 8. TRAINER
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# ==============================
# 9. TRAIN
# ==============================
print("Starting training on subset 5001-12000...")
trainer.train()

# ==============================
# 10. SAVE MODEL
# ==============================
save_path = "./model_5001_12000"
print(f"Saving model to {save_path}...")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Training Complete — Model saved in", save_path)
