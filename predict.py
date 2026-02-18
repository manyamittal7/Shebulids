import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load saved model
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

device = torch.device("cpu")
model.to(device)
model.eval()

# Get label mapping
id2label = model.config.id2label

# Test sentence
text = "John lives in New York"

inputs = tokenizer(text, return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
pred_ids = predictions[0].cpu().numpy()

print("\nPredictions:\n")
for token, label_id in zip(tokens, pred_ids):
    label_name = id2label[label_id]
    print(token, "->", label_name)
