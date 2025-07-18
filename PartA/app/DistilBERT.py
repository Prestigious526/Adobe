import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from statistics import mean, stdev
import re

# === LABEL DEFINITIONS ===
LABELS = ["Title", "H1", "H2", "H3", "H4", "NotHeading"]
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

# === LOAD AND BUILD TRAINING EXAMPLES ===
def build_training_examples(raw_lines, annotation_json):
    annotated_headings = set()
    for h in annotation_json.get("outline", []):
        page = h["page"]
        text = h["text"].strip().lower()
        annotated_headings.add((page, text))

    title_texts = annotation_json.get("title", "").lower().split()
    examples = []

    for line in raw_lines:
        page = line["page"]
        text = line["text"].strip()
        clean_text = text.lower()

        # Match
        if any(t in clean_text for t in title_texts):
            label = "Title"
        elif (page, clean_text) in annotated_headings:
            label = next((h["level"] for h in annotation_json["outline"]
                          if h["page"] == page and h["text"].strip().lower() == clean_text), "NotHeading")
        else:
            label = "NotHeading"

        examples.append({
            "text": text,
            "features": [
                line["font_size"], line["bold"], line["length"], line["is_upper"],
                line["indent"], line["spacing_before"], line["spacing_after"]
            ],
            "label": label2id[label]
        })

    return examples

def load_all_training_examples(output_dir):
    examples = []

    for filename in os.listdir(output_dir):
        if not filename.endswith("_raw.json"):
            continue

        base_name = filename.replace("_raw.json", "")
        raw_path = os.path.join(output_dir, filename)
        annotated_path = os.path.join(output_dir, base_name + ".json")

        if not os.path.exists(annotated_path):
            print(f"❌ Missing: {annotated_path}")
            continue

        raw_lines = json.load(open(raw_path))
        annotated = json.load(open(annotated_path))

        file_examples = build_training_examples(raw_lines, annotated)
        examples.extend(file_examples)
        print(f"✅ Processed: {base_name} → {len(file_examples)} lines")

    print(f"🎯 Total examples: {len(examples)}")
    return examples

# === DATASET CLASS ===
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class HeadingDataset(Dataset):
    def __init__(self, data, max_len=64):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = tokenizer(item["text"], truncation=True, padding='max_length',
                        max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "layout_features": torch.tensor(item["features"], dtype=torch.float),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

# === MODEL ===
class HeadingClassifier(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.layout_proj = nn.Linear(7, 64)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, layout_features):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        layout_out = self.layout_proj(layout_features)
        combined = torch.cat([text_out, layout_out], dim=1)
        return self.classifier(combined)

# === TRAINING FUNCTION ===
def train_model(model, dataset, epochs=3, batch_size=16, lr=2e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            layout_features = batch["layout_features"]
            labels = batch["label"]

            outputs = model(input_ids, attention_mask, layout_features)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📘 Epoch {epoch+1}/{epochs} — Loss: {total_loss:.4f}")

# === MAIN ENTRY ===
if __name__ == "__main__":
    output_dir = "./output"
    examples = load_all_training_examples(output_dir)
    dataset = HeadingDataset(examples)
    model = HeadingClassifier(num_labels=len(LABELS))
    train_model(model, dataset)
    torch.save(model.state_dict(), "distilbert_heading_model.pt")
    print("✅ Model saved to distilbert_heading_model.pt")
