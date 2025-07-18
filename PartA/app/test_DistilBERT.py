import os
import json
import torch
from transformers import DistilBertTokenizer
from DistilBERT import HeadingClassifier, id2label  # from training script
from extract_structure import extract_line_features, filter_candidates  # your existing layout extractor

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def prepare_features(line):
    return [
        line["font_size"], line["bold"], line["length"], line["is_upper"],
        line["indent"], line["spacing_before"], line["spacing_after"]
    ]

def predict(model, line):
    model.eval()
    text = line["text"]
    layout = prepare_features(line)

    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    layout_tensor = torch.tensor([layout], dtype=torch.float)

    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            layout_features=layout_tensor
        )
        pred = torch.argmax(logits, dim=1).item()
        return id2label[pred]

def test_model_on_pdfs(model_path="distilbert_heading_model.pt"):
    input_dir = "/app/input" if os.getenv("DOCKER") == "true" else "./input"
    output_dir = "/app/test_output" if os.getenv("DOCKER") == "true" else "./test_output"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = HeadingClassifier(num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    for filename in os.listdir(input_dir):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)
        print(f"🔍 Processing: {filename}")

        x = extract_line_features(pdf_path)
        lines = filter_candidates(x)
        output = {"title": "", "outline": []}

        for line in lines:
            label = predict(model, line)
            if label == "NotHeading":
                continue
            elif label == "Title" and not output["title"]:
                output["title"] = line["text"]
            else:
                output["outline"].append({
                    "level": label,
                    "text": line["text"],
                    "page": line["page"]  # 0-based already
                })

        # Save result
        out_file = os.path.join(output_dir, filename.replace(".pdf", ".json"))
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"✅ Saved: {out_file}")

if __name__ == "__main__":
    test_model_on_pdfs()
