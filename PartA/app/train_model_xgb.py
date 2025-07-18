import os
import json
import fitz  # PyMuPDF
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Folder paths
PDF_DIR = "../data/pdfs"
LABEL_DIR = "../data/labels"
MODEL_PATH = "heading_model_xgb.joblib"

# Label mappings
LABEL_MAP = {"H1": 1, "H2": 2, "H3": 3, "BODY": 0}
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}


def is_bold(font_name: str) -> bool:
    return "bold" in font_name.lower() or "black" in font_name.lower()


def extract_features(span, previous_top: float):
    font_size = round(span["size"], 1)
    bold = 1 if is_bold(span["font"]) else 0
    top = round(span["bbox"][1], 1)
    spacing = round(abs(top - previous_top), 1)
    return [font_size, bold, top, spacing]


def load_json_label(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train():
    X = []
    y = []

    for pdf_file in os.listdir(PDF_DIR):
        if not pdf_file.endswith(".pdf"):
            continue

        json_file = pdf_file.replace(".pdf", ".json")
        json_path = os.path.join(LABEL_DIR, json_file)
        if not os.path.exists(json_path):
            print(f"⚠️ Skipping {pdf_file} — no label found")
            continue

        label_data = load_json_label(json_path)
        if "outline" not in label_data or not label_data["outline"]:
            print(f"⚠️ Skipping {pdf_file} — outline missing or empty")
            continue

        label_map = {(item["text"].strip(), item["page"]): LABEL_MAP[item["level"]]
                     for item in label_data["outline"] if item["level"] in LABEL_MAP}

        pdf_path = os.path.join(PDF_DIR, pdf_file)
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            previous_top = 0

            for b in blocks:
                for line in b.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text or len(text) < 3:
                            continue

                        features = extract_features(span, previous_top)
                        previous_top = round(span["bbox"][1], 1)
                        label = label_map.get((text, page_num), 0)  # default: BODY

                        X.append(features)
                        y.append(label)

    if not X:
        print("❌ No training data found. Check your labels.")
        return

    print(f"✅ Training on {len(X)} samples")

    # Train XGBoost model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ✅ Fix for class mismatch issue
    all_labels = list(REVERSE_MAP.keys())  # [0, 1, 2, 3]
    target_names = [REVERSE_MAP[i] for i in all_labels]

    print("📊 Classification report:\n", classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=target_names,
        zero_division=0
    ))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
