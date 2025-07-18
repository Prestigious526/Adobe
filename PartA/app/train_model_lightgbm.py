import os
import json
import fitz
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

PDF_DIR = "../data/pdfs"
LABEL_DIR = "../data/labels"
MODEL_PATH = "heading_model_lightgbm.joblib"

LABEL_MAP = {"H1": 1, "H2": 2, "H3": 3, "BODY": 0}
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}


def is_bold(font):
    return "bold" in font.lower() or "black" in font.lower()


def extract_features(span, prev_top):
    font_size = round(span["size"], 1)
    bold = 1 if is_bold(span["font"]) else 0
    top = round(span["bbox"][1], 1)
    spacing = round(abs(top - prev_top), 1)
    return [font_size, bold, top, spacing]


def load_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train():
    X, y = [], []

    for pdf_file in os.listdir(PDF_DIR):
        if not pdf_file.endswith(".pdf"):
            continue

        label_file = pdf_file.replace(".pdf", ".json")
        label_path = os.path.join(LABEL_DIR, label_file)
        if not os.path.exists(label_path):
            print(f"⚠️ Skipping {pdf_file} — label missing")
            continue

        data = load_labels(label_path)
        if "outline" not in data or not data["outline"]:
            print(f"⚠️ Skipping {pdf_file} — empty outline")
            continue

        labels = {(item["text"].strip(), item["page"]): LABEL_MAP[item["level"]]
                  for item in data["outline"] if item["level"] in LABEL_MAP}

        doc = fitz.open(os.path.join(PDF_DIR, pdf_file))

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            prev_top = 0

            for block in blocks:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text or len(text) < 3:
                            continue

                        features = extract_features(span, prev_top)
                        prev_top = round(span["bbox"][1], 1)
                        label = labels.get((text, page_num), 0)

                        X.append(features)
                        y.append(label)

    if not X:
        print("❌ No training data found.")
        return

    print(f"✅ Collected {len(X)} samples")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    all_labels = list(REVERSE_MAP.keys())
    target_names = [REVERSE_MAP[i] for i in all_labels]

    print("📊 Evaluation:\n", classification_report(
        y_test, y_pred,
        labels=all_labels,
        target_names=target_names,
        zero_division=0
    ))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
