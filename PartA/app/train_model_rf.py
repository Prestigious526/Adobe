import os
import json
import fitz  # PyMuPDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = "../data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
MODEL_PATH = "heading_model_rf.joblib"


def is_bold(font_name: str):
    return "bold" in font_name.lower() or "black" in font_name.lower()


def extract_features(span, previous_top):
    font_size = round(span["size"], 1)
    bold = 1 if is_bold(span["font"]) else 0
    top = round(span["bbox"][1], 1)
    spacing = round(abs(top - previous_top), 1)
    return [font_size, bold, top, spacing]


def load_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train():
    X = []
    y = []

    for pdf_name in os.listdir(PDF_DIR):
        if not pdf_name.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, pdf_name)
        json_name = pdf_name.replace(".pdf", ".json")
        json_path = os.path.join(LABEL_DIR, json_name)

        if not os.path.exists(json_path):
            print(f"⚠️ No label found for {pdf_name}, skipping...")
            continue

        print(f"📄 Processing {pdf_name}")
        label_data = load_labels(json_path)
        labeled_headings = {(item["text"].strip(), item["page"]): item["level"]
                            for item in label_data.get("outline", [])}

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

                        label = labeled_headings.get((text, page_num + 1), "BODY")
                        X.append(features)
                        y.append(label)

    if not X:
        print("❌ No data collected — check your PDFs and JSONs.")
        return

    print(f"✅ Extracted {len(X)} samples")

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("🧪 Evaluation:\n", classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
