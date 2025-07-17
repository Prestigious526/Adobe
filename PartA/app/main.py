import fitz  # PyMuPDF
import os
import json
import joblib

MODEL_PATH = "heading_model_mlp.joblib"
SCALER_PATH = "scaler.joblib"

def is_bold(font_name: str):
    return "bold" in font_name.lower() or "black" in font_name.lower()

def extract_features(span, previous_top):
    font_size = round(span["size"], 1)
    bold = 1 if is_bold(span["font"]) else 0
    top = round(span["bbox"][1], 1)
    spacing = round(abs(top - previous_top), 1)
    return [font_size, bold, top, spacing]

def extract_headings(pdf_path, model, scaler):
    doc = fitz.open(pdf_path)
    outline = []
    title = None

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

                    scaled_features = scaler.transform([features])
                    pred = model.predict(scaled_features)[0]
                    if pred != "BODY":
                        outline.append({
                            "level": str(pred),         # Ensure it's a string like "H1"
                            "text": str(text),          # Just to be safe
                            "page": int(page_num + 1)   # Convert numpy.int64 → Python int
                        })


                    # Detect title: first large bold text on page 1
                    if page_num == 0 and not title:
                        if features[0] >= 15 and features[1] == 1:
                            title = text

    return {
        "title": str(title or "Untitled"),
        "outline": outline
    }

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    input_dir = "/app/input" if os.getenv("DOCKER") == "true" else "./input"
    output_dir = "/app/output" if os.getenv("DOCKER") == "true" else "./output"

    os.makedirs(output_dir, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)
        result = extract_headings(pdf_path, model, scaler)

        output_filename = filename.replace(".pdf", ".json")
        output_path = os.path.join(output_dir, output_filename)
        save_json(result, output_path)
        print(f"✅ Processed: {filename} → {output_filename}")

if __name__ == "__main__":
    main()
