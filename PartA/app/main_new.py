import os
import json
import joblib
from extract_structure import extract_line_features_with_text_stats, filter_candidates
import numpy as np
MODEL_PATH = "heading_model_lgbm.joblib"
SCALER_PATH = "scaler.joblib"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)




def apply_model(lines, model, scaler, feature_keys):
    outline = []
    for line in lines:
        features = [line.get(k, 0) for k in feature_keys]
        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        if pred != "BODY":
            outline.append({
                "level": pred,
                "text": line["text"],
                "page": line["page"]
            })
    return outline

def main():
    input_dir = "/app/input" if os.getenv("DOCKER") == "true" else "./input"
    output_dir = "/app/test_output" if os.getenv("DOCKER") == "true" else "./test_output"
    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_keys = [
        "font_size", "bold", "spacing_before", "spacing_after",
        "indent", "length", "is_upper", "line_top", "line_bottom", 
        "ends_with_colon", "is_short", "is_numbered", "first_page",
        "page", "num_words", "num_verbs", "num_nouns", "num_adjectives",
        "num_adverbs", "num_pronouns", "num_cardinals", "num_conjunctions",
        "num_predeterminers", "num_interjections"
    ]

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)
        lines = extract_line_features_with_text_stats(pdf_path)
        filtered_lines = filter_candidates(lines)

        outline_raw = apply_model(filtered_lines, model, scaler, feature_keys)

        outline = []
        for line in outline_raw:
            level = line["level"]
            if isinstance(level, int) or (isinstance(level, str) and level.isdigit()):
                level_str = f"H{int(level)+1}"
            else:
                level_str = str(level)
            outline.append({
                "level": level_str,
                "text": line["text"],
                "page": line["page"]
            })

        title_line = next((l for l in outline if l["level"] == "H1" and l["page"] == 0), None)
        title = title_line["text"] if title_line else ""
        if title_line:
            outline = [l for l in outline if l != title_line]

        result = {
            "title": title,
            "outline": outline
        }

        output_filename = filename.replace(".pdf", ".json")
        output_path = os.path.join(output_dir, output_filename)
        save_json(result, output_path)
        print(f"Processed: {filename} → {output_filename}")



if __name__ == "__main__":
    main()
