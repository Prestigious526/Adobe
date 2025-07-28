import os
import json
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Label mapping
LABEL_MAP = {"BODY": 0, "H1": 1, "H2": 2, "H3": 3}
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}

# Paths
INPUT_DIR = "./output"
MODEL_PATH = "heading_model_lgbm.joblib"
SCALER_PATH = "scaler.joblib"

# Features used for training
FEATURE_KEYS = [
    "font_size", "bold", "spacing_before", "spacing_after",
    "indent", "length", "is_upper", "line_top", "line_bottom", 
    "ends_with_colon", "is_short", "is_numbered", "first_page",
    "page", "num_words", "num_verbs", "num_nouns", "num_adjectives",
    "num_adverbs", "num_pronouns", "num_cardinals", "num_conjunctions",
    "num_predeterminers", "num_interjections"
]
  
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def match_label(raw_line, gt_outline):
    """Match based on exact text + page."""
    for item in gt_outline:
        if item["text"].strip() == raw_line["text"].strip() and item["page"] == raw_line["page"]:
            return item["level"]
    return "BODY"

def extract_features_and_labels():
    X, y = [], []
    skipped = 0

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith("_raw.json"):
            continue

        base = fname.replace("_raw.json", "")
        raw_path = os.path.join(INPUT_DIR, f"{base}_raw.json")
        label_path = os.path.join(INPUT_DIR, f"{base}.json")

        if not os.path.exists(label_path):
            print(f"⚠️ Missing ground truth for {base}, skipping")
            skipped += 1
            continue

        raw_lines = load_json(raw_path)
        try:
            gt = load_json(label_path)
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed for {label_path} → {e}")
            continue

        gt_outline = gt.get("outline", [])

        for line in raw_lines:
            label = match_label(line, gt_outline)
            if label not in LABEL_MAP:
                continue

            feat = [line.get(k, 0) for k in FEATURE_KEYS]
            X.append(feat)
            y.append(LABEL_MAP[label])

    print(f"Samples collected: {len(X)}")
    print(f"Files skipped (no GT): {skipped}")
    return np.array(X), np.array(y)

def train():
    X, y = extract_features_and_labels()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[REVERSE_MAP[i] for i in sorted(REVERSE_MAP)]))

if __name__ == "__main__":
    train()
