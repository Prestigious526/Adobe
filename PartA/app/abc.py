import os
import json
from extract_structure import extract_line_features_with_text_stats, filter_candidates
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def save_json(data, output_path):
    """Save a Python object as a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    input_dir = "/app/input" if os.getenv("DOCKER") == "true" else "./input"
    output_dir = "/app/output" if os.getenv("DOCKER") == "true" else "./output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)

        print(f"Processing: {filename}")

        all_lines = extract_line_features_with_text_stats(pdf_path)

        candidates = filter_candidates(all_lines)

        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_raw.json"
        output_path = os.path.join(output_dir, output_filename)
        save_json(candidates, output_path)

        print(f"Saved → {output_filename}")

if __name__ == "__main__":
    main()
