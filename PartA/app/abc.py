import os
import json
from extract_structure import filter_candidates, extract_line_features


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    input_dir = "/app/input" if os.getenv("DOCKER") == "true" else "./input"
    output_dir = "/app/output" if os.getenv("DOCKER") == "true" else "./output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)

        # Step 1: Extract lines with layout features
        all_lines = extract_line_features(pdf_path)

        # Step 2: Apply filtering constraints
        candidates = filter_candidates(all_lines)

        # Step 3: Save filtered result to JSON with _raw suffix
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_raw.json"
        output_path = os.path.join(output_dir, output_filename)
        save_json(candidates, output_path)

        print(f"Processed: {filename} → {output_filename}")


if __name__ == "__main__":
    main()
