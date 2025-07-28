
import os
import json
import re
import fitz  
import nltk
import pandas as pd
from collections import defaultdict, Counter
from statistics import mean, stdev
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

def extract_line_features_with_text_stats(pdf_path):

    doc = fitz.open(pdf_path)
    all_lines = []
    alnum_pattern = re.compile(r"[A-Za-z0-9]")

    for page_index, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        page_lines = []

        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                raw_text = "".join(span["text"] for span in spans)
                text = re.sub(r"\s{2,}", " ", raw_text).strip()
                if not text or not alnum_pattern.search(text):
                    continue

                first_span = spans[0]
                font_size = first_span.get("size", 0)
                font_flags = first_span.get("flags", 0)
                is_bold = bool(font_flags & 2)
                is_upper = text.isupper()
                indent = round(first_span.get("origin", [0])[0], 2)
                y_top = line["bbox"][1]
                y_bottom = line["bbox"][3]
                num_words = len(text.split())

                # POS tagging
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                tag_counts = Counter(tag for _, tag in pos_tags)

                page_lines.append({
                    "text": text,
                    "font_size": font_size,
                    "bold": int(is_bold),
                    "length": len(text),
                    "is_upper": int(is_upper),
                    "indent": indent,
                    "line_top": y_top,
                    "line_bottom": y_bottom,
                    "spacing_before": None, 
                    "spacing_after": None,
                    "ends_with_colon": int(text.endswith(":")),
                    "is_short": int(num_words <= 8),
                    "is_numbered": int(bool(re.match(r"^\d+(\.\d+)*", text))),
                    "first_page": int(page_index == 0),
                    "page": page_index,
                    "num_words": num_words,
                    "num_verbs": tag_counts["VB"] + tag_counts["VBD"] + tag_counts["VBG"] + tag_counts["VBN"] + tag_counts["VBP"] + tag_counts["VBZ"],
                    "num_nouns": tag_counts["NN"] + tag_counts["NNS"] + tag_counts["NNP"] + tag_counts["NNPS"],
                    "num_adjectives": tag_counts["JJ"] + tag_counts["JJR"] + tag_counts["JJS"],
                    "num_adverbs": tag_counts["RB"] + tag_counts["RBR"] + tag_counts["RBS"],
                    "num_pronouns": tag_counts["PRP"] + tag_counts["PRP$"] + tag_counts["WP"] + tag_counts["WP$"],
                    "num_cardinals": tag_counts["CD"],
                    "num_conjunctions": tag_counts["CC"],
                    "num_predeterminers": tag_counts["PDT"],
                    "num_interjections": tag_counts["UH"]
                })

        for i, line in enumerate(page_lines):
            line["spacing_before"] = round(abs(line["line_top"] - (page_lines[i - 1]["line_bottom"] if i > 0 else 0)), 2)
            line["spacing_after"] = round(abs((page_lines[i + 1]["line_top"] - line["line_bottom"]) if i < len(page_lines) - 1 else 0), 2)
            all_lines.append(line)

    return all_lines  


def merge_similar_multiline_rows(df):
    if df.empty:
        return df

    df = df.sort_values(by=["page", "line_top"]).reset_index(drop=True)
    merged_rows = []
    current = df.iloc[0].to_dict()

    for i in range(1, len(df)):
        row = df.iloc[i]
        keys_to_match = [
            "font_size", "bold", "is_upper", "ends_with_colon",
            "is_short", "is_numbered", "page", "first_page"
        ]
        if all(current.get(k) == row[k] for k in keys_to_match):
            current["text"] += " " + row["text"]
            current["length"] += row["length"]
            current["num_words"] += row["num_words"]
            current["line_top"] = min(current["line_top"], row["line_top"])
            current["line_bottom"] = max(current["line_bottom"], row["line_bottom"])
        else:
            merged_rows.append(current)
            current = row.to_dict()

    merged_rows.append(current)
    return pd.DataFrame(merged_rows)


def filter_candidates(lines, z=0.25, remove_repetitive_headers=True):
    from statistics import mean, stdev

    alnum_pattern = re.compile(r"[A-Za-z0-9]")
    lines_by_page = defaultdict(list)
    for line in lines:
        lines_by_page[line["page"]].append(line)

    filtered = []
    text_position_counts = defaultdict(int)
    page_heights = {}

    for page, page_lines in lines_by_page.items():
        page_heights[page] = max((l.get("line_bottom", 0) for l in page_lines), default=800)
        for line in page_lines:
            key = (line["text"].strip().lower(), round(line.get("line_top", 0), 1))
            text_position_counts[key] += 1

    for page, page_lines in lines_by_page.items():
        font_sizes = [l["font_size"] for l in page_lines]
        mu = mean(font_sizes)
        sigma = stdev(font_sizes) if len(font_sizes) > 1 else 0
        font_threshold = mu + z * sigma

        sorted_lines = sorted(page_lines, key=lambda x: x["line_top"])
        first_line = sorted_lines[0]["text"].strip()
        last_line = sorted_lines[-1]["text"].strip()

        for line in page_lines:
            text = line["text"].strip()
            alnum_count = sum(c.isalnum() for c in text)
            y_top = line.get("line_top", 0)
            key = (text.lower(), round(y_top, 1))

            if text == last_line:
                continue
            if remove_repetitive_headers and text_position_counts[key] >= 3:
                continue
            if alnum_count < 3:
                continue
            if len(text) > 100 or len(text.split()) > 15:
                continue
            if line["font_size"] < font_threshold:
                continue

            filtered.append(line)

    filtered_df = pd.DataFrame(filtered)
    merged_df = merge_similar_multiline_rows(filtered_df)
    return merged_df.to_dict(orient="records")
