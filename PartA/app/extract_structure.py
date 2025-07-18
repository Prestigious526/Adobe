# import fitz
# from statistics import mean, stdev
# import re

# def extract_line_features(pdf_path):
#     """
#     Extracts all lines from PDF with raw layout features.
#     Does NOT apply any filtering or alphanumeric checks.
#     """
#     doc = fitz.open(pdf_path)
#     all_lines = []

#     for page_index in range(len(doc)):
#         page = doc[page_index]
#         blocks = page.get_text("dict")["blocks"]

#         page_lines = []

#         for block in blocks:
#             for line in block.get("lines", []):
#                 spans = line.get("spans", [])
#                 if not spans:
#                     continue

#                 text = " ".join(span["text"].strip() for span in spans).strip()
#                 if not text:
#                     continue

#                 first_span = spans[0]
#                 font_size = first_span.get("size", 0)
#                 font_flags = first_span.get("flags", 0)
#                 is_bold = bool(font_flags & 2)
#                 is_upper = text.isupper()
#                 indent = round(first_span.get("origin", [0])[0], 2)
#                 line_top = line["bbox"][1]
#                 line_bottom = line["bbox"][3]

#                 page_lines.append({
#                     "text": text,
#                     "font_size": font_size,
#                     "bold": int(is_bold),
#                     "length": len(text),
#                     "is_upper": int(is_upper),
#                     "indent": indent,
#                     "line_top": line_top,
#                     "line_bottom": line_bottom,
#                     "page": page_index
#                 })

#         # Add spacing_before and spacing_after
#         for i, line in enumerate(page_lines):
#             spacing_before = round(abs(line["line_top"] - (page_lines[i-1]["line_bottom"] if i > 0 else 0)), 2)
#             spacing_after = round(abs((page_lines[i+1]["line_top"] - line["line_bottom"]) if i < len(page_lines)-1 else 0), 2)

#             all_lines.append({
#                 "text": line["text"],
#                 "font_size": line["font_size"],
#                 "bold": line["bold"],
#                 "length": line["length"],
#                 "is_upper": line["is_upper"],
#                 "indent": line["indent"],
#                 "spacing_before": spacing_before,
#                 "spacing_after": spacing_after,
#                 "page": line["page"]
#             })

#     return all_lines



# def filter_candidates(lines, z=0.25):
#     """
#     Applies all layout-aware constraints to detect heading candidates:
#     1. ≥ 3 alphanumeric characters
#     2. ≤ 100 characters
#     3. ≤ 20 words
#     4. font_size ≥ mean + z * std_dev (per page)
#     """
#     alnum_pattern = re.compile(r"[A-Za-z0-9]")
#     lines_by_page = {}
#     for line in lines:
#         lines_by_page.setdefault(line["page"], []).append(line)

#     filtered = []

#     for page, page_lines in lines_by_page.items():
#         font_sizes = [l["font_size"] for l in page_lines]

#         if len(font_sizes) < 2:
#             font_threshold = font_sizes[0] if font_sizes else 0
#         else:
#             mu = mean(font_sizes)
#             sigma = stdev(font_sizes)
#             font_threshold = mu + z * sigma

#         for line in page_lines:
#             text = line["text"]

#             if len(text) > 100:
#                 continue
#             if len(text.split()) > 20:
#                 continue
#             if sum(c.isalnum() for c in text) < 3:
#                 continue
#             if line["font_size"] < font_threshold:
#                 continue

#             filtered.append(line)

#     return filtered


import fitz  # PyMuPDF
import re
from statistics import mean, stdev
from collections import defaultdict

def extract_line_features(pdf_path):
    """
    Extracts layout-aware line features from a PDF.
    Each line includes font size, bold, spacing, indent, etc.
    """
    doc = fitz.open(pdf_path)
    all_lines = []
    alnum_pattern = re.compile(r"[A-Za-z0-9]")

    for page_index in range(len(doc)):
        page = doc[page_index]
        blocks = page.get_text("dict")["blocks"]
        page_lines = []

        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                raw_text = "".join(span["text"] for span in spans)
                text = re.sub(r"\s{2,}", " ", raw_text).strip()
                if not text:
                    continue

                first_span = spans[0]
                font_size = first_span.get("size", 0)
                font_flags = first_span.get("flags", 0)
                is_bold = bool(font_flags & 2)
                is_upper = text.isupper()
                indent = round(first_span.get("origin", [0])[0], 2)
                line_top = line["bbox"][1]
                line_bottom = line["bbox"][3]

                page_lines.append({
                    "text": text,
                    "font_size": font_size,
                    "bold": int(is_bold),
                    "length": len(text),
                    "is_upper": int(is_upper),
                    "indent": indent,
                    "line_top": line_top,
                    "line_bottom": line_bottom,
                    "page": page_index
                })

        # Compute spacing before/after for each line
        for i, line in enumerate(page_lines):
            spacing_before = round(abs(line["line_top"] - (page_lines[i - 1]["line_bottom"] if i > 0 else 0)), 2)
            spacing_after = round(abs((page_lines[i + 1]["line_top"] - line["line_bottom"]) if i < len(page_lines) - 1 else 0), 2)

            all_lines.append({
                "text": line["text"],
                "font_size": line["font_size"],
                "bold": line["bold"],
                "length": line["length"],
                "is_upper": line["is_upper"],
                "indent": line["indent"],
                "spacing_before": spacing_before,
                "spacing_after": spacing_after,
                "page": line["page"],
                "line_top": line["line_top"],
                "line_bottom": line["line_bottom"]
            })

    return all_lines



def filter_candidates(lines, z=0.25, remove_repetitive_headers=True):
    """
    Filters heading lines based on font, structure, and repetition patterns.
    No longer filters based on raw top/bottom spacing, but removes:
    - Lines repeated at same position across pages
    - First/last line on page
    """
    alnum_pattern = re.compile(r"[A-Za-z0-9]")
    lines_by_page = defaultdict(list)

    for line in lines:
        lines_by_page[line["page"]].append(line)

    filtered = []
    text_position_counts = defaultdict(int)
    page_heights = {}

    # Estimate page height from line_bottom
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

        # Find first and last line on this page
        sorted_lines = sorted(page_lines, key=lambda x: x["line_top"])
        first_line = sorted_lines[0]["text"].strip()
        last_line = sorted_lines[-1]["text"].strip()

        for line in page_lines:
            text = line["text"].strip()
            alnum_count = sum(c.isalnum() for c in text)
            y_top = line.get("line_top", 0)
            key = (text.lower(), round(y_top, 1))

            # 🔴 Skip first or last line on page (usually header/footer)
            if text == last_line:
                continue

            # 🔴 Skip lines repeated across ≥ 3 pages
            if remove_repetitive_headers and text_position_counts[key] >= 3:
                continue

            # 🔸 Layout + Textual Constraints
            if alnum_count < 3:
                continue
            if len(text) > 100 or len(text.split()) > 15:
                continue
            if line["font_size"] < font_threshold:
                continue

            filtered.append(line)

    return filtered
