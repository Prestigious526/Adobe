from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import joblib, hashlib
from pathlib import Path

def load(pdf_path: Path):
    """
    Parse PDF into a list of text blocks with font size and coordinates.
    Caches parsed result in /tmp/<hash>.pkl.
    """
    hash_id = hashlib.md5(str(pdf_path).encode()).hexdigest()
    cache_path = Path(f"/tmp/{hash_id}.pkl")
    if cache_path.exists():
        return joblib.load(cache_path)

    blocks = []
    for page_num, page_layout in enumerate(extract_pages(pdf_path)):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if not text:
                    continue
                # Average font size
                font_sizes = [obj.size for line in element for obj in line if isinstance(obj, LTChar)]
                avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                x0, y0, _, _ = element.bbox
                blocks.append({
                    "text": text,
                    "font_size": avg_font,
                    "x0": x0,
                    "y0": y0,
                    "page": page_num + 1
                })
    joblib.dump(blocks, cache_path)
    return blocks