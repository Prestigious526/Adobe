import json
from pathlib import Path

def load_persona(persona_path: Path):
    """
    Load persona + job from challenge1b_input.json.
    """
    with open(persona_path, 'r') as f:
        data = json.load(f)
    persona = data.get('persona', {})
    job = data.get('job_to_be_done', '')
    return persona, job

def section_slices(blocks, outline):
    """
    Merge blocks between headings into section chunks.
    """
    sections = []
    if not outline:
        return sections

    for i, heading in enumerate(outline):
        start = heading["start_idx"]
        end = outline[i + 1]["start_idx"] if i + 1 < len(outline) else len(blocks)
        text = " ".join([b["text"] for b in blocks[start:end]])
        sections.append({
            "page": heading["page"],
            "level": heading["level"],
            "title": heading["title"],
            "text": text
        })
    return sections