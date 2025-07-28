from pathlib import Path
import sys
from . import loader, outline, utils, embed, rank, summarise, schema

def process(collection_path: Path, output_dir: Path):
    persona_file = collection_path / "challenge1b_input.json"
    persona, job = utils.load_persona(persona_file)
    persona_text = f"{persona} {job}"

    pdf_dir = collection_path / "PDFs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sections = []
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        blocks = loader.load(pdf_file)
        outline_data = outline.build(blocks)
        sections = utils.section_slices(blocks, outline_data)
        embeddings = embed.encode([s["text"] for s in sections])
        ranked_sections = rank.select(sections, embeddings, persona_text)
        refined_sections = [summarise.refine(s) for s in ranked_sections]
        
        # Add document info to sections
        for section in refined_sections:
            section["document"] = pdf_file.name
        
        all_sections.extend(refined_sections)
    
    # Generate single consolidated output
    json_str = schema.output(collection_path, persona_file, all_sections)
    (output_dir / "challenge1b_output.json").write_text(json_str)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m app.main <collection_path> <output_dir>")
        sys.exit(1)
    process(Path(sys.argv[1]), Path(sys.argv[2]))