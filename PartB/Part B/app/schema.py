from pydantic import BaseModel, Field
from typing import List
import json
from datetime import datetime

class ExtractedSection(BaseModel):
    document: str
    section_title: str
    importance_rank: int
    page_number: int

class SubsectionAnalysis(BaseModel):
    document: str
    refined_text: str
    page_number: int

class Metadata(BaseModel):
    input_documents: List[str]
    persona: str
    job_to_be_done: str
    processing_timestamp: str

class Payload(BaseModel):
    metadata: Metadata
    extracted_sections: List[ExtractedSection]
    subsection_analysis: List[SubsectionAnalysis]

def output(collection_path, persona_file, all_sections):
    # Load persona info
    with open(persona_file, 'r') as f:
        persona_data = json.load(f)
    
    # Extract document names
    input_docs = [doc["filename"] for doc in persona_data.get("documents", [])]
    persona = persona_data.get("persona", {}).get("role", "")
    job = persona_data.get("job_to_be_done", {}).get("task", "")
    
    # Process sections
    extracted_sections = []
    subsection_analysis = []
    
    for section in all_sections:
        # Add to extracted sections
        extracted_sections.append(ExtractedSection(
            document=section["document"],
            section_title=section["title"],
            importance_rank=section["importance_rank"],
            page_number=section["page"]
        ))
        
        # Add to subsection analysis
        if "subsection" in section:
            subsection_analysis.append(SubsectionAnalysis(
                document=section["document"],
                refined_text=section["subsection"]["refined_text"],
                page_number=section["page"]
            ))
    
    payload = Payload(
        metadata=Metadata(
            input_documents=input_docs,
            persona=persona,
            job_to_be_done=job,
            processing_timestamp=datetime.now().isoformat()
        ),
        extracted_sections=extracted_sections,
        subsection_analysis=subsection_analysis
    )
    return payload.model_dump_json(indent=2)