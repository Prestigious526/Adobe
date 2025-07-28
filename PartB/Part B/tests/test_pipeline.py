import json
from pathlib import Path
import subprocess
import pytest

# Paths for testing
BASE_PATH = Path(__file__).resolve().parent.parent
COLLECTION_PATH = BASE_PATH / "Challenge_1b" / "Collection 1"
OUTPUT_PATH = BASE_PATH / "test_output"

@pytest.mark.order(1)
def test_pipeline_execution():
    """
    Run pipeline on Collection 1 and verify JSON files are generated.
    """
    # Ensure output folder exists and is empty
    if OUTPUT_PATH.exists():
        for file in OUTPUT_PATH.glob("*.json"):
            file.unlink()
    else:
        OUTPUT_PATH.mkdir()

    # Execute the pipeline
    subprocess.run(
        ["python3", "-m", "app.main", str(COLLECTION_PATH), str(OUTPUT_PATH)],
        check=True
    )

    # Check that JSON files are generated
    generated_files = list(OUTPUT_PATH.glob("*.json"))
    assert len(generated_files) > 0, "No output JSON files generated"

@pytest.mark.order(2)
def test_schema_and_comparison():
    """
    Validate generated JSON against schema and compare with ground-truth.
    """
    ground_truth_path = COLLECTION_PATH / "challenge1b_output.json"
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    generated_files = list(OUTPUT_PATH.glob("*.json"))
    assert generated_files, "Run pipeline first"

    # Basic schema validation: keys
    for gen_file in generated_files:
        with open(gen_file) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "extracted_sections" in data
        assert "subsection_analysis" in data
        assert isinstance(data["extracted_sections"], list)
        assert isinstance(data["subsection_analysis"], list)

        # Check important fields in metadata
        assert "input_documents" in data["metadata"]
        assert "persona" in data["metadata"]
        assert "job_to_be_done" in data["metadata"]
        assert "processing_timestamp" in data["metadata"]

        # Check important fields in extracted_sections
        for section in data["extracted_sections"]:
            assert "document" in section
            assert "section_title" in section
            assert "importance_rank" in section
            assert "page_number" in section

        # Check important fields in subsection_analysis
        for subsection in data["subsection_analysis"]:
            assert "document" in subsection
            assert "refined_text" in subsection
            assert "page_number" in subsection

    # Check that we have reasonable number of sections (not exact match)
    assert len(data["extracted_sections"]) > 0, "Should have extracted sections"
    assert len(data["subsection_analysis"]) > 0, "Should have subsection analysis"
    
    # Check that metadata matches ground truth structure
    assert data["metadata"]["persona"] == ground_truth["metadata"]["persona"]
    assert data["metadata"]["job_to_be_done"] == ground_truth["metadata"]["job_to_be_done"]
    assert len(data["metadata"]["input_documents"]) == len(ground_truth["metadata"]["input_documents"])