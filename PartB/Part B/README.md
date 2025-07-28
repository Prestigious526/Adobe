# Persona-Driven Document Intelligence (Part B)

Advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases.

## Features

- **PDF Processing**: Extract text blocks with font sizes and coordinates
- **Structure Detection**: Use K-means clustering to identify headings (H1, H2, H3)
- **Embedding**: Encode text using SentenceTransformers (all-MiniLM-L6-v2)
- **Ranking**: Hybrid approach combining cosine similarity and TF-IDF
- **Summarization**: Use T5-small for text refinement
- **Persona-Based Analysis**: Extract content relevant to specific user personas
- **Multi-Collection Support**: Process multiple PDF collections simultaneously

## Project Structure

```
Part B/
├── app/                    # Main application code
│   ├── main.py            # Main processing pipeline
│   ├── loader.py          # PDF parsing and caching
│   ├── outline.py         # Document structure extraction
│   ├── embed.py           # Text embedding
│   ├── rank.py            # Content ranking
│   ├── summarise.py       # Text summarization
│   ├── schema.py          # Output JSON structure
│   └── utils.py           # Helper functions
├── tests/                  # Test files
│   └── test_pipeline.py   # Pipeline validation tests
├── Challenge_1b/          # Challenge datasets
│   ├── Collection 1/      # Travel Planning
│   ├── Collection 2/      # Adobe Acrobat Learning
│   └── Collection 3/      # Recipe Collection
├── Dockerfile             # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Option 1: Local Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import torch, transformers, sentence_transformers; print('All dependencies installed successfully!')"
   ```

### Option 2: Docker Installation

1. **Build the Docker image:**
   ```bash
   docker build -t pdi .
   ```

## Usage

### Running the System

#### Method 1: Local Execution

```bash
# Process a single collection
python -m app.main "Challenge_1b/Collection 1" "output"

# Process multiple collections
python -m app.main "Challenge_1b/Collection 2" "output_collection2"
python -m app.main "Challenge_1b/Collection 3" "output_collection3"
```

#### Method 2: Docker Execution

```bash
# Process a collection using Docker
docker run --rm -v "$(pwd)/Challenge_1b/Collection 1:/app/input" -v "$(pwd)/output:/app/output" pdi "/app/input" "/app/output"

# Process multiple collections
docker run --rm -v "$(pwd)/Challenge_1b/Collection 2:/app/input" -v "$(pwd)/output_collection2:/app/output" pdi "/app/input" "/app/output"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_pipeline.py -v
```

## Input Format

The system expects a collection directory with the following structure:

```
Collection/
├── challenge1b_input.json    # Input configuration
├── PDFs/                     # PDF files to process
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
└── challenge1b_output.json   # Expected output (optional)
```

### Input JSON Structure

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [
    {"filename": "doc.pdf", "title": "Title"}
  ],
  "persona": {"role": "User Persona"},
  "job_to_be_done": {"task": "Use case description"}
}
```

## Output Format

The system generates a consolidated JSON file with the following structure:

```json
{
  "metadata": {
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "processing_timestamp": "2025-07-28T09:15:19.843466"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```

## Available Collections

### Collection 1: Travel Planning
- **Persona**: Travel Planner
- **Task**: Plan a 4-day trip for 10 college friends to South of France
- **Documents**: 7 travel guides

### Collection 2: Adobe Acrobat Learning
- **Persona**: HR Professional
- **Task**: Create and manage fillable forms for onboarding and compliance
- **Documents**: 15 Acrobat guides

### Collection 3: Recipe Collection
- **Persona**: Food Contractor
- **Task**: Prepare vegetarian buffet-style dinner menu for corporate gathering
- **Documents**: 9 cooking guides

## Performance

- **Processing Speed**: ~30 seconds for 7 PDFs
- **Memory Usage**: Efficient with caching
- **Output Quality**: High relevance to persona requirements
- **Scalability**: Handles 7-15 PDFs per collection

## Troubleshooting

### Common Issues

1. **GPU Requirements**: The system works on CPU, no GPU required
2. **Memory Issues**: Large PDFs may require more RAM
3. **Docker Issues**: Ensure Docker is running and has sufficient resources

### Error Messages

- `No such file or directory: 'python'`: Use `python3` instead
- `RuntimeError: No GPU found`: System works on CPU, ignore this warning
- `ConvergenceWarning`: Normal for K-means clustering, doesn't affect results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is part of the Adobe Hackathon challenge.
