# RAG PDF Extractor with Local Ollama

A powerful RAG (Retrieval Augmented Generation) pipeline that extracts structured information from PDF documents using local LLaMA 3.2 via Ollama.

## Features

- 📄 Extract text from PDF documents
- 🔍 Semantic search using embeddings (all-mpnet-base-v2)
- 🤖 Local LLM processing with Ollama (LLaMA 3.2)
- 📊 Export results to Excel
- 🎯 Smart key-value extraction with fallback heuristics
- 🔄 Automatic deduplication and normalization

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally

## Installation

### Step 1: Install Ollama

**For Windows:**
1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Run the installer
3. Open a terminal and run:
```bash
ollama pull llama3.2
```

**For macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pdfplumber pandas sentence-transformers faiss-cpu requests numpy openpyxl
```

## Setup

### Step 3: Start Ollama Server

Open a terminal and run:
```bash
ollama serve
```

Keep this terminal running while using the application.

### Step 4: Configure Your PDF Path

1. Open `main.py`
2. Update the `PDF_PATH` variable (line 28) with your PDF file path:
```python
PDF_PATH = "path/to/your/document.pdf"  # Change this to your PDF location
```

Example:
- Windows: `PDF_PATH = "C:\\Users\\YourName\\Documents\\resume.pdf"`
- Mac/Linux: `PDF_PATH = "/Users/YourName/Documents/resume.pdf"`

## Usage

Run the script:
```bash
python main.py
```

The script will:
1. Extract text from your PDF
2. Create semantic chunks for better retrieval
3. Extract key information using AI
4. Save results to `Output_improved.xlsx`

## What Gets Extracted

The pipeline automatically extracts:
- Personal information (name, DOB, nationality, etc.)
- Educational background (degrees, colleges, scores)
- Work experience (companies, roles, salaries)
- Technical skills and certifications
- And more...

## Output

Results are saved in `Output_improved.xlsx` with three columns:
- **Key**: The information type
- **Value**: The extracted value
- **Comments**: Source context from the PDF

## Customization

### Adjust Retrieval Settings

In `main.py`, you can modify:
- `CHUNK_SIZE`: Size of text chunks (default: 400 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100 characters)
- `TOP_K`: Number of chunks to retrieve per key (default: 6)

### Add Custom Keys

Edit the `keys` list in `main()` function (around line 252) to extract additional information:
```python
keys = [
    "First Name",
    "Last Name",
    # Add your custom keys here
    "Custom Field Name",
]
```

## Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check if the server is accessible at `http://localhost:11434`

### "Model not found" error
- Pull the model again: `ollama pull llama3.2`

### Slow performance
- First run downloads the embedding model (~420MB)
- Subsequent runs are faster
- Reduce `TOP_K` for faster processing

### Empty extractions
- Check if your PDF contains selectable text (not scanned images)
- Increase `TOP_K` for better retrieval
- The script has fallback heuristics for common patterns

## Technical Details

- **Embedding Model**: all-mpnet-base-v2 (sentence-transformers)
- **LLM**: LLaMA 3.2 via Ollama
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: pdfplumber

## License

This project is provided as-is for educational and personal use.

## Support

For issues or questions:
1. Ensure all dependencies are installed
2. Check that Ollama is running
3. Verify your PDF path is correct
4. Check the console output for error messages
