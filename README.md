# Financial Document RAG System

A Retrieval-Augmented Generation (RAG) system for querying financial documents (SEC 10-K filings).

## Architecture

```
PDF Documents → Parser → Chunker → Embeddings → ChromaDB
                                                    ↓
User Query → Embed → Vector Search → Rerank → LLM → Answer
```

## Features

- **Document Ingestion**: Parse PDFs with layout awareness using Unstructured
- **Smart Chunking**: Recursive text splitting with overlap for context preservation
- **Local Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Storage**: ChromaDB with persistent storage
- **Re-ranking**: Cross-encoder for improved retrieval precision
- **LLM Generation**: Google Gemini for answer synthesis
- **Web UI**: Streamlit interface for interactive Q&A

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bharat0812/financial-rag.git
   cd financial-rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or: source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   copy .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

5. **Add documents**
   - Place PDF files in `data/documents/`

6. **Ingest documents**
   ```bash
   python ingest.py
   ```

7. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
financial-rag/
├── data/documents/      # Place your PDFs here
├── src/
│   ├── ingestion/       # PDF parsing and chunking
│   ├── embedding/       # Embedding model wrapper
│   ├── retrieval/       # Vector store and reranking
│   ├── generation/      # LLM integration (Gemini)
│   ├── evaluation/      # Evaluation and experiment tracking
│   └── pipeline.py      # End-to-end orchestration
├── evaluation_data/     # Test sets for evaluation
├── experiments/         # Experiment logs and results
├── app.py               # Streamlit UI
├── ingest.py            # Document ingestion script
├── evaluate.py          # Evaluation and experiment script
└── requirements.txt
```

## Evaluation & Experiment Tracking

Track and compare different configurations to optimize your RAG system.

### Parameters You Can Tune

| Parameter | Description | Values to Try |
|-----------|-------------|---------------|
| `chunk_size` | Characters per chunk | 400, 600, 800, 1000, 1200 |
| `chunk_overlap` | Overlap between chunks | 100, 200, 300 |
| `top_k_retrieval` | Chunks to retrieve initially | 10, 20, 30 |
| `top_k_rerank` | Chunks after reranking | 3, 5, 10 |
| `use_reranker` | Enable cross-encoder reranking | True, False |
| `embedding_model` | Sentence transformer model | all-MiniLM-L6-v2, all-mpnet-base-v2 |
| `llm_model` | LLM for generation | gemini-2.5-flash |

### Running Evaluations

```bash
# Run a single experiment with default config
python evaluate.py

# Run with custom parameters
python evaluate.py --chunk-size 600 --top-k 10 --no-reranker

# Run parameter sweep (tests multiple combinations)
python evaluate.py --sweep

# Compare all experiments
python evaluate.py --compare

# Export results to CSV for analysis
python evaluate.py --export-csv

# Generate test questions using LLM
python evaluate.py --generate-tests
```

### Metrics Tracked

- **Precision@K**: Of retrieved chunks, how many are relevant?
- **Recall@K**: Of all relevant chunks, how many were retrieved?
- **Latency**: Response time in milliseconds
- **Answer Score**: LLM-judged answer quality (optional)

### Example Workflow

```bash
# 1. Generate test questions
python evaluate.py --generate-tests

# 2. Edit evaluation_data/test_set.json with expected answers

# 3. Run parameter sweep
python evaluate.py --sweep

# 4. Compare results
python evaluate.py --compare

# 5. Export to CSV for detailed analysis
python evaluate.py --export-csv
```

## GCP Production Mapping

| Local Component | GCP Equivalent |
|-----------------|----------------|
| all-MiniLM-L6-v2 | Vertex AI text-embedding-004 |
| ChromaDB | Vertex AI Vector Search |
| Gemini API | Vertex AI Gemini Pro |
| Local Streamlit | Cloud Run |

## License

MIT
