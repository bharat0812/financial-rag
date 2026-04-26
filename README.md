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
│   ├── generation/      # LLM integration
│   └── pipeline.py      # End-to-end orchestration
├── app.py               # Streamlit UI
├── ingest.py            # Document ingestion script
└── requirements.txt
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
