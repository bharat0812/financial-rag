"""
Document Ingestion Script.
Parses PDFs and stores embeddings in the vector database.

Usage:
    python ingest.py                    # Ingest all PDFs in data/documents/
    python ingest.py --clear            # Clear existing data and re-ingest
    python ingest.py --file doc.pdf     # Ingest a specific file
"""
import argparse
from pathlib import Path

from src.config import DOCUMENTS_DIR
from src.ingestion import parse_pdf, chunk_documents
from src.ingestion.parser import parse_directory
from src.retrieval import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest financial documents into vector store")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingesting")
    parser.add_argument("--file", type=str, help="Ingest a specific PDF file")
    parser.add_argument("--dir", type=str, help="Directory containing PDFs (default: data/documents/)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size (default from config)")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk overlap (default from config)")
    args = parser.parse_args()
    
    documents_dir = Path(args.dir) if args.dir else DOCUMENTS_DIR
    
    print("=" * 50)
    print("Financial Document RAG - Ingestion")
    print("=" * 50)
    
    vector_store = VectorStore()
    
    if args.clear:
        print("\nClearing existing data...")
        vector_store.clear()
        print("Done.")
    
    stats = vector_store.get_collection_stats()
    print(f"\nCurrent collection: {stats['name']} ({stats['count']} chunks)")
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
        print(f"\nParsing: {file_path}")
        parsed_docs = [parse_pdf(file_path)]
    else:
        pdf_files = list(documents_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"\nNo PDF files found in {documents_dir}")
            print("Please add PDF files to the data/documents/ directory.")
            return
        
        print(f"\nFound {len(pdf_files)} PDF file(s) in {documents_dir}")
        parsed_docs = parse_directory(documents_dir)
    
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP
    chunk_size = args.chunk_size or CHUNK_SIZE
    chunk_overlap = args.chunk_overlap or CHUNK_OVERLAP
    
    print(f"\nChunking {len(parsed_docs)} document(s)...")
    print(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    chunks = chunk_documents(parsed_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Total chunks: {len(chunks)}")
    
    print(f"\nAdding chunks to vector store...")
    vector_store.add_chunks(chunks)
    
    stats = vector_store.get_collection_stats()
    print(f"\nFinal collection: {stats['name']} ({stats['count']} chunks)")
    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
