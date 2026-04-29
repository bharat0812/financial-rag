"""
Text Chunking module for splitting documents into smaller pieces.
Uses recursive character splitting with configurable overlap.

WHY CHUNKING MATTERS:
- LLMs have limited context windows (can't fit a whole 10-K filing)
- Embeddings work better on focused topics, not entire documents
- Smaller chunks → more precise retrieval
- Trade-off: too small = lose context, too large = retrieve irrelevant info

DESIGN DECISIONS:
- 800 char chunks with 200 char overlap (default)
- Overlap prevents losing context at chunk boundaries
- Page metadata is preserved for citation in answers
"""
from typing import List, Dict, Any
from dataclasses import dataclass

# RecursiveCharacterTextSplitter tries natural boundaries first (paragraphs,
# then sentences, then words) before falling back to character splits.
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.
    
    Using @dataclass auto-generates __init__, __repr__, __eq__ methods.
    Cleaner than writing a regular class for simple data containers.
    """
    text: str                  # The actual chunk content
    metadata: Dict[str, Any]   # Source file, page numbers, chunk_index
    chunk_id: str              # Unique ID like "nvidia-10k.pdf_42"


def create_chunks(
    text: str,
    source_filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split plain text into overlapping chunks.
    Used as fallback when parser doesn't provide page-level elements.
    """
    # Separator hierarchy: try paragraph breaks first, fall back to characters.
    # This keeps semantic units together when possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    text_chunks = splitter.split_text(text)
    
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk = Chunk(
            text=chunk_text,
            metadata={
                "source": source_filename,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            },
            chunk_id=f"{source_filename}_{i}",
        )
        chunks.append(chunk)
    
    return chunks


def create_chunks_with_elements(
    elements: List[Dict[str, Any]],
    source_filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Create chunks from parsed document elements, preserving page information.
    
    KEY FIX: Originally treated each element (page) as one chunk regardless of
    size, making chunk_size parameter useless. Now properly splits large
    elements while keeping page-level metadata for citations.
    
    This matters for evaluation - we can compare different chunk sizes
    and see real differences in retrieval quality.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    chunk_index = 0
    
    for element in elements:
        element_text = element.get("text", "")
        page = element.get("page", 1)
        
        # Skip empty/whitespace-only elements (common in PDFs with weird formatting)
        if not element_text.strip():
            continue
        
        # Small element fits in one chunk - preserve as-is
        if len(element_text) <= chunk_size:
            chunk = Chunk(
                text=element_text.strip(),
                metadata={
                    "source": source_filename,
                    "chunk_index": chunk_index,
                    "pages": [page],  # List allows future multi-page chunks
                },
                chunk_id=f"{source_filename}_{chunk_index}",
            )
            chunks.append(chunk)
            chunk_index += 1
        else:
            # Large element (typically a full page) needs splitting.
            # Each sub-chunk inherits the parent page number.
            sub_chunks = splitter.split_text(element_text)
            for sub_chunk in sub_chunks:
                chunk = Chunk(
                    text=sub_chunk.strip(),
                    metadata={
                        "source": source_filename,
                        "chunk_index": chunk_index,
                        "pages": [page],
                    },
                    chunk_id=f"{source_filename}_{chunk_index}",
                )
                chunks.append(chunk)
                chunk_index += 1
    
    return chunks


def chunk_documents(
    parsed_docs: List[Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Process multiple parsed documents into chunks.
    
    Routes to the appropriate chunking function based on whether the parser
    extracted structured elements (pages) or just plain text.
    """
    all_chunks = []
    
    for doc in parsed_docs:
        # Use page-aware chunking when available - preserves citation metadata
        if doc.elements:
            chunks = create_chunks_with_elements(
                elements=doc.elements,
                source_filename=doc.filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            # Fallback path: parser only gave us raw text (no structure)
            chunks = create_chunks(
                text=doc.content,
                source_filename=doc.filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        
        all_chunks.extend(chunks)
        print(f"  - {doc.filename}: {len(chunks)} chunks")
    
    return all_chunks
