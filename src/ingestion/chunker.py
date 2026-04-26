"""
Text Chunking module for splitting documents into smaller pieces.
Uses recursive character splitting with configurable overlap.
"""
from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str


def create_chunks(
    text: str,
    source_filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The full text to split
        source_filename: Name of the source file for metadata
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of Chunk objects with metadata
    """
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
    
    Args:
        elements: List of parsed elements with page numbers
        source_filename: Name of the source file
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Chunk objects with page metadata
    """
    current_text = ""
    current_pages = set()
    chunks = []
    chunk_index = 0
    
    for element in elements:
        element_text = element.get("text", "")
        page = element.get("page", 1)
        
        if len(current_text) + len(element_text) > chunk_size and current_text:
            chunk = Chunk(
                text=current_text.strip(),
                metadata={
                    "source": source_filename,
                    "chunk_index": chunk_index,
                    "pages": sorted(list(current_pages)),
                },
                chunk_id=f"{source_filename}_{chunk_index}",
            )
            chunks.append(chunk)
            chunk_index += 1
            
            overlap_start = max(0, len(current_text) - chunk_overlap)
            current_text = current_text[overlap_start:]
            current_pages = {page}
        
        current_text += element_text + "\n\n"
        current_pages.add(page)
    
    if current_text.strip():
        chunk = Chunk(
            text=current_text.strip(),
            metadata={
                "source": source_filename,
                "chunk_index": chunk_index,
                "pages": sorted(list(current_pages)),
            },
            chunk_id=f"{source_filename}_{chunk_index}",
        )
        chunks.append(chunk)
    
    return chunks


def chunk_documents(parsed_docs: List[Any]) -> List[Chunk]:
    """
    Process multiple parsed documents into chunks.
    
    Args:
        parsed_docs: List of ParsedDocument objects
        
    Returns:
        List of all chunks from all documents
    """
    all_chunks = []
    
    for doc in parsed_docs:
        if doc.elements:
            chunks = create_chunks_with_elements(
                elements=doc.elements,
                source_filename=doc.filename,
            )
        else:
            chunks = create_chunks(
                text=doc.content,
                source_filename=doc.filename,
            )
        
        all_chunks.extend(chunks)
        print(f"  - {doc.filename}: {len(chunks)} chunks")
    
    return all_chunks
