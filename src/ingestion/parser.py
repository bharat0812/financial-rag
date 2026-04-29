"""
PDF Document Parser using Unstructured library.
Extracts text and metadata from financial documents.

WHY PARSING IS HARD:
- PDFs are layout-based, not content-based (designed for printing)
- Tables, columns, headers, footers all need to be handled
- Page numbers need to be tracked for citations
- Different PDFs have different structures (scanned vs. text-based)

DESIGN DECISIONS:
- Primary: 'unstructured' library (handles complex layouts well)
- Fallback: PyMuPDF (faster but simpler extraction)
- Use 'fast' strategy (skip OCR) - assumes text-based PDFs
- Track page numbers in metadata for source citations
"""
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    """
    Represents a parsed document with its content and metadata.
    
    Two views of the same content:
    - content: full text (used by simple chunker)
    - elements: list of structured pieces with page numbers (used by smart chunker)
    """
    filename: str
    content: str                       # Concatenated full text
    elements: List[Dict[str, Any]]     # Structured elements with page metadata
    page_count: int


def parse_pdf(file_path: Path) -> ParsedDocument:
    """
    Parse a PDF file and extract text content with structure awareness.
    
    Returns ParsedDocument with both full text and per-element structure.
    Falls back to PyMuPDF if 'unstructured' isn't installed.
    """
    # Defensive import: 'unstructured' has heavy dependencies that may fail
    # on some systems. We provide a fallback so the app still works.
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        return _parse_pdf_fallback(file_path)
    
    # 'fast' strategy = no OCR, just text extraction.
    # Other options: 'hi_res' (better for scanned PDFs but slow),
    # 'ocr_only' (forces OCR), 'auto' (decides based on content).
    elements = partition_pdf(
        filename=str(file_path),
        strategy="fast",
        include_page_breaks=True,  # Needed to track page transitions
    )
    
    parsed_elements = []
    full_text_parts = []
    current_page = 1
    
    for element in elements:
        # Convert each element to a simple dict for downstream processing
        element_dict = {
            "type": type(element).__name__,  # e.g., "Title", "NarrativeText", "Table"
            "text": str(element),
            "page": current_page,
        }
        
        # Some elements know their actual page number; trust that over our counter
        if hasattr(element, "metadata"):
            if hasattr(element.metadata, "page_number") and element.metadata.page_number:
                current_page = element.metadata.page_number
                element_dict["page"] = current_page
        
        # PageBreak is a marker, not content. Bump the counter and skip.
        if element_dict["type"] == "PageBreak":
            current_page += 1
            continue
            
        parsed_elements.append(element_dict)
        full_text_parts.append(str(element))
    
    # Double newline preserves paragraph boundaries for the chunker
    full_text = "\n\n".join(full_text_parts)
    
    return ParsedDocument(
        filename=file_path.name,
        content=full_text,
        elements=parsed_elements,
        page_count=current_page,
    )


def _parse_pdf_fallback(file_path: Path) -> ParsedDocument:
    """
    Fallback PDF parser using PyMuPDF when Unstructured is not available.
    
    Simpler approach: each page becomes one element. Loses fine-grained
    structure (titles, tables) but still preserves page boundaries.
    Underscore prefix = "private" function (Python convention).
    """
    import fitz  # PyMuPDF - lighter weight than unstructured
    
    doc = fitz.open(file_path)
    full_text_parts = []
    elements = []
    
    # enumerate(..., start=1) starts page count from 1, not 0 (matches human reading)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text_parts.append(text)
        elements.append({
            "type": "Page",
            "text": text,
            "page": page_num,
        })
    
    doc.close()  # Always close PDF handles to free memory
    
    return ParsedDocument(
        filename=file_path.name,
        content="\n\n".join(full_text_parts),
        elements=elements,
        page_count=len(elements),
    )


def parse_directory(directory: Path) -> List[ParsedDocument]:
    """
    Parse all PDF files in a directory.
    Used by ingest.py when ingesting a whole folder of documents.
    """
    # glob("*.pdf") finds all .pdf files (case-sensitive on Linux)
    pdf_files = list(directory.glob("*.pdf"))
    documents = []
    
    for pdf_file in pdf_files:
        print(f"Parsing: {pdf_file.name}")
        doc = parse_pdf(pdf_file)
        documents.append(doc)
        print(f"  - Extracted {len(doc.elements)} elements, {doc.page_count} pages")
    
    return documents
