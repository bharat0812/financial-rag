"""
PDF Document Parser using Unstructured library.
Extracts text and metadata from financial documents.
"""
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    """Represents a parsed document with its content and metadata."""
    filename: str
    content: str
    elements: List[Dict[str, Any]]
    page_count: int


def parse_pdf(file_path: Path) -> ParsedDocument:
    """
    Parse a PDF file and extract text content with structure awareness.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        ParsedDocument with extracted content and metadata
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        return _parse_pdf_fallback(file_path)
    
    elements = partition_pdf(
        filename=str(file_path),
        strategy="fast",
        include_page_breaks=True,
    )
    
    parsed_elements = []
    full_text_parts = []
    current_page = 1
    
    for element in elements:
        element_dict = {
            "type": type(element).__name__,
            "text": str(element),
            "page": current_page,
        }
        
        if hasattr(element, "metadata"):
            if hasattr(element.metadata, "page_number") and element.metadata.page_number:
                current_page = element.metadata.page_number
                element_dict["page"] = current_page
        
        if element_dict["type"] == "PageBreak":
            current_page += 1
            continue
            
        parsed_elements.append(element_dict)
        full_text_parts.append(str(element))
    
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
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(file_path)
    full_text_parts = []
    elements = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text_parts.append(text)
        elements.append({
            "type": "Page",
            "text": text,
            "page": page_num,
        })
    
    doc.close()
    
    return ParsedDocument(
        filename=file_path.name,
        content="\n\n".join(full_text_parts),
        elements=elements,
        page_count=len(elements),
    )


def parse_directory(directory: Path) -> List[ParsedDocument]:
    """
    Parse all PDF files in a directory.
    
    Args:
        directory: Path to directory containing PDFs
        
    Returns:
        List of ParsedDocument objects
    """
    pdf_files = list(directory.glob("*.pdf"))
    documents = []
    
    for pdf_file in pdf_files:
        print(f"Parsing: {pdf_file.name}")
        doc = parse_pdf(pdf_file)
        documents.append(doc)
        print(f"  - Extracted {len(doc.elements)} elements, {doc.page_count} pages")
    
    return documents
