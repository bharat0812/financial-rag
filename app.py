"""
Streamlit Web Application for Financial Document Q&A.

WHY STREAMLIT?
- Build web UIs in pure Python (no JS/HTML/CSS)
- Perfect for ML prototypes and internal tools
- Trade-off: less customizable than React, less scalable
- For production: would migrate to FastAPI backend + React frontend

UI FLOW:
- Sidebar: settings (reranker on/off, mock mode)
- Main area: question input + example buttons + answer display
- Sources are shown with expanders for transparency (verify the answer)

Usage:
    streamlit run app.py
"""
import streamlit as st
from pathlib import Path

from src.config import GOOGLE_API_KEY
from src.pipeline import RAGPipeline, create_pipeline
from src.retrieval import VectorStore


# st.set_page_config must be called BEFORE any other st commands.
# Sets the browser tab title, favicon, and layout.
st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="wide",  # Use full browser width instead of narrow centered column
)


def load_pipeline(use_reranker: bool = True, use_mock: bool = False) -> RAGPipeline:
    """Load the RAG pipeline. Thin wrapper around the factory function."""
    return create_pipeline(use_reranker=use_reranker, use_mock_llm=use_mock)


def get_collection_stats():
    """
    Get vector store statistics.
    
    Wrapped in try/except so the app loads even if ChromaDB has issues
    (e.g., first run, corrupted index, missing directory).
    """
    try:
        vs = VectorStore()
        return vs.get_collection_stats()
    except Exception:
        return {"count": 0, "name": "Not initialized"}


def main():
    # Page header - st.title is for the biggest heading
    st.title("📊 Financial Document Q&A")
    st.markdown("Ask questions about your financial documents using RAG.")
    
    # SIDEBAR: settings and metadata that stay visible while users interact
    with st.sidebar:
        st.header("Settings")
        
        # Toggle reranking - lets users see speed vs accuracy trade-off
        use_reranker = st.checkbox("Use Re-ranker", value=True, help="Improves accuracy but slower")
        # Auto-enable mock mode if no API key (so app still works for demo)
        # `not bool(GOOGLE_API_KEY)`: empty string → False → not False → True
        use_mock = st.checkbox("Mock Mode (no API)", value=not bool(GOOGLE_API_KEY), help="Test without Gemini API")
        
        st.divider()
        
        # Show current state of vector store - helps users know if ingestion ran
        st.header("Collection Info")
        stats = get_collection_stats()
        st.metric("Documents Indexed", stats["count"])
        
        if stats["count"] == 0:
            st.warning("No documents indexed. Run `python ingest.py` first.")
        
        st.divider()
        
        # Architecture summary - useful for demos and self-documentation
        st.header("About")
        st.markdown("""
        **Architecture:**
        - Embeddings: all-MiniLM-L6-v2
        - Vector DB: ChromaDB
        - Reranker: Cross-Encoder
        - LLM: Gemini 2.5 Flash
        
        **How it works:**
        1. Your question is embedded
        2. Similar chunks are retrieved
        3. Chunks are re-ranked for relevance
        4. LLM generates answer from context
        """)
    
    # EMPTY STATE: show instructions if no documents indexed yet
    # Better UX than a broken-looking search box
    if stats["count"] == 0:
        st.info("👆 Please ingest documents first using `python ingest.py`")
        
        st.subheader("Quick Start")
        st.code("""
# 1. Add PDF files to data/documents/

# 2. Run ingestion
python ingest.py

# 3. Refresh this page
        """, language="bash")
        return  # Early return - don't render the search interface
    
    # Build pipeline AFTER we know there's data to query
    pipeline = load_pipeline(use_reranker=use_reranker, use_mock=use_mock)
    
    # SEARCH INTERFACE
    st.subheader("Ask a Question")
    
    # Two-column layout: 80% for input, 20% for button (more text room)
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="What was the total revenue for the fiscal year?",
            label_visibility="collapsed",  # Hide label since placeholder is enough
        )
    with col2:
        # type="primary" makes it the prominent button (blue)
        # use_container_width=True makes button fill its column
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # EXAMPLE QUESTIONS: lower the friction for users to try the system
    example_questions = [
        "What was the total revenue?",
        "What are the main risk factors?",
        "How did operating expenses change?",
        "What is the company's growth strategy?",
    ]
    
    st.markdown("**Example questions:**")
    # One column per example - laid out side-by-side
    cols = st.columns(len(example_questions))
    for i, q in enumerate(example_questions):
        # key= is required when generating buttons in a loop (Streamlit needs unique IDs)
        if cols[i].button(q, key=f"example_{i}", use_container_width=True):
            # Clicking an example fills the query and triggers search
            query = q
            search_button = True
    
    # Process query if user submitted something
    if query and search_button:
        # st.spinner shows a loading animation while the block runs
        with st.spinner("Searching and generating answer..."):
            try:
                # The actual RAG call - everything we built leads to this
                result = pipeline.query(
                    question=query,
                    top_k_retrieval=20,
                    top_k_rerank=5,
                )
                
                # ANSWER: render LLM output (markdown supports formatting)
                st.subheader("Answer")
                st.markdown(result["answer"])
                
                # SOURCES: critical for trust - users can verify the answer
                # Each source is collapsible to keep the UI clean
                st.subheader("Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"📄 {source['source']} (Pages: {source.get('pages', 'N/A')})"):
                        st.markdown(source["text_preview"])
                
                # RETRIEVAL DETAILS: shows pipeline internals for debugging/transparency
                with st.expander("Retrieval Details"):
                    info = result.get("retrieval_info", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieved", info.get("total_retrieved", "N/A"))
                    col2.metric("After Rerank", info.get("after_rerank", "N/A"))
                    col3.metric("Reranker Used", "Yes" if info.get("used_reranker") else "No")
                    
            except Exception as e:
                # Catch all errors so the app doesn't crash on API failures.
                # Show actionable advice (most common issue is missing API key).
                st.error(f"Error: {str(e)}")
                st.info("Make sure you have set GOOGLE_API_KEY in your .env file, or enable Mock Mode.")


# Streamlit runs this file top-to-bottom on every interaction (re-runs the script).
# This idiom still works for compatibility with `python app.py`.
if __name__ == "__main__":
    main()
