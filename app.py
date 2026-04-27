"""
Streamlit Web Application for Financial Document Q&A.

Usage:
    streamlit run app.py
"""
import streamlit as st
from pathlib import Path

from src.config import GOOGLE_API_KEY
from src.pipeline import RAGPipeline, create_pipeline
from src.retrieval import VectorStore


st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="wide",
)


def load_pipeline(use_reranker: bool = True, use_mock: bool = False) -> RAGPipeline:
    """Load the RAG pipeline."""
    return create_pipeline(use_reranker=use_reranker, use_mock_llm=use_mock)


def get_collection_stats():
    """Get vector store statistics."""
    try:
        vs = VectorStore()
        return vs.get_collection_stats()
    except Exception:
        return {"count": 0, "name": "Not initialized"}


def main():
    st.title("📊 Financial Document Q&A")
    st.markdown("Ask questions about your financial documents using RAG.")
    
    with st.sidebar:
        st.header("Settings")
        
        use_reranker = st.checkbox("Use Re-ranker", value=True, help="Improves accuracy but slower")
        use_mock = st.checkbox("Mock Mode (no API)", value=not bool(GOOGLE_API_KEY), help="Test without Gemini API")
        
        st.divider()
        
        st.header("Collection Info")
        stats = get_collection_stats()
        st.metric("Documents Indexed", stats["count"])
        
        if stats["count"] == 0:
            st.warning("No documents indexed. Run `python ingest.py` first.")
        
        st.divider()
        
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
    
    if stats["count"] == 0:
        st.info("👆 Please ingest documents first using `python ingest.py`")
        
        st.subheader("Quick Start")
        st.code("""
# 1. Add PDF files to data/documents/

# 2. Run ingestion
python ingest.py

# 3. Refresh this page
        """, language="bash")
        return
    
    pipeline = load_pipeline(use_reranker=use_reranker, use_mock=use_mock)
    
    st.subheader("Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="What was the total revenue for the fiscal year?",
            label_visibility="collapsed",
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    example_questions = [
        "What was the total revenue?",
        "What are the main risk factors?",
        "How did operating expenses change?",
        "What is the company's growth strategy?",
    ]
    
    st.markdown("**Example questions:**")
    cols = st.columns(len(example_questions))
    for i, q in enumerate(example_questions):
        if cols[i].button(q, key=f"example_{i}", use_container_width=True):
            query = q
            search_button = True
    
    if query and search_button:
        with st.spinner("Searching and generating answer..."):
            try:
                result = pipeline.query(
                    question=query,
                    top_k_retrieval=20,
                    top_k_rerank=5,
                )
                
                st.subheader("Answer")
                st.markdown(result["answer"])
                
                st.subheader("Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"📄 {source['source']} (Pages: {source.get('pages', 'N/A')})"):
                        st.markdown(source["text_preview"])
                
                with st.expander("Retrieval Details"):
                    info = result.get("retrieval_info", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieved", info.get("total_retrieved", "N/A"))
                    col2.metric("After Rerank", info.get("after_rerank", "N/A"))
                    col3.metric("Reranker Used", "Yes" if info.get("used_reranker") else "No")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure you have set GOOGLE_API_KEY in your .env file, or enable Mock Mode.")


if __name__ == "__main__":
    main()
