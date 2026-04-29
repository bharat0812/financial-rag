"""
End-to-end RAG Pipeline.
Orchestrates retrieval and generation for question answering.

THIS IS THE ONLINE PIPELINE (runs on every user query):
- ingest.py = OFFLINE pipeline (build the index, runs once)
- pipeline.py = ONLINE pipeline (use the index, runs on each query)

THE FULL FLOW:
    Question → Embed → Vector Search (top 20) → Rerank (top 5) → LLM → Answer

WHY THIS DESIGN?
- Each component is independent and replaceable
- Easy to test, easy to swap (e.g., different LLM provider)
- Dependency injection makes it flexible
- Factory function provides simple default usage
"""
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import TOP_K_RETRIEVAL, TOP_K_RERANK
from .retrieval import VectorStore, Reranker
from .generation import LLM


class RAGPipeline:
    """
    Main RAG pipeline for question answering over financial documents.
    
    Composes VectorStore + Reranker + LLM into a single query() interface.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[LLM] = None,
        use_reranker: bool = True,
    ):
        """
        DEPENDENCY INJECTION: components can be passed in or auto-created.
        
        Why? Lets us:
        - Swap mocks for testing (pass MockLLM instead of LLM)
        - Share components across pipelines (one embedder, multiple stores)
        - A/B test different configurations (with/without reranker)
        """
        # `vector_store or VectorStore()`: Python's "or" returns the first truthy value.
        # If user passed a vector_store, use it; otherwise create a default.
        self.vector_store = vector_store or VectorStore()
        # use_reranker flag controls whether reranking happens at all
        self.reranker = reranker if use_reranker else None
        self.llm = llm or LLM()
        self.use_reranker = use_reranker
        
        # Edge case: user wants reranker but didn't provide one - create default
        if use_reranker and reranker is None:
            self.reranker = Reranker()
    
    def query(
        self,
        question: str,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
        top_k_rerank: int = TOP_K_RERANK,
        filter_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        The complete flow: retrieve → rerank → generate.
        Returns answer + sources + retrieval metadata for transparency.
        """
        # Optional metadata filtering - useful for "ask about specific document"
        # Translates to ChromaDB's where clause: {"source": "nvidia-10k.pdf"}
        filter_metadata = None
        if filter_source:
            filter_metadata = {"source": filter_source}
        
        # STAGE 1: RETRIEVE
        # Cast a wide net (top 20) - the reranker will narrow it down.
        # Wider net = better recall, but more work for the reranker.
        retrieved_docs = self.vector_store.search(
            query=question,
            top_k=top_k_retrieval,
            filter_metadata=filter_metadata,
        )
        
        # STAGE 2: RERANK (optional)
        # Cross-encoder is more accurate than embedding similarity.
        # Worth the extra latency for better precision.
        if self.use_reranker and self.reranker:
            reranked_docs = self.reranker.rerank(
                query=question,
                documents=retrieved_docs,
                top_k=top_k_rerank,
            )
        else:
            # No reranker: just take the top 5 from vector search
            reranked_docs = retrieved_docs[:top_k_rerank]
        
        # STAGE 3: GENERATE
        # LLM synthesizes a natural language answer from the top chunks.
        response = self.llm.generate(
            query=question,
            context_docs=reranked_docs,
        )
        
        # Attach pipeline metadata for debugging and the UI to display.
        # Useful for understanding "why did it return this answer?"
        response["retrieval_info"] = {
            "total_retrieved": len(retrieved_docs),
            "after_rerank": len(reranked_docs),
            "used_reranker": self.use_reranker,
        }
        
        return response
    
    def get_relevant_chunks(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get relevant chunks without generating an answer.
        
        Use cases:
        - Debugging: "what chunks does retrieval find for this question?"
        - Evaluation: test retrieval without LLM costs/latency
        - Building ground truth test sets
        """
        # top_k * 2 retrieves more candidates so reranker has options to choose from.
        # Rule of thumb: rerank 2-4x your final desired count.
        docs = self.vector_store.search(query=question, top_k=top_k * 2)
        
        if self.use_reranker and self.reranker:
            docs = self.reranker.rerank(question, docs, top_k=top_k)
        else:
            docs = docs[:top_k]
        
        return docs


def create_pipeline(
    use_reranker: bool = True,
    use_mock_llm: bool = False,
) -> RAGPipeline:
    """
    Factory function to create a configured RAG pipeline.
    
    DESIGN PATTERN: FACTORY
    Hides the complexity of wiring components together.
    Users just say what they want; the factory builds it.
    """
    # Local import: only loaded when needed, keeps top-level imports clean
    from .generation.llm import MockLLM
    
    # Create each component with sensible defaults
    vector_store = VectorStore()
    reranker = Reranker() if use_reranker else None
    # Ternary: use MockLLM for testing, real LLM for production
    llm = MockLLM() if use_mock_llm else LLM()
    
    return RAGPipeline(
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
        use_reranker=use_reranker,
    )
