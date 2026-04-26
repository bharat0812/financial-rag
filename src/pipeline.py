"""
End-to-end RAG Pipeline.
Orchestrates retrieval and generation for question answering.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import TOP_K_RETRIEVAL, TOP_K_RERANK
from .retrieval import VectorStore, Reranker
from .generation import LLM


class RAGPipeline:
    """
    Main RAG pipeline for question answering over financial documents.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[LLM] = None,
        use_reranker: bool = True,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: VectorStore instance
            reranker: Reranker instance
            llm: LLM instance
            use_reranker: Whether to use reranking (slower but more accurate)
        """
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker if use_reranker else None
        self.llm = llm or LLM()
        self.use_reranker = use_reranker
        
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
        
        Args:
            question: User's question
            top_k_retrieval: Number of documents to retrieve
            top_k_rerank: Number of documents after reranking
            filter_source: Optional filter by source document
            
        Returns:
            Dict with answer, sources, and retrieval info
        """
        filter_metadata = None
        if filter_source:
            filter_metadata = {"source": filter_source}
        
        retrieved_docs = self.vector_store.search(
            query=question,
            top_k=top_k_retrieval,
            filter_metadata=filter_metadata,
        )
        
        if self.use_reranker and self.reranker:
            reranked_docs = self.reranker.rerank(
                query=question,
                documents=retrieved_docs,
                top_k=top_k_rerank,
            )
        else:
            reranked_docs = retrieved_docs[:top_k_rerank]
        
        response = self.llm.generate(
            query=question,
            context_docs=reranked_docs,
        )
        
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
        Useful for debugging retrieval.
        
        Args:
            question: The query
            top_k: Number of chunks to return
            
        Returns:
            List of relevant chunks with scores
        """
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
    
    Args:
        use_reranker: Whether to use the reranker
        use_mock_llm: Use mock LLM (for testing without API)
        
    Returns:
        Configured RAGPipeline instance
    """
    from .generation.llm import MockLLM
    
    vector_store = VectorStore()
    reranker = Reranker() if use_reranker else None
    llm = MockLLM() if use_mock_llm else LLM()
    
    return RAGPipeline(
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
        use_reranker=use_reranker,
    )
