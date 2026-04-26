"""
Re-ranking module using Cross-Encoder models.
Improves retrieval precision by re-scoring candidate documents.
"""
from typing import List, Dict, Any

from ..config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:
    """
    Cross-encoder based reranker for improving retrieval results.
    """
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = TOP_K_RERANK,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents with 'text' field
            top_k: Number of top results to return
            
        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []
        
        pairs = [[query, doc["text"]] for doc in documents]
        
        scores = self.model.predict(pairs)
        
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
        
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]


class NoOpReranker:
    """
    No-op reranker that just returns top-k results without reranking.
    Useful for faster inference when reranking is not needed.
    """
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = TOP_K_RERANK,
    ) -> List[Dict[str, Any]]:
        """Return top-k documents without reranking."""
        return documents[:top_k]
