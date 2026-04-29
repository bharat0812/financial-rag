"""
Re-ranking module using Cross-Encoder models.
Improves retrieval precision by re-scoring candidate documents.

THE TWO-STAGE RETRIEVAL PATTERN:
- Stage 1 (vector search): fast but approximate. Cast a wide net (top 20).
- Stage 2 (reranking): slow but precise. Refine to top 5.

WHY NOT JUST USE THE RERANKER?
- Cross-encoder must process every (query, doc) pair separately
- For 1000 documents → 1000 forward passes → too slow
- Bi-encoder (embeddings) precomputes once, then search is fast

BI-ENCODER vs CROSS-ENCODER:
- Bi-encoder: encode query and doc SEPARATELY → compare vectors (fast)
- Cross-encoder: encode query+doc TOGETHER → output score (accurate)
- Cross-encoder captures word-level interactions between query and document
- Standard pattern in production search: bi-encoder retrieve + cross-encoder rerank

MODEL: ms-marco-MiniLM-L-6-v2
- Trained on Microsoft MARCO dataset (real Bing search queries)
- Knows what "relevant to a search query" looks like
- Small and fast enough for online reranking
"""
from typing import List, Dict, Any

from ..config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:
    """
    Cross-encoder based reranker for improving retrieval results.
    """
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None  # Lazy load - same pattern as Embedder
    
    @property
    def model(self):
        """Lazy load the model. See Embedder for full explanation of the pattern."""
        if self._model is None:
            # CrossEncoder is different from SentenceTransformer:
            # - SentenceTransformer: text → vector
            # - CrossEncoder: (text1, text2) → relevance score
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
        
        Takes the top-N from vector search and returns top-K with refined ordering.
        """
        # Defensive guard: empty input → empty output (don't crash)
        if not documents:
            return []
        
        # Build (query, document) pairs for the cross-encoder.
        # Same query repeated, paired with each candidate document.
        pairs = [[query, doc["text"]] for doc in documents]
        
        # predict() runs the cross-encoder on all pairs in a single batch.
        # Returns a relevance score for each pair (higher = more relevant).
        scores = self.model.predict(pairs)
        
        # Attach rerank score to each document for later inspection/debugging.
        # float() converts from numpy scalar - cleaner for JSON serialization.
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
        
        # Sort by rerank score descending (highest relevance first).
        # lambda is an inline function - common Python idiom for sort keys.
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return only the top K - the whole point of reranking is filtering down
        return reranked[:top_k]


class NoOpReranker:
    """
    No-op reranker that just returns top-k results without reranking.
    
    DESIGN PATTERN: STRATEGY (or NULL OBJECT)
    Same interface as Reranker but does nothing useful.
    Allows easy A/B testing - swap implementations without changing pipeline code.
    
    Use cases:
    - Latency-sensitive paths where reranking is too slow
    - Measuring reranker impact (compare with vs without)
    - Testing the rest of the pipeline in isolation
    """
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = TOP_K_RERANK,
    ) -> List[Dict[str, Any]]:
        """Return top-k documents without reranking - just slice the list."""
        return documents[:top_k]
