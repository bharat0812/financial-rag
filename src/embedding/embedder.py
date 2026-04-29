"""
Embedding module using Sentence Transformers.
Converts text into dense vector representations.

WHAT ARE EMBEDDINGS?
- Numerical representation of text where similar meanings have similar vectors
- "revenue" and "sales" → similar vectors (semantic similarity)
- "revenue" and "purple" → very different vectors
- Enables "find similar" search instead of just keyword matching

WHY all-MiniLM-L6-v2?
- Small (~80MB) and fast (~1000 texts/sec on CPU)
- Good quality for general English text
- 384-dimensional vectors (manageable size)
- Free and runs locally (no API calls, no data leakage)

ALTERNATIVES:
- OpenAI text-embedding-3-small: better quality, costs money, requires network
- BGE / E5 models: better quality but bigger
- Multilingual models: paraphrase-multilingual-MiniLM-L12-v2
"""
from typing import List, Union
import numpy as np

from ..config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class Embedder:
    """
    Wrapper for embedding models.
    
    Uses the WRAPPER pattern - hides the SentenceTransformer implementation
    behind a clean interface. Easy to swap to a different embedding model later.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        # Don't load the model yet - see the @property below for why
        self._model = None
    
    @property
    def model(self):
        """
        Lazy load the model to avoid slow imports.
        
        LAZY LOADING PATTERN:
        - SentenceTransformer takes ~2 seconds to load
        - If we load in __init__, every Embedder() call costs 2s even if unused
        - Instead, load only on first access; store and reuse afterwards
        - @property makes self.model look like an attribute but runs this code
        """
        if self._model is None:
            # Local import: heavy library only loaded when actually needed
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension. ChromaDB needs this for index setup."""
        return EMBEDDING_DIMENSION
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        Accepts both single string and list for caller convenience.
        Returns numpy array of shape (n_texts, dimension).
        """
        # Normalize input: model.encode() expects a list, even for a single text
        if isinstance(texts, str):
            texts = [texts]
        
        # encode() does: tokenize → forward pass through transformer → pool → normalize
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 10,  # Only show bar for large batches
            convert_to_numpy=True,              # Return numpy array, not torch tensor
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        Returns 1D array (just the vector, not wrapped in a list).
        """
        # [0] unwraps the single-element batch back to a single vector
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents.
        Returns 2D array of shape (n_docs, dimension).
        Just a clearly-named alias for embed() to make caller code self-documenting.
        """
        return self.embed(documents)
