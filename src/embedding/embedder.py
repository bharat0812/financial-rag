"""
Embedding module using Sentence Transformers.
Converts text into dense vector representations.
"""
from typing import List, Union
import numpy as np

from ..config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class Embedder:
    """
    Wrapper for embedding models.
    Uses Sentence Transformers for local embedding generation.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model to avoid slow imports."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIMENSION
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Numpy array of embeddings, shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: The query text
            
        Returns:
            1D numpy array of the embedding
        """
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            2D numpy array of embeddings
        """
        return self.embed(documents)
