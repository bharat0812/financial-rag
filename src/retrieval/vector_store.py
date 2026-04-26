"""
Vector Store module using ChromaDB.
Handles storage and retrieval of document embeddings.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..config import CHROMA_DIR, COLLECTION_NAME, TOP_K_RETRIEVAL
from ..embedding import Embedder
from ..ingestion.chunker import Chunk


class VectorStore:
    """
    ChromaDB-based vector store for document retrieval.
    """
    
    def __init__(
        self,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedder: Optional[Embedder] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedder: Embedder instance (creates new one if not provided)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            return
        
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedder.embed_documents(texts)
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            print(f"  Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    
    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with text, metadata, and score
        """
        query_embedding = self.embedder.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )
        
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "score": 1 - results["distances"][0][i] if results["distances"] else 1,
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
        }
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
