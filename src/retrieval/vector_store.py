"""
Vector Store module using ChromaDB.
Handles storage and retrieval of document embeddings.

WHY A VECTOR DATABASE?
- Regular DBs (Postgres, MySQL) optimize for exact match and range queries
- Vector DBs optimize for "find K most similar" queries
- They use approximate nearest neighbor (ANN) algorithms for speed:
  - Brute force: compare query to ALL vectors → O(n)
  - HNSW (what ChromaDB uses): build a graph, traverse smartly → O(log n)

WHY ChromaDB?
- Open source, runs locally, persistent storage
- Easy Python API, good for prototypes
- Production alternatives: Pinecone, Weaviate, Qdrant, Vertex AI Vector Search

KEY CONCEPT: COSINE SIMILARITY
- Measures angle between vectors, ignores magnitude
- Score 1.0 = identical direction (same meaning)
- Score 0.0 = perpendicular (unrelated)
- Score -1.0 = opposite (rarely happens with embeddings)
- Better than Euclidean for text because document length doesn't skew results
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
    Wraps ChromaDB to provide a clean interface for our pipeline.
    """
    
    def __init__(
        self,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedder: Optional[Embedder] = None,
    ):
        """
        DEPENDENCY INJECTION: embedder can be passed in or auto-created.
        Lets us share one embedder across components (saves memory) or
        inject a mock embedder for testing.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        # If no embedder provided, create a default one
        self.embedder = embedder or Embedder()
        
        # Create directory if it doesn't exist (parents=True handles nested paths)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # PersistentClient saves to disk - data survives restarts.
        # InMemoryClient exists too, but loses data on restart.
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),  # Disable phone-home
        )
        
        # get_or_create handles both first-run and subsequent runs in one call.
        # "hnsw:space": "cosine" tells ChromaDB to use cosine similarity for search.
        # Other options: "l2" (Euclidean), "ip" (inner product).
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store.
        
        Process: extract data → embed all texts → insert in batches.
        """
        if not chunks:
            return  # Defensive guard against empty input
        
        # List comprehensions extract parallel arrays from the chunk objects.
        # ChromaDB's API takes parallel lists, not a list of structs.
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Embed ALL texts at once - much faster than one at a time (vectorized)
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedder.embed_documents(texts)
        
        # BATCHING PATTERN:
        # - Inserting 10K+ chunks at once can OOM or hit ChromaDB limits
        # - Batch size 100 keeps memory stable and shows progress
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))  # Don't go past array end
            self.collection.add(
                ids=ids[i:batch_end],
                # .tolist() converts numpy array to Python list (ChromaDB requirement)
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
        
        This is the heart of retrieval - converts query to vector,
        finds K nearest neighbors, formats results.
        """
        # Step 1: Embed the query into a vector
        query_embedding = self.embedder.embed_query(query)
        
        # Step 2: Search ChromaDB
        # query_embeddings is a list because ChromaDB supports batch queries
        # (one query at a time here, but the API expects a list)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,  # Optional: filter by metadata, e.g., {"source": "nvidia.pdf"}
            include=["documents", "metadatas", "distances"],  # What fields to return
        )
        
        # Step 3: Format ChromaDB's nested response into a flat list
        # ChromaDB returns nested arrays (one inner list per query) - we unwrap
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    # Convert distance (lower=better) to score (higher=better) for caller convenience
                    "score": 1 - results["distances"][0][i] if results["distances"] else 1,
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection - used by app.py for the UI."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
        }
    
    def clear(self) -> None:
        """
        Clear all documents from the collection.
        
        Drop-and-recreate is faster than deleting docs one-by-one.
        Used by ingest.py --clear before re-ingesting with new chunk sizes.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
