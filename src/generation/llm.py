"""
LLM module for answer generation using Google Gemini.

THIS IS THE "G" IN RAG (Retrieval-Augmented GENERATION):
- Takes retrieved chunks + user question → natural language answer
- Without this, you'd just show raw text chunks (bad UX)

WHY GEMINI?
- Free tier available (1500 requests/day)
- Large context window (1M+ tokens) - can fit lots of chunks
- Fast (gemini-2.5-flash optimized for speed)
- Good at following instructions (important for grounded answers)

KEY CHALLENGE: HALLUCINATION
- LLMs tend to invent facts when they don't know
- Critical for finance (made-up numbers = lawsuit risk)
- Mitigations:
  1. System prompt: "ONLY use the provided context"
  2. Allow "I don't know" responses
  3. Low temperature (0.1) for factual responses
  4. Evaluation to measure faithfulness
"""
from typing import List, Dict, Any, Optional

from ..config import GOOGLE_API_KEY, LLM_MODEL


# SYSTEM PROMPT: tells the LLM its role and rules.
# Engineered carefully to prevent hallucination and encourage citations.
# {context} placeholder gets filled in with retrieved chunks at query time.
SYSTEM_PROMPT = """You are a helpful financial analyst assistant. Your task is to answer questions about financial documents based on the provided context.

Instructions:
1. Answer questions using ONLY the information provided in the context below
2. If the answer cannot be found in the context, say "I cannot find this information in the provided documents"
3. When citing numbers or specific facts, mention which document they come from
4. Be precise with financial figures - don't round unless asked
5. If the question is ambiguous, ask for clarification

Context from financial documents:
{context}

---
Answer the following question based on the context above."""


class LLM:
    """
    Wrapper for Google Gemini LLM.
    
    Same WRAPPER pattern as Embedder/Reranker - hides the underlying SDK
    behind a simple interface. Easy to swap providers (OpenAI, Anthropic, etc.).
    """
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        api_key: Optional[str] = GOOGLE_API_KEY,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self._model = None  # Lazy load - same pattern throughout codebase
        
        # Warn early but don't fail - useful for offline development with MockLLM
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not set. LLM generation will fail.")
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            # Local import - SDK is heavy and only needed if generating
            import google.generativeai as genai
            
            # Fail fast if key missing at the point of actual use
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY is required. Set it in .env file.")
            
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Generate an answer based on query and context.
        
        Process: format chunks with citations → build prompt → call API → return result.
        max_tokens=4096 prevents runaway costs but allows detailed answers.
        """
        # We build TWO things in parallel:
        # 1. context_parts: text fed to the LLM (with source labels for citations)
        # 2. sources: structured metadata returned to caller (for UI display + evaluation)
        context_parts = []
        sources = []
        
        # enumerate(..., 1) starts at 1 instead of 0 (more human-friendly)
        for i, doc in enumerate(context_docs, 1):
            # .get() with defaults handles missing fields gracefully
            source = doc.get("metadata", {}).get("source", f"Document {i}")
            pages = doc.get("metadata", {}).get("pages", [])
            
            # Build a citation label like "[nvidia-10k.pdf, page 15]"
            # Singular vs plural based on number of pages in this chunk
            source_label = f"[{source}"
            if pages:
                source_label += f", pages {pages[0]}-{pages[-1]}" if len(pages) > 1 else f", page {pages[0]}"
            source_label += "]"
            
            # The label appears BEFORE the chunk text so the LLM sees the source
            context_parts.append(f"{source_label}:\n{doc['text']}")
            sources.append({
                "source": source,
                "pages": pages,
                # CRITICAL: chunk_index is needed for evaluation metrics (Precision@K).
                # Without this, retrieved chunks can't be matched to ground truth.
                "chunk_index": doc.get("metadata", {}).get("chunk_index", 0),
                # Truncate to keep response payloads small
                "text_preview": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
            })
        
        # Join chunks with clear separator so LLM can distinguish them
        context = "\n\n---\n\n".join(context_parts)
        
        # Final prompt: system instructions + context + question
        prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nQuestion: {query}"
        
        # try/except prevents API errors from crashing the app.
        # Common failures: rate limits (429), expired keys (401), network issues.
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    # Temperature controls randomness:
                    # 0.0 = fully deterministic (can get stuck on bad outputs)
                    # 0.1 = mostly deterministic but allows tiny variation (our choice)
                    # 0.7+ = creative/varied (good for writing, bad for facts)
                    "temperature": 0.1,
                },
            )
            answer = response.text
        except Exception as e:
            # Return error in the answer field so the UI can display it
            answer = f"Error generating response: {str(e)}"
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query,  # Echoed back for logging/debugging
        }


class MockLLM:
    """
    Mock LLM for testing without API calls.
    
    DESIGN PATTERN: TEST DOUBLE / NULL OBJECT
    Same interface as LLM but returns canned responses.
    Critical for: unit tests, CI/CD without API keys, development without quota.
    """
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return a mock response. Mirrors LLM.generate() structure exactly."""
        sources = []
        for doc in context_docs:
            sources.append({
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "pages": doc.get("metadata", {}).get("pages", []),
                # Same chunk_index requirement as real LLM - evaluation needs this
                "chunk_index": doc.get("metadata", {}).get("chunk_index", 0),
                "text_preview": doc["text"][:200] + "...",
            })
        
        return {
            "answer": f"[Mock Response] Based on {len(context_docs)} documents, here's information about: {query}",
            "sources": sources,
            "query": query,
        }
