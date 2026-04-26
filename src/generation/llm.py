"""
LLM module for answer generation using Google Gemini.
"""
from typing import List, Dict, Any, Optional

from ..config import GOOGLE_API_KEY, LLM_MODEL


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
    """
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        api_key: Optional[str] = GOOGLE_API_KEY,
    ):
        """
        Initialize the LLM.
        
        Args:
            model_name: Gemini model name
            api_key: Google API key
        """
        self.model_name = model_name
        self.api_key = api_key
        self._model = None
        
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not set. LLM generation will fail.")
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            import google.generativeai as genai
            
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY is required. Set it in .env file.")
            
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate an answer based on query and context.
        
        Args:
            query: User's question
            context_docs: List of relevant documents with 'text' and 'metadata'
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        context_parts = []
        sources = []
        
        for i, doc in enumerate(context_docs, 1):
            source = doc.get("metadata", {}).get("source", f"Document {i}")
            pages = doc.get("metadata", {}).get("pages", [])
            
            source_label = f"[{source}"
            if pages:
                source_label += f", pages {pages[0]}-{pages[-1]}" if len(pages) > 1 else f", page {pages[0]}"
            source_label += "]"
            
            context_parts.append(f"{source_label}:\n{doc['text']}")
            sources.append({
                "source": source,
                "pages": pages,
                "text_preview": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nQuestion: {query}"
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            answer = response.text
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query,
        }


class MockLLM:
    """
    Mock LLM for testing without API calls.
    """
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return a mock response."""
        sources = []
        for doc in context_docs:
            sources.append({
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "pages": doc.get("metadata", {}).get("pages", []),
                "text_preview": doc["text"][:200] + "...",
            })
        
        return {
            "answer": f"[Mock Response] Based on {len(context_docs)} documents, here's information about: {query}",
            "sources": sources,
            "query": query,
        }
