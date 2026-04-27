"""
Test Set Generator.
Creates evaluation test sets from documents.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import random


class TestSetGenerator:
    """
    Generates test sets for RAG evaluation.
    
    Can create test sets:
    1. Manually (you provide questions/answers)
    2. Semi-automatically (LLM generates Q&A from chunks)
    """
    
    def __init__(self, vector_store=None, llm=None):
        """
        Initialize generator.
        
        Args:
            vector_store: VectorStore to sample chunks from
            llm: LLM for generating questions (optional)
        """
        self.vector_store = vector_store
        self.llm = llm
    
    def generate_from_chunks(
        self,
        num_questions: int = 20,
        chunks_per_question: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate test questions from random chunks using LLM.
        
        Args:
            num_questions: Number of questions to generate
            chunks_per_question: Chunks to use per question
            
        Returns:
            List of test cases
        """
        if not self.vector_store or not self.llm:
            raise ValueError("Need both vector_store and llm for auto-generation")
        
        # Get all chunks (sample from collection)
        all_results = self.vector_store.collection.get(
            limit=num_questions * 2,
            include=["documents", "metadatas"]
        )
        
        if not all_results["documents"]:
            return []
        
        test_cases = []
        used_indices = set()
        
        for i in range(min(num_questions, len(all_results["documents"]))):
            # Pick a random unused chunk
            available = [j for j in range(len(all_results["documents"])) if j not in used_indices]
            if not available:
                break
            
            idx = random.choice(available)
            used_indices.add(idx)
            
            chunk_text = all_results["documents"][idx]
            metadata = all_results["metadatas"][idx] if all_results["metadatas"] else {}
            
            # Generate Q&A using LLM
            prompt = f"""Based on the following text from a financial document, generate ONE specific factual question and its answer.

Text:
{chunk_text[:1500]}

Generate a question that:
1. Can be answered directly from this text
2. Is specific (asks about numbers, names, dates, or specific facts)
3. Would be useful for someone analyzing financial documents

Respond in this exact JSON format:
{{"question": "your question here", "answer": "the answer from the text"}}

JSON:"""
            
            try:
                response = self.llm.generate(
                    query=prompt,
                    context_docs=[],
                    max_tokens=200,
                )
                
                # Parse JSON from response
                answer_text = response["answer"].strip()
                # Find JSON in response
                start = answer_text.find("{")
                end = answer_text.rfind("}") + 1
                if start >= 0 and end > start:
                    qa = json.loads(answer_text[start:end])
                    
                    chunk_id = f"{metadata.get('source', 'unknown')}_{metadata.get('chunk_index', i)}"
                    
                    test_cases.append({
                        "question": qa["question"],
                        "expected_answer": qa["answer"],
                        "relevant_chunk_ids": [chunk_id],
                        "source_chunk": chunk_text[:500],
                    })
                    print(f"Generated Q{len(test_cases)}: {qa['question'][:60]}...")
                    
            except Exception as e:
                print(f"Failed to generate Q&A: {e}")
                continue
        
        return test_cases
    
    def create_manual_test_set(self) -> List[Dict[str, Any]]:
        """
        Create a manual test set template.
        Returns example test cases that you should customize.
        """
        return [
            {
                "question": "What was NVIDIA's total revenue for fiscal year 2024?",
                "expected_answer": "NVIDIA's total revenue for fiscal year 2024 was $60.9 billion.",
                "relevant_chunk_ids": [],  # Fill in after identifying relevant chunks
            },
            {
                "question": "What are the main risk factors mentioned in the document?",
                "expected_answer": "",  # Fill in expected answer
                "relevant_chunk_ids": [],
            },
            {
                "question": "What was the year-over-year revenue growth?",
                "expected_answer": "",
                "relevant_chunk_ids": [],
            },
            # Add more questions...
        ]
    
    def save_test_set(self, test_cases: List[Dict], filename: str = "test_set.json"):
        """Save test set to file."""
        filepath = Path("evaluation_data") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(test_cases, f, indent=2)
        
        print(f"Saved {len(test_cases)} test cases to {filepath}")
        return filepath
    
    def load_test_set(self, filename: str = "test_set.json") -> List[Dict]:
        """Load test set from file."""
        filepath = Path("evaluation_data") / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Test set not found: {filepath}")
        
        with open(filepath, "r") as f:
            test_cases = json.load(f)
        
        print(f"Loaded {len(test_cases)} test cases from {filepath}")
        return test_cases


# Pre-built test sets for financial documents
FINANCIAL_TEST_SET = [
    {
        "question": "What was the total revenue for the fiscal year?",
        "category": "financial_metrics",
    },
    {
        "question": "What was the gross profit margin?",
        "category": "financial_metrics",
    },
    {
        "question": "What are the main business segments?",
        "category": "business_overview",
    },
    {
        "question": "What are the key risk factors mentioned?",
        "category": "risk_factors",
    },
    {
        "question": "What was the research and development expense?",
        "category": "financial_metrics",
    },
    {
        "question": "Who are the main competitors mentioned?",
        "category": "competitive_landscape",
    },
    {
        "question": "What is the company's growth strategy?",
        "category": "strategy",
    },
    {
        "question": "What was the operating income?",
        "category": "financial_metrics",
    },
    {
        "question": "What geographic regions does the company operate in?",
        "category": "business_overview",
    },
    {
        "question": "What was the year-over-year revenue change?",
        "category": "financial_metrics",
    },
]
