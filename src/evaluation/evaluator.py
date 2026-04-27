"""
RAG Evaluation Module.
Measures retrieval and generation quality.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class EvaluationResult:
    """Results from a single evaluation."""
    question: str
    expected_answer: Optional[str]
    actual_answer: str
    retrieved_chunks: List[str]
    relevant_chunks: List[str]
    precision: float
    recall: float
    latency_ms: float
    answer_score: Optional[float] = None


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance.
    
    Metrics:
    - Retrieval: Precision@K, Recall@K, MRR
    - Generation: Answer relevance, faithfulness (requires LLM judge)
    - Performance: Latency
    """
    
    def __init__(self, pipeline, llm_judge=None):
        """
        Initialize evaluator.
        
        Args:
            pipeline: RAGPipeline instance to evaluate
            llm_judge: Optional LLM for judging answer quality
        """
        self.pipeline = pipeline
        self.llm_judge = llm_judge
    
    def evaluate_retrieval(
        self,
        question: str,
        relevant_chunk_ids: List[str],
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality for a single question.
        
        Args:
            question: The test question
            relevant_chunk_ids: List of chunk IDs that are relevant
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with precision, recall, mrr
        """
        chunks = self.pipeline.get_relevant_chunks(question, top_k=top_k)
        
        retrieved_ids = []
        for chunk in chunks:
            chunk_id = chunk.get("metadata", {}).get("source", "") + "_" + str(chunk.get("metadata", {}).get("chunk_index", 0))
            retrieved_ids.append(chunk_id)
        
        relevant_set = set(relevant_chunk_ids)
        retrieved_set = set(retrieved_ids)
        
        hits = len(relevant_set & retrieved_set)
        precision = hits / len(retrieved_set) if retrieved_set else 0
        recall = hits / len(relevant_set) if relevant_set else 0
        
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        return {
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "hits": hits,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_set),
        }
    
    def evaluate_single(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        relevant_chunk_ids: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation on a single question.
        
        Args:
            question: The test question
            expected_answer: Ground truth answer (optional)
            relevant_chunk_ids: Known relevant chunks (optional)
            
        Returns:
            EvaluationResult with all metrics
        """
        start_time = time.time()
        result = self.pipeline.query(question)
        latency_ms = (time.time() - start_time) * 1000
        
        retrieved_ids = []
        for source in result.get("sources", []):
            chunk_id = f"{source['source']}_{source.get('chunk_index', 0)}"
            retrieved_ids.append(chunk_id)
        
        precision = 0.0
        recall = 0.0
        if relevant_chunk_ids:
            relevant_set = set(relevant_chunk_ids)
            retrieved_set = set(retrieved_ids)
            hits = len(relevant_set & retrieved_set)
            precision = hits / len(retrieved_set) if retrieved_set else 0
            recall = hits / len(relevant_set) if relevant_set else 0
        
        answer_score = None
        if expected_answer and self.llm_judge:
            answer_score = self._judge_answer(
                question=question,
                expected=expected_answer,
                actual=result["answer"],
            )
        
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            actual_answer=result["answer"],
            retrieved_chunks=retrieved_ids,
            relevant_chunks=relevant_chunk_ids or [],
            precision=precision,
            recall=recall,
            latency_ms=latency_ms,
            answer_score=answer_score,
        )
    
    def evaluate_test_set(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate on a full test set.
        
        Args:
            test_cases: List of test cases with format:
                {
                    "question": str,
                    "expected_answer": str (optional),
                    "relevant_chunk_ids": List[str] (optional)
                }
                
        Returns:
            Aggregated metrics across all test cases
        """
        results = []
        total_precision = 0
        total_recall = 0
        total_latency = 0
        total_answer_score = 0
        answer_score_count = 0
        
        for i, case in enumerate(test_cases):
            print(f"Evaluating {i+1}/{len(test_cases)}: {case['question'][:50]}...")
            
            result = self.evaluate_single(
                question=case["question"],
                expected_answer=case.get("expected_answer"),
                relevant_chunk_ids=case.get("relevant_chunk_ids"),
            )
            results.append(result)
            
            total_precision += result.precision
            total_recall += result.recall
            total_latency += result.latency_ms
            
            if result.answer_score is not None:
                total_answer_score += result.answer_score
                answer_score_count += 1
        
        n = len(test_cases)
        
        return {
            "num_questions": n,
            "avg_precision": total_precision / n if n else 0,
            "avg_recall": total_recall / n if n else 0,
            "avg_latency_ms": total_latency / n if n else 0,
            "avg_answer_score": total_answer_score / answer_score_count if answer_score_count else None,
            "results": results,
        }
    
    def _judge_answer(
        self,
        question: str,
        expected: str,
        actual: str,
    ) -> float:
        """
        Use LLM to judge answer quality.
        
        Returns score from 0-1.
        """
        if not self.llm_judge:
            return 0.0
        
        prompt = f"""You are evaluating a question-answering system. Score the actual answer compared to the expected answer.

Question: {question}

Expected Answer: {expected}

Actual Answer: {actual}

Score from 1-5:
5 = Perfect - captures all key information correctly
4 = Good - mostly correct with minor omissions
3 = Partial - some correct information but missing key points
2 = Poor - mostly incorrect or irrelevant
1 = Wrong - completely incorrect or off-topic

Respond with ONLY a number (1-5)."""

        try:
            response = self.llm_judge.generate(
                query=prompt,
                context_docs=[],
                max_tokens=10,
            )
            score = int(response["answer"].strip()[0])
            return (score - 1) / 4  # Normalize to 0-1
        except:
            return 0.0
