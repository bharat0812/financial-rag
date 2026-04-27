"""
Compare retrieval metrics across different chunk sizes.
For each chunk size, auto-generates ground truth chunk IDs and evaluates.
"""
import json
import subprocess
import sys
from pathlib import Path

from src.retrieval import VectorStore
from src.pipeline import create_pipeline
from src.evaluation import RAGEvaluator, ExperimentTracker
from src.evaluation.experiment_tracker import ExperimentConfig
from src.config import EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL

# Test questions (company-specific)
QUESTIONS = [
    {"q": "What was NVIDIA's total revenue for fiscal year 2024?", "company": "nvidia",
     "expected": "NVIDIA's total revenue for fiscal year 2024 was $60.9 billion, up 126% from fiscal year 2023."},
    {"q": "What was NVIDIA's gross margin percentage in fiscal year 2024?", "company": "nvidia",
     "expected": "NVIDIA's gross margin was 72.7% in fiscal year 2024."},
    {"q": "What are NVIDIA's two main reportable business segments?", "company": "nvidia",
     "expected": "NVIDIA has two reportable segments: Compute & Networking, and Graphics."},
    {"q": "What was NVIDIA's operating income for fiscal year 2024?", "company": "nvidia",
     "expected": "NVIDIA's operating income was $33.0 billion in fiscal year 2024."},
    {"q": "What are the key risk factors mentioned in NVIDIA's 10-K?", "company": "nvidia",
     "expected": "Key risk factors include demand volatility, supply chain, export controls, and competition."},
    {"q": "What are Apple's reportable geographic segments?", "company": "apple",
     "expected": "Apple's segments are Americas, Europe, Greater China, Japan, and Rest of Asia Pacific."},
    {"q": "What was Apple's total gross margin for 2022?", "company": "apple",
     "expected": "Apple's total gross margin for 2022 was $170.8 billion."},
    {"q": "What are the main risk factors disclosed in Apple's 10-K?", "company": "apple",
     "expected": "Main risk factors include economic conditions, competition, and supply chain disruptions."},
    {"q": "What was Apple's Services revenue performance in 2022?", "company": "apple",
     "expected": "Services net sales increased due to higher advertising, cloud services, and App Store revenue."},
    {"q": "What are Apple's main product categories?", "company": "apple",
     "expected": "Apple's main products are iPhone, Mac, iPad, Wearables/Home/Accessories, and Services."},
]

CHUNK_CONFIGS = [
    {"chunk_size": 400, "chunk_overlap": 100},
    {"chunk_size": 800, "chunk_overlap": 200},
    {"chunk_size": 1200, "chunk_overlap": 300},
]


def ingest_with_config(chunk_size: int, chunk_overlap: int):
    """Re-ingest documents with specific chunk configuration."""
    print(f"\n{'='*60}")
    print(f"Ingesting with chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "ingest.py", "--clear", 
         "--chunk-size", str(chunk_size),
         "--chunk-overlap", str(chunk_overlap)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    # Extract chunk count from output
    for line in result.stdout.split('\n'):
        if 'Total chunks:' in line:
            print(line)
        if 'Final collection:' in line:
            print(line)
    
    return True


def find_relevant_chunks(vs: VectorStore, question: str, company: str, top_k: int = 3):
    """Find relevant chunk IDs for a question."""
    results = vs.search(question, top_k=10)
    
    relevant = []
    for r in results:
        source = r['metadata'].get('source', '').lower()
        if company in source:
            chunk_id = f"{r['metadata']['source']}_{r['metadata'].get('chunk_index', 0)}"
            relevant.append(chunk_id)
    
    return relevant[:top_k]


def build_test_set_for_config(chunk_size: int):
    """Build test set with chunk IDs for current configuration."""
    vs = VectorStore()
    
    test_cases = []
    for item in QUESTIONS:
        chunk_ids = find_relevant_chunks(vs, item["q"], item["company"])
        test_cases.append({
            "question": item["q"],
            "expected_answer": item["expected"],
            "relevant_chunk_ids": chunk_ids,
            "company": item["company"],
        })
    
    # Save test set
    filename = f"evaluation_data/test_set_chunk{chunk_size}.json"
    with open(filename, "w") as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"Saved test set to {filename}")
    return test_cases


def run_evaluation(chunk_size: int, chunk_overlap: int, test_cases: list, use_reranker: bool = True):
    """Run evaluation and return results."""
    config = ExperimentConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k_retrieval=20,
        top_k_rerank=5,
        use_reranker=use_reranker,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=LLM_MODEL,
        llm_provider="gemini",
        llm_temperature=0.1,
        notes=f"Chunk size comparison: {chunk_size} chars",
    )
    
    # Create pipeline
    pipeline = create_pipeline(use_reranker=use_reranker)
    evaluator = RAGEvaluator(pipeline=pipeline)
    
    print(f"\nEvaluating {len(test_cases)} questions...")
    results = evaluator.evaluate_test_set(test_cases)
    
    # Log experiment
    tracker = ExperimentTracker()
    tracker.log_experiment(config, results)
    
    return results


def main():
    print("="*70)
    print("CHUNK SIZE COMPARISON EXPERIMENT")
    print("="*70)
    print(f"Testing configurations: {[c['chunk_size'] for c in CHUNK_CONFIGS]}")
    print(f"Questions: {len(QUESTIONS)}")
    
    all_results = []
    
    for config in CHUNK_CONFIGS:
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        
        # Step 1: Ingest
        if not ingest_with_config(chunk_size, chunk_overlap):
            continue
        
        # Step 2: Build test set with correct chunk IDs
        print(f"\nBuilding test set for chunk_size={chunk_size}...")
        test_cases = build_test_set_for_config(chunk_size)
        
        # Step 3: Evaluate
        print(f"\nRunning evaluation for chunk_size={chunk_size}...")
        results = run_evaluation(chunk_size, chunk_overlap, test_cases)
        
        all_results.append({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "precision": results["avg_precision"],
            "recall": results["avg_recall"],
            "latency_ms": results["avg_latency_ms"],
        })
        
        print(f"\n>>> chunk_size={chunk_size}: Precision={results['avg_precision']:.3f}, Recall={results['avg_recall']:.3f}, Latency={results['avg_latency_ms']:.0f}ms")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Chunk Size':<12} {'Overlap':<10} {'Precision':<12} {'Recall':<12} {'Latency':<12}")
    print("-"*58)
    for r in all_results:
        print(f"{r['chunk_size']:<12} {r['chunk_overlap']:<10} {r['precision']:<12.3f} {r['recall']:<12.3f} {r['latency_ms']:<12.0f}ms")
    
    # Export comparison
    tracker = ExperimentTracker()
    tracker.export_to_csv()
    print(f"\nExported to experiments/experiments.csv")


if __name__ == "__main__":
    main()
