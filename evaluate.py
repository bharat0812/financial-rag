"""
RAG Evaluation Script.
Run experiments with different configurations and track results.

Usage:
    # Run with default config
    python evaluate.py
    
    # Run with custom parameters
    python evaluate.py --chunk-size 600 --top-k 10 --no-reranker
    
    # Compare all experiments
    python evaluate.py --compare
    
    # Generate test set
    python evaluate.py --generate-tests
"""
import argparse
import json
from pathlib import Path

from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, TOP_K_RERANK,
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL
)
from src.retrieval import VectorStore, Reranker
from src.generation import LLM
from src.pipeline import RAGPipeline
from src.evaluation import RAGEvaluator, ExperimentTracker, TestSetGenerator
from src.evaluation.experiment_tracker import ExperimentConfig


def create_pipeline_with_config(config: ExperimentConfig) -> RAGPipeline:
    """Create a pipeline with specific configuration."""
    from src.generation.llm import MockLLM
    
    vector_store = VectorStore()
    
    reranker = None
    if config.use_reranker:
        reranker = Reranker(model_name=config.reranker_model)
    
    if config.llm_provider == "mock":
        llm = MockLLM()
    else:
        llm = LLM(model_name=config.llm_model)
    
    return RAGPipeline(
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
        use_reranker=config.use_reranker,
    )


def run_experiment(
    config: ExperimentConfig,
    test_cases: list,
    tracker: ExperimentTracker,
    use_llm_judge: bool = False,
) -> dict:
    """Run a single experiment with given config."""
    print(f"\n{'='*60}")
    print("Running Experiment")
    print(f"{'='*60}")
    print(f"Config: chunk_size={config.chunk_size}, top_k={config.top_k_retrieval}, reranker={config.use_reranker}")
    
    pipeline = create_pipeline_with_config(config)
    
    llm_judge = None
    if use_llm_judge:
        llm_judge = LLM()
    
    evaluator = RAGEvaluator(pipeline=pipeline, llm_judge=llm_judge)
    
    results = evaluator.evaluate_test_set(test_cases)
    
    tracker.log_experiment(config, results, save_detailed=True)
    
    return results


def run_parameter_sweep(
    test_cases: list,
    tracker: ExperimentTracker,
):
    """Run experiments with different parameter combinations."""
    
    # Define parameter combinations to test
    chunk_sizes = [400, 800, 1200]
    top_k_values = [10, 20, 30]
    use_reranker_values = [True, False]
    
    print(f"\nRunning parameter sweep:")
    print(f"  Chunk sizes: {chunk_sizes}")
    print(f"  Top-K values: {top_k_values}")
    print(f"  Reranker: {use_reranker_values}")
    print(f"  Total experiments: {len(chunk_sizes) * len(top_k_values) * len(use_reranker_values)}")
    
    for chunk_size in chunk_sizes:
        for top_k in top_k_values:
            for use_reranker in use_reranker_values:
                config = ExperimentConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_size // 4,  # 25% overlap
                    top_k_retrieval=top_k,
                    top_k_rerank=5 if use_reranker else top_k,
                    use_reranker=use_reranker,
                    embedding_model=EMBEDDING_MODEL,
                    reranker_model=RERANKER_MODEL,
                    llm_model=LLM_MODEL,
                    llm_provider="gemini",
                    llm_temperature=0.1,
                    notes=f"Parameter sweep",
                )
                
                try:
                    run_experiment(config, test_cases, tracker)
                except Exception as e:
                    print(f"Experiment failed: {e}")
                    continue


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    
    # Config options
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=TOP_K_RETRIEVAL)
    parser.add_argument("--top-k-rerank", type=int, default=TOP_K_RERANK)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--llm-provider", choices=["gemini", "mock"], default="gemini")
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL)
    parser.add_argument("--notes", type=str, default="")
    
    # Actions
    parser.add_argument("--compare", action="store_true", help="Compare all experiments")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--generate-tests", action="store_true", help="Generate test set")
    parser.add_argument("--test-file", type=str, default="test_set.json", help="Test set file")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM to judge answers")
    parser.add_argument("--export-csv", action="store_true", help="Export results to CSV")
    
    args = parser.parse_args()
    
    tracker = ExperimentTracker()
    
    # Compare experiments
    if args.compare:
        if not tracker.experiments:
            print("No experiments found. Run some experiments first.")
            return
        tracker.compare_experiments()
        
        best = tracker.get_best_experiment("avg_precision")
        if best:
            print(f"\nBest experiment: {best.experiment_id}")
            print(f"  Precision: {best.avg_precision:.3f}")
            print(f"  Config: chunk={best.config.chunk_size}, top_k={best.config.top_k_retrieval}, reranker={best.config.use_reranker}")
        return
    
    # Export to CSV
    if args.export_csv:
        tracker.export_to_csv()
        return
    
    # Generate test set
    if args.generate_tests:
        vector_store = VectorStore()
        llm = LLM() if args.llm_provider == "gemini" else None
        
        generator = TestSetGenerator(vector_store=vector_store, llm=llm)
        
        if llm:
            print("Generating test questions using LLM...")
            test_cases = generator.generate_from_chunks(num_questions=10)
        else:
            print("Creating manual test set template...")
            test_cases = generator.create_manual_test_set()
        
        generator.save_test_set(test_cases, args.test_file)
        return
    
    # Load test set
    test_file = Path("evaluation_data") / args.test_file
    if not test_file.exists():
        print(f"Test set not found: {test_file}")
        print("Run with --generate-tests first, or create evaluation_data/test_set.json manually.")
        
        # Create a minimal test set
        print("\nCreating minimal test set for demo...")
        test_cases = [
            {"question": "What was the total revenue?"},
            {"question": "What are the main risk factors?"},
            {"question": "What is the company's growth strategy?"},
        ]
        Path("evaluation_data").mkdir(exist_ok=True)
        with open(test_file, "w") as f:
            json.dump(test_cases, f, indent=2)
        print(f"Created {test_file} with {len(test_cases)} questions.")
    else:
        with open(test_file) as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases from {test_file}")
    
    # Parameter sweep
    if args.sweep:
        run_parameter_sweep(test_cases, tracker)
        tracker.compare_experiments()
        return
    
    # Single experiment
    config = ExperimentConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k_retrieval=args.top_k,
        top_k_rerank=args.top_k_rerank,
        use_reranker=not args.no_reranker,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        llm_temperature=0.1,
        notes=args.notes,
    )
    
    run_experiment(config, test_cases, tracker, use_llm_judge=args.use_llm_judge)


if __name__ == "__main__":
    main()
