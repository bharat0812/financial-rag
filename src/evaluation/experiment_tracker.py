"""
Experiment Tracking Module.
Logs experiments with different parameters for comparison.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    # Chunking parameters
    chunk_size: int
    chunk_overlap: int
    
    # Retrieval parameters
    top_k_retrieval: int
    top_k_rerank: int
    use_reranker: bool
    
    # Model parameters
    embedding_model: str
    reranker_model: str
    llm_model: str
    llm_provider: str
    llm_temperature: float
    
    # Optional notes
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_hash(self) -> str:
        """Generate unique hash for this config."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass 
class ExperimentResult:
    """Results from an experiment."""
    experiment_id: str
    timestamp: str
    config: ExperimentConfig
    
    # Metrics
    avg_precision: float
    avg_recall: float
    avg_latency_ms: float
    avg_answer_score: Optional[float]
    num_questions: int
    
    # Additional details
    detailed_results: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


class ExperimentTracker:
    """
    Tracks experiments and enables comparison.
    
    Usage:
        tracker = ExperimentTracker()
        
        # Log an experiment
        tracker.log_experiment(config, results)
        
        # Compare experiments
        tracker.compare_experiments()
        
        # Get best config
        best = tracker.get_best_experiment("avg_precision")
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize tracker.
        
        Args:
            experiments_dir: Directory to store experiment logs
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.experiments_dir / "experiments.json"
        self.experiments: List[ExperimentResult] = []
        self._load_experiments()
    
    def _load_experiments(self):
        """Load existing experiments from disk."""
        if self.experiments_file.exists():
            with open(self.experiments_file, "r") as f:
                data = json.load(f)
                for exp_dict in data:
                    config = ExperimentConfig(**exp_dict["config"])
                    exp_dict["config"] = config
                    self.experiments.append(ExperimentResult(**exp_dict))
    
    def _save_experiments(self):
        """Save experiments to disk."""
        data = [exp.to_dict() for exp in self.experiments]
        with open(self.experiments_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def log_experiment(
        self,
        config: ExperimentConfig,
        eval_results: Dict[str, Any],
        save_detailed: bool = False,
    ) -> ExperimentResult:
        """
        Log a new experiment.
        
        Args:
            config: Experiment configuration
            eval_results: Results from RAGEvaluator.evaluate_test_set()
            save_detailed: Whether to save per-question results
            
        Returns:
            ExperimentResult object
        """
        experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.get_hash()}"
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            avg_precision=eval_results["avg_precision"],
            avg_recall=eval_results["avg_recall"],
            avg_latency_ms=eval_results["avg_latency_ms"],
            avg_answer_score=eval_results.get("avg_answer_score"),
            num_questions=eval_results["num_questions"],
            detailed_results=[
                {
                    "question": r.question,
                    "precision": r.precision,
                    "recall": r.recall,
                    "latency_ms": r.latency_ms,
                    "answer_score": r.answer_score,
                }
                for r in eval_results.get("results", [])
            ] if save_detailed else None,
        )
        
        self.experiments.append(result)
        self._save_experiments()
        
        # Also save detailed results separately
        if save_detailed and eval_results.get("results"):
            detail_file = self.experiments_dir / f"{experiment_id}_details.json"
            with open(detail_file, "w") as f:
                json.dump(result.detailed_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Experiment logged: {experiment_id}")
        print(f"{'='*60}")
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: ExperimentResult):
        """Pretty print an experiment result."""
        print(f"\nConfig:")
        print(f"  Chunk size: {result.config.chunk_size}")
        print(f"  Chunk overlap: {result.config.chunk_overlap}")
        print(f"  Top-K retrieval: {result.config.top_k_retrieval}")
        print(f"  Top-K rerank: {result.config.top_k_rerank}")
        print(f"  Use reranker: {result.config.use_reranker}")
        print(f"  Embedding model: {result.config.embedding_model}")
        print(f"  LLM: {result.config.llm_provider}/{result.config.llm_model}")
        
        print(f"\nResults ({result.num_questions} questions):")
        print(f"  Precision@K: {result.avg_precision:.3f}")
        print(f"  Recall@K: {result.avg_recall:.3f}")
        print(f"  Avg Latency: {result.avg_latency_ms:.0f}ms")
        if result.avg_answer_score is not None:
            print(f"  Answer Score: {result.avg_answer_score:.3f}")
    
    def compare_experiments(
        self,
        metric: str = "avg_precision",
        top_n: int = 10,
    ) -> List[ExperimentResult]:
        """
        Compare experiments by a metric.
        
        Args:
            metric: Metric to sort by
            top_n: Number of top experiments to show
            
        Returns:
            Sorted list of experiments
        """
        sorted_exps = sorted(
            self.experiments,
            key=lambda x: getattr(x, metric) or 0,
            reverse=True,
        )[:top_n]
        
        print(f"\n{'='*80}")
        print(f"Top {len(sorted_exps)} Experiments by {metric}")
        print(f"{'='*80}")
        
        headers = ["Rank", "ID", "Precision", "Recall", "Latency", "Chunk", "TopK", "Rerank"]
        print(f"{headers[0]:<5} {headers[1]:<20} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<8} {headers[6]:<6} {headers[7]:<6}")
        print("-" * 80)
        
        for i, exp in enumerate(sorted_exps, 1):
            print(
                f"{i:<5} "
                f"{exp.experiment_id[:18]:<20} "
                f"{exp.avg_precision:.3f}     "
                f"{exp.avg_recall:.3f}     "
                f"{exp.avg_latency_ms:>6.0f}ms  "
                f"{exp.config.chunk_size:<8} "
                f"{exp.config.top_k_retrieval:<6} "
                f"{'Yes' if exp.config.use_reranker else 'No':<6}"
            )
        
        return sorted_exps
    
    def get_best_experiment(self, metric: str = "avg_precision") -> Optional[ExperimentResult]:
        """Get the best experiment by a metric."""
        if not self.experiments:
            return None
        return max(self.experiments, key=lambda x: getattr(x, metric) or 0)
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get a specific experiment by ID."""
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                return exp
        return None
    
    def export_to_csv(self, filename: str = "experiments.csv"):
        """Export all experiments to CSV for analysis."""
        import csv
        
        filepath = self.experiments_dir / filename
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "experiment_id", "timestamp",
                "chunk_size", "chunk_overlap", 
                "top_k_retrieval", "top_k_rerank", "use_reranker",
                "embedding_model", "reranker_model", "llm_model", "llm_provider",
                "avg_precision", "avg_recall", "avg_latency_ms", "avg_answer_score",
                "num_questions", "notes"
            ])
            
            # Data
            for exp in self.experiments:
                writer.writerow([
                    exp.experiment_id, exp.timestamp,
                    exp.config.chunk_size, exp.config.chunk_overlap,
                    exp.config.top_k_retrieval, exp.config.top_k_rerank, exp.config.use_reranker,
                    exp.config.embedding_model, exp.config.reranker_model, 
                    exp.config.llm_model, exp.config.llm_provider,
                    exp.avg_precision, exp.avg_recall, exp.avg_latency_ms, exp.avg_answer_score,
                    exp.num_questions, exp.config.notes
                ])
        
        print(f"Exported {len(self.experiments)} experiments to {filepath}")
