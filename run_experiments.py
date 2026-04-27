"""
Run comprehensive experiments with different RAG configurations.
Tests different chunk sizes, overlaps, and reranker settings.
"""
import subprocess
import sys
import json
from datetime import datetime


CHUNK_CONFIGS = [
    {"chunk_size": 400, "chunk_overlap": 50},
    {"chunk_size": 400, "chunk_overlap": 100},
    {"chunk_size": 800, "chunk_overlap": 100},
    {"chunk_size": 800, "chunk_overlap": 200},
    {"chunk_size": 1200, "chunk_overlap": 200},
    {"chunk_size": 1200, "chunk_overlap": 300},
]

RERANKER_OPTIONS = [True, False]


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    print("=" * 70)
    print("RAG Parameter Sweep Experiments")
    print("=" * 70)
    print(f"Testing {len(CHUNK_CONFIGS)} chunk configurations x {len(RERANKER_OPTIONS)} reranker options")
    print(f"Total experiments: {len(CHUNK_CONFIGS) * len(RERANKER_OPTIONS)}")
    
    experiments_run = 0
    
    for config in CHUNK_CONFIGS:
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        
        # Re-ingest with this chunk configuration
        print(f"\n\n{'#'*70}")
        print(f"# Re-ingesting with chunk_size={chunk_size}, overlap={chunk_overlap}")
        print(f"{'#'*70}")
        
        ingest_cmd = [
            sys.executable, "ingest.py",
            "--clear",
            "--chunk-size", str(chunk_size),
            "--chunk-overlap", str(chunk_overlap),
        ]
        
        if not run_command(ingest_cmd, f"Ingest (chunk={chunk_size}, overlap={chunk_overlap})"):
            print(f"ERROR: Ingestion failed for chunk_size={chunk_size}")
            continue
        
        # Run evaluation with and without reranker
        for use_reranker in RERANKER_OPTIONS:
            reranker_str = "with" if use_reranker else "without"
            notes = f"chunk={chunk_size}, overlap={chunk_overlap}, {reranker_str} reranker"
            
            eval_cmd = [
                sys.executable, "evaluate.py",
                "--chunk-size", str(chunk_size),
                "--chunk-overlap", str(chunk_overlap),
                "--notes", notes,
            ]
            
            if not use_reranker:
                eval_cmd.append("--no-reranker")
            
            if run_command(eval_cmd, f"Evaluate ({notes})"):
                experiments_run += 1
            else:
                print(f"WARNING: Evaluation failed for {notes}")
    
    # Final comparison
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT SWEEP COMPLETE - {experiments_run} experiments run")
    print(f"{'='*70}")
    
    # Run comparison
    print("\nGenerating comparison...")
    subprocess.run([sys.executable, "evaluate.py", "--compare"])
    
    # Export to CSV
    print("\nExporting to CSV...")
    subprocess.run([sys.executable, "evaluate.py", "--export-csv"])
    
    print("\n" + "=" * 70)
    print("All experiments complete! Check experiments/experiments.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
