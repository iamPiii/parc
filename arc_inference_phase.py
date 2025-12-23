"""
ARC-AGI-2 INFERENCE PHASE SCRIPT

This runs in the **inference container**, where internet access IS blocked

- Load the tasks and the assets you prepared in prep phase (models, weights, etc.)
- Run your solver(s) on each test input
- Produce predictions and write them to `results.json`

You ARE allowed to:
- Replace the example `ARCSolver` with your own algorithm/model
- Change how many tasks you solve and in what order
- Add more logging / metrics, as long as the outputs still make sense

You MUST NOT:
- Attempt network calls (it will fail)
- Change the basic structure of `results["predictions"]`:
    * list of dicts with at least:
        - "problem_index" - provided in input 
        - "task_hash" - provided in input
        - "predicted_output"

The validator calls `run_inference_phase(input_dir, output_dir)` or the CLI here

NOTE: NVARC solver uses test-time training, expect ~10-20 minutes per puzzle
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

from arc_solver_nvarc import ARCSolver
from arc_utils import load_input_data, save_output_data


def run_inference_phase(input_dir: Path, output_dir: Path) -> None:
    """Inference phase: solve ARC-AGI-2 problems and save predictions"""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE - Solving ARC-AGI-2 Problems")
    print("=" * 60)

    try:
        print(f"\n[1/4] Loading input data from {input_dir}..")
        data = load_input_data(input_dir)
        problems: List[Dict[str, Any]] = data["tasks"]

        print("[2/4] Initializing ARC solver (NVARC model)..")
        solver = ARCSolver()

        predictions: List[Dict[str, Any]] = []
        total_solve_time = 0

        print(f"\n[3/4] Solving {len(problems)} problem(s) with NVARC..")
        
        for i in range(len(problems)):
            problem = problems[i]
            if "train_examples" not in problem:
                print(f"    ✗ Problem {i} missing 'train_examples' field")
                print(f"      Available keys: {list(problem.keys())}")
                continue

            print(f"\n  Problem {i + 1}/{len(problems)}:")
            print(f"    - Training examples: {len(problem['train_examples'])}")
            print(
                f"    - Test input shape: {len(problem['test_input'])}"
                f"x{len(problem['test_input'][0])}"
            )
            print(f"    - Starting NVARC solve (test-time training + beam search)...")
            
            try:
                start_time = time.time()
                predicted_output = solver.solve(
                    train_examples=problem["train_examples"],
                    test_input=problem["test_input"],
                )
                solve_time = time.time() - start_time
                total_solve_time += solve_time

                print(
                    f"    - Predicted output shape: {len(predicted_output)}"
                    f"x{len(predicted_output[0])}"
                )
                print(f"    ✓ Solved successfully in {solve_time:.1f}s ({solve_time/60:.1f} min)")

                prediction_entry = {
                    "problem_index": i,
                    "task_hash": problem.get("task_hash"),
                    "predicted_output": predicted_output,
                    "metadata": problem.get("metadata", {}),
                }
                predictions.append(prediction_entry)

            except Exception as e:
                solve_time = time.time() - start_time if 'start_time' in locals() else 0
                print(f"    ✗ Error solving problem {i} after {solve_time:.1f}s: {e}")
                import traceback

                traceback.print_exc()
                
                # Add failed prediction with None output
                prediction_entry = {
                    "problem_index": i,
                    "task_hash": problem.get("task_hash"),
                    "predicted_output": None,
                    "error": str(e),
                    "metadata": problem.get("metadata", {}),
                }
                predictions.append(prediction_entry)

        print(f"\n[4/4] Saving predictions to {output_dir}..")
        num_solved = sum(1 for p in predictions if p.get("predicted_output") is not None)
        results = {
            "phase": "inference",
            "status": "success",
            "num_problems_solved": num_solved,
            "total_solve_time_seconds": total_solve_time,
            "predictions": predictions,
        }

        save_output_data(results, output_dir)

        print("\n" + "=" * 60)
        print(f"INFERENCE PHASE COMPLETED - Solved {num_solved}/{len(problems)} problems")
        print(f"Total solve time: {total_solve_time:.1f}s ({total_solve_time/60:.1f} min)")
        if num_solved > 0:
            print(f"Average time per puzzle: {total_solve_time/num_solved:.1f}s ({total_solve_time/num_solved/60:.1f} min)")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: NVARC inference phase failed: {e}")
        import traceback

        traceback.print_exc()

        results = {
            "phase": "inference",
            "status": "failed",
            "error": str(e),
            "predictions": predictions if 'predictions' in locals() else [],
        }
        save_output_data(results, output_dir)

        print("\n" + "=" * 60)
        print("INFERENCE PHASE COMPLETED - Status: failed")
        print("=" * 60)

        sys.exit(1)


def _cli() -> int:
    """CLI entry point for running only the inference phase"""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Inference Phase Script")
    parser.add_argument("--input", type=str, required=True, help="Input directory path")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"\nPhase: inference")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    run_inference_phase(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except Exception as e:
        print(f"\nERROR (inference phase): {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
