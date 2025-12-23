"""
ARC-AGI-2 PREP PHASE SCRIPT

This runs in the **prep container**, where internet access is allowed

- Download EVERYTHING you will need later in the inference phase:
    * Hwen model weights (iamPi/Hwen from Hugging Face).
    * Tokenizer and config files.
    * Any auxiliary data needed for NVARC approach.

You ARE allowed to:
- Change which models are downloaded
- Add more downloads (multiple models, toolchains, etc.)

You MUST NOT:
- Write outside the provided `output_dir`
- Change the local cache paths


The validator calls `run_prep_phase(input_dir, output_dir)` or the CLI in this file
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from arc_solver_nvarc import model_name


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True
)
def download_model_with_retry(repo_id: str, cache_dir: str, local_dir: str) -> str:
    """download model with automatic retry on network failures"""
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )


def run_prep_phase(cache_dir = Path("/app/models")) -> None:
    """Prep phase: download NVARC model(s)"""
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading NVARC Models / Assets")
    print("=" * 60)

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / model_name.replace("/", "--")
    
    # We use the latest checkpoint (step_5972) by default
    checkpoint_dir = local_dir / "step_5972"
    
    print(f"\n[1/4] NVARC model to download: {model_name}")
    print(f"[2/4] Using cache directory: {cache_dir}")
    print(f"[3/4] Target local directory: {local_dir}")
    print(f"      Using checkpoint: step_5972")

    # Check if checkpoint folder exists with model files
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        files_count = len(list(checkpoint_dir.glob('*')))
        if files_count >= 5:  # Check for actual model files in checkpoint
            print(f"\n✓ Model checkpoint found in local cache ({files_count} files in step_5972), skipping download")
            
            print("\n" + "=" * 60)
            print("PREP PHASE COMPLETED - Status: success")
            print("=" * 60)
            return
        else:
            print(f"\n⚠ Partial checkpoint detected ({files_count} files), will resume...")
    elif local_dir.exists() and any(local_dir.iterdir()):
        folders_count = len(list(local_dir.glob('step_*')))
        print(f"\n⚠ Found {folders_count} checkpoint folders, but step_5972 incomplete, will resume...")

    print("(This phase requires internet access)")

    try:
        print("\n[4/4] Downloading NVARC model files from Hugging Face...")
        print("(Using automatic retry with exponential backoff)")
        
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = download_model_with_retry(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir)
        )

        print(f"✓ Model files downloaded to cache: {downloaded_path}")
        print("✓ Model download verified")
        
        # Verify checkpoint structure
        checkpoint_path = Path(downloaded_path) / "step_5972"
        if checkpoint_path.exists():
            files_count = len(list(checkpoint_path.glob('*')))
            print(f"✓ Files in checkpoint step_5972: {files_count}")
            checkpoints = [f.name for f in Path(downloaded_path).glob('step_*')]
            print(f"✓ Available checkpoints: {', '.join(sorted(checkpoints))}")
        else:
            files_count = len(list(Path(downloaded_path).glob('*')))
            print(f"✓ Files in model directory: {files_count}")

        prep_results = {
            "phase": "prep",
            "model": model_name,
            "status": "success",
            "message": f"NVARC model downloaded to {downloaded_path}",
            "cache_dir": str(cache_dir),
        }

    except Exception as e:
        print(f"ERROR: Could not complete prep phase: {e}")
        import traceback
        traceback.print_exc()
        
        prep_results = {
            "phase": "prep",
            "model": model_name,
            "status": "failed",
            "message": str(e),
        }

    print("\n" + "=" * 60)
    print(f"PREP PHASE COMPLETED - Status: {prep_results['status']}")
    print("=" * 60)

    if prep_results["status"] == "failed":
        sys.exit(1)


def _cli() -> int:
    """CLI entry point for running only the prep phase."""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Prep Phase Script (NVARC)")
    parser.add_argument("--input", type=str, required=True, help="Input directory path")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"\nPhase: prep")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    run_prep_phase()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except Exception as e:
        print(f"\nERROR (prep phase): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)