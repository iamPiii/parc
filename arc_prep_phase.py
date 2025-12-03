"""
ARC-AGI-2 PREP PHASE SCRIPT

This runs in the **prep container**, where internet access is allowed

- Download EVERYTHING you will need later in the inference phase:
    * VARC model weights (Hugging Face).
    * Any auxiliary data, vocab files, tokenizers...

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

# VARC imports
try:
    from arc_solver_varc import DEFAULT_VARC_REPO_ID, DEFAULT_VARC_CACHE_DIR
except ImportError:
    DEFAULT_VARC_REPO_ID = "VisionARC/offline_train_ViT"
    DEFAULT_VARC_CACHE_DIR = "/app/models"


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
        local_dir=local_dir,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )


def run_prep_phase(cache_dir = Path("/app/models")) -> None:
    """Prep phase: download VARC model checkpoint"""
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading VARC Model Checkpoint")
    print("=" * 60)

    varc_repo_id = DEFAULT_VARC_REPO_ID
    varc_cache_dir = Path(DEFAULT_VARC_CACHE_DIR)
    varc_local_dir = varc_cache_dir / varc_repo_id.replace("/", "--")
    
    print(f"\n[1/3] VARC checkpoint to download: {varc_repo_id}")
    print(f"[2/3] Cache directory: {varc_cache_dir}")
    print(f"[3/3] Target local directory: {varc_local_dir}")

    # Check if already downloaded
    checkpoint_files = list(varc_local_dir.glob("*.pth")) + list(varc_local_dir.glob("*.pt"))
    if varc_local_dir.exists() and checkpoint_files:
        print(f"\n✓ VARC checkpoint already exists at {varc_local_dir}")
        print(f"  Found checkpoint: {checkpoint_files[0].name}")
        
        prep_results = {
            "phase": "prep",
            "model": varc_repo_id,
            "status": "success",
            "message": f"VARC checkpoint already cached at {varc_local_dir}",
            "cache_dir": str(varc_cache_dir),
        }
        
        print("\n" + "=" * 60)
        print("PREP PHASE COMPLETED - Status: success")
        print("=" * 60)
        return

    print("(This phase requires internet access)")

    try:
        print("\n[Downloading] VARC checkpoint from Hugging Face...")
        print("(Using automatic retry with exponential backoff)")
        
        varc_cache_dir.mkdir(parents=True, exist_ok=True)
        varc_local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = download_model_with_retry(
            repo_id=varc_repo_id,
            cache_dir=str(varc_cache_dir.parent),
            local_dir=str(varc_local_dir)
        )

        # Verify checkpoint file exists
        checkpoint_files = list(Path(downloaded_path).glob("*.pth")) + list(Path(downloaded_path).glob("*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint file (*.pth or *.pt) found in {downloaded_path}")

        # Check which checkpoint was downloaded
        checkpoint_best = Path(downloaded_path) / "checkpoint_best.pt"
        checkpoint_final = Path(downloaded_path) / "checkpoint_final.pt"
        if checkpoint_best.exists():
            checkpoint_name = "checkpoint_best.pt"
        elif checkpoint_final.exists():
            checkpoint_name = "checkpoint_final.pt"
        else:
            checkpoint_name = checkpoint_files[0].name

        print(f"✓ VARC checkpoint downloaded to: {downloaded_path}")
        print(f"✓ Checkpoint file: {checkpoint_name}")
        files_count = len(list(Path(downloaded_path).glob('*')))
        print(f"✓ Total files in directory: {files_count}")

        prep_results = {
            "phase": "prep",
            "model": varc_repo_id,
            "status": "success",
            "message": f"VARC checkpoint downloaded to {downloaded_path}",
            "cache_dir": str(varc_cache_dir),
        }

    except Exception as e:
        print(f"ERROR: Could not complete prep phase: {e}")
        import traceback
        traceback.print_exc()
        
        prep_results = {
            "phase": "prep",
            "model": varc_repo_id,
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
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Prep Phase Script")
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