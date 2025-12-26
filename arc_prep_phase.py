"""
ARC-AGI-2 PREP PHASE SCRIPT

This runs in the **prep container**, where internet access is allowed

- Download EVERYTHING you will need later in the inference phase:
    * VARC model weights (Hugging Face).
    * NVARC model weights (Qwen LLM from Hugging Face).
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
import os
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

# NVARC imports
try:
    from arc_solver_nvarc import DEFAULT_NVARC_REPO_ID, DEFAULT_NVARC_CACHE_DIR
except ImportError:
    DEFAULT_NVARC_REPO_ID = "iamPi/Hwen-HF"
    DEFAULT_NVARC_CACHE_DIR = "/app/models"

# Determine which solver(s) to download models for
# NOTE: VARC temporarily disabled, using NVARC by default
ARC_SOLVER = os.environ.get("ARC_SOLVER", "nvarc")  # "nvarc" (default), "varc", or "both"


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


def download_varc_model() -> dict:
    """Download VARC (Vision Transformer) model checkpoint."""
    print("\n" + "-" * 40)
    print("Downloading VARC Model Checkpoint")
    print("-" * 40)

    varc_repo_id = DEFAULT_VARC_REPO_ID
    varc_cache_dir = Path(DEFAULT_VARC_CACHE_DIR)
    varc_local_dir = varc_cache_dir / varc_repo_id.replace("/", "--")

    print(f"  Repository: {varc_repo_id}")
    print(f"  Cache directory: {varc_cache_dir}")
    print(f"  Target local directory: {varc_local_dir}")

    # Check if already downloaded
    checkpoint_files = list(varc_local_dir.glob("*.pth")) + list(varc_local_dir.glob("*.pt"))
    if varc_local_dir.exists() and checkpoint_files:
        print(f"\n  ✓ VARC checkpoint already exists at {varc_local_dir}")
        print(f"    Found checkpoint: {checkpoint_files[0].name}")
        return {
            "model": "varc",
            "repo_id": varc_repo_id,
            "status": "success",
            "message": f"VARC checkpoint already cached at {varc_local_dir}",
        }

    try:
        print("\n  [Downloading] VARC checkpoint from Hugging Face...")

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

        print(f"  ✓ VARC checkpoint downloaded to: {downloaded_path}")
        print(f"  ✓ Checkpoint file: {checkpoint_name}")
        files_count = len(list(Path(downloaded_path).glob('*')))
        print(f"  ✓ Total files in directory: {files_count}")

        return {
            "model": "varc",
            "repo_id": varc_repo_id,
            "status": "success",
            "message": f"VARC checkpoint downloaded to {downloaded_path}",
        }

    except Exception as e:
        print(f"  ERROR: Could not download VARC model: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": "varc",
            "repo_id": varc_repo_id,
            "status": "failed",
            "message": str(e),
        }


def download_nvarc_model() -> dict:
    """Download NVARC (Qwen LLM) model checkpoint."""
    print("\n" + "-" * 40)
    print("Downloading NVARC Model Checkpoint (Qwen LLM)")
    print("-" * 40)

    nvarc_repo_id = DEFAULT_NVARC_REPO_ID
    nvarc_cache_dir = Path(DEFAULT_NVARC_CACHE_DIR)
    nvarc_local_dir = nvarc_cache_dir / nvarc_repo_id.replace("/", "--")

    print(f"  Repository: {nvarc_repo_id}")
    print(f"  Cache directory: {nvarc_cache_dir}")
    print(f"  Target local directory: {nvarc_local_dir}")

    # Check if already downloaded (for LLM, check for config.json or model files)
    config_file = nvarc_local_dir / "config.json"
    model_files = list(nvarc_local_dir.glob("*.safetensors")) + list(nvarc_local_dir.glob("*.bin"))
    if nvarc_local_dir.exists() and (config_file.exists() or model_files):
        print(f"\n  ✓ NVARC model already exists at {nvarc_local_dir}")
        if model_files:
            print(f"    Found model file: {model_files[0].name}")
        return {
            "model": "nvarc",
            "repo_id": nvarc_repo_id,
            "status": "success",
            "message": f"NVARC model already cached at {nvarc_local_dir}",
        }

    try:
        print("\n  [Downloading] NVARC model from Hugging Face...")

        nvarc_cache_dir.mkdir(parents=True, exist_ok=True)
        nvarc_local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = download_model_with_retry(
            repo_id=nvarc_repo_id,
            cache_dir=str(nvarc_cache_dir.parent),
            local_dir=str(nvarc_local_dir)
        )

        # Verify model files exist
        model_files = list(Path(downloaded_path).glob("*.safetensors")) + list(Path(downloaded_path).glob("*.bin"))
        config_file = Path(downloaded_path) / "config.json"

        if not config_file.exists() and not model_files:
            raise FileNotFoundError(f"No model files found in {downloaded_path}")

        print(f"  ✓ NVARC model downloaded to: {downloaded_path}")
        if model_files:
            print(f"  ✓ Model file: {model_files[0].name}")
        files_count = len(list(Path(downloaded_path).glob('*')))
        print(f"  ✓ Total files in directory: {files_count}")

        return {
            "model": "nvarc",
            "repo_id": nvarc_repo_id,
            "status": "success",
            "message": f"NVARC model downloaded to {downloaded_path}",
        }

    except Exception as e:
        print(f"  ERROR: Could not download NVARC model: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": "nvarc",
            "repo_id": nvarc_repo_id,
            "status": "failed",
            "message": str(e),
        }


def run_prep_phase(input_dir: Path = None, output_dir: Path = None) -> None:
    """Prep phase: download model checkpoints based on ARC_SOLVER setting.

    Args:
        input_dir: Input directory (not used, but required by validator interface)
        output_dir: Output directory (not used, but required by validator interface)
    """
    print("\n" + "=" * 60)
    print("PREP PHASE - Downloading Model Checkpoints")
    print("=" * 60)
    print(f"Solver mode: {ARC_SOLVER}")
    print("(This phase requires internet access)")

    results = []

    # Download VARC model if needed
    if ARC_SOLVER in ("varc", "both"):
        varc_result = download_varc_model()
        results.append(varc_result)

    # Download NVARC model if needed
    if ARC_SOLVER in ("nvarc", "both"):
        nvarc_result = download_nvarc_model()
        results.append(nvarc_result)

    # Check overall status
    all_success = all(r["status"] == "success" for r in results)

    print("\n" + "=" * 60)
    print(f"PREP PHASE COMPLETED - Status: {'success' if all_success else 'failed'}")
    for r in results:
        print(f"  {r['model'].upper()}: {r['status']}")
    print("=" * 60)

    if not all_success:
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

    run_prep_phase(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except Exception as e:
        print(f"\nERROR (prep phase): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)