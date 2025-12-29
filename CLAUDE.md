# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **ARC-AGI-2 Solver** for the Abstraction and Reasoning Corpus competition. It implements two complementary solving approaches:

- **NVARC** (default): LLM-based solver using a fine-tuned Qwen model with beam search inference
- **VARC**: Vision Transformer-based solver treating ARC as image-to-image translation with test-time training (TTT)

The project integrates with **Hone Subnet** - a Bittensor subnet for benchmarking ARC solvers in secure GPU sandboxes.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run phases (main execution flow)
python arc_main.py --phase prep --input /input --output /output
python arc_main.py --phase inference --input /input --output /output

# Select solver via environment variable
ARC_SOLVER=nvarc python arc_main.py --phase inference ...  # LLM-based (default)
ARC_SOLVER=varc python arc_main.py --phase inference ...   # Vision-based

# Docker build and run
docker build -t arc-agi-2 .
docker run --gpus all -v /input:/input -v /output:/output arc-agi-2
```

## Architecture

### Two-Phase Execution Model

1. **Prep Phase** (`arc_prep_phase.py`) - Has internet access
   - Downloads models from Hugging Face
   - NVARC: `iamPi/Hwen-HF`
   - VARC: `VisionARC/offline_train_ViT`
   - Caches to `/app/models`

2. **Inference Phase** (`arc_inference_phase.py`) - No internet, GPU isolated
   - Loads cached models
   - Reads tasks from `/input/miner_current_dataset.json`
   - Writes predictions to `/output/results.json`

### Solver Interface

Both solvers implement the same interface:

```python
class ARCSolver:
    def solve(
        self,
        train_examples: List[Dict],  # [{"input": grid, "output": grid}, ...]
        test_input: List[List[int]]  # 2D grid of ints 0-9
    ) -> List[List[int]]             # 2D grid prediction
```

### Key Files

| File | Purpose |
|------|---------|
| `arc_main.py` | CLI wrapper routing to phase scripts |
| `arc_prep_phase.py` | Model downloading (internet enabled) |
| `arc_inference_phase.py` | Inference execution (internet disabled) |
| `arc_solver_nvarc.py` | LLM solver with Qwen + turbo DFS beam search |
| `arc_solver_varc.py` | Vision solver with ViT + test-time training |
| `arc_utils.py` | Shared I/O utilities |

### NVARC Implementation Details

- Uses fixed 16-token ARC vocabulary: digits 0-9, newline (Ċ), and special tokens
- Grid-to-string conversion for tokenization
- Turbo DFS beam search for efficient inference
- No padding/attention_mask (matches original NVARC implementation)

### VARC Implementation Details

- 18M parameter Vision Transformer (or 55M UNet variant)
- Test-time training on each task's training examples
- Multi-view inference with color permutation augmentation
- Image size: 64x64, patch size: 2

## Data Formats

**Input** (`miner_current_dataset.json`):
```json
{
  "tasks": [{
    "task_hash": "abc123",
    "train_examples": [{"input": [[0,1],[2,3]], "output": [[3,2],[1,0]]}],
    "test_input": [[1,2],[3,4]]
  }]
}
```

**Output** (`results.json`):
```json
{
  "phase": "inference",
  "status": "success",
  "predictions": [{
    "problem_index": 0,
    "task_hash": "abc123",
    "predicted_output": [[4,3],[2,1]]
  }]
}
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARC_SOLVER` | `nvarc` | Solver selection: `nvarc` or `varc` |
| `NVARC_REPO_ID` | `iamPi/Hwen-HF` | HuggingFace model repo |
| `VARC_REPO_ID` | `VisionARC/offline_train_ViT` | VARC checkpoint repo |
| `NVARC_CACHE_DIR` | `/app/models` | Model cache directory |
| `VARC_CACHE_DIR` | `/app/models` | VARC cache directory |

## Subdirectories

- `NVARC/` - Original NVARC implementation (reference code with training)
- `VARC/` - Vision ARC approach with offline training, TTT scripts, and model architectures
- `hone/` - Bittensor subnet integration (validator, miner, sandbox runner)

## Docker Environment

- Base: `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04`
- Python: 3.11
- PyTorch: 2.7.0+cu126
- Torch compilation disabled (`TORCH_COMPILE=0`, `TORCHINDUCTOR_DISABLE=1`)

## Running details
This parc/ repo is on my local machine. The actual server is run on a GPU-available VM with the following steps:
1. Clone the hone_server/ repo
2. Run the setup_server.sh script
3. Run the run_server.sh script
4. Run the send_request.sh script, which creates a job on the server where this repository but the hone/ and hone_server/ repos are cloned and a docker image is built and run.
5. The server will be running on http://localhost:8080

## Focus areas
- You will only work with the parc/ repo without making any changes to the hone/ and hone_server/ repos, these are read-only.