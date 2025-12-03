import os
from pathlib import Path
from typing import List, Dict, Optional

import torch

from VARC.src.ARC_ViT import ARCViT  # or ARC_UNet if you prefer
from VARC.utils.eval_utils import IGNORE_INDEX, PAD_INDEX


# Hugging Face repo ID for VARC checkpoint
DEFAULT_VARC_REPO_ID = os.environ.get(
    "VARC_REPO_ID",
    "VisionARC/offline_train_ViT",
)

# Local cache directory for VARC checkpoints
DEFAULT_VARC_CACHE_DIR = os.environ.get(
    "VARC_CACHE_DIR",
    "VARC/saves",
)


class ARCSolver:
    """
    ARC solver backed by a pretrained VARC (Vision ARC) model.

    Public interface:
    - constructor exists
    - solve(train_examples, test_input) -> 2D int grid
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        image_size: int = 30,
        num_colors: int = 12,  # 10 colors + IGNORE_INDEX + PAD_INDEX
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        patch_size: int = 2,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.num_colors = num_colors

        # Determine checkpoint path
        if checkpoint_path:
            # Use explicit local path if provided
            ckpt_path = checkpoint_path
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"VARC checkpoint not found at {ckpt_path}. "
                    f"Please ensure the checkpoint was downloaded during prep phase."
                )
        else:
            # Look for checkpoint in local cache (downloaded during prep phase)
            repo_id = repo_id or DEFAULT_VARC_REPO_ID
            cache_dir = cache_dir or DEFAULT_VARC_CACHE_DIR
            
            local_dir = Path(cache_dir) / repo_id.replace("/", "--")
            
            # Check if checkpoint exists locally
            checkpoint_files = list(local_dir.glob("*.pth")) + list(local_dir.glob("*.pt"))
            if checkpoint_files:
                ckpt_path = str(checkpoint_files[0])
                print(f"✓ Found cached VARC checkpoint: {ckpt_path}")
            else:
                raise FileNotFoundError(
                    f"VARC checkpoint not found in cache directory: {local_dir}\n"
                    f"Expected repo: {repo_id}\n"
                    f"Cache dir: {cache_dir}\n"
                    f"Please ensure the checkpoint was downloaded during prep phase.\n"
                    f"You can also set VARC_CHECKPOINT env var or pass checkpoint_path explicitly."
                )

        # NOTE: we treat "this ARC-AGI-2 task" as a single task_id = 0
        num_tasks = 1

        # Build model architecture matching the offline‑trained VARC config
        self.model = ARCViT(
            num_tasks=num_tasks,
            image_size=image_size,
            num_colors=num_colors,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_task_tokens=1,
            patch_size=patch_size,
        ).to(self.device)

        # Load weights
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        # Strip possible DDP prefixes
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        print(f"VARC ARCSolver initialized from {ckpt_path} on {self.device}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]],
    ) -> List[List[int]]:
        """
        Args:
            train_examples: List of {'input': grid, 'output': grid}.
            test_input: test input grid.

        Returns:
            2D grid (list of lists) of ints in [0, 9].
        """
        # For now we ignore train_examples (no full TTT here) and run a
        # single forward pass conditioned on a fixed task token.
        logits = self._forward_single_example(test_input)
        pred_grid = logits.argmax(dim=1)[0].cpu().tolist()

        # Strip padding / border if present
        pred_grid = self._strip_padding(pred_grid)

        # Clamp values to valid ARC colors 0–9
        pred_grid = [[int(max(0, min(9, v))) for v in row] for row in pred_grid]

        return pred_grid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_single_example(self, input_grid: List[List[int]]) -> torch.Tensor:
        """
        Run VARC on a single input grid.

        Returns:
            logits: (1, num_colors, H, W) tensor
        """
        h = len(input_grid)
        w = len(input_grid[0]) if h > 0 else 0
        if h == 0 or w == 0:
            raise ValueError("Empty input grid.")

        if h > self.image_size - 2 or w > self.image_size - 2:
            raise ValueError(
                f"Grid {h}x{w} is too large for image_size={self.image_size}."
            )

        # Place the grid on a 30x30 canvas with a 1‑pixel border (like VARC loader)
        canvas = torch.full(
            (self.image_size, self.image_size),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.long)

        x_offset, y_offset = 1, 1
        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = torch.tensor(
            input_grid, dtype=torch.long
        )
        mask[y_offset:y_offset + h, x_offset:x_offset + w] = 1

        # Add PAD border on the right/bottom so `_extrac_grid`‑style post‑processing works.
        canvas[y_offset:y_offset + h, x_offset + w] = PAD_INDEX
        canvas[y_offset + h, x_offset:x_offset + w + 1] = PAD_INDEX
        mask[y_offset:y_offset + h + 1, x_offset:x_offset + w + 1] = 1

        canvas = canvas.unsqueeze(0).to(self.device)         # (1, H, W)
        mask = mask.unsqueeze(0).to(self.device)             # (1, H, W)
        task_ids = torch.zeros(1, dtype=torch.long).to(self.device)  # single task_id = 0

        with torch.no_grad():
            logits = self.model(canvas, task_ids, attention_mask=mask)

        return logits

    def _strip_padding(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Remove VARC PAD_INDEX border and trailing padding, returning a tight grid.
        """
        import numpy as np

        arr = np.array(grid)
        # Remove rows/cols that are entirely IGNORE_INDEX or PAD_INDEX
        mask = ~np.isin(arr, [IGNORE_INDEX, PAD_INDEX])
        if not mask.any():
            return [[0]]  # extremely degenerate case

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        cropped = arr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
        return cropped.tolist()