import os
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.amp import GradScaler
from numpy.random import RandomState

from VARC.src.ARC_ViT import ARCViT
from VARC.src.ARC_loader import pad_grid_with_translation, resolution_augmentation
from VARC.utils.eval_utils import IGNORE_INDEX, PAD_INDEX
from VARC.utils.lr_scheduler import get_cosine_schedule_with_warmup
from VARC.utils.preprocess import get_basic_augmenters
from VARC.utils.arclib.arc import Example, Task
from VARC.utils.arclib.augmenters import PermuteColors


# Hugging Face repo ID for VARC checkpoint
DEFAULT_VARC_REPO_ID = os.environ.get(
    "VARC_REPO_ID",
    "VisionARC/offline_train_ViT",
)

# Local cache directory for VARC checkpoints.
DEFAULT_VARC_CACHE_DIR = os.environ.get("VARC_CACHE_DIR", "/app/models")

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
        image_size: int = 64,
        num_colors: int = 12,  # 10 colors + IGNORE_INDEX + PAD_INDEX
        embed_dim: int = 512,
        depth: int = 10,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        patch_size: int = 2,
        device: Optional[str] = None,
        # TTT hyperparameters
        ttt_epochs: int = 100,
        ttt_warmup_epochs: int = 10,
        ttt_learning_rate: float = 3e-4,
        ttt_batch_size: int = 8,
        enable_ttt: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.num_colors = num_colors

        # TTT hyperparameters
        self.ttt_epochs = ttt_epochs
        self.ttt_warmup_epochs = ttt_warmup_epochs
        self.ttt_learning_rate = ttt_learning_rate
        self.ttt_batch_size = ttt_batch_size
        self.enable_ttt = enable_ttt
        
        # Determine checkpoint path
        ckpt_path: Optional[str] = None
        if checkpoint_path is not None:
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
            checkpoint_best = local_dir / "checkpoint_best.pt"
            checkpoint_final = local_dir / "checkpoint_final.pt"
            
            if checkpoint_best.exists():
                ckpt_path = str(checkpoint_best)
                print(f"✓ Found cached VARC checkpoint: {ckpt_path} (best checkpoint)")
            elif checkpoint_final.exists():
                ckpt_path = str(checkpoint_final)
                print(f"✓ Found cached VARC checkpoint: {ckpt_path} (final checkpoint)")
            else:
                raise FileNotFoundError(
                    f"VARC checkpoint not found in cache directory: {local_dir}\n"
                    f"Expected repo: {repo_id}\n"
                    f"Cache dir: {cache_dir}\n"
                    f"Please ensure the checkpoint was downloaded during prep phase.\n"
                    f"You can also set VARC_CHECKPOINT env var or pass checkpoint_path explicitly."
                )

        # Load checkpoint metadata (before building model) to discover task-token count
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint)
        # Strip possible DDP prefixes
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        task_token_weight = state_dict.get("task_token_embed.weight")
        num_tasks = task_token_weight.shape[0] if task_token_weight is not None else 1
        self.original_num_tasks = num_tasks

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
        if self.enable_ttt and train_examples:
            # Perform test-time training on training examples
            model = self._test_time_train(train_examples)
        else:
            # Use original model (zero-shot)
            model = self.model

        # Run inference on test input with the (possibly fine-tuned) model
        logits = self._forward_single_example(test_input, model=model)
        pred_grid = logits.argmax(dim=1)[0].cpu().tolist()

        # Strip padding / border if present
        pred_grid = self._strip_padding(pred_grid)

        # Clamp values to valid ARC colors 0–9
        pred_grid = [[int(max(0, min(9, v))) for v in row] for row in pred_grid]

        return pred_grid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _test_time_train(self, train_examples: List[Dict]) -> torch.nn.Module:
        """
        Perform test-time training on training examples with augmentation.
        
        Returns:
            Fine-tuned model (deep copy, original model unchanged)
        """

        # Prepare augmented training data with unique task IDs
        train_data, num_augmented_tasks = self._prepare_ttt_data_with_augmentation(train_examples)

        
        # Deep copy model to avoid modifying the original
        model = deepcopy(self.model)
        
        # Expand model to support all augmented tasks if needed
        if num_augmented_tasks > self.original_num_tasks:
            model = self._expand_model_task_tokens(model, num_augmented_tasks)
        
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.ttt_learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        
        scaler = GradScaler(enabled=(self.device.type == "cuda"))
        autocast_device_type = self.device.type if self.device.type in {"cuda", "cpu", "mps"} else "cuda"
        
        # Calculate total training steps (steps = batches across all epochs)
        steps_per_epoch = (len(train_data) + self.ttt_batch_size - 1) // self.ttt_batch_size
        total_training_steps = steps_per_epoch * self.ttt_epochs
        num_warmup_steps = steps_per_epoch * self.ttt_warmup_epochs
        
        # Create cosine learning rate scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )
        
        # Training loop
        for epoch in range(self.ttt_epochs):
            random.shuffle(train_data)
            
            # Process in batches
            for i in range(0, len(train_data), self.ttt_batch_size):
                batch = train_data[i:i + self.ttt_batch_size]
                
                # Stack inputs, masks, targets, task_ids
                inputs = torch.stack([item["input"] for item in batch]).to(self.device)
                attention_masks = torch.stack([item["attention_mask"] for item in batch]).to(self.device)
                targets = torch.stack([item["target"] for item in batch]).to(self.device)
                task_ids = torch.tensor([item["task_id"] for item in batch], dtype=torch.long).to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with mixed precision
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    logits = model(inputs, task_ids, attention_mask=attention_masks)
                    num_colors = logits.size(1)
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
                    loss = F.cross_entropy(
                        logits_flat,
                        targets.view(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate scheduler (step per batch)
                scheduler.step()
        
        model.eval()
        return model
    
    def _prepare_ttt_data_with_augmentation(self, train_examples: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Prepare training examples for TTT with augmentation.
        Creates augmented versions following VARC's approach:
        - For each training example, create a Task where that example is the test_example
          and other examples are train_examples
        - Augment the entire task (both train and test examples together)
        - Each augmented task gets a unique task_id
        
        Returns:
            Tuple of (list of dicts with 'input', 'attention_mask', 'target', 'task_id', and number of unique tasks)
        """
        # 5 Geometric augmenters used by VARC for TTT data generation
        basic_augmenters = get_basic_augmenters()
        
        data = []
        task_id_counter = 0
        rng_np = RandomState(42)  # For augmenters
        
        # Convert train_examples to ARC Example format
        arc_examples = []
        for example in train_examples:
            if "input" not in example or "output" not in example:
                continue
            
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])
            
            # Check size constraints
            h = max(len(input_grid), len(output_grid))
            w = max(input_grid.shape[1] if len(input_grid.shape) > 1 else 0,
                   output_grid.shape[1] if len(output_grid.shape) > 1 else 0)
            
            if h > self.image_size - 2 or w > self.image_size - 2:
                continue  # Skip examples that are too large
            
            arc_examples.append(Example(input=input_grid, output=output_grid))
        
        if not arc_examples:
            return data, task_id_counter
        
        # Create a single task with all examples as train_examples
        # For TTT, we use all examples as training data (no test_example needed during training)
        # We'll use the first example as a placeholder test_example for the Task structure
        initial_task = Task(
            name="",
            train_examples=arc_examples,
            test_example=arc_examples[0]  # Placeholder, not used during training
        )
        
        # Augment the task (all examples get augmented together)
        augmented_tasks = []
        # Add original (identity) version
        augmented_tasks.append(initial_task)

        # For each basic augmenter:
        #   - Add one geometric variant
        #   - Add 9 color‑permuted variants of that geometric variant
        NUM_COLOR_PERMUTES = 9
        for augmenter in basic_augmenters:
            try:
                geom_task = augmenter.apply_to_task(
                    initial_task, to_input=True, to_output=True, rng=rng_np
                )
                if not (geom_task.max_height() <= self.image_size - 2 and
                        geom_task.max_width() <= self.image_size - 2):
                    continue
                augmented_tasks.append(geom_task)

                # 9 color permutations of this geometric variant
                for _ in range(NUM_COLOR_PERMUTES):
                    perm_augmenter = PermuteColors()
                    perm_task = perm_augmenter.apply_to_task(
                        geom_task, to_input=True, to_output=True, rng=rng_np
                    )
                    if (perm_task.max_height() <= self.image_size - 2 and
                            perm_task.max_width() <= self.image_size - 2):
                        augmented_tasks.append(perm_task)
            except Exception:
                continue  # Skip if augmentation fails
        
        # Process each augmented task and assign a DISTINCT task_id to each augmented task
        for aug_task in augmented_tasks:

            # Process all train_examples (these are the actual training examples)
            for train_example in aug_task.train_examples:
                aug_input = train_example.input
                aug_output = train_example.output

                # Convert to python lists for resolution_augmentation
                input_grid_list = aug_input.tolist() if isinstance(aug_input, np.ndarray) else aug_input
                output_grid_list = aug_output.tolist() if isinstance(aug_output, np.ndarray) else aug_output

                example = {"input": input_grid_list, "output": output_grid_list}

                # --- Resolution augmentation (same as ARCDataset.process_per_example) ---
                max_cur_y = len(example["input"])
                max_cur_x = len(example["input"][0])
                if "output" in example:
                    max_cur_y = max(max_cur_y, len(example["output"]))
                    max_cur_x = max(max_cur_x, len(example["output"][0]))

                max_img_size = self.image_size - 2  # leave 1‑cell border on each side
                # Apply resolution_augmentation (scales up grids)
                example, scale_factor = resolution_augmentation(
                    example, max_cur_x, max_cur_y, rng_np, img_size=max_img_size
                )

                max_cur_x = len(example["input"][0])
                max_cur_y = len(example["input"])

                # --- Random translation within the canvas (same as ARCDataset.process_per_example) ---
                if max_img_size > max_cur_x:
                    x_offset = rng_np.randint(1, max_img_size - max_cur_x) if max_img_size > max_cur_x else 1
                else:
                    x_offset = 1
                if max_img_size > max_cur_y:
                    y_offset = rng_np.randint(1, max_img_size - max_cur_y) if max_img_size > max_cur_y else 1
                else:
                    y_offset = 1

                # Process input grid
                input_tensor, input_mask, _, _ = pad_grid_with_translation(
                    example["input"], self.image_size, x_offset, y_offset, output_shape=False
                )

                # Process output grid (target)
                target_tensor, target_mask, _, _ = pad_grid_with_translation(
                    example["output"], self.image_size, x_offset, y_offset, output_shape=True
                )

                # Set invalid positions to IGNORE_INDEX
                target_tensor = target_tensor.clone()
                target_tensor[target_mask == 0] = IGNORE_INDEX

                data.append({
                    "input": input_tensor,
                    "attention_mask": input_mask,
                    "target": target_tensor,
                    "task_id": task_id_counter,
                })

            # Increment task_id for the next augmented task
            task_id_counter += 1
        
        return data, task_id_counter
    
    def _expand_model_task_tokens(self, model: torch.nn.Module, num_tasks: int) -> torch.nn.Module:
        """
        Expand model to support more task tokens by reinitializing task_token_embed.
        
        Args:
            model: Model to expand
            num_tasks: Number of tasks to support
            
        Returns:
            Model with expanded task token embeddings
        """
        if num_tasks <= model.task_token_embed.num_embeddings:
            return model
        
        # Get current embedding dimension
        embed_dim = model.embed_dim
        num_task_tokens = model.num_task_tokens
        
        # Create new task token embedding with more tasks
        new_task_token_embed = torch.nn.Embedding(
            num_tasks,
            embed_dim * num_task_tokens,
        )
        
        # Initialize new embeddings
        torch.nn.init.trunc_normal_(new_task_token_embed.weight, std=0.02)
        
        # Copy existing weights if possible
        if model.task_token_embed.num_embeddings != 0:
            old_num = model.task_token_embed.num_embeddings
            new_task_token_embed.weight.data[:old_num] = model.task_token_embed.weight.data
        
        # Replace the embedding
        model.task_token_embed = new_task_token_embed
        model = model.to(self.device)
        
        return model

    def _forward_single_example(self, input_grid: List[List[int]], model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """
        Run VARC on a single input grid.

        Args:
            input_grid: Input grid to process
            model: Model to use (defaults to self.model)

        Returns:
            logits: (1, num_colors, H, W) tensor
        """
        if model is None:
            model = self.model
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
            logits = model(canvas, task_ids, attention_mask=mask)

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