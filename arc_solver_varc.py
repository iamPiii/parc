import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable

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

        # Multi‑view inference hyperparameters
        self.num_inference_attempts = 10
        
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
        # Deterministic base seed per problem
        seed_input = json.dumps(test_input)
        base_seed = int.from_bytes(seed_input.encode("utf-8"), "little") % (2**31)

        # Build auxiliary tasks (identity + 5 geoms × 10 color perms) once
        aux_tasks = self._build_auxiliary_tasks(
            train_examples=train_examples,
            test_input=test_input,
            base_seed=base_seed,
        )

        if self.enable_ttt and train_examples:
            # Perform test-time training on auxiliary tasks
            model = self._test_time_train(aux_tasks)
        else:
            # Use original model (zero-shot)
            model = self.model

        # Multi‑view inference with the same auxiliary tasks
        pred_grid = self._multi_view_inference(
            test_input,
            model=model,
            aux_tasks=aux_tasks,
            base_seed=base_seed,
        )

        # Clamp values to valid ARC colors 0–9
        pred_grid = [[int(max(0, min(9, v))) for v in row] for row in pred_grid]

        return pred_grid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _test_time_train(self, aux_tasks: List[Dict]) -> torch.nn.Module:
        """
        Perform test-time training on training examples with augmentation.
        
        Returns:
            Fine-tuned model (deep copy, original model unchanged)
        """

        # Prepare augmented training data with unique task IDs (aligned with aux tasks)
        train_data, num_augmented_tasks = self._prepare_ttt_data_from_auxiliary_tasks(aux_tasks)
        if not train_data:
            print("TTT skipped: no usable augmented training data.")
            return self.model

        
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
    
    def _prepare_ttt_data_from_auxiliary_tasks(self, aux_tasks: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Prepare TTT batches directly from pre-built auxiliary tasks.
        
        Each auxiliary task already encodes the geometric + color transform. We only apply
        resolution augmentation and random translation like ARCDataset.process_per_example,
        and we keep the provided task_id for its task embedding.
        """
        data: List[Dict] = []
        if not aux_tasks:
            return data, 0

        rng_py = random.Random(random.randint(0, 2**31 - 1))
        max_img_size = self.image_size - 2  # leave border on each side

        for aux in aux_tasks:
            task: Task = aux["task"]
            task_id = aux["task_id"]

            for train_example in task.train_examples:
                aug_input = train_example.input
                aug_output = train_example.output

                input_grid_list = aug_input.tolist() if isinstance(aug_input, np.ndarray) else aug_input
                output_grid_list = aug_output.tolist() if isinstance(aug_output, np.ndarray) else aug_output
                example = {"input": input_grid_list, "output": output_grid_list}

                # Resolution augmentation (identical to ARCDataset.process_per_example)
                max_cur_y = len(example["input"])
                max_cur_x = len(example["input"][0])
                max_cur_y = max(max_cur_y, len(example["output"]))
                max_cur_x = max(max_cur_x, len(example["output"][0]))

                max_len = max(max_cur_x, max_cur_y)
                max_scale_factor = (max_img_size // max_len) if max_len > 0 else 1

                if max_scale_factor > 1:
                    example, scale_factor = resolution_augmentation(
                        example, max_cur_x, max_cur_y, rng_py, img_size=max_img_size
                    )
                else:
                    scale_factor = 1

                max_cur_x *= scale_factor
                max_cur_y *= scale_factor

                # Skip if grids are too large after scaling
                if max_cur_y > max_img_size or max_cur_x > max_img_size:
                    continue

                # Random translation
                x_offset = rng_py.randint(1, max_img_size - max_cur_x + 1) if max_img_size > max_cur_x else 1
                y_offset = rng_py.randint(1, max_img_size - max_cur_y + 1) if max_img_size > max_cur_y else 1

                input_tensor, input_mask, _, _ = pad_grid_with_translation(
                    example["input"], self.image_size, x_offset, y_offset, output_shape=False
                )
                target_tensor, target_mask, _, _ = pad_grid_with_translation(
                    example["output"], self.image_size, x_offset, y_offset, output_shape=True
                )
                target_tensor = target_tensor.clone()
                target_tensor[target_mask == 0] = IGNORE_INDEX

                data.append(
                    {
                        "input": input_tensor,
                        "attention_mask": input_mask,
                        "target": target_tensor,
                        "task_id": task_id,
                    }
                )

        num_tasks = len({item["task_id"] for item in data}) if data else len(aux_tasks)
        return data, num_tasks
    
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

    # ------------------------------------------------------------------
    # Inference helpers: multi‑view + majority vote
    # ------------------------------------------------------------------

    def _multi_view_inference(
        self,
        input_grid: List[List[int]],
        model: Optional[torch.nn.Module] = None,
        aux_tasks: Optional[List[Dict]] = None,
        base_seed: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Run multi-view inference by reusing VARC's augmentation strategy:
        - Build a Task from train_examples + the current test input.
        - Apply the same geometric + color augmenters (identity + 5 basics + 9 perms).
        - For each augmented test example, run resolution augmentation + translation
          exactly like ARCDataset.process_per_example.
        - Run the model on each view, undo transforms, downsample if needed, revert
          color maps, then majority vote.
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

        # Deterministic base seed per problem if not provided
        if base_seed is None:
            seed_input = json.dumps(input_grid)
            base_seed = int.from_bytes(seed_input.encode("utf-8"), "little") % (2**31)

        # Build augmented auxiliary tasks (51) if not precomputed
        if aux_tasks is None:
            raise ValueError("aux_tasks must be provided for inference to align with TTT.")

        if not aux_tasks:
            raise RuntimeError(
                "Failed to build auxiliary tasks. This should not happen if train_examples are provided."
            )

        predictions: List[List[List[int]]] = []

        # For each auxiliary task, run num_inference_attempts
        # stochastic views (resolution + translation), then aggregate over all.
        rng_seq = RandomState(base_seed or 1)

        def next_seed() -> int:
            return int(rng_seq.randint(0, 2**31 - 1))

        for _ in range(self.num_inference_attempts):
            for aux in aux_tasks:
                task_obj: Task = aux["task"]
                undo_fn = aux["undo_fn"]
                color_inverse_map = aux["color_inverse_map"]
                task_id = aux["task_id"]

                # Compute max dimensions from AUGMENTED train examples in this task
                max_aug_train_output_h = 0
                max_aug_train_output_w = 0
                for train_ex in task_obj.train_examples:
                    out_h = train_ex.output.shape[0] if len(train_ex.output.shape) >= 2 else 0
                    out_w = train_ex.output.shape[1] if len(train_ex.output.shape) >= 2 else 0
                    max_aug_train_output_h = max(max_aug_train_output_h, out_h)
                    max_aug_train_output_w = max(max_aug_train_output_w, out_w)

                example_dict = {
                    "input": deepcopy(task_obj.test_example.input.tolist()),
                    "output": deepcopy(task_obj.test_example.output.tolist()),
                }
                view = self._process_example_for_inference(
                    example=example_dict,
                    seed=next_seed(),
                    max_train_output_h=max_aug_train_output_h,
                    max_train_output_w=max_aug_train_output_w,
                )
                if view is None:
                    continue

                inputs = view["inputs"].unsqueeze(0).to(self.device)
                attention_mask = view["attention_mask"].unsqueeze(0).to(self.device)
                task_ids = torch.tensor([task_id], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    logits = model(inputs, task_ids, attention_mask=attention_mask)

                pred_full = logits.argmax(dim=1)[0].cpu().tolist()
                pred_grid = self._extract_output_grid(
                    pred_full,
                    x_offset=view["x_offset"],
                    y_offset=view["y_offset"],
                )

                # Undo spatial transform
                pred_arr = undo_fn(np.array(pred_grid, dtype=int))
                pred_grid = pred_arr.tolist()

                # Downsample if resolution augmentation scaled the example
                if view["scale_factor"] > 1:
                    pred_grid = self._downsample_by_scale(pred_grid, view["scale_factor"])

                # Revert color permutations if any
                if color_inverse_map:
                    pred_grid = self._apply_color_inverse(pred_grid, color_inverse_map)

                predictions.append(pred_grid)

        voted = self._get_majority_vote(predictions)
        if not voted:
            return predictions[-1]

        return voted[0]["prediction"]

    @staticmethod
    def _get_majority_vote(predictions: List[List[List[int]]]) -> List[Dict]:
        """
        Majority vote over a list of grid predictions.
        Mirrors VARC's utils.eval_utils_ttt.get_majority_vote.
        """
        vote_count: Dict[str, int] = {}
        list_map: Dict[str, List[List[int]]] = {}

        for grid in predictions:
            label = json.dumps(grid)
            list_map[label] = grid
            if label not in vote_count:
                vote_count[label] = 0
            vote_count[label] += 1

        if not vote_count:
            return []

        sorted_votes = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)
        sorted_lists = [
            {"prediction": list_map[item[0]], "votes": item[1]} for item in sorted_votes
        ]
        return sorted_lists

    @staticmethod
    def _downsample_by_scale(predict_grid: List[List[int]], scale_factor: int) -> List[List[int]]:
        """
        Downsample a high‑resolution prediction grid back to the base resolution
        by block‑wise majority vote, mirroring VARC's eval_utils_ttt logic.
        """
        if scale_factor <= 1:
            return predict_grid

        h = len(predict_grid)
        w = len(predict_grid[0]) if h > 0 else 0
        if h == 0 or w == 0:
            return predict_grid

        downsampled: List[List[int]] = []
        for i in range(0, h, scale_factor):
            row: List[int] = []
            for j in range(0, w, scale_factor):
                block: List[int] = []
                for di in range(scale_factor):
                    for dj in range(scale_factor):
                        yi = i + di
                        xj = j + dj
                        if yi < h and xj < w:
                            block.append(int(predict_grid[yi][xj]))
                if block:
                    counts = np.bincount(block)
                    majority_value = int(np.argmax(counts))
                    row.append(majority_value)
            if row:
                downsampled.append(row)

        return downsampled

    # ------------------------------------------------------------------
    # Augmented view generation helpers
    # ------------------------------------------------------------------

    def _build_auxiliary_tasks(
        self,
        train_examples: Optional[List[Dict]],
        test_input: List[List[int]],
        base_seed: int,
    ) -> List[Dict]:
        """
        Build 51 tasks: 1 original + 5 geometric transforms × 10 color permutations.
        Each task gets a distinct task_id and an undo/color map for inference.
        """
        arc_train_examples: List[Example] = []
        for example in train_examples:
            if "input" not in example or "output" not in example:
                continue
            arc_train_examples.append(
                Example(input=np.array(example["input"]), output=np.array(example["output"]))
            )

        if not arc_train_examples:
            return []

        test_example = Example(
            input=np.array(test_input),
            output=np.array(test_input),
        )

        base_task = Task(
            name="",
            train_examples=arc_train_examples,
            test_example=test_example,
        )

        rng_np = RandomState(base_seed or 1)
        aux_tasks: List[Dict] = []

        def next_seed() -> int:
            return int(rng_np.randint(0, 2**31 - 1))

        task_id_counter = 0

        # Original task (identity, no color permutation)
        aux_tasks.append(
            {
                "task": base_task,
                "undo_fn": self._identity_np,
                "color_inverse_map": None,
                "task_id": task_id_counter,
            }
        )
        task_id_counter += 1

        basic_augmenters = get_basic_augmenters()
        NUM_COLOR_PERMUTES = 9

        for augmenter in basic_augmenters:
            geom_task = augmenter.apply_to_task(
                base_task,
                rng=RandomState(next_seed()),
                to_input=True,
                to_output=True,
            )
            if not self._task_fits_canvas(geom_task):
                continue

            undo_fn = self._get_undo_fn_for_augmenter(augmenter)

            # Add geometric-only auxiliary task
            aux_tasks.append(
                {
                    "task": geom_task,
                    "undo_fn": undo_fn,
                    "color_inverse_map": None,
                    "task_id": task_id_counter,
                }
            )
            task_id_counter += 1

            for _ in range(NUM_COLOR_PERMUTES):
                perm_augmenter = PermuteColors()
                perm_task = perm_augmenter.apply_to_task(
                    geom_task,
                    rng=RandomState(next_seed()),
                    to_input=True,
                    to_output=True,
                    use_test_output=False,
                )
                if not self._task_fits_canvas(perm_task):
                    continue
                color_map = getattr(perm_augmenter, "_color_map", None)
                color_inverse = {v: k for k, v in color_map.items()} if color_map else None
                aux_tasks.append(
                    {
                        "task": perm_task,
                        "undo_fn": undo_fn,
                        "color_inverse_map": color_inverse,
                        "task_id": task_id_counter,
                    }
                )
                task_id_counter += 1

        return aux_tasks

    def _process_example_for_inference(
        self,
        example: Dict[str, List[List[int]]],
        seed: int,
        max_train_output_h: int = 0,
        max_train_output_w: int = 0,
    ) -> Optional[Dict]:
        """
        Mirror ARCDataset.process_per_example for a single example (resolution aug +
        translation + padding). Returns tensors and metadata required for inference.
        """
        max_img_size = self.image_size - 2
        example_copy = {
            "input": deepcopy(example["input"]),
            "output": deepcopy(example["output"]),
        }

        rng = random.Random(seed or 1)

        # Compute bounds from test input + max train output dimensions
        test_input_h = len(example_copy["input"])
        test_input_w = len(example_copy["input"][0]) if test_input_h > 0 else 0
        max_cur_y = max(test_input_h, max_train_output_h)
        max_cur_x = max(test_input_w, max_train_output_w)

        if max_cur_y == 0 or max_cur_x == 0:
            return None

        if max_cur_y > max_img_size or max_cur_x > max_img_size:
            return None

        max_len = max(max_cur_x, max_cur_y)
        max_scale_factor = (max_img_size // max_len) if max_len > 0 else 1

        if max_scale_factor > 1:
            example_copy, scale_factor = resolution_augmentation(
                example_copy, max_cur_x, max_cur_y, rng, img_size=max_img_size
            )
        else:
            scale_factor = 1

        max_cur_x = max_cur_x * scale_factor
        max_cur_y = max_cur_y * scale_factor

        # Check if scaled dimensions fit
        if max_cur_y > max_img_size or max_cur_x > max_img_size:
            return None

        # Translation offsets: use theoretical scaled dimensions
        if max_img_size > max_cur_x:
            x_offset = rng.randint(1, max_img_size - max_cur_x)
        else:
            x_offset = 1

        if max_img_size > max_cur_y:
            y_offset = rng.randint(1, max_img_size - max_cur_y)
        else:
            y_offset = 1

        input_tensor, input_mask, _, _ = pad_grid_with_translation(
            example_copy["input"], self.image_size, x_offset, y_offset, output_shape=False
        )

        return {
            "inputs": input_tensor,
            "attention_mask": input_mask,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "scale_factor": scale_factor,
        }

    def _task_fits_canvas(self, task: Task) -> bool:
        max_dim = self.image_size - 2
        return task.max_height() <= max_dim and task.max_width() <= max_dim

    @staticmethod
    def _identity_np(arr: np.ndarray) -> np.ndarray:
        return arr

    def _get_undo_fn_for_augmenter(self, augmenter) -> Callable[[np.ndarray], np.ndarray]:
        from VARC.utils.arclib.augmenters import Rotate, Flip  # local import to avoid cycle

        if isinstance(augmenter, Rotate):
            angle = augmenter.angle
            if angle == 90:
                return lambda arr: np.rot90(arr, k=3)
            if angle == 180:
                return lambda arr: np.rot90(arr, k=2)
            if angle == 270:
                return lambda arr: np.rot90(arr, k=1)
        elif isinstance(augmenter, Flip):
            axis = augmenter.axis
            if axis == 0:
                return lambda arr: np.flipud(arr)
            if axis == 1:
                return lambda arr: np.fliplr(arr)
        return self._identity_np

    @staticmethod
    def _apply_color_inverse(
        grid: List[List[int]], color_inverse_map: Dict[int, int]
    ) -> List[List[int]]:
        return [
            [color_inverse_map.get(int(value), int(value)) for value in row]
            for row in grid
        ]


    def _extract_output_grid(self, pred: List[List[int]], x_offset: int, y_offset: int) -> List[List[int]]:
        """
        Extract output grid from model prediction using VARC's extrac_grid logic.
        
        Args:
            pred: Full canvas prediction (image_size x image_size)
            x_offset: X offset where input was placed
            y_offset: Y offset where input was placed
            
        Returns:
            Extracted output grid (cropped to actual output size)
        """
        
        np_predict = np.array(pred).reshape(self.image_size, self.image_size)
        np_predict_grid = np_predict[y_offset:, x_offset:]
        
        # Find the actual output dimensions by scanning until PAD_INDEX
        len_x, len_y = 0, 0
        while len_x < np_predict_grid.shape[1] and np_predict_grid[0][len_x] != PAD_INDEX:
            len_x += 1

        while len_y < np_predict_grid.shape[0] and np_predict_grid[len_y][0] != PAD_INDEX:
            len_y += 1
        
        # If PAD_INDEX not found, fall back to finding actual content region
        if len_x >= np_predict_grid.shape[1] or len_y >= np_predict_grid.shape[0]:
            mask = ~np.isin(np_predict_grid, [IGNORE_INDEX, PAD_INDEX])
            if mask.any():
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                if len(rows) > 0 and len(cols) > 0:
                    len_y = rows.max() + 1
                    len_x = cols.max() + 1
                else:
                    return [[0]]
            else:
                return [[0]]
        
        # Extract the actual output grid
        predict_grid = np_predict_grid[:len_y, :len_x].tolist()
        
        if not predict_grid or (len_y == 0 or len_x == 0):
            return [[0]]
        
        return predict_grid