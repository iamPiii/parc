"""
ARC-AGI-2 solver using NVARC approach with test-time training

THIS FILE IMPLEMENTS THE NVARC METHODOLOGY:

- You can:
    * Replace this with your own solver (different architecture, approach, etc.)
    * Modify training hyperparameters or augmentation strategies
    * Add more sophisticated beam search or scoring methods

- You MUST preserve:
    * The class name `ARCSolver` (or at least the public interface used by
      the inference script).
    * The method signature:
          solve(train_examples: List[Dict], test_input: List[List[int]]) -> List[List[int]]
    * The output format: a rectangular 2D grid of integers 0-9

- You MUST NOT:
    * Perform any network calls in `solve()` (no internet during inference)
    * Read or depend on ground-truth `test_output` (it will fail)
"""

import gc
import io
import time
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Union
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
import logging

logging.disable(logging.WARNING)

# This is the model *downloaded in prep phase* by default
# The prep-phase script imports this name
model_name = "iamPi/Hwen"


# ------------------------------------------------------------------
# NVARC Constants and Vocabulary
# ------------------------------------------------------------------

ARC_VOCAB = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "Ċ": 10, "<|im_end|>": 15,
}
ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def convert_grid_to_string(grid) -> str:
    """Convert 2D grid to string representation"""
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()


def is_valid_solution(guess):
    """Check if output is valid"""
    return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)


# ------------------------------------------------------------------
# NVARC Components
# ------------------------------------------------------------------

class QwenFormatter:
    """Formats ARC data for Qwen model - exact implementation from notebook"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fmt_query(self, query) -> str:
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return convert_grid_to_string(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge=False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += f"<|im_start|>user\n{grid_input}<|im_end|><|im_start|>assistant\n{grid_output}<|im_end|>"
        if test is not None:
            text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
        return text

    def max_new_tokens(self):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows=30):
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except:
            pass
        return None


class QwenDataCollatorForCompletionOnlyLM:
    """Data collator that masks everything except assistant responses"""
    
    def __init__(self, tokenizer, mlm=False):
        from transformers import DataCollatorForLanguageModeling
        self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)
    
    def __call__(self, examples):
        batch = self.collator(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


class UnslothFixedTrainer:
    """Unsloth trainer with fix for view tensor issue"""
    
    def __init__(self, **kwargs):
        from unsloth import UnslothTrainer
        self.trainer = UnslothTrainer(**kwargs)
        
    def train(self):
        return self.trainer.train()
    
    @property
    def accelerator(self):
        return self.trainer.accelerator
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.trainer.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.trainer.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.trainer.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:
    """Turbo DFS beam search - exact implementation from notebook"""
    
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    suffixes = defaultdict(list)
    candidates = dict()
    
    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))
    
    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x: x[0])
    
    while time.time() - start_time < 540 and time.time() < end_time:
        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0
        
        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1
        
        if num_alive_beams == 0:
            break
        
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        
        next_suffixes = turbo_dfs(
            model, outputs.logits[:, -1], max_new_tokens - 1, max_score,
            batch_scores, pos + 1, outputs.past_key_values, start_time, end_time,
        )
        
        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))
    
    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    """Inference wrapper for turbo DFS"""
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model, outputs.logits[:, -1], max_new_tokens, max_score,
        [0.0] * input_ids.size(0), input_ids.size(1),
        outputs.past_key_values, time.time(), end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x: x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    """Calculate scores for augmented samples"""
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length - 1:query_length - 1 + len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


# ------------------------------------------------------------------
# Main Solver Class
# ------------------------------------------------------------------

class ARCSolver:
    """
    ARC solver using NVARC approach with test-time training

    You can completely replace the internals of this class as long as:

    - The constructor still exists (signature can be extended)
    - `solve(train_examples, test_input)` still returns a 2D int grid
    """

    def __init__(self, use_vllm: bool = False, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.formatter = None
        
        # If no custom path provided, use default with latest checkpoint
        if model_path is None:
            # Try to find downloaded model with checkpoint folder
            from pathlib import Path
            default_cache = Path("/app/models") / model_name.replace("/", "--") / "step_5972"
            if default_cache.exists():
                self.model_path = str(default_cache)
            else:
                # Fallback to base model name (will download from HF)
                self.model_path = model_name
        else:
            self.model_path = model_path
            
        self.max_seq_length = 8192
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # NVARC hyperparameters (exact from notebook)
        self.peft_params = dict(
            r=256,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=42,
            use_rslora=True,
            loftq_config=None,
        )
        
        self.train_args = dict(
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            warmup_steps=0,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            learning_rate=5e-5,
            optim="adamw_torch",
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
            save_strategy="no",
            eval_strategy="no",
            logging_strategy="no",
            fp16=False,
            bf16=True,
            fsdp="",
            ddp_find_unused_parameters=False,
            dataloader_num_workers=0,
            gradient_checkpointing=False,
        )
        
        self.default_weights = None
        self.model_loaded = False
        
        print("🔧 ARCSolver initialized (NVARC with test-time training, device=%s)" % self.device)

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """
        Initialize NVARC model with Unsloth and LoRA

        Loads the Hwen model and sets up LoRA adapters for test-time training.
        Note: Hwen model has nested checkpoint folders (step_5400, step_5600, step_5800, step_5972).
        We use step_5972 (latest checkpoint) by default.
        """
        if self.model_loaded:
            return
            
        try:
            from unsloth import FastLanguageModel
            from peft import get_peft_model_state_dict
            
            print(f"📥 Loading NVARC model from: {self.model_path}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                full_finetuning=False,
                load_in_4bit=False,
                use_gradient_checkpointing=False,
                max_seq_length=self.max_seq_length,
            )
            
            self.model = FastLanguageModel.get_peft_model(self.model, **self.peft_params)
            
            # Convert float32 to bfloat16
            for name, param in self.model.named_parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.to(torch.bfloat16)
            
            # Store default weights for reset between puzzles
            self.default_weights = get_peft_model_state_dict(self.model, adapter_name="default")
            self.default_weights = {k: v.clone().detach() for k, v in self.default_weights.items()}
            
            self.formatter = QwenFormatter(tokenizer=self.tokenizer)
            self.model_loaded = True
            
            print("✓ NVARC model loaded successfully")
            
        except ImportError:
            print("⚠ Unsloth or dependencies not installed")
            self.model_loaded = False
            raise
        except Exception as e:
            print(f"⚠ Failed to initialize NVARC model: {e}")
            self.model_loaded = False
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]],
    ) -> List[List[int]]:
        """
        Learn from training examples and apply to test input

        Args:
            train_examples: List of dicts with 'input' and 'output' grids.
            test_input: The test input grid to solve.

        Returns:
            A 2D grid (list of lists) of ints in [0, 9].
        """
        # Initialize model if not already loaded
        if not self.model_loaded:
            self._init_model()
        
        # Try NVARC approach with test-time training
        try:
            puzzle_data = {
                "train": train_examples,
                "test": [{"input": test_input}]
            }
            
            result = self._solve_with_nvarc(puzzle_data)
            
            if result and self._is_valid_output(result):
                print("    ✓ Solved using NVARC")
                return result
            else:
                print("    ⚠ NVARC returned invalid output, falling back to identity")
        except Exception as e:
            print(f"    ⚠ NVARC solve failed: {e}, falling back to identity")
        
        # Fallback: return identity transform
        return self._fallback_solve(test_input, train_examples)

    # ------------------------------------------------------------------
    # NVARC solving
    # ------------------------------------------------------------------

    def _solve_with_nvarc(self, puzzle_data: Dict) -> Optional[List[List[int]]]:
        """
        Solve using NVARC approach with test-time training
        
        Steps:
        1. Reset model weights
        2. Test-time training with augmentation (n=16, seed=1)
        3. Beam search inference with augmentation (n=2, seed=2)
        4. Augmented scoring
        5. Return best solution
        """
        from unsloth import FastLanguageModel, UnslothTrainingArguments
        from datasets import Dataset
        from peft import set_peft_model_state_dict
        from arc_loader_nvarc import ArcDataset
        
        start_time = time.time()
        end_time = time.time() + 1200  # 20 minutes max
        
        # Reset model weights
        torch.cuda.reset_peak_memory_stats()
        set_peft_model_state_dict(self.model, self.default_weights.copy(), adapter_name="default")
        
        # Set to training mode
        self.model = FastLanguageModel.for_training(self.model)
        
        # Create ArcDataset from puzzle data
        puzzle_key = "temp_puzzle"
        arc_dataset = ArcDataset(
            queries={puzzle_key: puzzle_data},
            replies={},
            keys=[puzzle_key],
            is_orig=False
        )
        
        # Create augmented training dataset (n=16, seed=1)
        train_ds = arc_dataset.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=self.formatter, name="text", max_len=self.max_seq_length)
        train_list = train_ds.as_list(self.formatter)
        
        # Setup data collator
        collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=self.tokenizer, mlm=False)
        
        # Test-time training
        print(f"    🔄 Test-time training with {len(train_list)} augmented samples...")
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            trainer = UnslothFixedTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_list),
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                args=UnslothTrainingArguments(**self.train_args),
            )
            stats = trainer.train()
            self.model = trainer.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
            del trainer
        
        # Set to inference mode
        self.model = FastLanguageModel.for_inference(self.model)
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"    ✓ Training complete (loss: {stats.training_loss:.6f})")
        
        # Split for multiple test inputs
        puzzle_ds_multi = arc_dataset.split_multi_replies()
        
        # Create augmented evaluation dataset (n=2, seed=2)
        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
        eval_ds = eval_ds.cut_to_len(formatter=self.formatter, name="input", max_len=self.max_seq_length - self.formatter.max_new_tokens())
        
        # Group augmented samples by test ID
        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)
        
        # Create batches
        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
        
        # Perform inference with beam search and augmented scoring
        max_new_tokens = self.formatter.max_new_tokens()
        max_score = -np.log(0.2)
        known_scores = {}
        all_solutions = defaultdict(list)
        
        with torch.inference_mode():
            for subkeys in batches[:1]:  # Process first batch
                if time.time() - start_time > 600:  # 10 minute timeout
                    break
                
                print(f"    🔍 Decoding batch...")
                
                # Prepare batch tokens
                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, self.formatter)
                    tokens.append(self.tokenizer.encode(data["input"]))
                
                # Run beam search
                dfs_result = inference_turbo_dfs(self.model, tokens, max_new_tokens, max_score, end_time)
                
                # Process results
                for subkey_id, scored_beams in dfs_result:
                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    
                    for beam_score, beam_tokens in scored_beams[:3]:  # Top 3 beams
                        array = self.formatter.convert_tokens_to_array(beam_tokens)
                        if array is None:
                            continue
                        
                        # Invert augmentation to get original solution
                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                        grid_id = (bk, tuple(map(tuple, solution)))
                        
                        # Score with augmented samples
                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"    📊 Scoring solution...")
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024 ** 2)
                            aug_dataset = aug_dataset.cut_to_len(formatter=self.formatter, name="input", max_len=self.max_seq_length - max_new_tokens)
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(self.formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], self.tokenizer, self.model)
                            augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], self.tokenizer, self.model)
                            augmented_scores = augmented_scores1 + augmented_scores2
                            known_scores[grid_id] = augmented_scores
                        
                        # Store solution with scores
                        total_score = beam_score + np.mean(augmented_scores)
                        all_solutions[bk].append({
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "total_score": total_score,
                            "solution": solution,
                        })
        
        # Return best solution based on total score
        if all_solutions:
            puzzle_key_with_id = list(all_solutions.keys())[0]
            solutions = all_solutions[puzzle_key_with_id]
            solutions.sort(key=lambda x: x["total_score"])
            best_solution = solutions[0]["solution"]
            print(f"    ✓ Best solution: beam_score={solutions[0]['beam_score']:.4f}, aug_score={np.mean(solutions[0]['score_aug']):.4f}")
            return best_solution.tolist()
        
        return None

    # ------------------------------------------------------------------
    # Fallback / validation helpers
    # ------------------------------------------------------------------

    def _fallback_solve(
        self,
        test_input: List[List[int]],
        train_examples: List[Dict],
    ) -> List[List[int]]:
        """Simple fallback when NVARC fails"""
        # Return identity transform as fallback
        return [row[:] for row in test_input]

    def _is_valid_output(self, grid: List[List[int]]) -> bool:
        """Check if output grid is rectangular and within allowed size / values"""
        if not grid or not grid[0]:
            return False
        
        if len(grid) > 30 or len(grid[0]) > 30:
            return False
        
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                return False
            for val in row:
                if not isinstance(val, int) or not (0 <= val <= 9):
                    return False
        
        return True
