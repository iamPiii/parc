"""
NVARC ARC Solver - LLM-based approach using Qwen model with Unsloth/LoRA

This solver uses a fine-tuned Qwen language model to solve ARC tasks.
It performs test-time training (TTT) on training examples using LoRA adapters,
then uses beam search for inference.

Adapted from NVARC implementation for the ARC-AGI-2 competition.
"""

import gc
import io
import os
import time
import json
import logging
import sys
import faulthandler
import signal
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

# Enable faulthandler to get Python traceback on segfault
faulthandler.enable(file=sys.stderr)

# Signal handler for SIGSEGV
def sigsegv_handler(signum, frame):
    print("\n[FATAL] SIGSEGV (Segmentation Fault) caught!", file=sys.stderr, flush=True)
    print(f"Signal: {signum}, Frame: {frame}", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    sys.stderr.flush()
    sys.exit(139)

signal.signal(signal.SIGSEGV, sigsegv_handler)

# Disable PyTorch JIT compilation to prevent issues
os.environ['TORCH_COMPILE'] = '0'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
print("[STARTUP] Using standard transformers + peft (no Unsloth)", flush=True)

import numpy as np
import torch

# Also disable dynamo after torch import
torch._dynamo.config.disable = True
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

logging.disable(logging.WARNING)

# Hugging Face repo ID for NVARC checkpoint
DEFAULT_NVARC_REPO_ID = os.environ.get(
    "NVARC_REPO_ID",
    "iamPi/Hwen-HF",
)

# Local cache directory for NVARC checkpoints
DEFAULT_NVARC_CACHE_DIR = os.environ.get("NVARC_CACHE_DIR", "/app/models")


# ============================================================================
# Token/Vocabulary Constants
# ============================================================================

ARC_VOCAB = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "Ċ": 10,  # newline
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


# ============================================================================
# Grid Conversion Utilities
# ============================================================================

def convert_grid_to_string(grid) -> str:
    """Convert a 2D grid to string representation."""
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()


def is_valid_solution(guess) -> bool:
    """Check if a guess is a valid ARC solution."""
    return (isinstance(guess, np.ndarray) and
            guess.ndim == 2 and
            all(0 < x <= 30 for x in guess.shape))


def shuffled(data_list):
    """Return a shuffled copy of the list."""
    return np.random.permutation(data_list).tolist()


def permute_mod(a, descriptor, invert=False):
    """Apply color permutation to a grid."""
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation) == list(range(10))
    a = np.asarray(a)
    if a.ndim == 3:
        if not invert:
            permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim == 2
        if invert:
            permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a


def permute_rnd_all_(query):
    """Generate random color permutation descriptor."""
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))


# ============================================================================
# Qwen Formatter
# ============================================================================

class QwenFormatter:
    """Formats ARC grids for Qwen model input/output."""

    def __init__(self, tokenizer: AutoTokenizer):
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


# ============================================================================
# ARC Dataset Handler
# ============================================================================

class NVARCDataset:
    """Dataset handler for NVARC approach."""

    @staticmethod
    def forward_mod(a, key, use_perm=True):
        if a is None:
            return a
        for op in key.split('.')[1:]:
            if op == 'rot90':
                a = np.rot90(a)
            elif op == 'transpose':
                a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'):
                a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):
                a = np.copy(a)
            elif op.startswith('out') or op.startswith('ex') or op.startswith('run'):
                pass
            else:
                raise NotImplementedError(f"Operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True):
        if a is None:
            return a
        for op in key.split('.')[1:][::-1]:
            if op == 'rot90':
                a = np.rot90(a, k=3)
            elif op == 'transpose':
                a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'):
                a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):
                a = np.copy(a)
            elif op.startswith('out') or op.startswith('ex') or op.startswith('run'):
                pass
            else:
                raise NotImplementedError(f"Operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies=None, keys=None, is_orig=False):
        if replies is None:
            replies = {}
        if keys is not None:
            keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    def split_multi_replies(self):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def shuffled(self):
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    @staticmethod
    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys=[k for d in datasets for k in d.keys],
        )

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries = {}
        replies = {}
        keys = []
        for k0 in self.keys:
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if t == 'input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        return self.__class__(queries=queries, replies=replies, keys=keys)

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None:
            stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter: QwenFormatter):
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ''
        text = train + query + reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train + query, text=text)

    def as_list(self, formatter: QwenFormatter):
        return [self.get(key, formatter) for key in self.keys]

    def get_length(self, key, formatter: QwenFormatter, name, max_of_transposed=False):
        if formatter is None:
            if name == 'input':
                return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name == 'reply':
                return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else:
                assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None:
                    self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer.encode(ds.get(key, formatter=formatter)[name])) for ds in datasets)

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            reply = temp_ds.replies.get(key)
            while max_len < temp_ds.get_length(key, formatter=formatter, name=name):
                query = temp_ds.queries[key]

                # Safety check: keep at least one training example
                if len(query.get('train', [])) <= 1:
                    print(f"[WARNING] Cannot cut further - only {len(query.get('train', []))} training example(s) left for key {key}")
                    break

                if not key.split('.')[-1].startswith('ex'):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k == 'train' else v) for k, v in query.items()}
                if reply is not None:
                    temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None:
                new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None:
                p = p[:keep_max]
            new_key = f'{key}.ex' + ('-' if (p.max() > 9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k == 'train' else v) for k, v in self.queries[key].items()}
            if key in self.replies:
                new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def augment(self, n=1, shfl_keys=False, seed=42):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d


# ============================================================================
# Standard Trainer (no Unsloth)
# ============================================================================
# Using standard transformers Trainer - no custom overrides needed


# ============================================================================
# Data Collator for Completion-Only LM
# ============================================================================

class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Data collator that only computes loss on assistant completions.

    Matches original NVARC implementation exactly.
    """

    def torch_call(self, examples: list) -> dict:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            # Match original NVARC: use numpy directly on tensor
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


# ============================================================================
# Beam Search DFS Inference
# ============================================================================

def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, timeout_seconds) -> dict:
    """Depth-first beam search for generation."""
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

    while time.time() - start_time < timeout_seconds:
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
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens - 1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos + 1,
            cache=outputs.past_key_values,
            start_time=start_time,
            timeout_seconds=timeout_seconds,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, timeout_seconds):
    """Run beam search DFS inference."""
    import sys
    print(f"[DEBUG DFS] Starting inference_turbo_dfs with {len(prefix_tokens)} sequences", flush=True)
    sys.stdout.flush()

    # Pad sequences to same length (like calc_scores in original NVARC)
    batch_lengths = [len(tokens) for tokens in prefix_tokens]
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in prefix_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    print(f"[DEBUG DFS] Running model forward pass (first call may take 60-120s for compilation)...", flush=True)
    sys.stdout.flush()

    # Sync CUDA to ensure previous operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    forward_start = time.time()

    # Start a heartbeat thread to show progress during compilation
    import threading
    heartbeat_active = threading.Event()
    heartbeat_active.set()

    def heartbeat():
        while heartbeat_active.is_set():
            elapsed = time.time() - forward_start
            print(f"[DEBUG DFS] Still running... elapsed: {elapsed:.1f}s", flush=True)
            sys.stdout.flush()
            time.sleep(10)  # Print every 10 seconds

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)

        # Stop heartbeat
        heartbeat_active.clear()

        # Sync again to ensure forward pass is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_time = time.time() - forward_start
        print(f"[DEBUG DFS] Model forward pass completed in {forward_time:.2f}s", flush=True)
        sys.stdout.flush()
    except Exception as e:
        heartbeat_active.clear()
        forward_time = time.time() - forward_start
        print(f"[DEBUG DFS] ERROR in model forward pass after {forward_time:.2f}s: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    print(f"[DEBUG DFS] Starting turbo_dfs recursion...", flush=True)
    sys.stdout.flush()
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        timeout_seconds=timeout_seconds,
    )
    print(f"[DEBUG DFS] turbo_dfs recursion completed, got {len(suffixes)} suffix groups", flush=True)
    sys.stdout.flush()

    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x: x[0])
        result.append((batch_id, sorted_beams))
    print(f"[DEBUG DFS] Returning {len(result)} results", flush=True)
    sys.stdout.flush()
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    """Calculate scores for query-answer pairs."""
    import sys
    print(f"[DEBUG CALC_SCORES] Starting calc_scores with {len(queries)} queries", flush=True)
    sys.stdout.flush()

    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    print(f"[DEBUG CALC_SCORES] Tokenizing {len(queries)} query-answer pairs...", flush=True)
    sys.stdout.flush()
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    print(f"[DEBUG CALC_SCORES] Tokenization complete, max_len={max(batch_lengths)}", flush=True)
    sys.stdout.flush()

    max_len = max(batch_lengths)
    padded_tokens = []
    print(f"[DEBUG CALC_SCORES] Padding tokens to max_len={max_len}...", flush=True)
    sys.stdout.flush()
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    print(f"[DEBUG CALC_SCORES] Creating tensor...", flush=True)
    sys.stdout.flush()
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    print(f"[DEBUG CALC_SCORES] Tensor created: shape={input_ids.shape}, device={input_ids.device}", flush=True)
    sys.stdout.flush()

    print(f"[DEBUG CALC_SCORES] Running model forward pass...", flush=True)
    sys.stdout.flush()

    # Sync CUDA before forward pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    forward_start = time.time()
    try:
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)

        # Sync CUDA after forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_time = time.time() - forward_start
        print(f"[DEBUG CALC_SCORES] Model forward pass completed in {forward_time:.2f}s", flush=True)
        sys.stdout.flush()
    except Exception as e:
        forward_time = time.time() - forward_start
        print(f"[DEBUG CALC_SCORES] ERROR in model forward pass after {forward_time:.2f}s: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    print(f"[DEBUG CALC_SCORES] Computing log_softmax...", flush=True)
    sys.stdout.flush()
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    print(f"[DEBUG CALC_SCORES] Calculating scores...", flush=True)
    sys.stdout.flush()
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length - 1:query_length - 1 + len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    print(f"[DEBUG CALC_SCORES] Scores calculated, returning {len(result)} scores", flush=True)
    sys.stdout.flush()
    return result


# ============================================================================
# Scoring/Selection Algorithms
# ============================================================================

def hashable(guess):
    return tuple(map(tuple, guess))


def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


# ============================================================================
# Main ARCSolver Class
# ============================================================================

class ARCSolver:
    """
    ARC solver using NVARC (LLM-based) approach with Qwen model.

    This solver performs test-time training using LoRA adapters,
    then uses beam search for inference.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 512,
        device: Optional[str] = None,
        # Inference hyperparameters
        inference_augment_n: int = 2,
        inference_timeout: float = 540.0,  # 9 minutes per puzzle
        beam_threshold: float = 0.2,
    ) -> None:
        """
        Initialize the NVARC solver (inference-only, no training).

        Args:
            checkpoint_path: Direct path to model checkpoint
            repo_id: HuggingFace repo ID (default: iamPi/Hwen-HF)
            cache_dir: Local cache directory for models
            max_seq_length: Maximum sequence length
            device: Device to use (cuda/cpu)
            inference_augment_n: Number of augmentations for inference
            inference_timeout: Timeout for inference in seconds
            beam_threshold: Probability threshold for beam search
        """
        # GPU detection and diagnostics
        print("\n" + "=" * 50)
        print("NVARC Solver - GPU Detection")
        print("=" * 50)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        else:
            print("WARNING: CUDA not available! Training will be slow on CPU.")
            import os
            print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            print(f"  NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
        print("=" * 50 + "\n")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length

        print(f"Using device: {self.device}")

        # Inference hyperparameters
        self.inference_augment_n = inference_augment_n
        self.inference_timeout = inference_timeout
        self.max_score = -np.log(beam_threshold)

        # Determine model path
        if checkpoint_path is not None:
            model_path = checkpoint_path
        else:
            repo_id = repo_id or DEFAULT_NVARC_REPO_ID
            cache_dir = cache_dir or DEFAULT_NVARC_CACHE_DIR
            local_dir = Path(cache_dir) / repo_id.replace("/", "--")

            if local_dir.exists():
                model_path = str(local_dir)
                print(f"Found cached NVARC model: {model_path}")
            else:
                # Try to use HuggingFace repo directly
                model_path = repo_id
                print(f"Using HuggingFace repo: {model_path}")

        # PEFT/LoRA configuration (for model structure, but no training)
        self.peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        )

        # Load model with transformers
        print(f"Loading model from {model_path}...")

        # Determine device_map based on CUDA availability
        if torch.cuda.is_available():
            device_map = "auto"
            print(f"Loading model to GPU (device_map={device_map})")
        else:
            device_map = "cpu"
            print(f"WARNING: Loading model to CPU - this will be slow!")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=(checkpoint_path is not None),
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=(checkpoint_path is not None),
            trust_remote_code=True,
        )

        # Apply PEFT/LoRA
        self.model = get_peft_model(self.model, self.peft_config)

        # Set model to eval mode for inference
        self.model.eval()

        # Store default weights for reset
        self.default_weights = get_peft_model_state_dict(self.model, adapter_name="default")
        self.default_weights = {k: v.clone().detach() for k, v in self.default_weights.items()}

        # Create formatter and collator
        self.formatter = QwenFormatter(tokenizer=self.tokenizer)
        self.max_new_tokens = self.formatter.max_new_tokens()

        self.collator = QwenDataCollatorForCompletionOnlyLM(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Print GPU memory stats
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

        print(f"NVARC ARCSolver initialized on {self.device}")

    def solve(
        self,
        train_examples: List[Dict],
        test_input: List[List[int]],
    ) -> List[List[int]]:
        """
        Solve an ARC task.

        Args:
            train_examples: List of {'input': grid, 'output': grid}
            test_input: Test input grid

        Returns:
            Predicted output grid (2D list of ints in [0, 9])
        """
        start_time = time.time()
        print(f"[DEBUG] Starting solve() - {len(train_examples)} training examples, test input shape: {len(test_input)}x{len(test_input[0])}")

        # Create a unique key for this task (no underscores to avoid issues with split_multi_replies)
        task_key = "puzzle"

        # Build dataset from examples
        queries = {
            task_key: {
                "train": train_examples,
                "test": [{"input": test_input, "output": [[0]]}],  # dummy output
            }
        }
        replies = {}  # No replies for test

        puzzle_ds = NVARCDataset(queries=queries, replies=replies, keys=[task_key])
        print("[DEBUG] Dataset created")

        # Reset model weights to default
        set_peft_model_state_dict(
            self.model,
            deepcopy(self.default_weights),
            adapter_name="default",
        )
        print("[DEBUG] Model weights reset to default")

        # Ensure model is in eval mode
        self.model.eval()

        print("[DEBUG] Collecting garbage...", flush=True)
        sys.stdout.flush()
        gc.collect()
        print("[DEBUG] Clearing CUDA cache...", flush=True)
        sys.stdout.flush()
        torch.cuda.empty_cache()
        print("[DEBUG] Memory cleared, starting inference...", flush=True)
        sys.stdout.flush()

        # Run inference
        print("[DEBUG] Calling _run_inference()...", flush=True)
        sys.stdout.flush()
        predictions = self._run_inference(puzzle_ds, start_time)
        print(f"[DEBUG] Inference completed, got {len(predictions)} predictions", flush=True)
        sys.stdout.flush()

        if not predictions:
            # Fallback: return the test input
            print("Warning: No valid predictions, returning input grid")
            return test_input

        # Select best prediction
        best_prediction = predictions[0]

        # Clamp values to valid ARC colors 0-9
        result = [[int(max(0, min(9, v))) for v in row] for row in best_prediction.tolist()]

        return result


    def _run_inference(self, puzzle_ds: NVARCDataset, start_time: float) -> List[np.ndarray]:
        """Run inference with augmentation and decoding."""
        import sys
        print("[DEBUG INFERENCE] Starting _run_inference()", flush=True)
        sys.stdout.flush()

        # Split for multi-reply handling
        print("[DEBUG INFERENCE] Calling split_multi_replies()...", flush=True)
        sys.stdout.flush()
        puzzle_ds_multi = puzzle_ds.split_multi_replies()
        print(f"[DEBUG INFERENCE] Split complete, {len(puzzle_ds_multi.keys)} keys", flush=True)
        sys.stdout.flush()

        # Augment for inference
        print(f"[DEBUG INFERENCE] Starting augmentation (n={self.inference_augment_n})...", flush=True)
        sys.stdout.flush()
        eval_ds = puzzle_ds_multi.augment(n=self.inference_augment_n, seed=2)
        print(f"[DEBUG INFERENCE] Augmentation complete, {len(eval_ds.keys)} keys", flush=True)
        sys.stdout.flush()

        print(f"[DEBUG INFERENCE] Cutting to max length {self.max_seq_length - self.max_new_tokens}...", flush=True)
        sys.stdout.flush()
        eval_ds = eval_ds.cut_to_len(
            formatter=self.formatter,
            name="input",
            max_len=self.max_seq_length - self.max_new_tokens
        )
        print(f"[DEBUG INFERENCE] Cut complete, {len(eval_ds.keys)} keys", flush=True)
        sys.stdout.flush()

        # Group by test ID
        print("[DEBUG INFERENCE] Grouping by test ID...", flush=True)
        sys.stdout.flush()
        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)
        print(f"[DEBUG INFERENCE] Grouped into {len(test_id_to_subkeys)} test IDs", flush=True)
        sys.stdout.flush()

        # Create batches for inference
        print("[DEBUG INFERENCE] Creating batches for inference...", flush=True)
        sys.stdout.flush()
        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset + 2] if len(subkeys) > offset + 1 else [])
            if batch:
                batches.append(batch)

            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset + 2] if len(subkeys) > offset + 1 else [])
            if batch:
                batches.append(batch)

        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset:offset + 2] if len(subkeys) > offset + 1 else [])
            if batch:
                batches.append(batch)

            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset:offset + 2] if len(subkeys) > offset + 1 else [])
            if batch:
                batches.append(batch)

        print(f"[DEBUG INFERENCE] Created {len(batches)} batches", flush=True)
        sys.stdout.flush()

        decoded_results = {}
        known_scores = {}

        print("[DEBUG INFERENCE] Entering inference_mode context...", flush=True)
        sys.stdout.flush()
        with torch.inference_mode():
            print(f"[DEBUG INFERENCE] Starting batch loop ({len(batches)} batches)...", flush=True)
            sys.stdout.flush()
            for batch_idx, subkeys in enumerate(batches):
                print(f"[DEBUG INFERENCE] === Batch {batch_idx + 1}/{len(batches)} (size={len(subkeys)}) ===", flush=True)
                sys.stdout.flush()

                if not subkeys:
                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1} is empty, skipping", flush=True)
                    sys.stdout.flush()
                    continue

                elapsed = time.time() - start_time
                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: elapsed time = {elapsed:.1f}s", flush=True)
                sys.stdout.flush()

                if elapsed > self.inference_timeout:
                    print(f"[DEBUG INFERENCE] Timeout reached after {elapsed:.1f}s", flush=True)
                    sys.stdout.flush()
                    print(f"Inference timeout after {elapsed:.1f}s")
                    break

                # Tokenize inputs
                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Tokenizing {len(subkeys)} inputs...", flush=True)
                sys.stdout.flush()
                tokens = []
                for subkey_idx, subkey in enumerate(subkeys):
                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Tokenizing subkey {subkey_idx + 1}/{len(subkeys)}: {subkey}", flush=True)
                    sys.stdout.flush()
                    data = eval_ds.get(subkey, self.formatter)
                    encoded = self.tokenizer.encode(data["input"])
                    tokens.append(encoded)
                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Subkey {subkey_idx + 1} tokenized ({len(encoded)} tokens)", flush=True)
                    sys.stdout.flush()

                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: All inputs tokenized", flush=True)
                sys.stdout.flush()

                # Run beam search
                remaining_time = self.inference_timeout - elapsed
                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Starting beam search (timeout={remaining_time:.1f}s)...", flush=True)
                sys.stdout.flush()
                try:
                    dfs_result = inference_turbo_dfs(
                        self.model, tokens,
                        self.max_new_tokens,
                        self.max_score,
                        remaining_time
                    )
                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam search completed, got {len(dfs_result)} results", flush=True)
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: ERROR in beam search: {e}", flush=True)
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise

                # Process results
                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Processing {len(dfs_result)} results...", flush=True)
                sys.stdout.flush()
                for result_idx, (subkey_id, scored_beams) in enumerate(dfs_result):
                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Processing result {result_idx + 1}/{len(dfs_result)}, subkey_id={subkey_id}, {len(scored_beams)} beams", flush=True)
                    sys.stdout.flush()

                    if subkey_id >= len(subkeys):
                        print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Skipping result {result_idx + 1} - subkey_id {subkey_id} >= {len(subkeys)}", flush=True)
                        sys.stdout.flush()
                        continue

                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Result {result_idx + 1} - subkey={subkey}, bk={bk}", flush=True)
                    sys.stdout.flush()

                    for beam_idx, (beam_score, beam_tokens) in enumerate(scored_beams):
                        print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Processing beam {beam_idx + 1}/{len(scored_beams)}, score={beam_score}, {len(beam_tokens)} tokens", flush=True)
                        sys.stdout.flush()
                        print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Converting beam {beam_idx + 1} to array...", flush=True)
                        sys.stdout.flush()
                        array = self.formatter.convert_tokens_to_array(beam_tokens)
                        if array is None:
                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - array is None, skipping", flush=True)
                            sys.stdout.flush()
                            continue

                        print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - inverting transformations...", flush=True)
                        sys.stdout.flush()
                        solution = NVARCDataset.invert_mod(array, subkey, inv_perm=True)

                        grid_id = (bk, hashable(solution))
                        print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - solution shape={solution.shape}", flush=True)
                        sys.stdout.flush()

                        if grid_id in known_scores:
                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - using cached scores", flush=True)
                            sys.stdout.flush()
                            augmented_scores = known_scores[grid_id]
                        else:
                            # Score the solution
                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - creating augmented dataset for scoring...", flush=True)
                            sys.stdout.flush()
                            aug_dataset = NVARCDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - augmenting dataset...", flush=True)
                            sys.stdout.flush()
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % (1024 ** 2))
                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - cutting dataset...", flush=True)
                            sys.stdout.flush()
                            aug_dataset = aug_dataset.cut_to_len(
                                formatter=self.formatter,
                                name="input",
                                max_len=self.max_seq_length - self.max_new_tokens
                            )

                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - preparing augmented queries...", flush=True)
                            sys.stdout.flush()
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(self.formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - prepared {len(aug_queries)} augmented queries", flush=True)
                            sys.stdout.flush()

                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - calculating scores...", flush=True)
                            sys.stdout.flush()
                            if len(aug_queries) >= 4:
                                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - calc_scores batch 1 (4 queries)...", flush=True)
                                sys.stdout.flush()
                                augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], self.tokenizer, self.model)
                                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - calc_scores batch 1 complete", flush=True)
                                sys.stdout.flush()
                                if len(aug_queries) > 4:
                                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - calc_scores batch 2 ({len(aug_queries) - 4} queries)...", flush=True)
                                    sys.stdout.flush()
                                    augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], self.tokenizer, self.model)
                                    print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - calc_scores batch 2 complete", flush=True)
                                    sys.stdout.flush()
                                else:
                                    augmented_scores2 = []
                                augmented_scores = augmented_scores1 + augmented_scores2
                            else:
                                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - calc_scores ({len(aug_queries)} queries)...", flush=True)
                                sys.stdout.flush()
                                augmented_scores = calc_scores(aug_queries, aug_answers, self.tokenizer, self.model)
                                print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - calc_scores complete", flush=True)
                                sys.stdout.flush()

                            known_scores[grid_id] = augmented_scores
                            print(f"[DEBUG INFERENCE] Batch {batch_idx + 1}: Beam {beam_idx + 1} - scores cached", flush=True)
                            sys.stdout.flush()

                        result_key = f"{subkey}.out{len(decoded_results.get(bk, {}))}"
                        if bk not in decoded_results:
                            decoded_results[bk] = {}
                        decoded_results[bk][result_key] = {
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "solution": solution,
                        }

        # Select best solutions using scoring algorithm
        if decoded_results:
            selected = {}
            for bk, v in decoded_results.items():
                selected[bk] = score_kgmon({k: g for k, g in v.items()})

            # Return ordered predictions
            predictions = []
            for bk in sorted(selected.keys()):
                predictions.extend(selected[bk])
            return predictions

        return []
