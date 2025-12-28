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

logging.disable(logging.WARNING)

# Hugging Face repo ID for NVARC checkpoint
DEFAULT_NVARC_REPO_ID = os.environ.get(
    "NVARC_REPO_ID",
    "iamPi/Hwen-HF",
)

# Local cache directory for NVARC checkpoints
DEFAULT_NVARC_CACHE_DIR = os.environ.get("NVARC_CACHE_DIR", "/app/models")


# ============================================================================
# Token/Vocabulary Constants - HARDCODED for 16-token ARC vocabulary
# ============================================================================
# Exact vocabulary from tokenizer:
# {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,
#  "Ċ":10,"user":11,"assistant":12,"<|endoftext|>":13,"<|im_start|>":14,"<|im_end|>":15}

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,  # newline
    "<|im_end|>": 15,  # EOS for generation
}

ARC_TOKENS = list(ARC_VOCAB.values())  # [0,1,2,3,4,5,6,7,8,9,10,15]
USER_TOKEN_ID = 11       # "user"
ASSISTANT_TOKEN_ID = 12  # "assistant"
PAD_ID = 13              # <|endoftext|>
EOS_ID = 15              # <|im_end|>


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

_turbo_dfs_debug_printed = False  # Only print debug once

def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:
    """Depth-first beam search for generation.

    EXACTLY matches original NVARC implementation.
    Uses TWO timeout conditions: 540 seconds from start AND absolute end_time.
    """
    global _turbo_dfs_debug_printed

    n = logits.size(0)

    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)

    # Debug: print info about ARC_TOKENS and scores on first call
    if not _turbo_dfs_debug_printed:
        _turbo_dfs_debug_printed = True
        print(f"\n{'='*60}")
        print(f"[DEBUG turbo_dfs] First call debug info:")
        print(f"{'='*60}")
        print(f"  ARC_TOKENS = {ARC_TOKENS}")
        print(f"  EOS_ID = {EOS_ID}, PAD_ID = {PAD_ID}")
        print(f"  max_score = {max_score:.4f} (threshold, prob > {np.exp(-max_score):.2%})")
        print(f"  logits shape = {logits.shape} (batch_size, vocab_size)")
        print(f"  vocab_size from logits = {logits.shape[-1]}")

        # Check if vocab size matches expectation
        vocab_size = logits.shape[-1]
        if vocab_size != 16:
            print(f"  *** WARNING: Expected vocab_size=16, got {vocab_size}! ***")
            print(f"  *** This suggests model has wrong vocabulary! ***")

        # Show scores for each ARC token
        log_softmax = logits.float().cpu().log_softmax(-1)
        print(f"\n  Scores for ALL ARC_TOKENS (batch 0):")
        passing_count = 0
        for t in ARC_TOKENS:
            if t < nll.size(1):
                score = nll[0, t].item()
                prob = torch.exp(log_softmax[0, t]).item()
                passes = score < max_score
                if passes:
                    passing_count += 1
                print(f"    Token {t:2d}: nll={score:8.4f}, prob={prob:.4f}, passes={passes}")
            else:
                print(f"    Token {t:2d}: OUT OF VOCAB RANGE!")
        print(f"  Total passing ARC tokens: {passing_count}/{len(ARC_TOKENS)}")

        # Show top 10 tokens by probability (to see what model actually predicts)
        probs = torch.exp(log_softmax[0])
        top_probs, top_indices = torch.topk(probs, min(10, vocab_size))
        print(f"\n  Top {len(top_probs)} tokens by probability (batch 0):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            is_arc = idx.item() in ARC_TOKENS
            print(f"    {i+1}. Token {idx.item():3d}: prob={prob.item():.4f} {'(ARC)' if is_arc else ''}")
        print(f"{'='*60}\n")

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
        candidates[i] = sorted(candidates[i], key=lambda x: x[0])  # [:5]

    # EXACT match to original: TWO timeout conditions
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
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens - 1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos + 1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    """Run beam search DFS inference.

    EXACTLY matches original NVARC implementation:
    - NO padding, NO attention_mask
    - Assumes all sequences in batch have same length
    - Uses end_time (absolute timestamp) for timeout
    """
    # Match original NVARC exactly: convert directly to tensor
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)

    # Debug: print input shape
    print(f"  [inference_turbo_dfs] input_ids shape: {input_ids.shape}")

    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    total_beams = 0
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x: x[0])
        result.append((batch_id, sorted_beams))
        total_beams += len(sorted_beams)

    # Debug: print beam counts
    print(f"  [inference_turbo_dfs] Returned {len(result)} batch results, {total_beams} total beams")

    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    """Calculate scores for query-answer pairs.

    EXACTLY matches original NVARC implementation.
    """
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

    IMPORTANT: This is inference-only without TTT (Test-Time Training).
    The pre-trained model is used directly without adding any LoRA adapter.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 8192,  # Match original NVARC
        device: Optional[str] = None,
        # Inference hyperparameters (match original NVARC)
        inference_augment_n: int = 2,
        inference_timeout: float = 1200.0,  # 20 minutes total per puzzle (original uses end_time)
        beam_threshold: float = 0.2,  # -np.log(0.2) = max_score
    ) -> None:
        """
        Initialize the NVARC solver (inference-only, no TTT).

        Args:
            checkpoint_path: Direct path to model checkpoint
            repo_id: HuggingFace repo ID (default: iamPi/Hwen-HF)
            cache_dir: Local cache directory for models
            max_seq_length: Maximum sequence length (8192 to match original)
            device: Device to use (cuda/cpu)
            inference_augment_n: Number of augmentations for inference
            inference_timeout: Total timeout for inference in seconds
            beam_threshold: Probability threshold for beam search
        """
        print("\n" + "=" * 50)
        print("NVARC Solver - Inference Only (No TTT)")
        print("=" * 50)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print("=" * 50 + "\n")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length

        # Inference hyperparameters - match original NVARC exactly
        self.inference_augment_n = inference_augment_n
        self.inference_timeout = inference_timeout
        self.max_score = -np.log(beam_threshold)  # Same as original

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
                model_path = repo_id
                print(f"Using HuggingFace repo: {model_path}")

        print(f"Loading model from {model_path}...")

        # Determine device_map based on CUDA availability
        if torch.cuda.is_available():
            device_map = "auto"
        else:
            device_map = "cpu"
            print("WARNING: Loading model to CPU - this will be slow!")

        # Load base model - NO PEFT adapter since we're not doing TTT
        # The checkpoint should already contain the SFT-trained weights
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

        # Verify tokenizer matches expected 16-token ARC vocabulary
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"Expected ARC_TOKENS: {ARC_TOKENS}")

        # Set model to eval mode for inference - NO PEFT adapter applied
        self.model.eval()

        # Create formatter - exactly as original
        self.formatter = QwenFormatter(tokenizer=self.tokenizer)
        self.max_new_tokens = self.formatter.max_new_tokens()

        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        print(f"NVARC ARCSolver initialized on {self.device} (inference-only, no TTT)")

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
        end_time = start_time + self.inference_timeout  # Absolute end time like original

        # Create a unique key for this task
        task_key = "puzzle"

        # Build dataset from examples
        queries = {
            task_key: {
                "train": train_examples,
                "test": [{"input": test_input, "output": [[0]]}],  # dummy output
            }
        }

        puzzle_ds = NVARCDataset(queries=queries, replies={}, keys=[task_key])

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        # Run inference
        predictions = self._run_inference(puzzle_ds, start_time, end_time)

        if not predictions:
            # Fallback: return the test input
            print("Warning: No valid predictions, returning input grid")
            return test_input

        # Select best prediction
        best_prediction = predictions[0]

        # Clamp values to valid ARC colors 0-9
        result = [[int(max(0, min(9, v))) for v in row] for row in best_prediction.tolist()]

        return result


    def _run_inference(self, puzzle_ds: NVARCDataset, start_time: float, end_time: float) -> List[np.ndarray]:
        """Run inference with augmentation and decoding.

        EXACTLY matches original NVARC worker function's inference part.
        """
        # Split for multi-reply handling
        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        # Augment for inference - exactly as original (n=2, seed=2)
        eval_ds = puzzle_ds_multi.augment(n=self.inference_augment_n, seed=2)
        eval_ds = eval_ds.cut_to_len(
            formatter=self.formatter,
            name="input",
            max_len=self.max_seq_length - self.max_new_tokens
        )

        # Group by test ID - exactly as original
        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)

        # Create batches - EXACTLY as original (lines 370-396)
        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            # 0: permute x 2
            # 4: rot90.rot90.permute x 2
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
            # 2: permute.rot90 x 2
            # 6: rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
        for test_id, subkeys in test_id_to_subkeys.items():
            # 8: transpose.permute x 2
            # 12: transpose.rot90.rot90.permute x 2
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
            # 10: transpose.rot90.permute x 2
            # 14: transpose.rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)

        decoded_results = {}
        known_scores = {}

        with torch.inference_mode():

            for subkeys in batches:

                spend_time = time.time() - start_time
                if spend_time > 1200 or time.time() > end_time:
                    print(f"Timeout after {spend_time:.1f}s")
                    break

                print(f"Decoding {subkeys}")

                # Tokenize inputs - exactly as original
                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, self.formatter)
                    tokens.append(self.tokenizer.encode(data["input"]))

                # Run beam search with end_time - exactly as original
                dfs_result = inference_turbo_dfs(
                    self.model, tokens,
                    self.max_new_tokens,
                    self.max_score,
                    end_time
                )

                # Process results - exactly as original
                for subkey_id, scored_beams in dfs_result:

                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]

                    for beam_score, beam_tokens in scored_beams:

                        array = self.formatter.convert_tokens_to_array(beam_tokens)
                        if array is None:
                            continue

                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                        grid_id = (bk, tuple(map(tuple, solution)))

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"Scoring {subkey}")
                            aug_dataset = NVARCDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % (1024 ** 2))
                            aug_dataset = aug_dataset.cut_to_len(
                                formatter=self.formatter,
                                name="input",
                                max_len=self.max_seq_length - self.max_new_tokens
                            )
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(self.formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            # Exactly as original: split into two batches of 4
                            augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], self.tokenizer, self.model)
                            augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], self.tokenizer, self.model)
                            augmented_scores = augmented_scores1 + augmented_scores2
                            known_scores[grid_id] = augmented_scores

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
