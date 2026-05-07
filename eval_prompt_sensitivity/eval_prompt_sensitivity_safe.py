"""Safe re-implementation of prompt-sensitivity evaluation.

Why this exists
---------------
The original `eval_prompt_sensitivity.py` reused a pre-tokenised input
tensor and sliced it across four consecutive `model.generate()` calls per
(model, target_k) pair.  On NVIDIA GPU (CUDA + bfloat16) this caused
50-100% of samples in the second / fourth batch calls to truncate after
~20 characters; the buggy `clean_grid` then hid the truncation by
padding with 'f's.  Training scripts never hit the bug because they
re-tokenise on every call and request one sample at a time.

This script mirrors the training pattern:
  - fresh tokenisation per batch (no slice reuse)
  - small batch size (default 5) — proven to produce full grids
    across ALL models in the diagnostic
  - per-batch truncation check + retry (up to MAX_RETRIES per batch)
  - CUDA cache flush between batches
  - resumable: skips (model, target_k) pairs already in the output file
  - extra `n_real_chars` and `is_truncated` columns for downstream filtering
    (the original 5 columns are kept first, so any existing plot code
    still loads correctly)
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[1])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

import csv
import gc
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(PROJECT_ROOT)
OUT_PATH = BASE / "eval_prompt_sensitivity" / "prompt_sensitivity_results.csv"

MODELS = {
    "SFT_Only_Base":         BASE / "only_sft/sft/final_sft_only_model",
    "CPT_SFT_Base":          BASE / "training/base/sft/final_sft_model",
    "DPO_Single_SFT":        BASE / "only_sft/dpo/single_target/dpo_optimized_single_model",
    "DPO_Multi_SFT":         BASE / "only_sft/dpo/multi_target/dpo_optimized_multi_lhs_model",
    "GRPO_Single_SFT":       BASE / "only_sft/grpo/single/grpo_sft_only_single_model",
    "GRPO_Multi_SFT":        BASE / "only_sft/grpo/multi/grpo_sft_only_multi_model",
    "DPO_Single_CPT_SFT":    BASE / "training/dpo/single_target/dpo_optimized_single_model",
    "DPO_Multi_CPT_SFT":     BASE / "training/dpo/multi_target/dpo_optimized_multi_lhs_model",
    "GRPO_Single_CPT_SFT":   BASE / "training/grpo/single/grpo_cpt_sft_single_model",
    "GRPO_Multi_CPT_SFT":    BASE / "training/grpo/multi/grpo_cpt_sft_multi_model",
}

TARGET_KS = [1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08]
SAMPLES_PER_TARGET = 100
BATCH_SIZE         = 5            # diagnose proved bs=5 works for ALL models
MAX_NEW            = 400
TEMPERATURE        = 1.0
MAX_RETRIES        = 2            # per batch, when truncation rate is high
TRUNCATE_THRESHOLD = 50           # real_chars below this is considered truncated
RETRY_FRACTION     = 0.4          # retry whole batch if more than 40% truncated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_FIELDS = [
    "model_name", "target_k", "sample_idx",   # original schema
    "gd_count", "grid_str",
    "n_real_chars", "is_truncated",            # new diagnostic columns
]


# ------------------------------------------------------------------ #
# Output cleaning                                                     #
# ------------------------------------------------------------------ #
def clean_grid(text: str) -> tuple[str, int]:
    """Return (grid_padded_to_289, n_real_valid_chars).

    `n_real_valid_chars` is the count of f/g/c characters BEFORE padding.
    Downstream code can flag samples with a small `n_real_chars` as
    truncated rather than guessing from trailing-f patterns.
    """
    pos = text.find(":\n")
    if pos != -1:
        body = text[pos + 2:]
    else:
        body = text
    body = body.replace(" ", "").replace("\n", "")
    valid = [c for c in body if c in ("f", "g", "c")]
    n_real = len(valid)
    if n_real < 289:
        valid = valid + ["f"] * (289 - n_real)
    return "".join(valid[:289]), n_real


# ------------------------------------------------------------------ #
# Resume support                                                      #
# ------------------------------------------------------------------ #
def already_done(out_path: Path) -> tuple[set[tuple[str, float]], int]:
    """Return (set_of_completed_pairs, total_completed_samples).

    A (model, target_k) pair is "complete" once it has SAMPLES_PER_TARGET
    rows in the existing output file.  Partially-done pairs are NOT
    counted as complete (we just regenerate them), but their existing
    rows still contribute to total_completed_samples for the progress bar.
    """
    if not out_path.exists():
        return set(), 0
    counts: dict[tuple[str, float], int] = {}
    with out_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model_name"], float(row["target_k"]))
            counts[key] = counts.get(key, 0) + 1
    complete = {k for k, n in counts.items() if n >= SAMPLES_PER_TARGET}
    total_done = sum(min(n, SAMPLES_PER_TARGET) for k, n in counts.items()
                     if k in complete)
    return complete, total_done


# ------------------------------------------------------------------ #
# Generation                                                          #
# ------------------------------------------------------------------ #
def generate_batch(model, tokenizer, prompt: str, batch_n: int):
    """Generate `batch_n` samples in a single fresh-tokenised batch.

    Returns a list of dicts: {grid, n_real_chars, gd_count, is_truncated}.
    """
    # FRESH tokenisation — never slice a pre-tokenised tensor.
    batched_prompts = [prompt] * batch_n
    inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    samples = []
    for txt in decoded:
        grid, n_real = clean_grid(txt)
        samples.append({
            "grid":         grid,
            "n_real_chars": n_real,
            "gd_count":     grid.count("g"),
            "is_truncated": n_real < TRUNCATE_THRESHOLD,
        })
    return samples


def generate_with_retry(model, tokenizer, prompt: str, batch_n: int, label: str,
                        pbar=None):
    """Generate a batch; retry whole batch up to MAX_RETRIES if truncated."""
    for attempt in range(MAX_RETRIES + 1):
        samples = generate_batch(model, tokenizer, prompt, batch_n)
        n_trunc = sum(s["is_truncated"] for s in samples)
        if n_trunc / batch_n <= RETRY_FRACTION:
            return samples, attempt
        if attempt < MAX_RETRIES:
            msg = (f"      [retry {attempt+1}/{MAX_RETRIES}] {label}: "
                   f"{n_trunc}/{batch_n} truncated, regenerating batch")
            if pbar is not None:
                pbar.write(msg)
            else:
                tqdm.write(msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return samples, MAX_RETRIES


# ------------------------------------------------------------------ #
# Per-model loop                                                      #
# ------------------------------------------------------------------ #
def run_model(model_name: str, model_path: Path, done: set, csv_writer, fout,
              overall_pbar):
    if not model_path.exists():
        overall_pbar.write(f"  [warn] checkpoint missing: {model_path}; skip")
        return {"trunc": 0, "retries": 0}

    overall_pbar.write(f"\n{'='*70}\n  Loading {model_name}\n  {model_path}\n{'='*70}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, local_files_only=True
    ).to(DEVICE).eval()
    overall_pbar.write(f"  load time: {time.time()-t0:.1f}s")

    stats = {"trunc": 0, "retries": 0}
    for k_val in TARGET_KS:
        if (model_name, k_val) in done:
            overall_pbar.write(f"  k={k_val:.2f} already done, skipping")
            continue

        prompt = f"Reactor Core Design (k={k_val:.5f}, fq=1.0000, fdh=1.0000):\n"
        n_done = 0
        while n_done < SAMPLES_PER_TARGET:
            batch_n = min(BATCH_SIZE, SAMPLES_PER_TARGET - n_done)
            samples, n_retries = generate_with_retry(
                model, tokenizer, prompt, batch_n,
                label=f"{model_name} k={k_val:.2f}",
                pbar=overall_pbar)
            stats["retries"] += n_retries
            for j, s in enumerate(samples):
                csv_writer.writerow({
                    "model_name":   model_name,
                    "target_k":     k_val,
                    "sample_idx":   n_done + j + 1,
                    "gd_count":     s["gd_count"],
                    "grid_str":     s["grid"],
                    "n_real_chars": s["n_real_chars"],
                    "is_truncated": int(s["is_truncated"]),
                })
                if s["is_truncated"]:
                    stats["trunc"] += 1
            fout.flush()
            n_done += batch_n

            overall_pbar.update(batch_n)
            overall_pbar.set_postfix_str(
                f"{model_name}|k={k_val:.2f}|trunc={stats['trunc']}|retries={stats['retries']}")

            # Reset CUDA state between batches to avoid the contamination
            # that broke the original eval.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return stats


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done, n_already = already_done(OUT_PATH)
    total_samples = len(MODELS) * len(TARGET_KS) * SAMPLES_PER_TARGET
    if done:
        print(f"Resuming: {len(done)} (model, target_k) pairs already complete "
              f"({n_already:,} / {total_samples:,} samples).")
    else:
        print(f"Starting fresh evaluation -> {OUT_PATH}")

    write_header = not OUT_PATH.exists() or OUT_PATH.stat().st_size == 0
    fout = OUT_PATH.open("a", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        fout.flush()

    print(f"\nDevice: {DEVICE}")
    print(f"Models: {len(MODELS)}, target_ks: {len(TARGET_KS)}, "
          f"samples/target: {SAMPLES_PER_TARGET}, batch: {BATCH_SIZE}")
    print(f"Total samples: {total_samples:,}\n")

    t_start = time.time()
    grand_stats = {"trunc": 0, "retries": 0}
    with tqdm(total=total_samples, initial=n_already,
              desc="overall", unit="sample", smoothing=0.05,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
              ) as overall_pbar:
        for model_name, model_path in MODELS.items():
            stats = run_model(model_name, model_path, done, writer, fout,
                              overall_pbar)
            grand_stats["trunc"] += stats["trunc"]
            grand_stats["retries"] += stats["retries"]

    fout.close()
    elapsed = time.time() - t_start
    print(f"\nDone. Wall time: {elapsed/60:.1f} min "
          f"(truncated: {grand_stats['trunc']}, retries: {grand_stats['retries']})")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
