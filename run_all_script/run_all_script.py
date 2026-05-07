"""
Master Runner Script: 5-Seed Reproducibility Experiment
========================================================
Executes the full training pipeline across 5 seeds.

Steps execute in a "Batch Parallel" fashion:
For each training script, all 5 seeds are started simultaneously.
The script waits for ALL 5 seeds to complete before moving to the next training stage.

Execution Order:
  SFT-Only (Seeds 0-4) -> DPO Single (Seeds 0-4) -> DPO Multi (Seeds 0-4) ...
"""
import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[1])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

import subprocess
import sys
import time
import os
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Configuration
# ============================================================
# Standard 5-seed set
SEEDS = [0, 1, 2, 3, 4]

# Max simultaneous processes (Matches 5 seeds)
MAX_PARALLEL = 2

BASE = PROJECT_ROOT

# ============================================================
# Scripts List
# ============================================================
SCRIPTS = {
    # --- Base Training ---
    "sft_only":     (f"{BASE}/only_sft/sft",                    "sft_only.py"),
    "cpt":          (f"{BASE}/training/base/cpt",                "cpt.py"),
    "sft_cpt":      (f"{BASE}/training/base/sft",                "sft.py"),

    # --- DPO (SFT-only base) ---
    "dpo_sft_single": (f"{BASE}/only_sft/dpo/single_target",    "single_dpo.py"),
    "dpo_sft_multi":  (f"{BASE}/only_sft/dpo/multi_target",     "LHS_multi_dpo.py"),

    # --- GRPO (SFT-only base) ---
    "grpo_sft_single": (f"{BASE}/only_sft/grpo/single",         "single.py"),
    "grpo_sft_multi":  (f"{BASE}/only_sft/grpo/multi",          "multi.py"),

    # --- DPO (CPT+SFT base) ---
    "dpo_cpt_single":  (f"{BASE}/training/dpo/single_target",   "single_dpo.py"),
    "dpo_cpt_multi":   (f"{BASE}/training/dpo/multi_target",    "LHS_multi_dpo.py"),

    # --- GRPO (CPT+SFT base) ---
    "grpo_cpt_single": (f"{BASE}/training/grpo/single",         "single.py"),
    "grpo_cpt_multi":  (f"{BASE}/training/grpo/multi",          "multi.py"),

    # --- GA (independent) ---
    "ga_baseline":     (f"{BASE}/ga",                            "ga_baseline.py"),
    "ga_constrained":  (f"{BASE}/ga",                            "ga_constrained.py"),
}

# ============================================================
# Execution Pipeline
# ============================================================
# Each step runs all 5 seeds in parallel, then waits.
PIPELINE = [
    # --- Phase 1: SFT Only Base ---
    ("sft_only",        "Parallel"),

    # --- Phase 2: RL on SFT-Only ---
    ("dpo_sft_single",  "Parallel"),
    ("dpo_sft_multi",   "Parallel"),
    ("grpo_sft_single", "Parallel"),
    ("grpo_sft_multi",  "Parallel"),

    # --- Phase 3: CPT Base ---
    ("cpt",             "Parallel"),
    ("sft_cpt",         "Parallel"),

    # --- Phase 4: RL on CPT+SFT ---
    ("dpo_cpt_single",  "Parallel"),
    ("dpo_cpt_multi",   "Parallel"),
    ("grpo_cpt_single", "Parallel"),
    ("grpo_cpt_multi",  "Parallel"),

    # --- Phase 5: GA (CPU-bound) ---
    ("ga_baseline",     "Sequential"),
    ("ga_constrained",  "Sequential"),
]

# ============================================================
# Environment
# ============================================================
CONDA_PREFIX = "/opt/homebrew/Caskroom/miniforge/base/envs/openmc/bin/python"
OPENMC_SCRIPTS = {
    "dpo_sft_single", "dpo_sft_multi",
    "grpo_sft_single", "grpo_sft_multi",
    "dpo_cpt_single", "dpo_cpt_multi",
    "grpo_cpt_single", "grpo_cpt_multi",
    "ga_baseline", "ga_constrained",
}

# ============================================================
# Logging
# ============================================================
LOG_DIR = os.path.join(BASE, "run_all_script", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f"master_{RUN_ID}.log")

_log_lock = threading.Lock()

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with _log_lock:
        print(line)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))

# ============================================================
# Runner
# ============================================================
def get_done_marker(key, seed):
    return os.path.join(LOG_DIR, f"{key}_seed{seed}.done")

def run_script(key, seed, to_file=False):
    if os.path.exists(get_done_marker(key, seed)):
        log(f"    ⏭️ SKIP: Seed {seed} for {key} (Already completed)")
        return "SKIPPED"

    cwd, script = SCRIPTS[key]
    python = CONDA_PREFIX if key in OPENMC_SCRIPTS else sys.executable
    cmd = [python, script, "--seed", str(seed)]

    log(f"    🚀 START: Seed {seed} for {key}")
    start = time.time()
    try:
        if to_file:
            task_log_path = os.path.join(LOG_DIR, f"{RUN_ID}_{key}_seed{seed}.log")
            with open(task_log_path, "w") as task_log:
                result = subprocess.run(cmd, cwd=cwd, stdout=task_log, stderr=subprocess.STDOUT, text=True)
        else:
            result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)

        elapsed = time.time() - start
        if result.returncode == 0:
            log(f"    ✅ DONE: Seed {seed} for {key} [{format_duration(elapsed)}]")
            with open(get_done_marker(key, seed), "w") as f:
                f.write(datetime.now().isoformat())
            return "SUCCESS"
        else:
            log(f"    ❌ FAIL: Seed {seed} for {key} [exit={result.returncode}] [{format_duration(elapsed)}]")
            return "FAILED"
    except Exception as e:
        log(f"    ❌ ERROR: Seed {seed} for {key} [{e}]")
        return "FAILED"

# ============================================================
# Main
# ============================================================
def main():
    tasks = []
    for step_key, mode in PIPELINE:
        for seed in SEEDS:
            tasks.append((step_key, seed, mode))

    total_start = time.time()
    total_tasks = len(tasks)
    completed = 0
    real_completed = 0
    failed = []

    log("=" * 70)
    log("🧬 MASTER RUNNER: Continuous Task Parallelization")
    log(f"   Mode: A global queue to unconditionally run up to {MAX_PARALLEL} tasks concurrently.")
    log(f"   Resume Capability: Enabled (skips tasks with .done marker).")
    log("=" * 70)
    log(f"Master log: {LOG_FILE}")
    log(f"Total tasks: {total_tasks}")
    log("=" * 70)

    log(f"\n📦 Submitting all {total_tasks} tasks to worker pool...")
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(run_script, key, seed, True): (key, seed) for key, seed, mode in tasks}
        
        for future in as_completed(futures):
            key, seed = futures[future]
            status = future.result()
            completed += 1
            
            if status == "SUCCESS":
                real_completed += 1
            elif status == "FAILED":
                failed.append((key, seed))
            
            # Progress
            pct = completed / total_tasks * 100
            elapsed = time.time() - total_start
            
            if real_completed > 0:
                eta = (elapsed / real_completed) * (total_tasks - completed)
                eta_str = format_duration(eta)
            else:
                eta_str = "Calculating..."
                
            log(f"  📊 Progress: {completed}/{total_tasks} ({pct:.1f}%) | ETA: {eta_str}")

    # Summary
    total_elapsed = time.time() - total_start
    log("\n" + "=" * 70)
    log("🏁 ALL EXPERIMENTS COMPLETE")
    log(f"   Total time: {format_duration(total_elapsed)}")
    if failed:
        log(f"   ❌ Failed ({len(failed)} tasks): {failed}")
    else:
        log("   ✅ All tasks succeeded!")
    log("=" * 70)

if __name__ == "__main__":
    main()
