import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[1])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

import os
import csv
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

base_path = PROJECT_ROOT

MODELS = {
    # Baselines
    "SFT_Only_Base": os.path.join(base_path, "only_sft/sft/final_sft_only_model"),
    "CPT_SFT_Base": os.path.join(base_path, "training/base/sft/final_sft_model"),
    
    # SFT Only (RL)
    "DPO_Single_SFT": os.path.join(base_path, "only_sft/dpo/single_target/dpo_optimized_single_model"),
    "DPO_Multi_SFT": os.path.join(base_path, "only_sft/dpo/multi_target/dpo_optimized_multi_lhs_model"),
    "GRPO_Single_SFT": os.path.join(base_path, "only_sft/grpo/single/grpo_sft_only_single_model"),
    "GRPO_Multi_SFT": os.path.join(base_path, "only_sft/grpo/multi/grpo_sft_only_multi_model"),
    
    # CPT+SFT (RL)
    "DPO_Single_CPT_SFT": os.path.join(base_path, "training/dpo/single_target/dpo_optimized_single_model"),
    "DPO_Multi_CPT_SFT": os.path.join(base_path, "training/dpo/multi_target/dpo_optimized_multi_lhs_model"),
    "GRPO_Single_CPT_SFT": os.path.join(base_path, "training/grpo/single/grpo_cpt_sft_single_model"),
    "GRPO_Multi_CPT_SFT": os.path.join(base_path, "training/grpo/multi/grpo_cpt_sft_multi_model")
}

TARGET_KS = [1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08]
SAMPLES_PER_TARGET = 100
BATCH_SIZE = 25  # Memory-safe batch size for NVIDIA GPU (CUDA)

OUTPUT_FILE = os.path.join(base_path, "eval_prompt_sensitivity", "prompt_sensitivity_results.csv")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_grid(grid_str):
    """Clean the LLM output to extract just the 289 char core."""
    grid_start = grid_str.find(":\n")
    if grid_start != -1:
        grid_text = grid_str[grid_start+2:]
    else:
        grid_text = grid_str
        
    cleaned = grid_text.replace(" ", "").replace("\n", "")
    valid_chars = [c for c in cleaned if c in ['f', 'g', 'c']]
    
    # Pad if necessary
    while len(valid_chars) < 289:
        valid_chars.append('f')
    return ''.join(valid_chars[:289])

def load_model(path):
    print(f"\n🔄 Loading Model: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(DEVICE)
    model.eval()
    return model, tokenizer

def main():
    # Write header if file doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "target_k", "sample_idx", "gd_count", "grid_str"])
            
    print(f"Target evaluations: {len(TARGET_KS)} targets * 100 samples * 10 models = 7,000 generations total")

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"⚠️ Model path not found: {model_path}. Skipping.")
            continue
            
        model, tokenizer = load_model(model_path)
        
        for k_val in TARGET_KS:
            prompt = f"Reactor Core Design (k={k_val:.5f}, fq=1.0000, fdh=1.0000):\n"
            
            # Prepare batched inputs
            batched_prompts = [prompt] * BATCH_SIZE
            inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(DEVICE)
            
            pbar = tqdm(total=SAMPLES_PER_TARGET, desc=f"{model_name} | K={k_val:.2f}", leave=False)
            
            generated_samples = []
            
            # Run in batches
            for i in range(0, SAMPLES_PER_TARGET, BATCH_SIZE):
                current_batch_size = min(BATCH_SIZE, SAMPLES_PER_TARGET - i)
                batch_inputs = {
                    "input_ids": inputs.input_ids[:current_batch_size],
                    "attention_mask": inputs.attention_mask[:current_batch_size]
                }
                
                with torch.no_grad():
                    outputs = model.generate(
                        **batch_inputs,
                        max_new_tokens=400,
                        temperature=1.0,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for idx, out_str in enumerate(decoded_outputs):
                    sample_idx = i + idx + 1
                    cleaned_grid = clean_grid(out_str)
                    gd_count = cleaned_grid.count('g')
                    generated_samples.append([model_name, k_val, sample_idx, gd_count, cleaned_grid])
                    
                pbar.update(current_batch_size)
            
            pbar.close()
            
            # Save results progressively
            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(generated_samples)
                
        # Free memory before loading next model
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✅ Evaluation complete. All results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
