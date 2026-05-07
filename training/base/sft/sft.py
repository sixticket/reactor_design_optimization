import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[3])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

import os
import csv
import pandas as pd
import numpy as np
import torch
import random
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator
)

# --- [CLI Arguments] ---
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
SEED = args.seed

# --- [Seed Setting] ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# --- [Configuration] ---
# 🚨 앞서 진행한 CPT 모델의 최종 저장 경로
CPT_MODEL_PATH = fos.path.join(PROJECT_ROOT, "training/base/cpt/final_model_seed{SEED}")
DATA_PATH = os.path.join(PROJECT_ROOT, "data_generation/reactor_10k_final.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training/base/sft")

LOG_FILE = os.path.join(OUTPUT_DIR, f"training_log_sft_fft_seed{SEED}.csv")
MAX_LENGTH = 384


# --- [1. Custom CSV Logger] ---
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss", "eval_loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                step = state.global_step
                epoch = logs.get("epoch", "")
                train_loss = logs.get("loss", "")
                eval_loss = logs.get("eval_loss", "")

                if train_loss != "" or eval_loss != "":
                    writer.writerow([step, epoch, train_loss, eval_loss])


# --- [2. Load Tokenizer] ---
print(f"🤖 Loading tokenizer from CPT model: {CPT_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(CPT_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token


# --- [3. Data Preprocessing (Conditional SFT)] ---
def preprocess_data(file_path):
    print(f"📂 Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    formatted_sequences = []
    for _, row in df.iterrows():
        raw_seq = row['grid_str']
        rows = [raw_seq[i:i + 17] for i in range(0, 289, 17)]
        grid_text = " \n ".join([" ".join(list(r)) for r in rows])

        # 타겟 물리 지표 프롬프트 추가
        prompt = (
            f"Reactor Core Design (k={row['k_eff']:.5f}, "
            f"fq={row['fq']:.4f}, fdh={row['fdh']:.4f}):\n{grid_text}"
        )
        formatted_sequences.append(prompt + tokenizer.eos_token)

    print(f"✅ Loaded {len(formatted_sequences)} samples.")
    return Dataset.from_dict({"text": formatted_sequences})

full_dataset = preprocess_data(DATA_PATH)

train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=SEED)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=SEED)

train_dataset = train_testvalid['train']
eval_dataset = test_valid['train']
test_dataset = test_valid['test']

test_dataset.to_json(os.path.join(OUTPUT_DIR, f"test_dataset_sft_seed{SEED}.json"))


# --- [4. Tokenization (Prompt Masking)] ---
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    labels = []
    for i, text in enumerate(examples["text"]):
        prompt_end = text.find(":\n") + 2
        prompt_tokens = len(tokenizer.encode(text[:prompt_end], add_special_tokens=True))
        
        label = list(tokenized["input_ids"][i])

        # 프롬프트 부분은 Loss 계산에서 제외 (-100)
        for j in range(prompt_tokens):
            label[j] = -100

        # 패딩 부분도 마스킹
        for j in range(len(label)):
            if tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                label[j] = -100

        labels.append(label)

    tokenized["labels"] = labels
    return tokenized

print("✂️ Tokenizing datasets for SFT...")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
eval_tokenized = eval_dataset.map(tokenize_function, batched=True)


# --- [5. Model Loading (CUDA & FFT)] ---
print(f"🤖 Loading CPT model for Full Fine-Tuning: {CPT_MODEL_PATH}...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

# LoRA 래핑 없이 원본 모델 그대로 로드
model = AutoModelForCausalLM.from_pretrained(
    CPT_MODEL_PATH,
    torch_dtype=torch.bfloat16, 
    attn_implementation="eager"
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Total parameters: {total_params:,}")
print(f"📊 Trainable parameters: {trainable_params:,} (100%)")


# --- [6. Training Arguments] ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  
    learning_rate=5e-5,  # 🚨 FFT에 맞춰 LR을 다시 안정적인 값으로 하향 조정
    num_train_epochs=10, 

    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",

    bf16=False, 
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",

    disable_tqdm=False,
    logging_first_step=True,
    gradient_checkpointing=True,
)


# --- [7. Trainer Initialization] ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=default_data_collator,
    callbacks=[CSVLoggerCallback(LOG_FILE)]
)


# --- [8. Execution] ---
print("🚀 Starting SFT (Full Fine-Tuning) on CUDA GPU...")
trainer.train()

# 모델 전체 가중치 저장
final_path = os.path.join(OUTPUT_DIR, f"final_sft_model_seed{SEED}")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"🎉 SFT Complete! Logs saved to {LOG_FILE}")
print(f"💾 Full model saved to {final_path}")