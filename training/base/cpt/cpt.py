import os
from pathlib import Path

# Project root: derived from REACTORFOLD_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[3])
PROJECT_ROOT = os.environ.get("REACTORFOLD_ROOT", _DEFAULT_ROOT)

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
MODEL_ID = "google/gemma-3-270m"
DATA_PATH = os.path.join(PROJECT_ROOT, "data_generation/reactor_10k_final.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training/base/cpt")

LOG_FILE = os.path.join(OUTPUT_DIR, f"training_log_base_cpt_seed{SEED}.csv")
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
print(f"🤖 Loading tokenizer: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token


# --- [3. Data Preprocessing (Pure CPT)] ---
def preprocess_data(file_path):
    print(f"📂 Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    formatted_sequences = []
    for _, row in df.iterrows():
        raw_seq = row['grid_str']
        # 격자 형태 유지 (17개마다 줄바꿈)
        rows = [raw_seq[i:i + 17] for i in range(0, 289, 17)]
        grid_text = " \n ".join([" ".join(list(r)) for r in rows])

        # 🚨 [수정됨] 성능 지표(프롬프트) 제거. 오직 격자 텍스트와 EOS 토큰만 추가
        formatted_sequences.append(grid_text + tokenizer.eos_token)

    print(f"✅ Loaded {len(formatted_sequences)} samples.")
    return Dataset.from_dict({"text": formatted_sequences})

full_dataset = preprocess_data(DATA_PATH)

train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=SEED)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=SEED)

train_dataset = train_testvalid['train']
eval_dataset = test_valid['train']
test_dataset = test_valid['test']

test_dataset.to_json(os.path.join(OUTPUT_DIR, f"test_dataset_seed{SEED}.json"))
print(f"📊 Dataset Split: Train({len(train_dataset)}), Valid({len(eval_dataset)}), Test({len(test_dataset)})")


# --- [4. Tokenization (Pure CPT)] ---
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    labels = []
    for i in range(len(examples["text"])):
        label = list(tokenized["input_ids"][i])

        # 🚨 [수정됨] 프롬프트 마스킹 로직 제거. 오직 패딩(padding) 토큰만 -100으로 마스킹
        for j in range(len(label)):
            if label[j] == tokenizer.pad_token_id:
                label[j] = -100

        labels.append(label)

    tokenized["labels"] = labels
    return tokenized

print("✂️ Tokenizing datasets...")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
eval_tokenized = eval_dataset.map(tokenize_function, batched=True)


# --- [5. Model Loading (CUDA 가속)] ---
print(f"🤖 Loading model: {MODEL_ID}...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
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
    learning_rate=5e-5,
    num_train_epochs=5,

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
print("🚀 Starting Pure CPT Full Fine-tuning on CUDA GPU...")
trainer.train()

final_path = os.path.join(OUTPUT_DIR, f"final_model_seed{SEED}")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"🎉 CPT Complete! Logs saved to {LOG_FILE}")
print(f"💾 Model saved to {final_path}")