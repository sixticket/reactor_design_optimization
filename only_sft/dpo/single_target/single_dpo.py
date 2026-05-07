import os
from pathlib import Path

# Project root: derived from REACTORFOLD_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[3])
PROJECT_ROOT = os.environ.get("REACTORFOLD_ROOT", _DEFAULT_ROOT)

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import tempfile
import shutil
import openmc
import argparse
import random

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
os.environ.setdefault(
    'OPENMC_CROSS_SECTIONS',
    os.path.expanduser('~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml'),
)

# SFT (SFT only) 모델 경로
BASE_DIR = os.path.join(PROJECT_ROOT, "only_sft/sft")
MODEL_PATH = os.path.join(BASE_DIR, f"final_sft_only_model_seed{SEED}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STEPS = 1000
LEARNING_RATE = 1e-5
BETA = 0.01

EXPECTED_GT = 25

# 🚨 고정된 단일 타겟 (Single Target)
TARGET_K_EFF = 1.05
TARGET_FQ = 1.00
TARGET_FDH = 1.00

OUTPUT_FILE = f"./dpo_single_target_seed{SEED}_results.csv"

# Physical Constants
pitch = 1.26
fuel_r = 0.4096
clad_id = 0.835
clad_od = 0.950
assembly_size = 17 * pitch

# Guide Tube Coordinates
GT_COORDS = [
    (2, 5), (2, 8), (2, 11), (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13), (14, 5), (14, 8), (14, 11)
]

GT_INDICES = set([r * 17 + c for r, c in GT_COORDS])

def fix_grid_gt(grid_str):
    grid_list = list(grid_str)
    if len(grid_list) < 289:
        grid_list.extend(['f'] * (289 - len(grid_list)))
    grid_list = grid_list[:289]

    for i in range(289):
        if i in GT_INDICES:
            grid_list[i] = 'c'  
        elif grid_list[i] == 'c':
            grid_list[i] = 'f'  
    return "".join(grid_list)


# --- [OpenMC Simulation Functions] ---
def create_materials():
    fuel = openmc.Material(name='UO2 3.1wt%')
    fuel.add_nuclide('U235', 0.031)
    fuel.add_nuclide('U238', 0.969)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.29769)

    gd = openmc.Material(name='Gd Poison 8wt%')
    gd.add_nuclide('U235', 0.031 * 0.92)
    gd.add_nuclide('U238', 0.969 * 0.92)
    gd.add_nuclide('O16', 2.0 * 0.92 + 3.0 * 0.08)
    gd.add_nuclide('Gd155', 0.08 * 0.148)
    gd.add_nuclide('Gd156', 0.08 * 0.199)
    gd.add_nuclide('Gd157', 0.08 * 0.156)
    gd.add_nuclide('Gd158', 0.08 * 0.249)
    gd.add_nuclide('Gd160', 0.08 * 0.218)
    gd.set_density('g/cm3', 10.5)

    zirc = openmc.Material(name='Zircaloy-4')
    for el, frac in [('Sn', 0.014), ('Fe', 0.0121), ('Cr', 0.0107), ('Ni', 0.0050), ('Zr', 0.9582)]:
        zirc.add_element(el, frac)
    zirc.set_density('g/cm3', 6.55)

    water = openmc.Material(name='PWR Water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.set_density('g/cm3', 0.701)

    materials = openmc.Materials([fuel, gd, zirc, water])
    materials.export_to_xml()
    return fuel, gd, zirc, water

def create_pin_universe(rod_type, fuel, gd, zirc, water):
    fuel_cyl = openmc.ZCylinder(r=fuel_r)
    clad_inner = openmc.ZCylinder(r=clad_id / 2)
    clad_outer = openmc.ZCylinder(r=clad_od / 2)
    box = openmc.model.RectangularPrism(width=pitch, height=pitch, boundary_type='reflective')
    fill = {0: fuel, 1: gd, 2: water}[rod_type]
    cells = [
        openmc.Cell(fill=fill, region=-fuel_cyl),
        openmc.Cell(fill=None, region=+fuel_cyl & -clad_inner),
        openmc.Cell(fill=zirc, region=+clad_inner & -clad_outer),
        openmc.Cell(fill=water, region=-box & +clad_outer)
    ]
    return openmc.Universe(cells=cells)

def create_assembly(grid, fuel, gd, zirc, water):
    lattice = openmc.RectLattice()
    lattice.pitch = (pitch, pitch)
    lattice.lower_left = (-assembly_size / 2, -assembly_size / 2)
    lattice.dimension = (17, 17)
    lattice.universes = np.array([
        [create_pin_universe(grid[i, j], fuel, gd, zirc, water) for j in range(17)]
        for i in range(17)
    ])
    outer_box = openmc.model.RectangularPrism(assembly_size, assembly_size, boundary_type='reflective')
    return openmc.Geometry([openmc.Cell(fill=lattice, region=-outer_box)])


def evaluate_grid(grid_str):
    g_count = grid_str.count('g')

    base_tmp_dir = tempfile.gettempdir()
    work_dir = tempfile.mkdtemp(prefix='dpo_run_', dir=base_tmp_dir)
    original_dir = os.getcwd()

    try:
        os.chdir(work_dir)
        fuel, gd, zirc, water = create_materials()

        grid = np.zeros((17, 17), dtype=int)
        for i, char in enumerate(grid_str):
            r, c = divmod(i, 17)
            if char == 'f': grid[r, c] = 0
            elif char == 'g': grid[r, c] = 1
            elif char == 'c': grid[r, c] = 2

        create_assembly(grid, fuel, gd, zirc, water).export_to_xml()

        settings = openmc.Settings()
        settings.batches = 30
        settings.inactive = 10
        settings.particles = 20000
        settings.output = {'summary': False}
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box(
                [-assembly_size / 2, -assembly_size / 2, 0],
                [assembly_size / 2, assembly_size / 2, 1]
            )
        )
        settings.export_to_xml()

        tallies = openmc.Tallies()
        mesh = openmc.RegularMesh()
        mesh.dimension = (17, 17, 1)
        mesh.lower_left = (-assembly_size / 2, -assembly_size / 2, 0)
        mesh.upper_right = (assembly_size / 2, assembly_size / 2, 1)
        power_tally = openmc.Tally(name='power')
        power_tally.filters = [openmc.MeshFilter(mesh)]
        power_tally.scores = ['fission']
        tallies.append(power_tally)
        tallies.export_to_xml()

        openmc.run(output=False)

        with openmc.StatePoint('statepoint.30.h5') as sp:
            k_eff = sp.keff.nominal_value
            power = sp.get_tally(name='power').mean.ravel()

            fq = power.max() / power.mean() if power.mean() > 0 else 99.9
            power_2d = power.reshape(17, 17)
            channel_powers = power_2d.mean(axis=0)
            fdh = channel_powers.max() / channel_powers.mean() if channel_powers.mean() > 0 else 99.9

        # 🚨 Single Target Fitness: 목표 K_eff(1.05)와의 거리에 비례하는 직접적인 페널티
        penalty_weight = 100.0
        penalty_k = penalty_weight * abs(k_eff - TARGET_K_EFF)
        fitness = 0.6 * fq + 0.4 * fdh + penalty_k

        return (fitness, k_eff, fq, fdh, g_count)

    except Exception as e:
        return (99.9, 0.0, 99.9, 99.9, g_count)
    finally:
        os.chdir(original_dir)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)


# --- [Model Functions] ---
def load_model(model_path):
    print(f"🤖 Loading FFT model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(DEVICE)

    model.train()
    for param in model.parameters():
        param.requires_grad = True

    print(f"✅ Full model loaded on {DEVICE}")
    return model, tokenizer

def create_prompt(k_eff, fq, fdh):
    return f"Reactor Core Design (k={k_eff:.5f}, fq={fq:.4f}, fdh={fdh:.4f}):\n"

def generate_grid(model, tokenizer, prompt, temperature=1.0):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    model.train()
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    grid_start = generated.find(":\n") + 2
    grid_text = generated[grid_start:]
    grid_str = grid_text.replace(" ", "").replace("\n", "")

    grid_str = validate_grid(grid_str)
    grid_str = fix_grid_gt(grid_str)
    return grid_str

def validate_grid(grid_str):
    valid_chars = [c for c in grid_str if c in ['f', 'g', 'c']]
    while len(valid_chars) < 289:
        valid_chars.append('f')
    return ''.join(valid_chars[:289])

def get_log_probs(model, tokenizer, prompt, grid_str):
    rows = [grid_str[i:i + 17] for i in range(0, 289, 17)]
    grid_text = " \n ".join([" ".join(list(r)) for r in rows])
    full_text = prompt + grid_text + tokenizer.eos_token

    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    labels = inputs["input_ids"][:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    
    grid_log_prob = token_log_probs[:, prompt_len:].sum()
    return grid_log_prob

def dpo_loss(model, tokenizer, prompt, chosen_grid, rejected_grid, beta=0.01):
    chosen_log_prob = get_log_probs(model, tokenizer, prompt, chosen_grid)
    rejected_log_prob = get_log_probs(model, tokenizer, prompt, rejected_grid)
    loss = -F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))
    return loss


# --- [Main Training Loop] ---
def run_online_dpo(model, tokenizer, num_steps, output_file):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "simulation_count", "elapsed_time_sec",
            "chosen_grid", "chosen_fitness", "chosen_k_eff", "chosen_fq", "chosen_fdh", "chosen_g_count",
            "rejected_grid", "rejected_fitness", "rejected_k_eff", "rejected_fq", "rejected_fdh", "rejected_g_count",
            "best_fitness", "best_k_eff", "best_fq", "best_fdh", "best_g_count",
            "loss"
        ])

    start_time = time.time()
    sim_count = 0
    best_ever = (99.9, 0.0, 99.9, 99.9, 0)

    pbar = tqdm(total=num_steps, desc="Online DPO (Single-Target: 1.05)", unit="step")

    for step in range(1, num_steps + 1):
        # 🚨 프롬프트에 단일 타겟(1.05)만 주입
        prompt = create_prompt(TARGET_K_EFF, TARGET_FQ, TARGET_FDH)

        grid1 = generate_grid(model, tokenizer, prompt, temperature=1.0)
        grid2 = generate_grid(model, tokenizer, prompt, temperature=1.0)

        metrics1 = evaluate_grid(grid1)
        metrics2 = evaluate_grid(grid2)
        sim_count += 2

        if metrics1[0] <= metrics2[0]:
            chosen_grid, chosen_metrics = grid1, metrics1
            rejected_grid, rejected_metrics = grid2, metrics2
        else:
            chosen_grid, chosen_metrics = grid2, metrics2
            rejected_grid, rejected_metrics = grid1, metrics1

        if chosen_metrics[0] < best_ever[0]:
            best_ever = chosen_metrics

        elapsed = time.time() - start_time

        if chosen_metrics[0] >= 99.0 and rejected_metrics[0] >= 99.0:
            loss_val = 0.0
        else:
            model.train()
            optimizer.zero_grad()
            loss = dpo_loss(model, tokenizer, prompt, chosen_grid, rejected_grid, beta=BETA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_val = loss.item()

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, sim_count, round(elapsed, 2),
                chosen_grid, round(chosen_metrics[0], 4), round(chosen_metrics[1], 5),
                round(chosen_metrics[2], 4), round(chosen_metrics[3], 4), chosen_metrics[4],
                rejected_grid, round(rejected_metrics[0], 4), round(rejected_metrics[1], 5),
                round(rejected_metrics[2], 4), round(rejected_metrics[3], 4), rejected_metrics[4],
                round(best_ever[0], 4), round(best_ever[1], 5),
                round(best_ever[2], 4), round(best_ever[3], 4), best_ever[4],
                round(loss_val, 4)
            ])

        pbar.update(1)
        pbar.set_postfix({
            "BestFit": f"{best_ever[0]:.3f}",
            "Gd": chosen_metrics[4],
            "Loss": f"{loss_val:.4f}"
        })

    pbar.close()
    return best_ever, sim_count


# --- [Main] ---
if __name__ == "__main__":
    print("🧹 Cleaning temp files...")
    base_tmp = tempfile.gettempdir()
    for item in os.listdir(base_tmp):
        if item.startswith('dpo_run_'):
            shutil.rmtree(os.path.join(base_tmp, item), ignore_errors=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"\n🚀 Starting Online DPO Optimization (FFT)")
    print(f"=" * 60)
    print(f"⚙️  Mode: SINGLE-TARGET (k={TARGET_K_EFF})")
    print(f"⚙️  Steps: {NUM_STEPS} (Simulations: {NUM_STEPS * 2})")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print("=" * 60)

    model, tokenizer = load_model(MODEL_PATH)

    best_metrics, total_sims = run_online_dpo(
        model, tokenizer,
        num_steps=NUM_STEPS,
        output_file=OUTPUT_FILE
    )

    final_path = f"./dpo_optimized_single_seed{SEED}_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\n" + "=" * 60)
    print("🏆 Online DPO Optimization Complete!")
    print(f"💾 Model saved to: {final_path}")
    print("=" * 60)