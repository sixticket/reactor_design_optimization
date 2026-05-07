import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[3])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

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
import concurrent.futures
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
BASE_DIR = os.path.join(PROJECT_ROOT, "training/base/sft")
MODEL_PATH = os.path.join(BASE_DIR, f"final_sft_model_seed{SEED}")
OUTPUT_FILE = f"./grpo_cpt_sft_single_seed{SEED}_results.csv"
FINAL_MODEL_DIR = f"./grpo_cpt_sft_single_seed{SEED}_model"

os.environ.setdefault(
    'OPENMC_CROSS_SECTIONS',
    os.path.expanduser('~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml'),
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUP_SIZE = 4
NUM_STEPS = 500
LEARNING_RATE = 1e-5

TARGET_K_EFF = 1.05
TARGET_FQ = 1.00
TARGET_FDH = 1.00

# --- [Physics Setup] ---
pitch, fuel_r, clad_id, clad_od = 1.26, 0.4096, 0.835, 0.950
assembly_size = 17 * pitch
GT_COORDS = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13), (5, 2), (5, 5), (5, 8), (5, 11), (5, 14), (8, 2), (8, 5), (8, 8), (8, 11), (8, 14), (11, 2), (11, 5), (11, 8), (11, 11), (11, 14), (13, 3), (13, 13), (14, 5), (14, 8), (14, 11)]
GT_INDICES = set([r * 17 + c for r, c in GT_COORDS])

def fix_grid_gt(grid_str):
    grid_list = list(grid_str.ljust(289, 'f'))[:289]
    for i in range(289):
        if i in GT_INDICES: grid_list[i] = 'c'
        elif grid_list[i] == 'c': grid_list[i] = 'f'
    return "".join(grid_list)

# --- [OpenMC Engine] ---
def create_materials():
    fuel = openmc.Material(name='UO2')
    fuel.add_nuclide('U235', 0.031); fuel.add_nuclide('U238', 0.969); fuel.add_nuclide('O16', 2.0); fuel.set_density('g/cm3', 10.29769)
    gd = openmc.Material(name='Gd')
    gd.add_nuclide('U235', 0.031*0.92); gd.add_nuclide('U238', 0.969*0.92); gd.add_nuclide('O16', 2.0*0.92+3.0*0.08)
    gd.add_nuclide('Gd155', 0.08*0.148); gd.add_nuclide('Gd156', 0.08*0.199); gd.add_nuclide('Gd157', 0.08*0.156)
    gd.add_nuclide('Gd158', 0.08*0.249); gd.add_nuclide('Gd160', 0.08*0.218); gd.set_density('g/cm3', 10.5)
    zirc = openmc.Material(name='Zirc')
    for el, frac in [('Sn', 0.014), ('Fe', 0.0121), ('Cr', 0.0107), ('Ni', 0.0050), ('Zr', 0.9582)]: zirc.add_element(el, frac)
    zirc.set_density('g/cm3', 6.55)
    water = openmc.Material(name='Water')
    water.add_nuclide('H1', 2.0); water.add_nuclide('O16', 1.0); water.add_s_alpha_beta('c_H_in_H2O'); water.set_density('g/cm3', 0.701)
    materials = openmc.Materials([fuel, gd, zirc, water])
    materials.export_to_xml()
    return fuel, gd, zirc, water

def create_assembly(grid, fuel, gd, zirc, water):
    fuel_cyl = openmc.ZCylinder(r=fuel_r); clad_inner = openmc.ZCylinder(r=clad_id/2); clad_outer = openmc.ZCylinder(r=clad_od/2)
    box = openmc.model.RectangularPrism(width=pitch, height=pitch, boundary_type='reflective')
    
    def get_univ(rod_type):
        fill = {0: fuel, 1: gd, 2: water}[rod_type]
        cells = [openmc.Cell(fill=fill, region=-fuel_cyl), openmc.Cell(fill=None, region=+fuel_cyl & -clad_inner),
                 openmc.Cell(fill=zirc, region=+clad_inner & -clad_outer), openmc.Cell(fill=water, region=-box & +clad_outer)]
        return openmc.Universe(cells=cells)
    
    lattice = openmc.RectLattice(); lattice.pitch = (pitch, pitch); lattice.lower_left = (-assembly_size/2, -assembly_size/2); lattice.dimension = (17, 17)
    lattice.universes = np.array([[get_univ(grid[i, j]) for j in range(17)] for i in range(17)])
    outer_box = openmc.model.RectangularPrism(assembly_size, assembly_size, boundary_type='reflective')
    return openmc.Geometry([openmc.Cell(fill=lattice, region=-outer_box)])

def evaluate_grid(grid_str, target_k):
    g_count = grid_str.count('g')
    work_dir = tempfile.mkdtemp(prefix='grpo_run_', dir=tempfile.gettempdir())
    original_dir = os.getcwd()

    try:
        os.chdir(work_dir)
        fuel, gd, zirc, water = create_materials()
        grid = np.zeros((17, 17), dtype=int)
        for i, char in enumerate(grid_str):
            if char == 'g': grid[divmod(i, 17)] = 1
            elif char == 'c': grid[divmod(i, 17)] = 2
        
        create_assembly(grid, fuel, gd, zirc, water).export_to_xml()
        settings = openmc.Settings(); settings.batches = 30; settings.inactive = 10; settings.particles = 20000; settings.output = {'summary': False}
        settings.source = openmc.IndependentSource(space=openmc.stats.Box([-assembly_size/2, -assembly_size/2, 0], [assembly_size/2, assembly_size/2, 1]))
        settings.export_to_xml()
        
        tallies = openmc.Tallies(); mesh = openmc.RegularMesh(); mesh.dimension = (17, 17, 1)
        mesh.lower_left = (-assembly_size/2, -assembly_size/2, 0); mesh.upper_right = (assembly_size/2, assembly_size/2, 1)
        pt = openmc.Tally(name='power'); pt.filters = [openmc.MeshFilter(mesh)]; pt.scores = ['fission']; tallies.append(pt)
        tallies.export_to_xml()

        openmc.run(output=False)
        with openmc.StatePoint('statepoint.30.h5') as sp:
            k_eff = sp.keff.nominal_value
            power = sp.get_tally(name='power').mean.ravel()
            fq = power.max() / power.mean() if power.mean() > 0 else 99.9
            fdh = power.reshape(17, 17).mean(axis=0).max() / power.reshape(17, 17).mean(axis=0).mean() if power.reshape(17, 17).mean(axis=0).mean() > 0 else 99.9

        fitness = 0.6 * fq + 0.4 * fdh + 100.0 * abs(k_eff - target_k)
        return (fitness, k_eff, fq, fdh, g_count)
    except Exception:
        return (99.9, 0.0, 99.9, 99.9, g_count)
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

# --- [Model Setup] ---
def generate_grid(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, temperature=1.0, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    grid_str = generated[generated.find(":\n") + 2:].replace(" ", "").replace("\n", "")
    return fix_grid_gt(''.join([c for c in grid_str if c in ['f', 'g', 'c']]))

def get_log_probs(model, tokenizer, prompt, grid_str):
    full_text = prompt + " \n ".join([" ".join(list(grid_str[i:i+17])) for i in range(0, 289, 17)]) + tokenizer.eos_token
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    logits = model(**inputs).logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, inputs["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)[:, prompt_len:].sum()

# --- [Main Execution] ---
if __name__ == "__main__":
    print("🧹 Cleaning temp files..."); [shutil.rmtree(os.path.join(tempfile.gettempdir(), d)) for d in os.listdir(tempfile.gettempdir()) if d.startswith('grpo_run_')]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True).to(DEVICE)
    model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["step", "target_k", "group_idx", "sim_count", "elapsed", "grid", "fitness", "k_eff", "fq", "fdh", "g_count", "loss"])

    start_time = time.time()
    sim_count = 0
    pbar = tqdm(total=NUM_STEPS, desc="GRPO CPT+SFT (Single)")
    
    for step in range(1, NUM_STEPS + 1):
        prompt = f"Reactor Core Design (k={TARGET_K_EFF:.5f}, fq={TARGET_FQ:.4f}, fdh={TARGET_FDH:.4f}):\n"
        grids = [generate_grid(model, tokenizer, prompt) for _ in range(GROUP_SIZE)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=GROUP_SIZE) as executor:
            metrics = list(executor.map(evaluate_grid, grids, [TARGET_K_EFF]*GROUP_SIZE))
        sim_count += GROUP_SIZE

        fitnesses = np.array([m[0] for m in metrics])
        advantages = (np.mean(fitnesses) - fitnesses) / (np.std(fitnesses) + 1e-8)
        best_idx = np.argmin(fitnesses)
        best = metrics[best_idx]

        if not np.all(fitnesses >= 99.0):
            model.train(); optimizer.zero_grad(); loss = 0.0
            for i in range(GROUP_SIZE): loss -= advantages[i] * get_log_probs(model, tokenizer, prompt, grids[i])
            loss = loss / GROUP_SIZE; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            loss_val = loss.item()
        else: loss_val = 0.0

        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(GROUP_SIZE):
                m = metrics[i]
                writer.writerow([step, TARGET_K_EFF, i, sim_count, round(time.time()-start_time, 2), grids[i], round(m[0], 4), round(m[1], 5), round(m[2], 4), round(m[3], 4), m[4], round(loss_val, 4)])

        pbar.update(1); pbar.set_postfix({"Fit": f"{best[0]:.3f}", "Gd": best[4], "Loss": f"{loss_val:.4f}"})

    pbar.close(); model.save_pretrained(FINAL_MODEL_DIR); tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("✅ CPT+SFT Single GRPO Complete!")