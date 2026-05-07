import os
from pathlib import Path

# Project root: derived from REACTORFOLD_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[2])
PROJECT_ROOT = os.environ.get("REACTORFOLD_ROOT", _DEFAULT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

def load_data():
    base_dir = os.path.join(PROJECT_ROOT, "training")
    
    # 1. DPO Single
    dpo_single_path = os.path.join(base_dir, "dpo/single_target/dpo_single_target_results.csv")
    dpo_single = pd.read_csv(dpo_single_path)
    
    # 2. GRPO Single
    grpo_single_path = os.path.join(base_dir, "grpo/single/grpo_cpt_sft_single_results.csv")
    grpo_single = pd.read_csv(grpo_single_path)
    
    # 3. DPO Multi
    dpo_multi_path = os.path.join(base_dir, "dpo/multi_target/dpo_multi_lhs_results.csv")
    dpo_multi = pd.read_csv(dpo_multi_path)
    
    # 4. GRPO Multi
    grpo_multi_path = os.path.join(base_dir, "grpo/multi/grpo_cpt_sft_multi_results.csv")
    grpo_multi = pd.read_csv(grpo_multi_path)
    
    return dpo_single, grpo_single, dpo_multi, grpo_multi

def plot_losses(dpo_single, grpo_single, dpo_multi, grpo_multi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 20

    # DPO Loss Plot (Single vs Multi)
    axes[0].plot(dpo_single['step'], dpo_single['loss'].rolling(window=window, min_periods=1).mean(), 
                 label='DPO (Single Target)', color='blue', alpha=0.8, linewidth=2)
    axes[0].plot(dpo_multi['step'], dpo_multi['loss'].rolling(window=window, min_periods=1).mean(), 
                 label='DPO (Multi Target)', color='cyan', alpha=0.8, linewidth=2)
    axes[0].set_title('DPO Training Loss')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss (Cross Entropy)')
    axes[0].legend()

    # GRPO Loss Plot (Single vs Multi)
    axes[1].plot(grpo_single['step'], grpo_single['loss'].rolling(window=window, min_periods=1).mean(), 
                 label='GRPO (Single Target)', color='red', alpha=0.8, linewidth=2)
    axes[1].plot(grpo_multi['step'], grpo_multi['loss'].rolling(window=window, min_periods=1).mean(), 
                 label='GRPO (Multi Target)', color='orange', alpha=0.8, linewidth=2)
    axes[1].set_title('GRPO Training Loss (Advantage Surrogate)')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Loss (PPO-style)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('dpo_grpo_loss_comparison.png', dpi=300)
    print("Saved dpo_grpo_loss_comparison.png")
    plt.close()

def plot_fitness_single_target(dpo_single, grpo_single):
    plt.figure(figsize=(10, 6))
    
    # For DPO, grab the best of the 2 samples in each step ('chosen_fitness')
    dpo_realtime = dpo_single['chosen_fitness']
    dpo_cummin = dpo_realtime.cummin()
    
    # For GRPO, grab the best of the 4 samples in each step ('best_fit')
    grpo_realtime = grpo_single['best_fit']
    grpo_cummin = grpo_realtime.cummin()
    
    # 1. Cumulative Minimum (Solid thick lines)
    plt.plot(dpo_single['simulation_count'], dpo_cummin, 
             label='DPO (Cumulative Min)', color='blue', linewidth=2.5)
    plt.plot(grpo_single['sim_count'], grpo_cummin, 
             label='GRPO (Cumulative Min)', color='red', linewidth=2.5)
             
    # 2. Real-time Step Best (Dashed/faded lines)
    plt.plot(dpo_single['simulation_count'], dpo_realtime, 
             label='DPO (Step Best)', color='cyan', linestyle='--', alpha=0.6, linewidth=1.5)
    plt.plot(grpo_single['sim_count'], grpo_realtime, 
             label='GRPO (Step Best)', color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
    
    plt.title('Optimization Trajectory: DPO vs GRPO (Single Target)')
    plt.xlabel('OpenMC Simulation Count')
    plt.ylabel('Minimum Fitness (Lower is better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dpo_grpo_single_target_fitness.png', dpi=300)
    print("Saved dpo_grpo_single_target_fitness.png")
    plt.close()

def plot_fitness_multi_target(dpo_multi, grpo_multi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 20

    # 1. Smoothed Fitness vs Sim Count
    # Multi-target fitness is noisy because target_k is randomized.
    # But a moving average shows if the model gets generally better.
    axes[0].plot(dpo_multi['simulation_count'], dpo_multi['chosen_fitness'].rolling(window=window, min_periods=1).mean(), 
                 label='DPO (Moving Avg)', color='cyan', linewidth=2)
    axes[0].plot(grpo_multi['sim_count'], grpo_multi['best_fit'].rolling(window=window, min_periods=1).mean(), 
                 label='GRPO (Moving Avg)', color='orange', linewidth=2)
    axes[0].set_title('Smoothed Fitness (Multi Target)')
    axes[0].set_xlabel('OpenMC Simulation Count')
    axes[0].set_ylabel('Fitness (Lower is better)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # 2. Distance from Target K vs Sim Count
    dpo_k_err = (dpo_multi['chosen_k_eff'] - dpo_multi['target_k']).abs() * 100000  # in pcm
    grpo_k_err = (grpo_multi['best_k'] - grpo_multi['target_k']).abs() * 100000  # in pcm
    
    axes[1].plot(dpo_multi['simulation_count'], dpo_k_err.rolling(window=window, min_periods=1).mean(), 
                 label='DPO (Moving Avg Error)', color='cyan', linewidth=2)
    axes[1].plot(grpo_multi['sim_count'], grpo_k_err.rolling(window=window, min_periods=1).mean(), 
                 label='GRPO (Moving Avg Error)', color='orange', linewidth=2)
    axes[1].set_title('Deviation from Target k_eff (Multi Target)')
    axes[1].set_xlabel('OpenMC Simulation Count')
    axes[1].set_ylabel('k_eff Absolute Error (pcm)')
    axes[1].set_yscale('log')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('dpo_grpo_multi_target_performance.png', dpi=300)
    print("Saved dpo_grpo_multi_target_performance.png")
    plt.close()

if __name__ == "__main__":
    print("Loading data...")
    dpo_single, grpo_single, dpo_multi, grpo_multi = load_data()
    
    print("Generating loss plots...")
    plot_losses(dpo_single, grpo_single, dpo_multi, grpo_multi)
    
    print("Generating single-target fitness plots...")
    plot_fitness_single_target(dpo_single, grpo_single)
    
    print("Generating multi-target performance plots...")
    plot_fitness_multi_target(dpo_multi, grpo_multi)
    
    print("Done!")
