import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[2])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def load_data():
    base_dir = PROJECT_ROOT

    # GA unconstrained
    ga_df = pd.read_csv(os.path.join(base_dir, "ga/ga_baseline_single_results.csv"))
    # GA constrained (16 Gd)
    ga16_df = pd.read_csv(os.path.join(base_dir, "ga/ga_baseline_single_gd16_results.csv"))
    # DPO single
    dpo_df = pd.read_csv(os.path.join(base_dir, "training/dpo/single_target/dpo_single_target_results.csv"))
    # GRPO single
    grpo_df = pd.read_csv(os.path.join(base_dir, "training/grpo/single/grpo_cpt_sft_single_results.csv"))

    return ga_df, ga16_df, dpo_df, grpo_df


def plot_vs_ga(ga_df, ga16_df, dpo_df, grpo_df):
    # --- Prepare DPO data (2 sims per step) ---
    dpo_step_best = dpo_df['chosen_fitness']
    dpo_cummin = dpo_step_best.cummin()
    dpo_sim = dpo_df['simulation_count']
    dpo_step_k = dpo_df['chosen_k_eff']
    dpo_cummin_k = dpo_step_k.copy()
    # Track k_eff of the cumulative best fitness
    dpo_best_idx = dpo_step_best.expanding().apply(lambda s: s.idxmin(), raw=False).astype(int)
    dpo_cummin_k = dpo_step_k.iloc[dpo_best_idx.values].values

    # --- Prepare GRPO data (4 sims per step) ---
    grpo_step_best = grpo_df['best_fit']
    grpo_cummin = grpo_step_best.cummin()
    grpo_sim = grpo_df['sim_count']
    grpo_step_k = grpo_df['best_k']
    grpo_best_idx = grpo_step_best.expanding().apply(lambda s: s.idxmin(), raw=False).astype(int)
    grpo_cummin_k = grpo_step_k.iloc[grpo_best_idx.values].values

    # Colors & styles
    c_dpo = '#1f77b4'
    c_grpo = '#ff7f0e'
    c_ga = '#2ca02c'
    c_ga16 = '#9467bd'

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # ========================================================
    # Subplot 1: DPO & GRPO — Step Best (dashed) + Cumulative Min (solid)
    # ========================================================
    ax = axes[0]
    # DPO
    ax.plot(dpo_sim, dpo_step_best, color=c_dpo, alpha=0.35, linewidth=1, linestyle='--')
    ax.plot(dpo_sim, dpo_cummin, color=c_dpo, linewidth=1.5, label='DPO (Cumulative Min)')
    # GRPO
    ax.plot(grpo_sim, grpo_step_best, color=c_grpo, alpha=0.35, linewidth=1, linestyle='--')
    ax.plot(grpo_sim, grpo_cummin, color=c_grpo, linewidth=1.5, label='GRPO (Cumulative Min)')
    # Light labels for dashed
    ax.plot([], [], color=c_dpo, alpha=0.5, linestyle='--', label='DPO (Step Best, per 2 sims)')
    ax.plot([], [], color=c_grpo, alpha=0.5, linestyle='--', label='GRPO (Step Best, per 4 sims)')

    ax.set_ylabel('Fitness (↓ lower is better)', fontsize=11)
    ax.set_title('(a) DPO & GRPO Single-Target Optimization Trajectory', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ========================================================
    # Subplot 2: Cumulative Min comparison — GA vs GA-16 vs DPO vs GRPO
    # ========================================================
    ax = axes[1]
    ax.plot(ga_df['simulation_count'], ga_df['best_fitness'],
            color=c_ga, linewidth=1.5, label='GA (No Gd Constraint)')
    ax.plot(ga16_df['simulation_count'], ga16_df['best_fitness'],
            color=c_ga16, linewidth=1.5, label='GA (Gd=16 Constraint)')
    ax.plot(dpo_sim, dpo_cummin, color=c_dpo, linewidth=1.5, label='DPO (Cumulative Min)')
    ax.plot(grpo_sim, grpo_cummin, color=c_grpo, linewidth=1.5, label='GRPO (Cumulative Min)')

    ax.set_ylabel('Fitness (↓ lower is better)', fontsize=11)
    ax.set_title('(b) Cumulative Best Fitness: GA vs DPO vs GRPO', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ========================================================
    # Subplot 3: Gd Count
    # ========================================================
    ax = axes[2]
    ax.plot(ga_df['simulation_count'], ga_df['best_gd'],
            color=c_ga, linewidth=1.5, label='GA (No Gd Constraint)')
    ax.plot(ga16_df['simulation_count'], ga16_df['best_gd'],
            color=c_ga16, linewidth=1.5, label='GA (Gd=16 Constraint)')
    # DPO chosen gd
    dpo_gd = dpo_df['chosen_g_count']
    ax.plot(dpo_sim, dpo_gd, color=c_dpo, linewidth=1.5, label='DPO (Chosen Gd)')
    # GRPO best gd
    ax.plot(grpo_sim, grpo_df['gd'], color=c_grpo, linewidth=1.5, label='GRPO (Best Gd)')

    ax.set_ylabel('Gadolinium (Gd) Count', fontsize=11)
    ax.set_title('(c) Gadolinium Pin Count per Step', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ========================================================
    # Subplot 4: K-effective (tracking the cumulative-best individual)
    # ========================================================
    ax = axes[3]
    ax.axhline(y=1.05, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label='Target $k_{eff}$ = 1.05')

    ax.plot(ga_df['simulation_count'], ga_df['best_k_eff'],
            color=c_ga, linewidth=1.5, label='GA (No Gd Constraint)')
    ax.plot(ga16_df['simulation_count'], ga16_df['best_k_eff'],
            color=c_ga16, linewidth=1.5, label='GA (Gd=16 Constraint)')
    ax.plot(dpo_sim, dpo_cummin_k, color=c_dpo, linewidth=1.5, label='DPO (Best $k_{eff}$)')
    ax.plot(grpo_sim, grpo_cummin_k, color=c_grpo, linewidth=1.5, label='GRPO (Best $k_{eff}$)')

    ax.set_ylabel('$k_{eff}$', fontsize=11)
    ax.set_xlabel('OpenMC Simulation Count', fontsize=12)
    ax.set_title('(d) $k_{eff}$ of Cumulative Best Individual', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Global x-limit
    max_sim = max(ga_df['simulation_count'].max(), ga16_df['simulation_count'].max(),
                  dpo_sim.max(), grpo_sim.max())
    for a in axes:
        a.set_xlim(0, max_sim)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, "plot/vs_ga/ga_vs_dpo_grpo_fitness.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading datasets...")
    ga_df, ga16_df, dpo_df, grpo_df = load_data()
    print("Generating comprehensive comparative plot...")
    plot_vs_ga(ga_df, ga16_df, dpo_df, grpo_df)
    print("Done!")
