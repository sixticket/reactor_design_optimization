import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[2])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os

sns.set_theme(style="white")

def parse_grid(grid_str):
    mapping = {'f': 0, 'g': 1, 'c': 2}
    data = [mapping.get(c, 0) for c in grid_str]
    if len(data) < 289: data.extend([0]*(289-len(data)))
    return np.array(data[:289]).reshape(17, 17)

def load_best_samples():
    base_dir = PROJECT_ROOT
    
    # NuScale 16-Gd
    nuscale_df = pd.read_csv(os.path.join(base_dir, "nuscale/nuscale_baseline_results.csv"))
    nuscale_16 = {
        'label': 'NuScale 16-Gd',
        'grid': nuscale_df.iloc[0]['grid'],
        'fit': nuscale_df.iloc[0]['fitness'],
        'gd': nuscale_df.iloc[0]['gd_count'],
        'keff': nuscale_df.iloc[0]['k_eff'],
        'color': '#8c564b'
    }
    # NuScale 24-Gd
    nuscale_24 = {
        'label': 'NuScale 24-Gd',
        'grid': nuscale_df.iloc[1]['grid'],
        'fit': nuscale_df.iloc[1]['fitness'],
        'gd': nuscale_df.iloc[1]['gd_count'],
        'keff': nuscale_df.iloc[1]['k_eff'],
        'color': '#9467bd'
    }
    
    # GA unconstrained
    ga_df = pd.read_csv(os.path.join(base_dir, "ga/ga_baseline_single_results.csv"))
    ga_best = ga_df.loc[ga_df['best_fitness'].idxmin()]
    ga_sample = {
        'label': 'GA (No Constraint)',
        'grid': ga_best['best_grid'],
        'fit': ga_best['best_fitness'],
        'gd': ga_best['best_gd'],
        'keff': ga_best['best_k_eff'],
        'color': '#2ca02c'
    }
    
    # GA constrained (Gd=16)
    ga16_df = pd.read_csv(os.path.join(base_dir, "ga/ga_baseline_single_gd16_results.csv"))
    ga16_best = ga16_df.loc[ga16_df['best_fitness'].idxmin()]
    ga16_sample = {
        'label': 'GA (Gd=16)',
        'grid': ga16_best['best_grid'],
        'fit': ga16_best['best_fitness'],
        'gd': ga16_best['best_gd'],
        'keff': ga16_best['best_k_eff'],
        'color': '#17becf'
    }
    
    # DPO
    dpo_df = pd.read_csv(os.path.join(base_dir, "training/dpo/single_target/dpo_single_target_results.csv"))
    dpo_best_idx = dpo_df['chosen_fitness'].idxmin()
    dpo_row = dpo_df.loc[dpo_best_idx]
    dpo_sample = {
        'label': 'DPO',
        'grid': dpo_row['chosen_grid'],
        'fit': dpo_row['chosen_fitness'],
        'gd': dpo_row['chosen_g_count'],
        'keff': dpo_row['chosen_k_eff'],
        'color': '#1f77b4'
    }
    
    # GRPO
    grpo_df = pd.read_csv(os.path.join(base_dir, "training/grpo/single/grpo_cpt_sft_single_results.csv"))
    grpo_row = grpo_df.loc[grpo_df['best_fit'].idxmin()]
    grpo_sample = {
        'label': 'GRPO',
        'grid': grpo_row['best_grid'],
        'fit': grpo_row['best_fit'],
        'gd': grpo_row['gd'],
        'keff': grpo_row['best_k'],
        'color': '#ff7f0e'
    }
    
    # Row 1: NuScale 16, NuScale 24, GA unconstrained
    # Row 2: GA Gd=16, DPO, GRPO
    return [nuscale_16, nuscale_24, ga_sample, ga16_sample, dpo_sample, grpo_sample]

def plot_best_reactors():
    samples = load_best_samples()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'hspace': 0.40})
    axes_flat = axes.flatten()
    grid_cmap = ListedColormap(['#E0E0E0', '#70C0F2', '#EE7950'])

    # Legend at the very top
    legend_patches = [
        Patch(facecolor='#E0E0E0', edgecolor='lightgray', label='Fuel'),
        Patch(facecolor='#70C0F2', label='Gd Poison'),
        Patch(facecolor='#EE7950', label='Guide Tube (GT)')
    ]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=13, frameon=False)

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        grid_matrix = parse_grid(sample['grid'])
        
        for spine in ax.spines.values():
            spine.set_edgecolor(sample['color'])
            spine.set_linewidth(3)
            
        sns.heatmap(grid_matrix, ax=ax, cmap=grid_cmap, cbar=False, square=True,
                    linewidths=0.5, linecolor='white', vmin=0, vmax=2)
                    
        ax.set_title(sample['label'], fontsize=16, fontweight='bold', color=sample['color'], pad=12)
        
        metrics_text = (f"Fitness: {sample['fit']:.3f} | Gd: {int(sample['gd'])}\n"
                        f"$k_{{eff}}$: {sample['keff']:.5f}")
        
        ax.text(0.5, -0.08, metrics_text, transform=ax.transAxes, ha='center', va='top', 
                fontsize=11, color='#333333', fontweight='semibold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4))
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PROJECT_ROOT, "plot/vs_ga/best_reactor_configurations.png"), dpi=300, bbox_inches='tight')
    print("Saved best_reactor_configurations.png")

if __name__ == "__main__":
    print("Extracting the absolute best configurations...")
    plot_best_reactors()
