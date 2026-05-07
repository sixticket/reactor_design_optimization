import os
from pathlib import Path

# Project root: derived from REACTORGEN_ROOT env var if set, otherwise
# inferred from this script's location.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[2])
PROJECT_ROOT = os.environ.get("REACTORGEN_ROOT", _DEFAULT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

def plot_prompt_sensitivity():
    data_path = os.path.join(PROJECT_ROOT, "eval_prompt_sensitivity/prompt_sensitivity_results.csv")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Split the models into two logical groups: SFT Only vs CPT+SFT
    sft_models = ["SFT_Only_Base", "DPO_Single_SFT", "DPO_Multi_SFT", "GRPO_Single_SFT", "GRPO_Multi_SFT"]
    cpt_sft_models = ["CPT_SFT_Base", "DPO_Single_CPT_SFT", "DPO_Multi_CPT_SFT", "GRPO_Single_CPT_SFT", "GRPO_Multi_CPT_SFT"]
    
    # Map raw model names to clean labels
    label_map = {
        "SFT_Only_Base": "Base (SFT Only)",
        "DPO_Single_SFT": "DPO Single",
        "DPO_Multi_SFT": "DPO Multi",
        "GRPO_Single_SFT": "GRPO Single",
        "GRPO_Multi_SFT": "GRPO Multi",
        
        "CPT_SFT_Base": "Base (CPT+SFT)",
        "DPO_Single_CPT_SFT": "DPO Single",
        "DPO_Multi_CPT_SFT": "DPO Multi",
        "GRPO_Single_CPT_SFT": "GRPO Single",
        "GRPO_Multi_CPT_SFT": "GRPO Multi"
    }
    
    # Predefined colors for standard methods
    color_palette = {
        "Base (SFT Only)": "grey", "Base (CPT+SFT)": "grey",
        "DPO Single": "#1f77b4", "DPO Multi": "#9467bd",
        "GRPO Single": "#ff7f0e", "GRPO Multi": "#d62728"
    }
    
    df['clean_label'] = df['model_name'].map(label_map)
    df['training_base'] = df['model_name'].apply(lambda x: 'SFT Only' if x in sft_models else 'CPT+SFT')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: SFT Only
    df_sft = df[df['training_base'] == 'SFT Only']
    sns.lineplot(data=df_sft, x="target_k", y="gd_count", hue="clean_label", 
                 ax=axes[0], marker="o", palette=color_palette, linewidth=2.5, err_style="band")
    axes[0].set_title("Prompt Sensitivity (SFT Only Baseline Checkpoints)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Provided Prompt $k_{eff}$ Target", fontsize=12)
    axes[0].set_ylabel("Generated Gadolinium (Gd) Count", fontsize=12)
    axes[0].get_legend().remove()
    
    # Plot 2: CPT + SFT
    df_cpt = df[df['training_base'] == 'CPT+SFT']
    sns.lineplot(data=df_cpt, x="target_k", y="gd_count", hue="clean_label", 
                 ax=axes[1], marker="o", palette=color_palette, linewidth=2.5, err_style="band")
    axes[1].set_title("Prompt Sensitivity (CPT+SFT Baseline Checkpoints)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Provided Prompt $k_{eff}$ Target", fontsize=12)
    axes[1].set_ylabel("")
    axes[1].get_legend().remove()
    
    # Global tweaks
    for ax in axes:
        ax.set_xticks(df['target_k'].unique())
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # Shared legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Optimization Method", loc='lower center',
              bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=11, title_fontsize=12, frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    save_path = os.path.join(PROJECT_ROOT, "plot/prompt_sensitivity/prompt_sensitivity_plot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved Prompt Sensitivity plot to: {save_path}")

if __name__ == "__main__":
    plot_prompt_sensitivity()
