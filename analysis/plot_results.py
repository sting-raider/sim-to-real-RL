"""
Plot Results for Sim-to-Real SO-100 Project.

Generates publication-quality charts for:
1. Sim vs Real success rate comparison
2. Learning curves (reward over training steps)
3. Transfer gap by randomization level
4. Ablation study results
"""

import argparse
import json
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Run: pip install matplotlib")

from pathlib import Path


def plot_transfer_gap(sim_results: list, real_results: list, output_path: str = None):
    """Plot sim vs real success rates and transfer gap."""
    if not HAS_MATPLOTLIB:
        print("Cannot plot without matplotlib.")
        return
    
    policies = []
    sim_rates = []
    real_rates = []
    
    for sim, real in zip(sim_results, real_results):
        policies.append(sim.get('name', sim.get('policy_name', 'Unknown')))
        sim_rates.append(sim.get('success_rate', 0) * 100)
        real_rates.append(real.get('success_rate', 0) * 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Side-by-side bar chart
    x = np.arange(len(policies))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, sim_rates, width, label='Simulation', color='#4C72B0')
    bars2 = ax1.bar(x + width/2, real_rates, width, label='Real World', color='#DD8452')
    
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Sim-to-Real Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Transfer gap
    gaps = [s - r for s, r in zip(sim_rates, real_rates)]
    colors = ['#C44E52' if g > 30 else '#8172B2' if g > 15 else '#55A868' for g in gaps]
    
    bars3 = ax2.bar(policies, gaps, color=colors)
    ax2.set_ylabel('Transfer Gap (%)')
    ax2.set_title('Performance Gap (Sim - Real)')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylim(-10, max(gaps) + 10 if gaps else 60)
    
    for i, (bar, gap) in enumerate(zip(bars3, gaps)):
        ax2.annotate(f'{gap:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C44E52', label='Large gap (>30%)'),
        Patch(facecolor='#8172B2', label='Medium gap (15-30%)'),
        Patch(facecolor='#55A868', label='Small gap (<15%)'),
    ]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.savefig("analysis/transfer_gap.png")
        print("Saved to: analysis/transfer_gap.png")
    
    plt.close()


def plot_ablation_study(ablation_results: list, output_path: str = None):
    """Plot ablation study: success rate vs randomization level."""
    if not HAS_MATPLOTLIB:
        print("Cannot plot without matplotlib.")
        return
    
    levels = []
    sim_rates = []
    real_rates = []
    sim_stds = []
    real_stds = []
    
    for result in ablation_results:
        levels.append(result.get('level', result.get('name', 'Unknown')))
        sim_rates.append(result.get('sim_success_rate', 0) * 100)
        real_rates.append(result.get('real_success_rate', 0) * 100)
        sim_stds.append(result.get('sim_std', 0) * 100)
        real_stds.append(result.get('real_std', 0) * 100)
    
    x = np.arange(len(levels))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate vs randomization
    ax1.errorbar(x, sim_rates, yerr=sim_stds, fmt='o-', color='#4C72B0', label='Simulation',
                 capsize=5, capthick=1, elinewidth=1)
    ax1.errorbar(x, real_rates, yerr=real_stds, fmt='s-', color='#DD8452', label='Real World',
                 capsize=5, capthick=1, elinewidth=1)
    
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Performance vs Randomization Level')
    ax1.set_xticks(x)
    _ = ax1.set_xticklabels(levels)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Transfer gap vs randomization
    gaps = [s - r for s, r in zip(sim_rates, real_rates)]
    ax2.plot(x, gaps, 'o-', color='#C44E52', linewidth=2, markersize=8)
    ax2.fill_between(x, 0, gaps, alpha=0.3, color='#C44E52')
    ax2.set_ylabel('Transfer Gap (%)')
    ax2.set_title('Gap vs Randomization Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.savefig("analysis/ablation_study.png")
        print("Saved to: analysis/ablation_study.png")
    
    plt.close()


def plot_learning_curve(timesteps: list, rewards: list, output_path: str = None):
    """Plot learning curve (reward over training steps)."""
    if not HAS_MATPLOTLIB:
        print("Cannot plot without matplotlib.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(timesteps, rewards, linewidth=2, color='#4C72B0')
    ax.fill_between(timesteps,
                    np.array(rewards) - np.array(rewards)*0.1,
                    np.array(rewards) + np.array(rewards)*0.1,
                    alpha=0.2, color='#4C72B0')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Learning Curve')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.savefig("analysis/learning_curve.png")
        print("Saved to: analysis/learning_curve.png")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for SO-100 results")
    parser.add_argument("--sim-results", type=str, nargs='+',
                       help="JSON files with sim evaluation results")
    parser.add_argument("--real-results", type=str, nargs='+',
                       help="JSON files with real evaluation results")
    parser.add_argument("--ablation", type=str, nargs='+',
                       help="JSON files with ablation study results")
    parser.add_argument("--output-dir", type=str, default="analysis",
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.sim_results and args.real_results:
        sim_data = []
        for f in args.sim_results:
            with open(f) as fp:
                sim_data.extend(json.load(fp))
        
        real_data = []
        for f in args.real_results:
            with open(f) as fp:
                real_data.extend(json.load(fp))
        
        plot_transfer_gap(sim_data, real_data,
                         output_path=f"{args.output_dir}/transfer_gap.png")
    
    if args.ablation:
        ablation_data = []
        for f in args.ablation:
            with open(f) as fp:
                ablation_data.extend(json.load(fp))
        
        plot_ablation_study(ablation_data,
                           output_path=f"{args.output_dir}/ablation_study.png")


if __name__ == "__main__":
    main()
