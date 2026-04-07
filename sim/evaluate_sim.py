"""
Evaluate policy performance in simulation.

Runs policy for N episodes and records success rate,
reward statistics, and episode lengths.

Usage:
    python evaluate_sim.py --model models/reach_none_20240101/final_model.zip
    python evaluate_sim.py --model models/grasp_medium_20240101/final_model.zip --episodes 100
"""

import argparse
import numpy as np
from pathlib import Path
import json

from envs.so100_env import SO100Env
from stable_baselines3 import SAC, PPO


def evaluate(model_path: str, task: str = "grasp", randomization: str = "none",
             episodes: int = 100, verbose: bool = True):
    """Evaluate trained policy in simulation.
    
    Args:
        model_path: Path to saved model
        task: Task name
        randomization: Randomization level
        episodes: Number of episodes to evaluate
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation results
    """
    # Load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)
    
    # Create environment
    env = SO100Env(task=task, randomization_level=randomization)
    
    # Evaluation metrics
    success_count = 0
    rewards = []
    episode_lengths = []
    final_distances = []
    
    print(f"Evaluating: {model_path}")
    print(f"Episodes: {episodes}")
    print("-" * 40)
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        final_dist = 0
        
        while not (done or truncated) and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            final_dist = info.get('final_distance', 0)
        
        if info.get('success', False):
            success_count += 1
        
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        final_distances.append(final_dist)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Steps={steps}, "
                  f"Success={info.get('success', False)}")
    
    # Compute statistics
    results = {
        'model_path': model_path,
        'task': task,
        'randomization': randomization,
        'episodes': episodes,
        'success_rate': success_count / episodes,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'mean_final_distance': np.mean(final_distances),
        'std_final_distance': np.std(final_distances),
    }
    
    # Print results
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Success Rate:        {results['success_rate']*100:.1f}%")
    print(f"Mean Reward:         {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} +/- {results['std_episode_length']:.1f}")
    print(f"Mean Final Distance: {results['mean_final_distance']:.4f} +/- {results['std_final_distance']:.4f}")
    print("=" * 40)
    
    return results


def compare_policies(models: list, **common_args):
    """Compare multiple policies.
    
    Args:
        models: List of (model_path, task, randomization) tuples
        common_args: Other args to pass to evaluate
    """
    all_results = []
    
    for model_path, task, rand in models:
        results = evaluate(model_path, task, rand, **common_args)
        results['policy_name'] = model_path.split('/')[-2]  # Folder name
        all_results.append(results)
    
    print("\n" + "=" * 60)
    print("POLICY COMPARISON")
    print("=" * 60)
    print(f"{'Policy':<30} {'Success%':<12} {'Mean Reward':<12} {'Ep Length':<12}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['policy_name']:<30} "
              f"{r['success_rate']*100:<12.1f} "
              f"{r['mean_reward']:<12.2f} "
              f"{r['mean_episode_length']:<12.1f}")
    
    print("=" * 60)
    
    return all_results


def save_results(results: list, output_path: str):
    """Save evaluation results to JSON.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SO-100 policy in simulation")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to saved model file")
    parser.add_argument("--task", type=str, default="grasp",
                       choices=["reach", "grasp"],
                       help="Task name")
    parser.add_argument("--randomization", type=str, default="none",
                       help="Randomization level used in training")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--compare", nargs='+', default=None,
                       help="Additional models to compare: model_path task rand")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Evaluate main model
    results = evaluate(
        args.model, args.task, args.randomization, args.episodes
    )
    
    # Save results
    if args.output:
        save_results([results], args.output)


if __name__ == "__main__":
    main()
