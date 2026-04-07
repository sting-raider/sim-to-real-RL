"""
Training Script for SO-100 Sim-to-Real Project.

Trains RL policies using SAC (Soft Actor-Critic) on parallel
simulation environments. Supports baseline and domain-randomized training.

Usage:
    python train.py --task reach --steps 500000
    python train.py --task grasp --randomization medium --steps 5000000
    python train.py --task grasp --randomization none --steps 5000000
    python train.py --config configs/domain_rand.yaml
"""

import argparse
import yaml
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Import environment
from envs.so100_env import SO100Env
from envs.domain_rand import get_preset_config

# SB3 imports
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

try:
    import wandb
    from stable_baselines3.common.callbacks import WandbCallback
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Weights & Biases not installed. Install with: pip install wandb")


def make_env(task: str, randomization_level: str = "none", rank: int = 0, seed: int = 0):
    """Environment factory for parallel execution.
    
    Args:
        task: Task name ('reach' or 'grasp')
        randomization_level: Randomization level
        rank: Environment rank (used for unique seeds)
        seed: Base random seed
        
    Returns:
        Environment creation function
    """
    def _init():
        env = SO100Env(
            task=task,
            randomization_level=randomization_level,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def load_config(config_path: str = None) -> dict:
    """Load training configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(args, config: dict = None):
    """Main training function.
    
    Args:
        args: Command line arguments
        config: Optional config dictionary from YAML
    """
    # Merge CLI args with config file
    task = config.get('task', args.task) if config else args.task
    randomization = config.get('randomization', args.randomization) if config else args.randomization
    steps = config.get('steps', args.steps) if config else args.steps
    num_envs = config.get('num_envs', args.num_envs) if config else args.num_envs
    algorithm = config.get('algorithm', args.algorithm) if config else args.algorithm
    
    print(f"\n{'='*60}")
    print(f"SO-100 Sim-to-Real Training")
    print(f"{'='*60}")
    print(f"Task:              {task}")
    print(f"Randomization:     {randomization}")
    print(f"Algorithm:         {algorithm}")
    print(f"Parallel envs:     {num_envs}")
    print(f"Total steps:       {steps:,}")
    print(f"{'='*60}\n")
    
    # Create output directories
    run_id = f"{task}_{randomization}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path("models") / run_id
    log_dir = Path("logs") / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create parallel environment
    print(f"Creating {num_envs} parallel environments...")
    env = SubprocVecEnv([
        make_env(task, randomization, rank=i) for i in range(num_envs)
    ])
    
    # Create evaluation environment
    eval_env = SO100Env(task=task, randomization_level=randomization)
    
    # Setup algorithm
    print(f"Initializing {algorithm}...")
    if algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=1,
            tensorboard_log=str(log_dir),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    elif algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(log_dir),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'sac' or 'ppo'.")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=steps // 10,
        save_path=str(save_dir),
        name_prefix=run_id,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(log_dir),
        eval_freq=steps // 20,
        n_eval_episodes=10,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # WandB callback (if available)
    if HAS_WANDB and not args.no_wandb:
        run_name = f"{task}_{randomization}"
        run = wandb.init(
            project="sim-to-real-arm",
            name=run_name,
            config={
                "task": task,
                "randomization": randomization,
                "algorithm": algorithm,
                "num_envs": num_envs,
                "total_steps": steps,
                "learning_rate": 3e-4,
            }
        )
        wandb_callback = WandbCallback(
            model_save_path=f"models/{run_id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    # Start training
    print(f"\nStarting training for {steps:,} steps...")
    print(f"Estimated time: {steps / (num_envs * 10000):.1f} hours\n")
    
    model.learn(
        total_timesteps=steps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = save_dir / "final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Close environment
    env.close()
    
    # Finish WandB run
    if HAS_WANDB and not args.no_wandb:
        run.finish()
    
    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Train SO-100 policy")
    parser.add_argument("--task", type=str, default="reach",
                       choices=["reach", "grasp"], help="Task to train")
    parser.add_argument("--randomization", type=str, default="none",
                       choices=["none", "low", "medium", "high", "extreme"],
                       help="Domain randomization level")
    parser.add_argument("--algorithm", type=str, default="sac",
                       choices=["sac", "ppo"], help="RL algorithm")
    parser.add_argument("--steps", type=int, default=5_000_000,
                       help="Total training steps")
    parser.add_argument("--num_envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    
    train(args, config)


if __name__ == "__main__":
    main()
