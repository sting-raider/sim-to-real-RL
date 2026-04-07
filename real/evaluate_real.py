"""
Real-World Evaluation Script.

Runs standardized 20-grasp evaluation on the real SO-100 arm.
Records results for comparison with simulation performance.

Usage:
    python evaluate_real.py --model models/baseline/final_model.zip --name "baseline"
    python evaluate_real.py --model models/domain_rand/final_model.zip --name "domain_rand"
    python evaluate_real.py --compare models/*/final_model.zip
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from real.inference import run_policy, RealSO100Env
from stable_baselines3 import SAC, PPO


def evaluate_real(model_path: str, name: str, attempts: int = 20, **kwargs):
    """Evaluate a single model on real hardware.
    
    Args:
        model_path: Path to trained model
        name: Human-readable name for this policy
        attempts: Number of grasp attempts
        kwargs: Additional args passed to run_policy
    """
    print(f"\nEvaluating: {name}")
    print(f"Model: {model_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 40)
    
    results = run_policy(model_path, attempts=attempts, **kwargs)
    
    return {
        'name': name,
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat(),
        'attempts': attempts,
        'successes': sum(1 for r in results if r['success']),
        'success_rate': sum(1 for r in results if r['success']) / attempts,
        'results': results,
    }


def save_results(all_results: list, output_dir: str = "models"):
    """Save evaluation results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / f"real_eval_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return output_path


def print_comparison(all_results: list):
    """Print comparison table of evaluation results."""
    print("\n" + "=" * 70)
    print("REAL-WORLD EVALUATION COMPARISON")
    print("=" * 70)
    print(f"{'Policy':<20} {'Success%':<12} {'Success':<12} {'Attempts':<12}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['name']:<20} {r['success_rate']*100:<12.1f} "
              f"{r['successes']:<12} {r['attempts']:<12}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies on real SO-100 arm")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--name", type=str, default="unnamed",
                       help="Name for this policy")
    parser.add_argument("--attempts", type=int, default=20,
                       help="Number of grasp attempts")
    parser.add_argument("--compare", nargs='+',
                       help="Multiple models to compare")
    parser.add_argument("--arm-port", type=str, default="/dev/ttyUSB0",
                       help="Arm serial port")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Camera calibration file")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    all_results = []
    
    if args.compare:
        # Compare multiple models
        for model_path in args.compare:
            name = Path(model_path).parent.name
            result = evaluate_real(
                model_path, name,
                attempts=args.attempts,
                arm_port=args.arm_port,
                camera_index=args.camera,
                calibration_path=args.calibration,
            )
            all_results.append(result)
            print_comparison(all_results)
    elif args.model:
        # Evaluate single model
        result = evaluate_real(
            args.model, args.name,
            attempts=args.attempts,
            arm_port=args.arm_port,
            camera_index=args.camera,
            calibration_path=args.calibration,
        )
        all_results.append(result)
    else:
        print("Provide --model for single evaluation or --compare for multiple models.")
        return
    
    print_comparison(all_results)
    save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
