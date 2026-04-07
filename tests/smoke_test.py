"""Smoke test — verify SO100Env runs without crashes."""

import os, sys, time
import numpy as np

# Add project root to path so imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sim.envs.so100_env import SO100Env


def test_env(task="reach", randomization="none", steps=50):
    print(f"\n{'='*50}")
    print(f"Testing: task={task}  randomization={randomization}")
    print(f"{'='*50}")

    env = SO100Env(task=task, randomization_level=randomization)
    obs, info = env.reset(seed=42)

    print(f"  Obs shape          : {obs.shape}")
    assert obs.shape == (21,), f"Expected (21,) got {obs.shape}"
    print(f"  Action space       : {env.action_space}")
    print(f"  Obs  space         : {env.observation_space}")
    print(f"  Robot URDF         : {env.urdf_path}")

    print(f"\n  Stepping {steps} times with random actions…")
    rewards = []
    for t in range(steps):
        action = env.action_space.sample()  # random [-1,1]^6
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            print(f"  Episode ended at step {t}: success={info['success']}, reward={info['episode_reward']:.2f}")
            obs, info = env.reset()

    env.close()

    print(f"\n  Reward: min={min(rewards):.3f}  max={max(rewards):.3f}  mean={np.mean(rewards):.3f}")
    print(f"  ✅ {task} env passed with {randomization} randomization")


if __name__ == "__main__":
    test_env(task="reach", randomization="none", steps=30)
    test_env(task="reach", randomization="medium", steps=30)
    test_env(task="grasp", randomization="none", steps=30)
    test_env(task="grasp", randomization="medium", steps=30)
    print(f"\n{'='*50}")
    print(f"  ALL SMOKE TESTS PASSED ✅")
    print(f"{'='*50}")
