"""
Reward Functions for SO-100 Sim-to-Real Training.

Hand-crafted reward functions for:
- Reach: Get end effector close to target
- Grasp: Reach, grasp, and lift objects
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward function weights."""
    # Weights for different reward components
    distance_weight: float = 1.0        # Reward for being close to object
    grasp_weight: float = 5.0           # Big reward for successful grasp
    lift_weight: float = 3.0            # Reward for lifting
    time_penalty: float = 0.01          # Penalty per timestep
    action_penalty: float = 0.001        # Penalty for large actions
    smoothness_penalty: float = 0.0001   # Penalty for jerky movements
    
    # Grasp detection thresholds
    grasp_threshold: float = 0.05       # Max distance to count as grasping
    lift_height_threshold: float = 0.1  # Min height to count as lifted
    
    # Shaping
    use_shaping_reward: bool = True     # Use potential-based shaping
    shaping_weight: float = 0.1         # Weight for potential function


def compute_reach_reward(
    state: Dict,
    prev_state: Dict,
    config: RewardConfig = RewardConfig()
) -> float:
    """Compute reward for reach task.
    
    Args:
        state: Current state dictionary
        prev_state: Previous state dictionary
        config: Reward configuration
        
    Returns:
        Computed reward
    """
    ee_pos = np.array(state['end_effector_pos'])
    obj_pos = np.array(state['object_position'])
    
    # Distance to object
    dist = np.linalg.norm(ee_pos - obj_pos)
    reward = -config.distance_weight * dist
    
    # Shaping: reward for reducing distance from previous step
    if config.use_shaping_reward:
        prev_dist = np.linalg.norm(
            np.array(prev_state['end_effector_pos']) - obj_pos
        )
        reward += config.shaping_weight * (prev_dist - dist)
    
    # Action penalty (encourage smooth movements)
    if 'action' in state:
        action = np.array(state['action'])
        reward -= config.action_penalty * np.sum(np.square(action))
    
    # Time penalty
    reward -= config.time_penalty
    
    return reward


def compute_grasp_reward(
    state: Dict,
    prev_state: Dict,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Compute reward for grasp task.
    
    Args:
        state: Current state dictionary
        prev_state: Previous state dictionary
        config: Reward configuration
        
    Returns:
        Computed reward
    """
    ee_pos = np.array(state['end_effector_pos'])
    obj_pos = np.array(state['object_position'])
    
    # Distance to object
    dist = np.linalg.norm(ee_pos - obj_pos)
    reward = -config.distance_weight * dist
    
    # Grasp detection
    if state.get('is_grasping', False):
        reward += config.grasp_weight
        
        # Lift reward
        obj_height = obj_pos[2]
        if obj_height > config.lift_height_threshold:
            reward += config.lift_weight
            
            # Additional reward for lifting higher
            lift_height = obj_height - config.lift_height_threshold
            reward += config.lift_weight * lift_height
    
    # Grasp progress shaping
    if config.use_shaping_reward:
        prev_dist = np.linalg.norm(
            np.array(prev_state['end_effector_pos']) - obj_pos
        )
        reward += config.shaping_weight * (prev_dist - dist)
        
        # Shaping for gripper opening/closing
        if 'gripper_state' in state and 'gripper_state' in prev_state:
            gripper_state = state['gripper_state']
            prev_gripper = prev_state['gripper_state']
            
            # Reward for closing gripper when close to object
            if dist < config.grasp_threshold * 3:
                target_gripper = 1.0  # Closed
                reward += config.shaping_weight * (
                    target_gripper - abs(gripper_state - target_gripper)
                )
            else:
                target_gripper = 0.0  # Open
                reward += config.shaping_weight * (
                    target_gripper - abs(gripper_state - target_gripper)
                )
    
    # Action penalty
    if 'action' in state:
        action = np.array(state['action'])
        reward -= config.action_penalty * np.sum(np.square(action))
    
    # Smoothness penalty
    if 'action' in state and 'prev_action' in state:
        action_diff = np.array(state['action']) - np.array(state['prev_action'])
        reward -= config.smoothness_penalty * np.sum(np.square(action_diff))
    
    # Time penalty
    reward -= config.time_penalty
    
    return reward


class AdaptiveReward:
    """Reward function that adapts difficulty based on performance.
    
    Makes the task harder as the agent gets better, preventing it from
    getting stuck on easy versions.
    """
    
    def __init__(self, initial_config: RewardConfig = None):
        self.config = initial_config or RewardConfig()
        self.success_rate = 0.5  # Initial estimate
        self.rolling_window = []
        self.window_size = 1000
    
    def update_success_rate(self, success: bool):
        """Update rolling success rate."""
        self.rolling_window.append(float(success))
        if len(self.rolling_window) > self.window_size:
            self.rolling_window.pop(0)
        
        self.success_rate = np.mean(self.rolling_window)
    
    def get_config(self) -> RewardConfig:
        """Get adapted reward config based on success rate."""
        config = RewardConfig()
        
        if self.success_rate > 0.8:
            # Agent is doing well, make it harder
            config.distance_weight *= 1.2
        elif self.success_rate < 0.2:
            # Agent is struggling, make it easier
            config.distance_weight *= 0.8
            config.shaping_weight *= 1.5
        
        return config
    
    def reset(self):
        """Reset success rate tracking."""
        self.rolling_window = []
        self.success_rate = 0.5
