"""
Domain Randomization Module for SO-100 Sim-to-Real Training.

Randomizes physics parameters, visual properties, and sensor noise
during simulation to produce policies robust to real-world variation.

Each episode samples new values from configured ranges. The agent
learns to handle ALL of these, becoming robust to the sim-to-real gap.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization ranges.
    
    All ranges are (min, max) tuples for uniform sampling.
    """
    # Physics randomization
    object_mass: Tuple[float, float] = (0.05, 0.3)          # kg
    table_friction: Tuple[float, float] = (0.3, 1.2)         # coefficient
    table_restitution: Tuple[float, float] = (0.0, 0.3)      # coefficient
    joint_damping: Tuple[float, float] = (0.8, 1.2)          # multiplier on nominal
    joint_friction: Tuple[float, float] = (0.8, 1.2)         # multiplier on nominal
    arm_mass: Tuple[float, float] = (0.9, 1.1)               # multiplier on nominal
    
    # Visual randomization
    lighting_intensity: Tuple[float, float] = (0.5, 1.5)     # multiplier
    lighting_position_noise: Tuple[float, float] = (-0.1, 0.1)  # meters
    object_color: bool = True                                # random RGB
    table_color: bool = False                                # random RGB
    background_color: bool = False                           # random RGB
    
    # Sensor randomization
    camera_position_noise: Tuple[float, float] = (-0.02, 0.02)  # meters
    camera_angle_noise: Tuple[float, float] = (-0.05, 0.05)     # radians
    joint_position_noise: float = 0.001                       # std in meters
    joint_velocity_noise: float = 0.01                        # std in m/s
    
    # Action randomization
    action_delay_prob: float = 0.0                           # probability of action delay
    action_noise_std: float = 0.0                            # std in action space


@dataclass
class RandomizationState:
    """Holds the current episode's randomization values."""
    # Physics
    object_mass: float = 0.1
    table_friction: float = 0.8
    table_restitution: float = 0.1
    joint_damping: float = 1.0
    joint_friction: float = 1.0
    arm_mass: float = 1.0
    
    # Visual
    lighting_intensity: float = 1.0
    lighting_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    table_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    
    # Sensor
    camera_position_noise: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_angle_noise: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Action
    action_noise: np.ndarray = field(default_factory=lambda: np.zeros(6))


class DomainRandomizer:
    """Applies domain randomization to the simulation environment.
    
    Usage:
        randomizer = DomainRandomizer(config)
        state = randomizer.sample()  # Call at start of each episode
        randomizer.apply_to_env(env, state)  # Apply to simulation
    """
    
    def __init__(self, config: Optional[RandomizationConfig] = None):
        self.config = config or RandomizationConfig()
    
    def sample(self) -> RandomizationState:
        """Sample new randomization values for a new episode."""
        state = RandomizationState()
        
        # Physics
        state.object_mass = np.random.uniform(*self.config.object_mass)
        state.table_friction = np.random.uniform(*self.config.table_friction)
        state.table_restitution = np.random.uniform(*self.config.table_restitution)
        state.joint_damping = np.random.uniform(*self.config.joint_damping)
        state.joint_friction = np.random.uniform(*self.config.joint_friction)
        state.arm_mass = np.random.uniform(*self.config.arm_mass)
        
        # Visual
        state.lighting_intensity = np.random.uniform(*self.config.lighting_intensity)
        state.lighting_position = tuple(np.random.uniform(
            *self.config.lighting_position_noise, size=3
        ))
        
        if self.config.object_color:
            state.object_color = tuple(np.random.uniform(0, 1, size=3))
        if self.config.table_color:
            state.table_color = tuple(np.random.uniform(0, 1, size=3))
        if self.config.background_color:
            state.background_color = tuple(np.random.uniform(0, 1, size=3))
        
        # Sensor
        state.camera_position_noise = tuple(np.random.uniform(
            *self.config.camera_position_noise, size=3
        ))
        state.camera_angle_noise = tuple(np.random.uniform(
            *self.config.camera_angle_noise, size=3
        ))
        
        # Action noise
        if self.config.action_noise_std > 0:
            state.action_noise = np.random.normal(
                0, self.config.action_noise_std, size=6
            )
        
        return state
    
    def apply_to_env(self, env, state: RandomizationState):
        """Apply randomization state to the simulation environment.
        
        This is environment-specific and should be adapted to your
        simulator (IsaacGym, Genesis, etc.).
        
        Args:
            env: The simulation environment instance
            state: The randomization state to apply
        """
        self._apply_physics(env, state)
        self._apply_visuals(env, state)
        self._apply_sensor_noise(env, state)
    
    def _apply_physics(self, env, state: RandomizationState):
        """Apply physics randomization."""
        # These are placeholders - actual implementation depends on simulator
        pass
    
    def _apply_visuals(self, env, state: RandomizationState):
        """Apply visual randomization."""
        # These are placeholders - actual implementation depends on simulator
        pass
    
    def _apply_sensor_noise(self, env, state: RandomizationState):
        """Apply sensor randomization."""
        # These are placeholders - actual implementation depends on simulator
        pass
    
    def add_noise_to_observations(self, obs: np.ndarray, state: RandomizationState) -> np.ndarray:
        """Add sensor noise to observation vector.
        
        Args:
            obs: Clean observation vector
            state: Current randomization state
            
        Returns:
            Noisy observation vector
        """
        noisy_obs = obs.copy()
        
        # First 6 values: joint positions
        noisy_obs[:6] += np.random.normal(0, self.config.joint_position_noise, 6)
        
        # Next 6: joint velocities
        noisy_obs[6:12] += np.random.normal(0, self.config.joint_velocity_noise, 6)
        
        # Positions (end effector, object, relative) get camera noise
        pos_noise = np.random.uniform(
            *self.config.camera_position_noise, size=9
        )
        noisy_obs[12:21] += pos_noise
        
        return noisy_obs
    
    def add_noise_to_actions(self, action: np.ndarray, state: RandomizationState) -> np.ndarray:
        """Add action noise and delay for robustness.
        
        Args:
            action: Clean action vector
            state: Current randomization state
            
        Returns:
            Perturbed action vector
        """
        noisy_action = action.copy()
        
        # Add action noise
        noisy_action += state.action_noise
        
        # Random action delay (drop action probabilistically)
        if self.config.action_delay_prob > 0:
            if np.random.random() < self.config.action_delay_prob:
                noisy_action = np.zeros_like(noisy_action)
        
        # Clip to valid range
        noisy_action = np.clip(noisy_action, -1.0, 1.0)
        
        return noisy_action
    
    def get_config_dict(self) -> Dict:
        """Return config as dictionary for logging."""
        return {
            'object_mass': self.config.object_mass,
            'table_friction': self.config.table_friction,
            'table_restitution': self.config.table_restitution,
            'joint_damping': self.config.joint_damping,
            'joint_friction': self.config.joint_friction,
            'arm_mass': self.config.arm_mass,
            'lighting_intensity': self.config.lighting_intensity,
            'camera_position_noise': self.config.camera_position_noise,
            'joint_position_noise': self.config.joint_position_noise,
            'joint_velocity_noise': self.config.joint_velocity_noise,
            'action_noise_std': self.config.action_noise_std,
            'action_delay_prob': self.config.action_delay_prob,
        }


# Preset configurations for different randomization levels
def get_preset_config(level: str = "medium") -> RandomizationConfig:
    """Get preset randomization configuration by level.
    
    Args:
        level: One of 'none', 'low', 'medium', 'high', 'extreme'
        
    Returns:
        RandomizationConfig instance
        
    Raises:
        ValueError: If level is not recognized
    """
    presets = {
        "none": RandomizationConfig(
            object_mass=(0.1, 0.1),
            table_friction=(0.8, 0.8),
            table_restitution=(0.1, 0.1),
            joint_damping=(1.0, 1.0),
            joint_friction=(1.0, 1.0),
            arm_mass=(1.0, 1.0),
            lighting_intensity=(1.0, 1.0),
            lighting_position_noise=(0.0, 0.0),
            object_color=False,
            table_color=False,
            background_color=False,
            camera_position_noise=(0.0, 0.0),
            camera_angle_noise=(0.0, 0.0),
            joint_position_noise=0.0,
            joint_velocity_noise=0.0,
            action_delay_prob=0.0,
            action_noise_std=0.0,
        ),
        "low": RandomizationConfig(
            object_mass=(0.08, 0.12),
            table_friction=(0.7, 0.9),
            joint_damping=(0.95, 1.05),
            joint_friction=(0.95, 1.05),
            arm_mass=(0.98, 1.02),
            lighting_intensity=(0.9, 1.1),
            lighting_position_noise=(-0.02, 0.02),
            object_color=True,
            camera_position_noise=(-0.005, 0.005),
            camera_angle_noise=(-0.01, 0.01),
            joint_position_noise=0.0005,
            joint_velocity_noise=0.005,
            action_noise_std=0.01,
            action_delay_prob=0.01,
        ),
        "medium": RandomizationConfig(
            object_mass=(0.05, 0.3),
            table_friction=(0.3, 1.2),
            joint_damping=(0.8, 1.2),
            joint_friction=(0.8, 1.2),
            arm_mass=(0.9, 1.1),
            lighting_intensity=(0.5, 1.5),
            lighting_position_noise=(-0.05, 0.05),
            object_color=True,
            camera_position_noise=(-0.02, 0.02),
            camera_angle_noise=(-0.05, 0.05),
            joint_position_noise=0.001,
            joint_velocity_noise=0.01,
            action_noise_std=0.02,
            action_delay_prob=0.02,
        ),
        "high": RandomizationConfig(
            object_mass=(0.02, 0.5),
            table_friction=(0.1, 2.0),
            joint_damping=(0.5, 1.5),
            joint_friction=(0.5, 1.5),
            arm_mass=(0.8, 1.2),
            lighting_intensity=(0.3, 2.0),
            lighting_position_noise=(-0.1, 0.1),
            object_color=True,
            table_color=True,
            background_color=True,
            camera_position_noise=(-0.05, 0.05),
            camera_angle_noise=(-0.1, 0.1),
            joint_position_noise=0.002,
            joint_velocity_noise=0.02,
            action_noise_std=0.05,
            action_delay_prob=0.05,
        ),
        "extreme": RandomizationConfig(
            object_mass=(0.01, 1.0),
            table_friction=(0.0, 3.0),
            joint_damping=(0.3, 2.0),
            joint_friction=(0.3, 2.0),
            arm_mass=(0.5, 1.5),
            lighting_intensity=(0.1, 3.0),
            lighting_position_noise=(-0.2, 0.2),
            object_color=True,
            table_color=True,
            background_color=True,
            camera_position_noise=(-0.1, 0.1),
            camera_angle_noise=(-0.2, 0.2),
            joint_position_noise=0.005,
            joint_velocity_noise=0.05,
            action_noise_std=0.1,
            action_delay_prob=0.1,
        ),
    }
    
    if level not in presets:
        raise ValueError(f"Unknown randomization level: {level}. "
                        f"Choose from: {list(presets.keys())}")
    
    return presets[level]
