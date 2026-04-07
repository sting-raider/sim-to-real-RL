"""
SO-100 Simulation Environment.

Defines the OpenAI Gym environment for training a 6-DOF SO-100
robot arm to reach and grasp objects. The environment supports:
- Reach task: Move end effector to target position
- Grasp task: Reach, grasp, and lift objects
- Domain randomization integration
- Parallel environment execution
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from .domain_rand import DomainRandomizer, RandomizationConfig, get_preset_config


class SO100Env(gym.Env):
    """SO-100 Robot Arm simulation environment.
    
    Observation Space (21-dimensional):
        - joint_positions: [6] current joint angles
        - joint_velocities: [6] current joint velocities
        - end_effector_pos: [3] x, y, z of gripper tip
        - object_position: [3] x, y, z of target object
        - object_to_gripper: [3] relative vector from gripper to object
    
    Action Space (6-dimensional):
        - Continuous values in [-1, 1] representing target joint angles
        - Maps to actual servo angles via linear scaling
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        task: str = "reach",
        randomization_level: str = "none",
        randomization_config: Optional[RandomizationConfig] = None,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.task = task
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        
        # Initialize domain randomizer
        if randomization_config:
            self.randomizer = DomainRandomizer(randomization_config)
        else:
            self.randomizer = DomainRandomizer(get_preset_config(randomization_level))
        
        # Action space: 6 continuous values [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation space: 21-dimensional vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )
        
        # Simulation state placeholders
        self.sim = None
        self.arm = None
        self.object = None
        self.gripper = None
        self.randomization_state = None
        
        # Episode tracking
        self.episode_reward = 0.0
        self.success = False
        
        # Initialize simulation
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """Initialize the physics simulation.
        
        This is where you connect to IsaacGym, Genesis, or other simulators.
        For development, this placeholder allows testing the environment logic.
        """
        # TODO: Replace with actual simulator initialization
        # Example for IsaacGym:
        # self.gym = gymapi.acquire_gym()
        # self.sim = self.gym.create_sim(...)
        # self._load_robot()
        # self._load_objects()
        
        # Placeholder for development
        self._state = self._get_initial_state()
    
    def _get_initial_state(self) -> Dict:
        """Get initial state for a new episode."""
        return {
            'joint_positions': np.zeros(6, dtype=np.float32),
            'joint_velocities': np.zeros(6, dtype=np.float32),
            'end_effector_pos': np.array([0.3, 0.0, 0.5], dtype=np.float32),
            'object_position': np.array([0.3, 0.0, 0.1], dtype=np.float32),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        # Sample new randomization values for this episode
        self.randomization_state = self.randomizer.sample()
        
        # Reset simulation state
        self._state = self._get_initial_state()
        self.current_step = 0
        self.episode_reward = 0.0
        self.success = False
        
        # Apply randomization to simulation
        if self.sim is not None:
            self.randomizer.apply_to_env(self.sim, self.randomization_state)
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.
        
        Args:
            action: Target joint positions in [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Apply action noise from domain randomization
        noisy_action = self.randomizer.add_noise_to_actions(
            action, self.randomization_state
        )
        
        # Execute action in simulation
        self._execute_action(noisy_action)
        
        # Get new state
        obs = self._get_observation()
        
        # Compute reward based on task
        reward = self._compute_reward()
        
        # Check termination
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps
        
        # Store episode info
        self.episode_reward += reward
        
        info = {
            'step': self.current_step,
            'reward': reward,
            'episode_reward': self.episode_reward,
            'success': self.success,
        }
        
        # Add noisy observation if randomization is enabled
        if self.randomization_state is not None:
            noisy_obs = self.randomizer.add_noise_to_observations(
                obs, self.randomization_state
            )
            return noisy_obs, reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: np.ndarray):
        """Execute action in the simulation.
        
        Args:
            action: Noisy action vector
        """
        # TODO: Implement actual simulator call
        # Example: self.gym.set_dof_position_target(self.sim, self.arm, action)
        pass
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector.
        
        Returns:
            21-dimensional observation vector
        """
        # TODO: Get actual values from simulator
        state = self._state
        
        obs = np.concatenate([
            state['joint_positions'],      # [6]
            state['joint_velocities'],     # [6]
            state['end_effector_pos'],     # [3]
            state['object_position'],      # [3]
            state['object_position'] - state['end_effector_pos'],  # [3]
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self) -> float:
        """Compute reward for current state.
        
        Returns:
            Reward value
        """
        ee_pos = self._state['end_effector_pos']
        obj_pos = self._state['object_position']
        
        # Distance to object
        dist_to_object = np.linalg.norm(ee_pos - obj_pos)
        reward = -dist_to_object
        
        if self.task == "grasp":
            # Grasping reward components
            grasp_reward = self._compute_grasp_reward()
            reward += grasp_reward
            
            # Lift reward
            lift_reward = self._compute_lift_reward()
            reward += lift_reward
        
        # Time penalty to encourage speed
        reward -= 0.01
        
        return reward
    
    def _compute_grasp_reward(self) -> float:
        """Compute reward for grasping the object."""
        # TODO: Implement actual grasp detection from simulator
        is_grasping = False
        return 5.0 if is_grasping else 0.0
    
    def _compute_lift_reward(self) -> float:
        """Compute reward for lifting the object."""
        # TODO: Implement actual lift detection from simulator
        object_lifted = False
        return 3.0 if object_lifted else 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        if self.task == "grasp":
            # Done if object is lifted above threshold
            ee_height = self._state['end_effector_pos'][2]
            if ee_height > 0.15:  # Arbitrary threshold
                self.success = True
                return True
        
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            # TODO: Implement visualization
            pass
        elif self.render_mode == 'rgb_array':
            # TODO: Return RGB array
            pass
    
    def close(self):
        """Clean up resources."""
        # TODO: Clean up simulator
        pass
    
    def get_randomization_info(self) -> Dict:
        """Get current randomization state for logging."""
        if self.randomization_state is None:
            return {}
        
        return {
            'object_mass': self.randomization_state.object_mass,
            'table_friction': self.randomization_state.table_friction,
            'joint_damping': self.randomization_state.joint_damping,
            'lighting_intensity': self.randomization_state.lighting_intensity,
            'camera_noise': self.randomization_state.camera_position_noise,
        }
