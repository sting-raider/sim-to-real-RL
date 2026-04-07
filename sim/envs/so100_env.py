"""
SO-100 Simulation Environment — PyBullet Backend.

A 6-DOF robotic arm environment with real physics simulation using PyBullet.

Tasks:
  - "reach":  Move end effector to a random target position
  - "grasp":  Reach, close gripper, grasp object, and lift it to a height

Observation Space (21-dim):
  [0:6]   joint_positions
  [6:12]  joint_velocities
  [12:15] end_effector_position
  [15:18] object_position
  [18:21] object_to_ee vector

Action Space (6-dim):
  Continuous [-1, 1] per joint → mapped to joint position targets
  Note: Joint 6 is gripper open/close (special handling)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional

# ─── Kinematic constants for the SO-100 ───────────────────────────────
# The SO-100 is a LeRobot 6-DOF arm + 1-DOF parallel gripper
# Joint 0–5: arm joints,  Joint 6: gripper (binary open/close)

GRIPPER_FINGER_WIDTH = 0.042             # max gap in metres
GRASP_DIST_THRESHOLD  = 0.03             # gripper width for "grasping"

# Nominal joint limits (radians)
JOINT_DEFAULTS = np.array([
    0.0,        # J0  pan   -180°…+180°
    -0.52,      # J1  tilt  -120°…+60°
    -0.52,      # J2  tilt  -60°…+120°
    0.79,       # J3  tilt  -120°…+60°
    0.0,        # J4  tilt  -120°…+60°
    0.0,        # J5  pan   -180°…+180°
], dtype=np.float32)

JOINT_LIMITS_LOW  = np.deg2rad([-160, -120, -60,  -120, -120, -160])
JOINT_LIMITS_HIGH = np.deg2rad([ 160,   60,  120,    60,   60,  160])

# ─── URDF builder ─────────────────────────────────────────────────────
SO100_URDF_TEMPLATE = """<?xml version="1.0" ?>
<robot name="so100">
  <material name="black"><color rgba="0.15 0.15 0.18 1.0"/></material>
  <material name="silver"><color rgba="0.6 0.6 0.65 1.0"/></material>
  <material name="red"><color rgba="0.95 0.1 0.1 1.0"/></material>

  <!-- Base -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual><origin xyz="0 0 0.02" rpy="0 0 0"/>
      <geometry><box size="0.12 0.12 0.04"/></geometry>
      <material name="silver"/></visual>
    <collision><origin xyz="0 0 0.02" rpy="0 0 0"/>
      <geometry><box size="0.12 0.12 0.04"/></geometry></collision>
  </link>

  <!-- Link 1 — pan base -->
  <link name="l1">
    <inertial><mass value="0.8"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.003"/>
    </inertial>
    <visual><origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><cylinder radius="0.04" length="0.1"/></geometry>
      <material name="black"/></visual>
    <collision><origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><cylinder radius="0.04" length="0.1"/></geometry></collision>
  </link>
  <joint name="j0" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.02" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="5" velocity="3" lower="{j0_l}" upper="{j0_h}"/>
  </joint>

  <!-- Link 2 — tilt up -->
  <link name="l2">
    <inertial><mass value="0.6"/>
      <origin xyz="0 0 0.12" rpy="0 0 0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.002"/>
    </inertial>
    <visual><origin xyz="0 0 0.12" rpy="0 0 0"/>
      <geometry><box size="0.05 0.04 0.24"/></geometry>
      <material name="silver"/></visual>
    <collision><origin xyz="0 0 0.12" rpy="0 0 0"/>
      <geometry><box size="0.05 0.04 0.24"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="5" velocity="3" lower="{j1_l}" upper="{j1_h}"/>
  </joint>

  <!-- Link 3 — upper arm -->
  <link name="l3">
    <inertial><mass value="0.5"/>
      <origin xyz="0 0.08 0" rpy="0 0 0"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.002"/>
    </inertial>
    <visual><origin xyz="0 0.08 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.16 0.04"/></geometry>
      <material name="black"/></visual>
    <collision><origin xyz="0 0.08 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.16 0.04"/></geometry></collision>
  </link>
  <joint name="j2" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="5" velocity="3" lower="{j2_l}" upper="{j2_h}"/>
  </joint>

  <!-- Link 4 — forearm -->
  <link name="l4">
    <inertial><mass value="0.35"/>
      <origin xyz="0 0.06 0" rpy="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.001"/>
    </inertial>
    <visual><origin xyz="0 0.06 0" rpy="0 0 0"/>
      <geometry><box size="0.035 0.12 0.035"/></geometry>
      <material name="silver"/></visual>
    <collision><origin xyz="0 0.06 0" rpy="0 0 0"/>
      <geometry><box size="0.035 0.12 0.035"/></geometry></collision>
  </link>
  <joint name="j3" type="revolute">
    <parent link="l3"/><child link="l4"/>
    <origin xyz="0 0.16 0" rpy="1.5708 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="5" velocity="3" lower="{j3_l}" upper="{j3_h}"/>
  </joint>

  <!-- Link 5 — wrist pitch -->
  <link name="l5">
    <inertial><mass value="0.2"/>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0005"/>
    </inertial>
    <visual><origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><cylinder radius="0.025" length="0.08"/></geometry>
      <material name="black"/></visual>
    <collision><origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><cylinder radius="0.025" length="0.08"/></geometry></collision>
  </link>
  <joint name="j4" type="revolute">
    <parent link="l4"/><child link="l5"/>
    <origin xyz="0 0.12 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="5" velocity="3" lower="{j4_l}" upper="{j4_h}"/>
  </joint>

  <!-- Link 6 — wrist roll / EE mount -->
  <link name="l6">
    <inertial><mass value="0.15"/>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0003"/>
    </inertial>
    <visual><origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.05"/></geometry>
      <material name="red"/></visual>
    <collision><origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.05"/></geometry></collision>
  </link>
  <joint name="j5" type="revolute">
    <parent link="l5"/><child link="l6"/>
    <origin xyz="0 0 0.08" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="3" velocity="3" lower="{j5_l}" upper="{j5_h}"/>
  </joint>

  <!-- Grip base (dummy) -->
  <link name="gripper_base">
    <inertial><mass value="0.1"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>
  <joint name="grip_base" type="fixed">
    <parent link="l6"/><child link="gripper_base"/>
    <origin xyz="0 0 0.06" rpy="0 0 0"/>
  </joint>

  <!-- Left finger -->
  <link name="finger_l">
    <inertial><mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="5e-6"/>
    </inertial>
    <visual><origin xyz="0.012 0 0" rpy="0 0 0"/>
      <geometry><box size="0.024 0.06 0.03"/></geometry>
      <material name="black"/></visual>
    <collision><origin xyz="0.012 0 0" rpy="0 0 0"/>
      <geometry><box size="0.024 0.06 0.03"/></geometry></collision>
  </link>
  <joint name="finger_left" type="prismatic">
    <parent link="gripper_base"/><child link="finger_l"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit effort="2" velocity="0.5" lower="-0.021" upper="0.021"/>
  </joint>

  <!-- Right finger -->
  <link name="finger_r">
    <inertial><mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="5e-6"/>
    </inertial>
    <visual><origin xyz="-0.012 0 0" rpy="0 0 0"/>
      <geometry><box size="0.024 0.06 0.03"/></geometry>
      <material name="black"/></visual>
    <collision><origin xyz="-0.012 0 0" rpy="0 0 0"/>
      <geometry><box size="0.024 0.06 0.03"/></geometry></collision>
  </link>
  <joint name="finger_right" type="prismatic">
    <parent link="gripper_base"/><child link="finger_r"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit effort="2" velocity="0.5" lower="-0.021" upper="0.021"/>
  </joint>
</robot>
"""


def _build_urdf() -> str:
    """Write SO-100 URDF to a temp file and return path."""
    limits = list(JOINT_LIMITS_LOW) + list(JOINT_LIMITS_HIGH)
    template = SO100_URDF_TEMPLATE.format(
        j0_l=limits[0],  j0_h=limits[6],
        j1_l=limits[1],  j1_h=limits[7],
        j2_l=limits[2],  j2_h=limits[8],
        j3_l=limits[3],  j3_h=limits[9],
        j4_l=limits[4],  j4_h=limits[10],
        j5_l=limits[5],  j5_h=limits[11],
    )
    path = Path(tempfile.gettempdir()) / "so100_generated.urdf"
    path.write_text(template)
    return str(path)


# ─── Environment ──────────────────────────────────────────────────────
class SO100Env(gym.Env):
    """SO-100 Robot Arm physics environment with PyBullet."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        task: str = "reach",
        randomization_level: str = "none",
        max_steps: int = 200,
        render_mode: str = None,
        urdf_dir: str = None,
    ):
        super().__init__()
        assert task in ("reach", "grasp"), f"Unknown task: {task}"
        self.task = task
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        # Domain randomization
        from .domain_rand import DomainRandomizer, get_preset_config
        self.randomizer = DomainRandomizer(get_preset_config(randomization_level))
        self.randomization_state = None

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # URDF
        if urdf_dir:
            self.urdf_path = os.path.join(urdf_dir, "so100.urdf")
            if not os.path.isfile(self.urdf_path):
                self.urdf_path = _build_urdf()
        else:
            self.urdf_path = _build_urdf()

        # PyBullet handles
        self.client_id = None
        self.robot_id = None
        self.object_id = None
        self.table_id = None
        self.arm_joint_indices = []
        self.gripper_joint_map = {}

        # Episode tracking
        self.episode_reward = 0.0
        self.success = False
        self.prev_ee_pos = None
        self.prev_action = None

        # Render connection (shared among workers via pybullet.DIRECT)
        self._start_pybullet()
        self.episode_reward = 0.0

    # ── PyBullet lifecycle ────────────────────────────────────────────
    def _start_pybullet(self):
        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._load_scene()

    def _load_scene(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=10,
            physicsClientId=self.client_id,
        )

        # Floor
        plane = p.loadURDF("plane.urdf",
                           physicsClientId=self.client_id)

        # Table
        self.table_id = p.loadURDF(
            "table/table.urdf", [0.6, 0, -0.62], [0, 0, 0, 1],
            useFixedBase=True, physicsClientId=self.client_id,
        )

        # Table top (flat box on top) so objects don't fall through gaps
        self.table_top = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.4, 0.3, 0.01],
                physicsClientId=self.client_id,
            ),
            basePosition=[0.6, 0, 0.599],
            physicsClientId=self.client_id,
        )

        # Robot
        self.robot_id = p.loadURDF(
            self.urdf_path, [0, 0, 0], [0, 0, 0, 1],
            useFixedBase=True, physicsClientId=self.client_id,
        )

        # Identify joints
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        self.arm_joint_indices = []
        self.gripper_joint_map = {}
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            name = info[1].decode("utf-8")
            if name in ("j0", "j1", "j2", "j3", "j4", "j5"):
                self.arm_joint_indices.append(i)
            elif name in ("finger_left", "finger_right"):
                self.gripper_joint_map[name] = i

    def _load_object(self, position):
        """Spawn a graspable box."""
        # Use box shape
        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.025, 0.025, 0.025],
            physicsClientId=self.client_id,
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.025, 0.025, 0.025],
            rgbaColor=[1.0, 0.2, 0.1, 1.0],
            physicsClientId=self.client_id,
        )
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position,
            physicsClientId=self.client_id,
        )
        return obj_id

    def _reset_scene(self):
        self._load_scene()

    def _get_initial_state(self):
        """Sample a new episode state with object on table."""
        # Random object position on table surface
        obj_x = 0.45 + np.random.uniform(-0.15, 0.15)
        obj_y = np.random.uniform(-0.15, 0.15)
        obj_z = 0.63  # slightly above table

        return {
            "joint_positions": np.copy(JOINT_DEFAULTS),
            "joint_velocities": np.zeros(6, dtype=np.float32),
            "end_effector_pos": np.array([0.35, 0.0, 0.55], dtype=np.float32),
            "object_position": np.array([obj_x, obj_y, obj_z], dtype=np.float32),
            "gripper_width": GRIPPER_FINGER_WIDTH,
        }

    # ── Gym interface ─────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_scene()
        self._state = self._get_initial_state()
        self.current_step = 0
        self.episode_reward = 0.0
        self.success = False
        self.prev_ee_pos = self._state["end_effector_pos"].copy()
        self.prev_action = np.zeros(6, dtype=np.float32)

        # Reset robot to default joint positions
        for i, jid in enumerate(self.arm_joint_indices):
            p.resetJointState(
                self.robot_id, jid, JOINT_DEFAULTS[i],
                0, physicsClientId=self.client_id,
            )
        # Open gripper
        for name, jid in self.gripper_joint_map.items():
            target = 0.021 if name == "finger_left" else -0.021
            p.resetJointState(self.robot_id, jid, target, 0, physicsClientId=self.client_id)

        # Create object
        obj_pos = self._state["object_position"]
        self.object_id = self._load_object(obj_pos)

        # Step simulation a few steps to settle
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client_id)

        # Sample randomization
        self.randomization_state = self.randomizer.sample()
        self._apply_randomization()

        return self._get_observation(), {}

    def _apply_randomization(self):
        """Apply domain randomization to current scene."""
        state = self.randomization_state
        if state is None:
            return
        # Adjust object mass
        if self.object_id is not None:
            p.changeDynamics(
                self.object_id, -1,
                mass=state.object_mass,
                lateralFriction=1.0,
                physicsClientId=self.client_id,
            )
        # Adjust table friction
        if self.table_id is not None:
            p.changeDynamics(
                self.table_id, -1,
                lateralFriction=state.table_friction,
                physicsClientId=self.client_id,
            )

    def step(self, action):
        self.current_step += 1

        # Add noise
        noisy_action = self.randomizer.add_noise_to_actions(
            action, self.randomization_state
        )

        # Map [-1,1] to joint angles and execute
        self._execute_action(noisy_action)

        # Step physics (4 sub-steps per env step for 1/60s real-time)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client_id)

        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps

        self.episode_reward += reward

        info = {
            "step": self.current_step,
            "reward": float(reward),
            "episode_reward": float(self.episode_reward),
            "success": bool(self.success),
        }

        noisy_obs = self.randomizer.add_noise_to_observations(
            obs, self.randomization_state
        )
        return noisy_obs, reward, terminated, truncated, info

    def _execute_action(self, action):
        """Convert normalised action [-1, 1]^6 to joint targets and send to sim."""
        # Scale arm joints
        for i, joint_idx in enumerate(self.arm_joint_indices):
            low = JOINT_LIMITS_LOW[i]
            high = JOINT_LIMITS_HIGH[i]
            target = float(low + 0.5 * (action[i] + 1.0) * (high - low))
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                positionGain=0.5,
                velocityGain=1.0,
                physicsClientId=self.client_id,
            )

        # Gripper: action[5] in [-1, 1]
        # -1 → fully open, +1 → fully closed
        gripper_val = action[5]
        left_target = 0.021 * (1.0 - gripper_val)     # open=0.021, closed=0
        right_target = -0.021 * (1.0 - gripper_val)

        for name, jid in self.gripper_joint_map.items():
            target = left_target if name == "finger_left" else right_target
            p.setJointMotorControl2(
                self.robot_id, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=10.0,
                physicsClientId=self.client_id,
            )

    def _get_observation(self):
        """Assemble the 21-D observation vector."""
        # Joint state
        raw = []
        for joint_idx in self.arm_joint_indices:
            pos, vel, _, _ = p.getJointState(
                self.robot_id, joint_idx, physicsClientId=self.client_id
            )
            raw.append((pos, vel))

        joint_positions = np.array([r[0] for r in raw], dtype=np.float32)
        joint_velocities = np.array([r[1] for r in raw], dtype=np.float32)

        # End effector (link 6 = the last physical link, index = num_joints - 1)
        num_j = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        ee_link = num_j - 1  # l6
        ee_state = p.getLinkState(
            self.robot_id, ee_link, physicsClientId=self.client_id
        )
        ee_pos = np.array(ee_state[0], dtype=np.float32)

        # Object
        obj_pos, obj_orn = p.getBasePositionAndOrientation(
            self.object_id, physicsClientId=self.client_id
        )
        obj_pos = np.array(obj_pos, dtype=np.float32)

        # Relative vector: object - ee
        obj_to_ee = obj_pos - ee_pos

        obs = np.concatenate([
            joint_positions,      # [6]
            joint_velocities,     # [6]
            ee_pos,               # [3]
            obj_pos,              # [3]
            obj_to_ee,            # [3]
        ]).astype(np.float32)

        self.prev_ee_pos = ee_pos.copy()
        self._state = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_pos": ee_pos,
            "object_position": obj_pos,
        }

        return obs

    def _compute_reward(self) -> float:
        """Task-specific reward computation."""
        ee_pos = self._state["end_effector_pos"]
        obj_pos = self._state["object_position"]
        dist = np.linalg.norm(ee_pos - obj_pos)

        reward = -1.0 * dist  # distance reward

        if self.task == "grasp":
            obj_h = obj_pos[2]
            if obj_h > 0.68:  # lifted above table
                reward += 3.0 + 5.0 * max(0, obj_h - 0.68)
            # Check gripper proximity bonus
            if dist < 0.05:
                reward += 1.0

        # Time penalty
        reward -= 0.01

        return float(reward)

    def _is_done(self) -> bool:
        if self.task == "grasp":
            obj_pos = self._state["object_position"]
            if obj_pos[2] > 0.72:
                self.success = True
                return True
            # Fail: object fell off table
            if obj_pos[2] < 0.5:
                self.success = False
                return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            w, h, rgb, depth, seg = p.getCameraImage(
                640, 480, renderer=p.ER_TINY_RENDERER,
                physicsClientId=self.client_id,
            )
            return np.array(rgb, dtype=np.uint8).reshape(h, w, 4)

    def close(self):
        p.disconnect(self.client_id)

    def get_randomization_info(self) -> dict:
        if self.randomization_state is None:
            return {}
        return {
            "object_mass": self.randomization_state.object_mass,
            "table_friction": self.randomization_state.table_friction,
            "joint_damping": self.randomization_state.joint_damping,
            "lighting_intensity": self.randomization_state.lighting_intensity,
            "camera_noise": self.randomization_state.camera_position_noise,
        }


# ── Gym registration ──────────────────────────────────────────────────
from gymnasium.envs.registration import register

register(
    id="SO100Reach-v0",
    entry_point="envs.so100_env:SO100Env",
    max_episode_steps=200,
    kwargs={"task": "reach"},
)

register(
    id="SO100Grasp-v0",
    entry_point="envs.so100_env:SO100Env",
    max_episode_steps=200,
    kwargs={"task": "grasp"},
)
