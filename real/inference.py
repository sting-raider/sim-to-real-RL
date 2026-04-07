"""
Real Robot Inference Pipeline.

Loads a trained policy and runs it on the real SO-100 arm
with webcam-based object detection.

Usage:
    python inference.py --model models/grasp_medium_20240101/final_model.zip
    python inference.py --model models/grasp_medium_20240101/final_model.zip --attempts 20
"""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from stable_baselines3 import SAC, PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("stable-baselines3 not installed. Run: pip install stable-baselines3")

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("ultralytics not installed. Run: pip install ultralytics")


class RealSO100Env:
    """Wrapper for the real SO-100 arm that mimics the sim environment interface.
    
    This allows the same policy to run on both sim and real hardware.
    """
    
    def __init__(self, arm_port: str = "/dev/ttyUSB0", camera_index: int = 0,
                 calibration_path: str = None):
        self.arm_port = arm_port
        self.camera_index = camera_index
        
        # Initialize arm
        try:
            from so100_sdk import SO100Arm
            self.arm = SO100Arm(port=arm_port)
            self.arm.enable_torque()
            self.arm.go_home()
            print(f"Connected to SO-100 on {arm_port}")
        except ImportError:
            print("so100_sdk not installed. Using mock arm.")
            self.arm = None
            self._mock_joint_positions = np.zeros(6, dtype=np.float32)
            self._mock_ee_pos = np.array([0.3, 0.0, 0.5], dtype=np.float32)
        
        # Initialize camera
        self.cam = cv2.VideoCapture(camera_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        # Initialize object detection
        if HAS_YOLO:
            self.detector = YOLO("yolov8n.pt").to("cuda" if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else "cpu")
            print(f"YOLOv8 loaded on {self.detector.device}")
        
        # Load camera calibration
        self.calibration_path = calibration_path
        self.calibration = None
        if calibration_path and Path(calibration_path).exists():
            data = np.load(calibration_path)
            self.calibration = data['homography']
            print(f"Loaded calibration from {calibration_path}")
        
        # State tracking
        self.current_step = 0
        self.max_steps = 200
    
    def get_object_position(self) -> np.ndarray:
        """Detect object position using camera and YOLO.
        
        Returns:
            3D position array [x, y, z] in world coordinates
        """
        if not self.cam.isOpened():
            return np.array([0.3, 0.0, 0.1], dtype=np.float32)
        
        ret, frame = self.cam.read()
        if not ret:
            print("Failed to read camera frame")
            return np.array([0.3, 0.0, 0.1], dtype=np.float32)
        
        if not HAS_YOLO:
            # Mock detection
            return np.array([0.3, 0.0, 0.1], dtype=np.float32)
        
        results = self.detector(frame, verbose=False)[0]
        
        if len(results.boxes) == 0:
            print("No object detected")
            return None
        
        # Get the first detected box
        box = results.boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Convert to world coordinates if calibration available
        if self.calibration is not None:
            px = np.array([[cx, cy, 1]], dtype=np.float32).T
            world = self.calibration @ px
            world /= world[2]
            world_x, world_y = world[0][0], world[1][0]
        else:
            # Normalize pixel coordinates as fallback
            h, w = frame.shape[:2]
            world_x = (cx / w) * 0.6 - 0.3  # Map to [-0.3, 0.3]
            world_y = (cy / h) * 0.4 - 0.2  # Map to [-0.2, 0.2]
        
        # Assume object is on table surface for z
        return np.array([world_x, world_y, 0.1], dtype=np.float32)
    
    def reset(self):
        """Reset arm to home position."""
        if self.arm:
            self.arm.go_home()
        else:
            self._mock_ee_pos = np.array([0.3, 0.0, 0.5], dtype=np.float32)
        
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action: np.ndarray):
        """Execute action on the real arm.
        
        Args:
            action: Target joint positions
        """
        self.current_step += 1
        
        if self.arm:
            self.arm.set_joint_targets(action)
            self.arm.step()
        else:
            # Mock action
            self._mock_ee_pos += action * 0.01
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector matching sim environment."""
        # Joint positions
        if self.arm:
            joint_pos = np.array(self.arm.get_joint_positions(), dtype=np.float32)
            joint_vel = np.array(self.arm.get_joint_velocities(), dtype=np.float32)
            ee_pos = np.array(self.arm.get_end_effector_position(), dtype=np.float32)
        else:
            joint_pos = self._mock_joint_positions
            joint_vel = np.zeros(6, dtype=np.float32)
            ee_pos = self._mock_ee_pos
        
        # Object position
        obj_pos = self.get_object_position()
        if obj_pos is None:
            # Use last known position or default
            obj_pos = np.array([0.3, 0.0, 0.1], dtype=np.float32)
        
        obs = np.concatenate([
            joint_pos,        # [6]
            joint_vel,        # [6]
            ee_pos,           # [3]
            obj_pos,          # [3]
            obj_pos - ee_pos  # [3]
        ])
        
        return obs.astype(np.float32)
    
    def is_grasping(self) -> bool:
        """Check if arm is currently grasping an object."""
        if self.arm:
            return self.arm.gripper_force() > 0.5
        return False
    
    def close(self):
        """Clean up resources."""
        if self.arm:
            self.arm.disable_torque()
        self.cam.release()


def run_policy(model_path: str, attempts: int = 20, max_steps: int = 200,
               arm_port: str = "/dev/ttyUSB0", camera_index: int = 0,
               calibration_path: str = None):
    """Run trained policy on real arm.
    
    Args:
        model_path: Path to saved policy
        attempts: Number of grasp attempts
        max_steps: Maximum steps per attempt
        arm_port: Serial port for arm
        camera_index: Camera device index
        calibration_path: Path to camera calibration file
    """
    # Load policy
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)
    
    print(f"\n{'='*50}")
    print(f"Real Robot Inference")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Attempts: {attempts}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*50}\n")
    
    # Initialize real environment
    env = RealSO100Env(
        arm_port=arm_port,
        camera_index=camera_index,
        calibration_path=calibration_path
    )
    
    # Results tracking
    success_count = 0
    results = []
    
    for attempt in range(attempts):
        print(f"\n--- Attempt {attempt + 1}/{attempts} ---")
        
        obs = env.reset()
        
        for step in range(max_steps):
            # Policy inference
            action, _ = model.predict(obs, deterministic=True)
            obs = env.step(action)
            
            # Check for grasp
            if env.is_grasping():
                # Check if object is lifted
                ee_height = obs[14]  # End effector z position
                if ee_height > 0.15:
                    print(f"  SUCCESS at step {step+1}")
                    success_count += 1
                    results.append({
                        'attempt': attempt + 1,
                        'success': True,
                        'steps': step + 1,
                    })
                    break
        else:
            print(f"  FAILED after {max_steps} steps")
            results.append({
                'attempt': attempt + 1,
                'success': False,
                'steps': max_steps,
            })
    
    # Summary
    success_rate = success_count / attempts * 100
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*50}")
    print(f"Success Rate: {success_rate:.0f}% ({success_count}/{attempts})")
    print(f"{'='*50}")
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run policy on real SO-100 arm")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to saved model file")
    parser.add_argument("--attempts", type=int, default=20,
                       help="Number of grasp attempts")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per attempt")
    parser.add_argument("--arm-port", type=str, default="/dev/ttyUSB0",
                       help="Serial port for arm connection")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Path to camera calibration file")
    
    args = parser.parse_args()
    
    run_policy(
        args.model,
        attempts=args.attempts,
        max_steps=args.max_steps,
        arm_port=args.arm_port,
        camera_index=args.camera,
        calibration_path=args.calibration,
    )


if __name__ == "__main__":
    main()
