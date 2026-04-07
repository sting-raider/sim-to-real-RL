import cv2
import numpy as np
from pathlib import Path


class CameraCalibrator:
    """Calibrates webcam to map pixel coordinates to real-world table coordinates.
    
    Uses a checkerboard or known reference points to compute the
    homography matrix between camera and table space.
    
    Usage:
        calibration = CameraCalibrator()
        calibration.capture_points()
        matrix = calibration.compute_calibration()
        calibration.save("calibration.npz")
    """
    
    def __init__(self, num_points: int = 4):
        self.num_points = num_points
        self.pixel_points = []
        self.world_points = []
        self.homography_matrix = None
    
    def capture_points(self, camera_index: int = 0):
        """Capture corresponding pixel-world point pairs interactively.
        
        Opens the camera feed and lets the user click points on screen.
        Then prompts for the real-world coordinates of each clicked point.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        print(f"Click {self.num_points} points on the table in the camera feed.")
        print("Press 'q' to quit, 's' to save a point, 'r' to reset all points.")
        
        pixel_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pixel_points.append((x, y))
                print(f"  Pixel point saved: ({x}, {y}) - {len(pixel_points)}/{self.num_points}")
        
        cv2.namedWindow("Calibration - Click Points")
        cv2.setMouseCallback("Calibration - Click Points", mouse_callback)
        
        while len(pixel_points) < self.num_points:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw clicked points
            for i, (px, py) in enumerate(pixel_points):
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (px + 10, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Calibration - Click Points", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Calibration cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                pixel_points.clear()
                print("Reset all points.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.pixel_points = pixel_points
        print(f"\nCaptured {len(self.pixel_points)} pixel points.")
        print("Now enter the real-world (x, y) coordinates in meters for each point:")
        
        for i in range(len(self.pixel_points)):
            while True:
                try:
                    x = float(input(f"Point {i+1} - X (meters): "))
                    y = float(input(f"Point {i+1} - Y (meters): "))
                    self.world_points.append([x, y])
                    break
                except ValueError:
                    print("Invalid number. Try again.")
        
        print("All points captured. Computing calibration...")
    
    def compute_calibration(self):
        """Compute the homography matrix from pixel to world coordinates."""
        if len(self.pixel_points) < 4:
            raise ValueError(f"Need at least 4 points, got {len(self.pixel_points)}")
        
        pixels = np.array(self.pixel_points, dtype=np.float32)
        worlds = np.array(self.world_points, dtype=np.float32)
        
        H, _ = cv2.findHomography(pixels, worlds)
        self.homography_matrix = H
        
        print("Homography matrix computed:")
        print(H)
        
        return H
    
    def pixel_to_world(self, px: float, py: float):
        """Convert pixel coordinates to world coordinates.
        
        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate
            
        Returns:
            (x, y) in world coordinates (meters)
        """
        if self.homography_matrix is None:
            raise RuntimeError("Calibration not computed yet. Call compute_calibration() first.")
        
        point = np.array([[px, py, 1]], dtype=np.float32).T
        result = self.homography_matrix @ point
        result /= result[2]
        
        return result[0][0], result[1][0]
    
    def save(self, path: str):
        """Save calibration matrix to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, homography=self.homography_matrix)
        print(f"Calibration saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load calibration matrix from file."""
        data = np.load(path)
        calib = cls()
        calib.homography_matrix = data['homography']
        return calib
    
    def test_calibration(self, camera_index: int = 0):
        """Test calibration by showing real-world coordinates on camera feed."""
        if self.homography_matrix is None:
            raise RuntimeError("Calibration not computed yet.")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        print("Testing calibration. Move the pointer around the table.")
        print("Press 'q' to quit.")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                world_x, world_y = self.pixel_to_world(x, y)
                print(f"  Pixel: ({x}, {y}) -> World: ({world_x:.3f}, {world_y:.3f})")
        
        cv2.namedWindow("Calibration Test")
        cv2.setMouseCallback("Calibration Test", mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Calibration Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Camera calibration for SO-100")
    parser.add_argument("--mode", choices=["calibrate", "test", "convert"],
                       default="calibrate", help="Operation mode")
    parser.add_argument("--checkpoint", type=str, default="vision/calibration.npz",
                       help="Path to save/load calibration")
    parser.add_argument("--px", type=float, help="Pixel x to convert")
    parser.add_argument("--py", type=float, help="Pixel y to convert")
    
    args = parser.parse_args()
    
    if args.mode == "calibrate":
        calib = CameraCalibrator()
        calib.capture_points()
        calib.compute_calibration()
        calib.save(args.checkpoint)
    elif args.mode == "test":
        calib = CameraCalibrator.load(args.checkpoint)
        calib.test_calibration()
    elif args.mode == "convert":
        if args.px is None or args.py is None:
            print("Provide --px and --py for conversion.")
            return
        calib = CameraCalibrator.load(args.checkpoint)
        world_x, world_y = calib.pixel_to_world(args.px, args.py)
        print(f"Pixel ({args.px}, {args.py}) -> World ({world_x:.3f}, {world_y:.3f}) m")


if __name__ == "__main__":
    main()
