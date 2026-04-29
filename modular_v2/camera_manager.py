"""
Camera management for Added-Mass-Lab GUI
Handles camera detection, validation, and configuration
"""

import cv2
from .config import DEFAULT_CAMERA_BACKEND, CAMERA_SCAN_RANGE


class CameraManager:
    """Handles camera detection, validation, and configuration"""
    
    def __init__(self):
        self.available_cameras = []
        self.selected_camera_id = None
        self.selected_resolution = None
        self.selected_framerate = None
        self.current_cap = None
        
    def detect_cameras(self, max_cameras=CAMERA_SCAN_RANGE):
        """Detect available cameras on the system"""
        available = []
        print("Detecting cameras...")
        
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, DEFAULT_CAMERA_BACKEND)
                if cap.isOpened():
                    # Try to read a frame to validate camera
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera name/info if possible
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_info = {
                            'id': i,
                            'name': f"Camera {i}",
                            'default_resolution': (width, height),
                            'default_fps': fps,
                            'validated': True
                        }
                        available.append(camera_info)
                        print(f"Found valid camera {i}: {width}x{height} @ {fps}fps")
                    cap.release()
                else:
                    print(f"Camera {i}: Not available")
            except Exception as e:
                print(f"Camera {i}: Error during detection - {e}")
                try:
                    cap.release()
                except:
                    pass
                
        self.available_cameras = available
        return available
    
    def configure_camera(self, camera_id):
        """Configure camera with optimal settings"""
        try:
            if self.current_cap:
                self.current_cap.release()
                
            # Try DirectShow backend first
            cap = cv2.VideoCapture(camera_id, DEFAULT_CAMERA_BACKEND)
            
            if not cap.isOpened():
                print(f"Failed to open camera {camera_id} with DirectShow")
                return None
                
            # Set MJPEG codec first
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Try resolutions in order of preference
            resolutions = [(1920, 1080), (1280, 720), (640, 480)]
            target_fps = 30
            
            configured_resolution = None
            
            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                
                # Verify the settings took effect
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Test if we can actually get frames at this resolution
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[1] >= width * 0.8:  # Allow some tolerance
                    configured_resolution = (actual_width, actual_height)
                    print(f"Camera {camera_id} configured: {actual_width}x{actual_height} @ {actual_fps}fps")
                    break
                    
            # Set MJPEG codec again to ensure it's applied
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            if configured_resolution:
                self.current_cap = cap
                self.selected_camera_id = camera_id
                self.selected_resolution = configured_resolution
                self.selected_framerate = actual_fps
                print(f"Camera manager state updated: ID={camera_id}, Resolution={configured_resolution}, FPS={actual_fps}")
                return cap
            else:
                cap.release()
                print(f"Failed to configure camera {camera_id}")
                return None
                
        except Exception as e:
            print(f"Error configuring camera {camera_id}: {e}")
            if 'cap' in locals():
                try:
                    cap.release()
                except:
                    pass
            return None

    def configure_camera_resolution(self, camera_id, target_resolution):
        """Configure camera to a specific resolution"""
        try:
            if self.current_cap:
                self.current_cap.release()
                
            # Try DirectShow backend first
            cap = cv2.VideoCapture(camera_id, DEFAULT_CAMERA_BACKEND)
            
            if not cap.isOpened():
                print(f"Failed to open camera {camera_id} with DirectShow")
                return None
                
            # Set MJPEG codec first
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            target_width, target_height = target_resolution
            target_fps = 30
            
            # Try to set the target resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            
            # Verify the settings took effect
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Test if we can actually get frames at this resolution
            ret, frame = cap.read()
            if ret and frame is not None:
                # Check if we got close to the target resolution (allow some tolerance)
                if (abs(actual_width - target_width) <= 10 and 
                    abs(actual_height - target_height) <= 10):
                    
                    # Set MJPEG codec again to ensure it's applied
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    
                    self.current_cap = cap
                    self.selected_camera_id = camera_id
                    self.selected_resolution = (actual_width, actual_height)
                    self.selected_framerate = actual_fps
                    print(f"Camera {camera_id} reconfigured to: {actual_width}x{actual_height} @ {actual_fps}fps")
                    return cap
            
            # If we couldn't set the target resolution, fall back to default configuration
            cap.release()
            print(f"Failed to configure camera {camera_id} to {target_width}x{target_height}, trying default configuration")
            return self.configure_camera(camera_id)
                
        except Exception as e:
            print(f"Error configuring camera {camera_id} to specific resolution: {e}")
            if 'cap' in locals():
                try:
                    cap.release()
                except:
                    pass
            return None

    def get_current_camera(self):
        """Get the currently configured camera capture object"""
        return self.current_cap
        
    def is_camera_configured(self):
        """Check if a camera is currently configured"""
        return self.current_cap is not None and self.selected_camera_id is not None
        
    def release_camera(self):
        """Explicitly release the current camera"""
        if self.current_cap:
            try:
                self.current_cap.release()
                print(f"Camera {self.selected_camera_id} released explicitly")
            except Exception as e:
                print(f"Error releasing camera: {e}")
            finally:
                self.current_cap = None
