"""
Calibration video processing for Added-Mass-Lab GUI
Processes calibration videos to extract camera parameters
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
import sys
import cv2
import numpy as np
from collections import Counter
from PIL import Image, ImageTk


# Canonical list of inner-corner counts as (cols, rows), conventionally
# listed long-axis-first so X = long edge.  All entries are physically
# distinct boards (no transpose pair) so cv2.findChessboardCorners picks
# one unambiguously.  Imported by calibration_recorder so the recording-
# time live-detection preview and the post-recording calibration analysis
# stay in sync — adding a board size here makes it available everywhere.
CHECKERBOARD_SIZES = [
    (9, 6),    # 10x7 squares — most common
    (8, 6),    # 9x7 squares
    (7, 5),    # 8x6 squares
    (10, 7),   # 11x8 squares
    (6, 4),    # 7x5 squares — small board
    (11, 8),   # 12x9 squares — large board
    (11, 7),   # 12x8 squares — elongated, 77 corners (high-precision option)
]


class CalibrationProcessor:
    """Processes calibration videos to extract camera parameters"""
    
    def __init__(self, parent, calibration_data):
        self.parent = parent
        self.calibration_data = calibration_data
        self.window = None
        self.processing = False
        
        # GUI elements
        self.progress_var = None
        self.status_label = None
        self.preview_label = None
        self.progress_bar = None
        
        # Processing data
        self.video_path = None
        # Single source of truth — see module-level CHECKERBOARD_SIZES.
        self.checkerboard_sizes = list(CHECKERBOARD_SIZES)
        self.detected_checkerboard_size = None

        # Lens model used for camera-matrix fitting.  Options exposed in the
        # checkerboard-confirmation dialog: "rational" (default), "fisheye",
        # and "scaramuzza".  Legacy saved calibrations with model_type "pinhole"
        # still load through main_gui's importer; the runtime dispatch in
        # corrections._apply_lens branches on all values.
        self.lens_model = "rational"
        # Default 1.0 matches CalibrationData.reset(); see that docstring for why.
        self.fisheye_balance = 1.0

        # Raw BGR frames collected during extract_calibration_data; used by
        # _fit_scaramuzza() which needs image files on disk for py-OCamCalib.
        self.calibration_frames = []

        # Completion callback
        self.completion_callback = None
        
    def get_video_resolution(self, video_path):
        """Get the resolution of a video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (width, height)
        except Exception as e:
            print(f"Error getting video resolution: {e}")
            return None
        
    def load_calibration_video(self, video_path=None):
        """Load and process calibration video"""
        if video_path is None:
            video_path = filedialog.askopenfilename(
                title="Load Calibration Video",
                filetypes=[("Video files", "*.avi *.mp4 *.mov"), ("All files", "*.*")]
            )
            
        if not video_path or not os.path.exists(video_path):
            return
            
        self.video_path = video_path
        self.show_processing_window()
        
    def show_processing_window(self):
        """Create and show the processing window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Processing Calibration Video")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_window_close)

        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Camera Calibration Processing",
                                font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Processing Status", padding="5")
        status_frame.pack(fill='x', pady=(0, 10))

        self.status_label = ttk.Label(status_frame, text="Initializing...")
        self.status_label.pack()

        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.pack(fill='x', pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=400)
        self.progress_bar.pack(pady=5)

        self.progress_text = ttk.Label(progress_frame, text="0%")
        self.progress_text.pack()

        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Detection Preview")
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))

        self.preview_label = ttk.Label(preview_frame, text="Processing will begin shortly...")
        self.preview_label.pack(expand=True)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')

        from .tooltip import ToolTip
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_processing)
        self.cancel_button.pack(side='right', padx=5)
        ToolTip(self.cancel_button, "Cancel calibration processing and close this window.")

        # Start processing in background thread
        self.processing = True
        processing_thread = threading.Thread(target=self.process_calibration, daemon=True)
        processing_thread.start()
        
    def process_calibration(self):
        """Main calibration processing pipeline"""
        try:
            # Step 1: Analyze video and detect checkerboard size
            self.update_status("Step 1: Analyzing checkerboard patterns...")
            result = self.detect_checkerboard_size()
            
            if result[0] is None:
                self.update_status("ERROR: No checkerboard patterns detected!")
                messagebox.showerror("Error", "No checkerboard patterns found in the video.")
                self.close_window()
                return
                
            checkerboard_size, detection_count, total_analyzed = result
            detection_percentage = (detection_count / total_analyzed) * 100 if total_analyzed > 0 else 0
            
            self.detected_checkerboard_size = checkerboard_size
            self.update_status(f"Detected checkerboard size: {checkerboard_size[0]}x{checkerboard_size[1]}")
            
            # Show confirmation dialog
            confirmed = self.show_checkerboard_confirmation(checkerboard_size, detection_percentage)
            if not confirmed:
                self.update_status("Calibration cancelled by user")
                self.close_window()
                return
            
            # Step 2: Extract calibration data
            self.update_status("Step 2: Extracting calibration points...")
            success = self.extract_calibration_data()
            
            if not success:
                self.update_status("ERROR: Failed to extract calibration data!")
                messagebox.showerror("Error", "Failed to extract sufficient calibration data.")
                self.close_window()
                return
                
            # Step 3: Fit camera model
            self.update_status("Step 3: Fitting camera distortion model...")
            success = self.fit_camera_model()
            
            if success:
                # Fisheye balance tuning is deferred to perspective_corrector,
                # where the captured perspective frame gives a meaningful scene
                # to tune against (and lets us auto-tune against detected
                # checkerboard rows/columns).
                self.update_status("Calibration completed successfully!")
                self.show_completion_dialog()
            else:
                self.update_status("ERROR: Camera model fitting failed!")
                # Surface the underlying reason where possible — _fit_fisheye
                # populates _last_fit_error with a multi-line diagnosis (the
                # rational fit currently has no equivalent).
                detail = getattr(self, '_last_fit_error', None)
                msg = "Failed to fit camera distortion model."
                if detail:
                    msg = f"{msg}\n\n{detail}"
                messagebox.showerror("Error", msg)
                
        except Exception as e:
            self.update_status(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"Calibration processing failed: {str(e)}")
        finally:
            self.processing = False
            
    def detect_checkerboard_size(self):
        """Detect the most likely checkerboard size from video frames"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_analyze = min(40, total_frames)
        
        # Select frame indices distributed over video length
        frame_indices = np.linspace(0, total_frames-1, frames_to_analyze, dtype=int)
        
        size_detections = Counter()
        # Track which frames contain each checkerboard size
        frames_by_size = {size: [] for size in self.checkerboard_sizes}
        
        for i, frame_idx in enumerate(frame_indices):
            if not self.processing:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Try each checkerboard size
                for size in self.checkerboard_sizes:
                    ret_corners, corners = cv2.findChessboardCorners(gray, size, None)
                    if ret_corners:
                        size_detections[size] += 1
                        frames_by_size[size].append(frame_idx)
                        break  # Only count the first successful detection per frame
                        
            # Update progress
            progress = (i + 1) / len(frame_indices) * 100
            self.update_progress(progress, f"Analyzing frame {i+1}/{len(frame_indices)}")
            
        cap.release()
        
        # Store the frames containing the detected checkerboard size
        if size_detections:
            most_common = size_detections.most_common(1)[0]
            detected_size = most_common[0]
            self.initial_valid_frames = frames_by_size[detected_size]
            print(f"Initial detection found {len(self.initial_valid_frames)} frames with {detected_size[0]}x{detected_size[1]} checkerboard")
            return detected_size, most_common[1], len(frame_indices)
        else:
            self.initial_valid_frames = []
            return None, 0, len(frame_indices)
            
    def show_checkerboard_confirmation(self, checkerboard_size, detection_percentage):
        """Show confirmation dialog for detected checkerboard size"""
        # This will be set by the dialog
        confirmation_result = [None]
        
        def show_dialog():
            # Create confirmation dialog
            dialog = tk.Toplevel(self.window)
            dialog.title("Confirm Checkerboard Size & Lens Model")
            dialog.geometry("460x490")
            dialog.resizable(False, False)
            dialog.transient(self.window)
            dialog.grab_set()  # Make dialog modal

            # Center the dialog on parent window
            dialog.update_idletasks()
            parent_x = self.window.winfo_x()
            parent_y = self.window.winfo_y()
            parent_width = self.window.winfo_width()
            parent_height = self.window.winfo_height()

            x = parent_x + (parent_width // 2) - (dialog.winfo_width() // 2)
            y = parent_y + (parent_height // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")

            # Main frame
            main_frame = ttk.Frame(dialog, padding="20")
            main_frame.pack(fill='both', expand=True)

            # Title
            title_label = ttk.Label(main_frame, text="Checkerboard Size Detected",
                                   font=('Arial', 14, 'bold'))
            title_label.pack(pady=(0, 10))

            # Detection information
            size_text = f"Detected size: {checkerboard_size[0]} × {checkerboard_size[1]} corners"
            size_label = ttk.Label(main_frame, text=size_text, font=('Arial', 12))
            size_label.pack(pady=(0, 3))

            percentage_text = f"Detection confidence: {detection_percentage:.1f}%"
            percentage_label = ttk.Label(main_frame, text=percentage_text, font=('Arial', 11))
            percentage_label.pack(pady=(0, 10))

            # Additional info
            info_text = f"This corresponds to a {checkerboard_size[0]+1} × {checkerboard_size[1]+1} square checkerboard."
            info_label = ttk.Label(main_frame, text=info_text, font=('Arial', 10),
                                  foreground='gray')
            info_label.pack(pady=(0, 15))

            # Lens-model selector.
            #   rational   - 8-coef Brown-Conrady; rectilinear / wide-angle ≤~150° FOV.
            #   fisheye    - cv2.fisheye equidistant; true fisheye, image circle
            #                smaller than sensor.  Use when rational leaves residual
            #                spherical curvature.
            #   scaramuzza - Scaramuzza omnidirectional polynomial (py-OCamCalib);
            #                most robust for true fisheye / omnidirectional lenses.
            #                Requires py-OCamCalib to be installed.
            model_frame = ttk.LabelFrame(main_frame, text="Lens model", padding="8")
            model_frame.pack(fill='x', pady=(0, 15))

            lens_model_var = tk.StringVar(value=self.lens_model)
            ttk.Radiobutton(
                model_frame,
                text="Rational (default — rectilinear / wide-angle up to ~150° FOV)",
                variable=lens_model_var, value="rational",
            ).pack(anchor='w')
            ttk.Radiobutton(
                model_frame,
                text="Fisheye (cv2.fisheye — equidistant, image circle smaller than sensor)",
                variable=lens_model_var, value="fisheye",
            ).pack(anchor='w')
            ttk.Radiobutton(
                model_frame,
                text="Scaramuzza / OCamCalib (most robust for true fisheye — requires py-OCamCalib)",
                variable=lens_model_var, value="scaramuzza",
            ).pack(anchor='w')

            # Question
            question_label = ttk.Label(main_frame, text="Proceed with calibration?",
                                      font=('Arial', 11))
            question_label.pack(pady=(0, 15))

            # Buttons frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x')

            def on_ok():
                self.lens_model = lens_model_var.get()
                confirmation_result[0] = True
                dialog.destroy()
                
            def on_cancel():
                confirmation_result[0] = False
                dialog.destroy()
            
            # Buttons
            from .tooltip import ToolTip
            cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
            cancel_btn.pack(side='right', padx=(5, 0))
            ToolTip(cancel_btn, "Cancel and go back to the previous step (do not use this detection).")
            ok_btn = ttk.Button(button_frame, text="OK", command=on_ok)
            ok_btn.pack(side='right')
            ToolTip(ok_btn, "Use this detected checkerboard size and continue with calibration.")
            
            # Wait for dialog to close
            dialog.wait_window()
        
        # Show dialog in main thread
        self.window.after(0, show_dialog)
        
        # Wait for result with simple polling
        import time
        while confirmation_result[0] is None:
            time.sleep(0.1)
            if self.window.winfo_exists():
                self.window.update()
            else:
                # Window was closed, treat as cancel
                return False
                
        return confirmation_result[0] == True
            
    def extract_calibration_data(self):
        """Extract calibration points from video using smart frame selection"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame_count = 40  # Target number of frames with valid checkerboards
        
        # Get the final set of frames containing valid checkerboards
        valid_frame_indices = self.get_valid_checkerboard_frames(cap, total_frames, target_frame_count)
        
        if len(valid_frame_indices) < 10:
            self.update_status(f"ERROR: Only found {len(valid_frame_indices)} frames with valid checkerboards (need at least 10)")
            cap.release()
            return False
        
        # Prepare object points
        checkerboard_size = self.detected_checkerboard_size
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane
        self.image_size = None
        self.calibration_frames = []  # raw BGR frames (for py-OCamCalib)
        
        successful_detections = 0
        
        # Process only the frames we know contain valid checkerboards
        for i, frame_idx in enumerate(valid_frame_indices):
            if not self.processing:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.image_size = gray.shape[::-1]  # (width, height)
                
                # Find checkerboard corners (should always succeed since we pre-validated)
                ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                
                if ret_corners:
                    successful_detections += 1
                    
                    # Refine corner positions
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Add object points and image points
                    self.objpoints.append(objp)
                    self.imgpoints.append(corners)
                    self.calibration_frames.append(frame.copy())
                    
                    # Show preview with detected corners
                    preview_frame = frame.copy()
                    cv2.drawChessboardCorners(preview_frame, checkerboard_size, corners, ret_corners)
                    
                    # Add text overlay
                    text = f"Valid Frame {successful_detections}/{len(valid_frame_indices)}: {checkerboard_size[0]}x{checkerboard_size[1]}"
                    cv2.putText(preview_frame, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Update preview in GUI
                    self.update_preview(preview_frame)
                else:
                    # This should not happen since we pre-validated frames
                    print(f"Warning: Frame {frame_idx} failed checkerboard detection despite pre-validation")
                    
            # Update progress
            progress = (i + 1) / len(valid_frame_indices) * 100
            self.update_progress(progress, f"Processing valid frames: {successful_detections}/{len(valid_frame_indices)}")
            
        cap.release()
        
        print(f"Calibration extraction completed: {successful_detections} valid frames processed")
        return successful_detections >= 10
        
    def get_valid_checkerboard_frames(self, cap, total_frames, target_count):
        """Smart frame selection algorithm to find frames with valid checkerboards"""
        checkerboard_size = self.detected_checkerboard_size
        
        # Start with frames we already know contain the checkerboard from initial detection
        valid_frames = self.initial_valid_frames.copy() if hasattr(self, 'initial_valid_frames') else []
        checked_frames = set(valid_frames)  # Track all frames we've already checked
        
        print(f"=== SMART FRAME SELECTION DEBUG ===")
        print(f"Total frames in video: {total_frames}")
        print(f"Target frames needed: {target_count}")
        print(f"Starting with {len(valid_frames)} pre-validated frames from initial detection")
        print(f"Initial valid frames: {sorted(valid_frames) if valid_frames else 'None'}")
        
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while len(valid_frames) < target_count and iteration < max_iterations:
            iteration += 1
            
            # Calculate how many more frames we need
            remaining_needed = target_count - len(valid_frames)
            
            # Get available frame indices from ENTIRE video (excluding already checked frames)
            available_frames = [i for i in range(total_frames) if i not in checked_frames]
            
            print(f"Iteration {iteration}: {len(available_frames)} frames available to check (out of {total_frames} total)")
            print(f"Already checked {len(checked_frames)} frames")
            
            if not available_frames:
                print(f"No more frames available to check. Found {len(valid_frames)} valid frames.")
                break
                
            # Determine how many frames to check this iteration
            # Check more frames early on, fewer as we get closer to target
            # Increased limits for searching 40 frames instead of 20
            frames_to_check = min(remaining_needed * 4, len(available_frames), 80)  # Check 4x what we need, max 80
            
            # Select frames evenly distributed across available frames in entire video
            if frames_to_check >= len(available_frames):
                frame_indices = available_frames
            else:
                step = len(available_frames) / frames_to_check
                frame_indices = [available_frames[int(i * step)] for i in range(frames_to_check)]
            
            print(f"Iteration {iteration}: Checking {len(frame_indices)} frames across entire video for {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard")
            print(f"Frame range being searched: {min(frame_indices)} to {max(frame_indices)} (selected from 0 to {total_frames-1})")
            
            # Check each frame for the target checkerboard
            iteration_valid_count = 0
            for i, frame_idx in enumerate(frame_indices):
                if not self.processing:
                    break
                    
                checked_frames.add(frame_idx)  # Mark as checked regardless of result
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                    
                    if ret_corners:
                        valid_frames.append(frame_idx)
                        iteration_valid_count += 1
                        if len(valid_frames) >= target_count:
                            break
                
                # Update progress for this iteration
                total_progress = ((iteration - 1) * 50 + (i + 1)) / (max_iterations * 50) * 50  # Use 50% of progress bar
                self.update_progress(total_progress, 
                                   f"Smart frame selection - Iteration {iteration}: {len(valid_frames)} valid frames found")
            
            print(f"Iteration {iteration} completed: Found {iteration_valid_count} new valid frames (total: {len(valid_frames)})")
        
        # Sort frame indices for sequential processing
        valid_frames.sort()
        
        print(f"Smart frame selection completed: {len(valid_frames)} valid frames found after {iteration} iterations")
        print(f"Valid frame indices: {valid_frames}")
        
        return valid_frames
        
    def fit_camera_model(self):
        """Fit camera distortion model to extracted data.

        Branches on self.lens_model (set by the checkerboard-confirmation
        dialog):
          - "rational" (default): 8-coef Brown-Conrady via
            _fit_pinhole(extra_flags=cv2.CALIB_RATIONAL_MODEL); fits
            rectilinear and moderate wide-angle lenses cleanly.
          - "fisheye": cv2.fisheye equidistant via _fit_fisheye(); required
            for true fisheye lenses (image circle smaller than sensor).

        The 5-coef pinhole fit (_fit_pinhole with no flags) is reachable
        only via legacy calibration JSONs whose model_type="pinhole".
        """
        if not hasattr(self, 'objpoints') or not hasattr(self, 'imgpoints'):
            return False

        try:
            self.update_status(f"Fitting {self.lens_model} camera model...")
            self.update_progress(50, "Computing camera matrix...")

            if self.lens_model == "scaramuzza":
                ok, scaramuzza_params, mean_error = self._fit_scaramuzza()
                if not ok:
                    return False
                self.update_progress(100, "Calibration complete!")
                self.calibration_data.set_calibration(
                    None, None, self.detected_checkerboard_size,
                    mean_error, self.image_size,
                    model_type="scaramuzza",
                    scaramuzza_params=scaramuzza_params,
                )
                return True

            if self.lens_model == "fisheye":
                ok, camera_matrix, dist_coeffs, mean_error = self._fit_fisheye()
            else:
                ok, camera_matrix, dist_coeffs, mean_error = self._fit_pinhole(
                    extra_flags=cv2.CALIB_RATIONAL_MODEL)

            if not ok:
                return False

            self.update_progress(100, "Calibration complete!")

            # Store calibration data globally
            self.calibration_data.set_calibration(
                camera_matrix, dist_coeffs, self.detected_checkerboard_size,
                mean_error, self.image_size,
                model_type=self.lens_model,
                fisheye_balance=self.fisheye_balance,
            )

            return True

        except Exception as e:
            print(f"Camera model fitting error: {e}")
            return False

    def _fit_pinhole(self, extra_flags=0):
        """Standard cv2.calibrateCamera path. Returns (ok, K, D, mean_err).

        Setting extra_flags=cv2.CALIB_RATIONAL_MODEL enables the 8-coef
        rational distortion model (k1..k6, p1, p2), which fits wide-angle
        lenses up to ~150 deg FOV considerably better than the default
        5-coef Brown-Conrady model. Runtime cv2.undistort handles the
        longer D vector transparently.
        """
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None,
            flags=extra_flags,
        )
        if not ret:
            return False, None, None, None

        self.update_progress(75, "Calculating calibration error...")

        total_error = 0.0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.objpoints)
        print(f"[{self.lens_model} fit] RMS-equivalent reprojection error: {mean_error:.4f} px "
              f"(D length = {len(dist_coeffs.flatten())})")
        return True, camera_matrix, dist_coeffs, mean_error

    def _fit_scaramuzza(self):
        """Calibrate using py-OCamCalib (Scaramuzza omnidirectional model).

        Returns (ok, params_dict, rms_error).  Saves the extracted calibration
        frames to a temp directory, runs py-OCamCalib's CalibrationEngine, loads
        the resulting JSON, cleans up, and returns the parameter dict.

        The JSON dict (with keys taylor_coefficient, distortion_center,
        stretch_matrix, inverse_poly, etc.) is stored directly in
        CalibrationData.scaramuzza_params and used at runtime by
        corrections._build_scaramuzza_maps().
        """
        import tempfile
        import shutil

        self._last_fit_error = None

        try:
            from pyocamcalib.modelling.calibration import CalibrationEngine
        except ImportError:
            self._last_fit_error = (
                "py-OCamCalib is not installed.\n"
                "Install with:\n"
                "  pip install git+https://github.com/jakarto3d/py-OCamCalib.git"
            )
            return False, None, None

        frames = self.calibration_frames
        if len(frames) < 5:
            self._last_fit_error = (
                f"Scaramuzza fit needs at least 5 calibration frames; "
                f"only {len(frames)} collected."
            )
            return False, None, None

        # Bundle adjustment cost scales with n_frames². Cap at 25 to keep
        # runtime under ~2 min; accuracy doesn't improve beyond ~20 frames.
        MAX_SCARAMUZZA_FRAMES = 25
        if len(frames) > MAX_SCARAMUZZA_FRAMES:
            step = len(frames) / MAX_SCARAMUZZA_FRAMES
            frames = [frames[int(i * step)] for i in range(MAX_SCARAMUZZA_FRAMES)]
            print(f"[scaramuzza fit] Downsampled to {len(frames)} frames for bundle adjustment")

        tmpdir = tempfile.mkdtemp(prefix="ocamcalib_")
        try:
            self.update_status(f"Scaramuzza fit: saving {len(frames)} frames to disk...")
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(tmpdir, f"frame_{i:04d}.png"), frame)

            cb_cols, cb_rows = self.detected_checkerboard_size
            engine = CalibrationEngine(
                tmpdir,
                (cb_rows, cb_cols),   # py-OCamCalib expects (rows, cols)
                "camera",
                square_size=1.0,
            )

            # Redirect stdout/stderr to StringIO while pyocamcalib runs so that
            # tqdm's progress writes don't crash a windowed PyInstaller bundle
            # where sys.stdout/sys.stderr are None (console=False).
            import io as _io
            _old_stdout, _old_stderr = sys.stdout, sys.stderr
            if sys.stdout is None:
                sys.stdout = _io.StringIO()
            if sys.stderr is None:
                sys.stderr = _io.StringIO()

            try:
                n_frames = len(frames)
                self.update_status(
                    f"Scaramuzza fit: detecting corners in {n_frames} frames "
                    f"(~{n_frames // 2}–{n_frames} s) ...")
                engine.detect_corners(check=False)
                n_detected = sum(1 for v in engine.detections.values() if v is not None)
                self.update_status(
                    f"Scaramuzza fit: corners found in {n_detected}/{n_frames} frames — "
                    f"running linear parameter estimation...")

                engine.estimate_fisheye_parameters()
                linear_rms = getattr(engine, 'rms_overall', None)
                rms_str = f" (linear RMS {linear_rms:.2f} px)" if linear_rms else ""
                self.update_status(
                    f"Scaramuzza fit: linear estimate done{rms_str} — "
                    f"fitting inverse polynomial...")

                engine.find_poly_inv()
                # Read results directly from engine attributes (avoid save_calibration's
                # hardcoded relative path which doesn't point inside tmpdir).
                params = {
                    "taylor_coefficient": engine.taylor_coefficient.tolist(),
                    "distortion_center":  list(engine.distortion_center),
                    "stretch_matrix":     engine.stretch_matrix.tolist(),
                    "inverse_poly":       engine.inverse_poly.tolist(),
                    "rms_overall":        float(engine.rms_overall),
                }

                rms = params["rms_overall"]
                self.update_status(
                    f"Scaramuzza fit complete — RMS {rms:.3f} px  "
                    f"({n_detected} frames, {len(params['taylor_coefficient'])}-coef Taylor, "
                    f"{len(params['inverse_poly'])}-coef inverse poly)")
                print(f"[scaramuzza fit] RMS: {rms:.4f} px")
                print(f"[scaramuzza fit] distortion_center: {params['distortion_center']}")
                print(f"[scaramuzza fit] stretch_matrix: {params['stretch_matrix']}")
                print(f"[scaramuzza fit] taylor_coefficient ({len(params['taylor_coefficient'])} coef)")
                print(f"[scaramuzza fit] inverse_poly ({len(params['inverse_poly'])} coef)")

            finally:
                sys.stdout = _old_stdout
                sys.stderr = _old_stderr

            return True, params, rms

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._last_fit_error = f"py-OCamCalib calibration failed:\n{exc}\n\n{tb}"
            if sys.stderr is not None:
                print(f"[scaramuzza fit] {exc}\n{tb}", file=sys.stderr)
            return False, None, None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _prescreen_fisheye_views(self, objp_fisheye, imgp_fisheye):
        """Drop views whose corner geometry would crash fisheye InitExtrinsics.

        cv2.fisheye.calibrate aborts with a 'fabs(norm_u1) > 0' assertion in
        cv::internal::InitExtrinsics when a view's back-projected corners are
        nearly degenerate (collinear after the inverse-fisheye remap).  That
        error path reports no view index, so our regex-based drop loop can't
        recover.  cv2.solvePnP with SOLVEPNP_IPPE (planar PnP) shares the same
        degeneracy conditions on planar targets — if IPPE can't solve a view,
        InitExtrinsics is likely to choke on it too.

        Returns (filtered_objp, filtered_imgp, dropped_indices).  Uses a
        crude image-centre guess for K because we don't have a calibrated K
        yet; that's fine for the degeneracy check, which is geometric.
        """
        W, H = self.image_size
        K_guess = np.array(
            [[W * 0.5, 0.0, W * 0.5],
             [0.0, W * 0.5, H * 0.5],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        good_objp, good_imgp, dropped = [], [], []
        for i, (op, ip) in enumerate(zip(objp_fisheye, imgp_fisheye)):
            obj_3d = op.reshape(-1, 3).astype(np.float64)
            img_2d = ip.reshape(-1, 1, 2).astype(np.float64)
            try:
                ok, _, _ = cv2.solvePnP(
                    obj_3d, img_2d, K_guess, None,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error:
                ok = False
            if ok:
                good_objp.append(op)
                good_imgp.append(ip)
            else:
                dropped.append(i)
        return good_objp, good_imgp, dropped

    def _try_fisheye_calibrate(self, objp, imgp, flags, criteria,
                                prune_on_check_cond, max_drop, K_init=None):
        """One pass of cv2.fisheye.calibrate with optional CHECK_COND pruning.

        Returns (ok, K, D, rms, error_msg).  Mutates objp/imgp in place when
        pruning views.  Pass prune_on_check_cond=False to surface the first
        cv2.error directly without retry.

        When K_init is provided (a 3×3 float64 matrix), it is used as the
        starting camera matrix and CALIB_USE_INTRINSIC_GUESS is added to flags
        automatically, bypassing OpenCV's fisheye auto-init which is brittle on
        vignetting-lens data (InitExtrinsics 'fabs(norm_u1) > 0' assertion).
        When K_init is None, K starts as zeros and OpenCV auto-initialises.
        """
        import re as _re
        N = len(objp)
        if K_init is not None:
            K = K_init.copy()
            flags = flags | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        else:
            K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
        dropped = 0
        last_err = "no attempt made"
        while True:
            try:
                rms, K, D, _, _ = cv2.fisheye.calibrate(
                    objp, imgp, self.image_size,
                    K, D, rvecs, tvecs, flags, criteria)
                return True, K, D, float(rms), None
            except cv2.error as e:
                last_err = str(e).strip().splitlines()[-1]
                if not prune_on_check_cond:
                    return False, None, None, None, last_err
                m = _re.search(r'input array (\d+)', str(e))
                if not m:
                    return False, None, None, None, last_err
                bad_idx = int(m.group(1))
                if bad_idx >= len(objp):
                    return False, None, None, None, (
                        f"reported view {bad_idx} but only {len(objp)} remain")
                if dropped >= max_drop:
                    return False, None, None, None, (
                        f"dropped {dropped} views and optimiser still unhappy")
                print(f"[fisheye fit] Dropping ill-conditioned view {bad_idx} "
                      f"(retry {dropped + 1}/{max_drop})")
                del objp[bad_idx]
                del imgp[bad_idx]
                del rvecs[bad_idx]
                del tvecs[bad_idx]
                dropped += 1
                K = K_init.copy() if K_init is not None else np.zeros((3, 3), dtype=np.float64)
                D = np.zeros((4, 1), dtype=np.float64)

    def _fit_fisheye(self):
        """cv2.fisheye.calibrate path. Returns (ok, K, D, mean_err).

        Two-stage strategy because the canonical recipe (CHECK_COND + iterative
        pruning) is brittle on real-world data — corners that pass cornerSubPix
        cleanly can still be flagged ill-conditioned by the optimizer's Jacobian
        check, and pruning more than half the views means we lose information
        the relaxed fit could have used:
          1. Canonical: CHECK_COND + RECOMPUTE_EXTRINSIC + FIX_SKEW with
             pruning loop (drop the offending view, retry).
          2. Permissive fallback: drop CHECK_COND so the optimizer keeps every
             view; bump iteration cap.  This converges where stage 1 won't on
             vignetting lenses where corners cluster near the image circle.
        On failure, sets self._last_fit_error with a human-readable diagnosis
        so fit_camera_model can show it in the UI.

        Image-point layout note: shape (N, 1, 2), dtype float64 is required.
        cv2.fisheye uses leading dims semantically and silently produces
        garbage with the wrong layout.
        """
        self._last_fit_error = None
        N_views = len(self.objpoints)
        objp_fisheye = [op.reshape(1, -1, 3).astype(np.float64)
                        for op in self.objpoints]
        imgp_fisheye = [ip.reshape(-1, 1, 2).astype(np.float64)
                        for ip in self.imgpoints]

        # Pre-screen views to drop ones with degenerate corner geometry that
        # would crash cv2.fisheye.InitExtrinsics (which gives no view index
        # for our drop loop to act on).
        objp_fisheye, imgp_fisheye, prescreen_dropped = (
            self._prescreen_fisheye_views(objp_fisheye, imgp_fisheye))
        if prescreen_dropped:
            print(f"[fisheye fit] Pre-screen dropped {len(prescreen_dropped)} "
                  f"views with degenerate geometry: {prescreen_dropped}")
        if len(objp_fisheye) < 5:
            self._last_fit_error = (
                f"Only {len(objp_fisheye)} views passed pre-screening "
                f"(out of {N_views}).  The calibration data has too many "
                f"views with degenerate corner geometry — re-record with "
                f"the checkerboard in more varied positions / orientations."
            )
            return False, None, None, None

        # Seed the fisheye fit with K from a rational (pinhole) fit on the same
        # data.  cv2.calibrateCamera's initialiser is robust to vignetting-lens
        # corner distributions; the resulting fx/fy give fisheye.InitExtrinsics a
        # well-conditioned focal-length estimate that avoids the
        # 'fabs(norm_u1) > 0' assertion which fires when OpenCV auto-initialises
        # fisheye K on the same data.  cx/cy are overwritten to image centre to
        # enforce radial symmetry — the fit is free to refine fx/fy and D.
        W, H = self.image_size
        K_seed = None
        ok_rational, K_rational, _, _ = self._fit_pinhole(
            extra_flags=cv2.CALIB_RATIONAL_MODEL)
        if ok_rational:
            K_seed = K_rational.copy().astype(np.float64)
            K_seed[0, 2] = W / 2.0
            K_seed[1, 2] = H / 2.0
            print(f"[fisheye fit] Rational K seed: "
                  f"fx={K_seed[0,0]:.1f}, fy={K_seed[1,1]:.1f}, "
                  f"cx={K_seed[0,2]:.1f}, cy={K_seed[1,2]:.1f}")
        else:
            print("[fisheye fit] Rational seed fit failed; using zero K (fisheye auto-init)")

        # ---- Stage 1: canonical recipe with CHECK_COND + pruning ----
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                 | cv2.fisheye.CALIB_CHECK_COND
                 | cv2.fisheye.CALIB_FIX_SKEW)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        ok, K, D, rms, stage1_err = self._try_fisheye_calibrate(
            objp_fisheye, imgp_fisheye, flags, criteria, prune_on_check_cond=True,
            max_drop=max(1, N_views // 2), K_init=K_seed,
        )

        if not ok:
            # ---- Stage 2: permissive fallback, no CHECK_COND ----
            # Re-run pre-screen on the original lists (stage 1 mutated
            # objp_fisheye/imgp_fisheye via its drop loop).
            print(f"[fisheye fit] Stage 1 failed ({stage1_err}); retrying without CHECK_COND.")
            objp_fisheye = [op.reshape(1, -1, 3).astype(np.float64)
                            for op in self.objpoints]
            imgp_fisheye = [ip.reshape(-1, 1, 2).astype(np.float64)
                            for ip in self.imgpoints]
            objp_fisheye, imgp_fisheye, _ = (
                self._prescreen_fisheye_views(objp_fisheye, imgp_fisheye))
            flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                     | cv2.fisheye.CALIB_FIX_SKEW)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            ok, K, D, rms, stage2_err = self._try_fisheye_calibrate(
                objp_fisheye, imgp_fisheye, flags, criteria, prune_on_check_cond=False,
                max_drop=0, K_init=K_seed,
            )
            if not ok:
                self._last_fit_error = (
                    f"Fisheye optimiser refused both fits.\n"
                    f"  Stage 1 (with CHECK_COND): {stage1_err}\n"
                    f"  Stage 2 (relaxed):         {stage2_err}\n\n"
                    f"This usually means the calibration views don't span enough\n"
                    f"of the lens FOV for a fisheye fit — try recording a video\n"
                    f"with the checkerboard moved across more of the image circle\n"
                    f"(centre, all four corners of the visible region, tilted)."
                )
                return False, None, None, None

        # Diagnostics: surface fit quality and the actual sensor coverage of
        # corner observations. With a vignetting lens, "good frame coverage"
        # in the GUI is not the same as good optical-axis-angle coverage,
        # and the spread numbers below are the honest measure.
        all_pts = np.concatenate([ip.reshape(-1, 2) for ip in self.imgpoints])
        x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
        W, H = self.image_size
        x_pct = 100.0 * (x_max - x_min) / W
        y_pct = 100.0 * (y_max - y_min) / H
        print(f"[fisheye fit] RMS reprojection error: {rms:.4f} px")
        print(f"[fisheye fit] K (raw) =\n{K}")
        print(f"[fisheye fit] D = {D.flatten()}")
        print(f"[fisheye fit] Corner spread: "
              f"x∈[{x_min:.0f}, {x_max:.0f}] of [0, {W}] ({x_pct:.1f}%);  "
              f"y∈[{y_min:.0f}, {y_max:.0f}] of [0, {H}] ({y_pct:.1f}%)")
        if x_pct < 80 or y_pct < 80:
            print(f"[fisheye fit] NOTE: corners cover < 80% of one sensor axis. "
                  f"This is normal for vignetting lenses; runtime undistort "
                  f"should use balance=1.0 (the default) to avoid extrapolating "
                  f"into untrained regions.")

        # Post-fit centring: the fisheye model is radially symmetric about
        # (cx, cy), so any asymmetry in the undistorted output comes from
        # cx/cy drifting off the image centre during fit (the optimiser
        # over-fits to wherever the corners cluster, which for a vignetting
        # lens is rarely the true optical centre).  We can't constrain this
        # at fit time — CALIB_FIX_PRINCIPAL_POINT trips an InitExtrinsics
        # assertion on this kind of data — so we overwrite cx/cy after the
        # fact.  Drift of a few percent of W has negligible effect in the
        # central tracking region; large drift is logged so the user knows
        # the fit was poorly constrained.
        cx_drift_px = K[0, 2] - W / 2.0
        cy_drift_px = K[1, 2] - H / 2.0
        cx_drift_pct = 100.0 * cx_drift_px / W
        cy_drift_pct = 100.0 * cy_drift_px / H
        print(f"[fisheye fit] cx/cy drift from image centre: "
              f"({cx_drift_px:+.1f}, {cy_drift_px:+.1f}) px "
              f"= ({cx_drift_pct:+.1f}%, {cy_drift_pct:+.1f}%)")
        if abs(cx_drift_pct) > 5.0 or abs(cy_drift_pct) > 5.0:
            print(f"[fisheye fit] WARNING: large principal-point drift suggests "
                  f"the calibration views don't span enough of the lens FOV. "
                  f"Re-centring to image centre may leave residual asymmetry.")
        K[0, 2] = W / 2.0
        K[1, 2] = H / 2.0
        print(f"[fisheye fit] K (cx/cy re-centred) =\n{K}")

        # cv2.fisheye.calibrate returns the overall RMS reprojection error
        # directly, in pixels — same units as the pinhole mean_error above.
        return True, K, D, float(rms)
            
    def update_status(self, message):
        """Update status label in GUI thread"""
        if self.status_label:
            self.window.after(0, lambda: self.status_label.configure(text=message))
            
    def update_progress(self, value, text=""):
        """Update progress bar in GUI thread"""
        if self.progress_var and self.progress_text:
            self.window.after(0, lambda: [
                self.progress_var.set(value),
                self.progress_text.configure(text=f"{value:.1f}% - {text}")
            ])
            
    def update_preview(self, frame):
        """Update preview image in GUI thread"""
        try:
            if self.preview_label and frame is not None:
                # Resize frame for preview
                height, width = frame.shape[:2]
                max_width, max_height = 400, 300
                
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                frame_resized = cv2.resize(frame, (new_width, new_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update preview in main thread
                self.window.after(0, self._update_preview_image, photo)
                
        except Exception as e:
            print(f"Preview update error: {e}")
            
    def _update_preview_image(self, photo):
        """Helper to update preview image in main thread"""
        try:
            if self.preview_label:
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
        except Exception as e:
            print(f"Preview image update error: {e}")
            
    def show_completion_dialog(self, show_popup=True):
        """Show calibration completion dialog"""
        # Show the popup if requested and no completion callback is set
        if show_popup and not self.completion_callback:
            info = self.calibration_data.get_calibration_info()
            def show_dialog_and_close():
                messagebox.showinfo("Calibration Complete", 
                                   f"Camera calibration successful!\n\n{info}")
                # Auto-close window after showing completion dialog
                self.close_window()
            self.window.after(0, show_dialog_and_close)
        
        # Call completion callback if set (for perspective correction)
        # Don't close window yet - let the callback chain handle it
        if self.completion_callback:
            self.completion_callback()
        
        # Only close window immediately if no popup and no callback
        elif not show_popup:
            self.window.after(100, self.close_window)
        
    def cancel_processing(self):
        """Cancel the processing"""
        self.processing = False
        self.close_window()
        
    def on_window_close(self):
        """Handle window close event"""
        self.cancel_processing()
        
    def close_window(self):
        """Close the processing window"""
        if self.window:
            self.window.after(0, self.window.destroy)
