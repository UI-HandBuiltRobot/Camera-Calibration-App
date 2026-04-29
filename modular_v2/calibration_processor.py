"""
Calibration video processing for Added-Mass-Lab GUI
Processes calibration videos to extract camera parameters
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
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

        # Lens model used for camera-matrix fitting. The GUI no longer
        # exposes a chooser — every new calibration uses the rational
        # (8-coef) Brown-Conrady fit, which is a strict superset of the
        # 5-coef pinhole model and handles wide-angle lenses cleanly.
        # Field retained so legacy saved calibrations with model_type
        # "pinhole" or "fisheye" still load through main_gui's calibration
        # importer; the runtime dispatch in corrections._apply_lens still
        # branches on these.
        self.lens_model = "rational"
        # Default 1.0 matches CalibrationData.reset(); see that docstring for why.
        self.fisheye_balance = 1.0

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
                self.update_status("Calibration completed successfully!")
                self.show_completion_dialog()
            else:
                self.update_status("ERROR: Camera model fitting failed!")
                messagebox.showerror("Error", "Failed to fit camera distortion model.")
                
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
            dialog.title("Confirm Checkerboard Size")
            dialog.geometry("420x260")
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
            title_label.pack(pady=(0, 15))

            # Detection information
            size_text = f"Detected size: {checkerboard_size[0]} × {checkerboard_size[1]} corners"
            size_label = ttk.Label(main_frame, text=size_text, font=('Arial', 12))
            size_label.pack(pady=(0, 5))

            percentage_text = f"Detection confidence: {detection_percentage:.1f}%"
            percentage_label = ttk.Label(main_frame, text=percentage_text, font=('Arial', 11))
            percentage_label.pack(pady=(0, 15))

            # Additional info
            info_text = f"This corresponds to a {checkerboard_size[0]+1} × {checkerboard_size[1]+1} square checkerboard."
            info_label = ttk.Label(main_frame, text=info_text, font=('Arial', 10),
                                  foreground='gray')
            info_label.pack(pady=(0, 20))

            # Question
            question_label = ttk.Label(main_frame, text="Proceed with calibration using this size?",
                                      font=('Arial', 11))
            question_label.pack(pady=(0, 20))

            # Buttons frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x')

            def on_ok():
                # Always rational; no longer user-selectable
                self.lens_model = "rational"
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

        Always uses the rational (8-coef) Brown-Conrady fit via
        _fit_pinhole(extra_flags=cv2.CALIB_RATIONAL_MODEL). The 5-coef
        pinhole and equidistant fisheye fits remain available as private
        methods (_fit_pinhole with no flags, _fit_fisheye) so legacy
        calibration JSONs that recorded model_type="pinhole" or "fisheye"
        still re-fit consistently if anything ever calls those paths
        directly, but the GUI no longer routes here.
        """
        if not hasattr(self, 'objpoints') or not hasattr(self, 'imgpoints'):
            return False

        try:
            self.update_status(f"Fitting {self.lens_model} camera model...")
            self.update_progress(50, "Computing camera matrix...")

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

    def _fit_fisheye(self):
        """cv2.fisheye.calibrate path. Returns (ok, K, D, mean_err).

        Canonical fisheye-calibration recipe, no embellishments:
          - object points:  shape (1, N, 3), dtype float64
          - image points:   shape (N, 1, 2), dtype float64 (cornerSubPix's
                            natural output shape — DO NOT reshape to (1, N, 2);
                            cv2.fisheye uses leading dims semantically and
                            silently produces garbage with the wrong layout).
          - K, D start as zeros; the function derives its own initial K.
          - Flags = RECOMPUTE_EXTRINSIC | CHECK_COND | FIX_SKEW.

        The only non-canonical piece is an iterative-pruning loop around
        CHECK_COND failures, since the canonical recipe is a one-shot call
        that aborts on the first ill-conditioned view. Pruning is the
        standard workaround documented in every fisheye tutorial.
        """
        N_views = len(self.objpoints)
        objp_fisheye = [op.reshape(1, -1, 3).astype(np.float64)
                        for op in self.objpoints]
        imgp_fisheye = [ip.reshape(-1, 1, 2).astype(np.float64)
                        for ip in self.imgpoints]

        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_views)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_views)]

        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                 | cv2.fisheye.CALIB_CHECK_COND
                 | cv2.fisheye.CALIB_FIX_SKEW)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # Iterative pruning: when CALIB_CHECK_COND rejects a view, drop it
        # and retry. The error message names the offending array index.
        # Cap retries at half the views so we don't loop forever on broken
        # data; pruning more than that means the dataset itself is unusable.
        import re as _re
        dropped = []
        max_drop = max(1, N_views // 2)
        while len(dropped) < max_drop:
            try:
                rms, K, D, _, _ = cv2.fisheye.calibrate(
                    objp_fisheye, imgp_fisheye, self.image_size,
                    K, D, rvecs, tvecs, flags, criteria)
                break
            except cv2.error as e:
                msg = str(e)
                m = _re.search(r'input array (\d+)', msg)
                if not m:
                    print(f"cv2.fisheye.calibrate failed (not a CHECK_COND error): {e}")
                    return False, None, None, None
                bad_idx = int(m.group(1))
                if bad_idx >= len(objp_fisheye):
                    print(f"cv2.fisheye.calibrate reported view {bad_idx} but only "
                          f"{len(objp_fisheye)} remain; aborting.")
                    return False, None, None, None
                dropped.append(bad_idx)
                print(f"[fisheye fit] Dropping ill-conditioned view {bad_idx} "
                      f"(retry {len(dropped)}/{max_drop})")
                del objp_fisheye[bad_idx]
                del imgp_fisheye[bad_idx]
                del rvecs[bad_idx]
                del tvecs[bad_idx]
                # Reset K/D to zeros so the optimizer derives a fresh
                # initial K on the smaller set, matching the canonical
                # recipe (which always starts from zeros).
                K = np.zeros((3, 3), dtype=np.float64)
                D = np.zeros((4, 1), dtype=np.float64)
        else:
            print(f"[fisheye fit] Aborting: dropped {max_drop} views and the "
                  f"optimizer is still unhappy. Calibration data is unusable.")
            return False, None, None, None

        if dropped:
            print(f"[fisheye fit] Used {len(objp_fisheye)} views "
                  f"({N_views} - {len(dropped)} dropped)")

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
        print(f"[fisheye fit] K =\n{K}")
        print(f"[fisheye fit] D = {D.flatten()}")
        print(f"[fisheye fit] Corner spread: "
              f"x∈[{x_min:.0f}, {x_max:.0f}] of [0, {W}] ({x_pct:.1f}%);  "
              f"y∈[{y_min:.0f}, {y_max:.0f}] of [0, {H}] ({y_pct:.1f}%)")
        if x_pct < 80 or y_pct < 80:
            print(f"[fisheye fit] NOTE: corners cover < 80% of one sensor axis. "
                  f"This is normal for vignetting lenses; runtime undistort "
                  f"should use balance=1.0 (the default) to avoid extrapolating "
                  f"into untrained regions.")

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
