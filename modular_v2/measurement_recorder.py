"""
Measurement video recording for Added-Mass-Lab GUI
Handles measurement video recording with calibration correction processing
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import os
import json
import cv2
import numpy as np
from PIL import Image, ImageTk

from .corrections import apply_corrections


class MeasurementVideoRecorder:
    """Handles measurement video recording with calibration correction processing"""
    
    def __init__(self, parent, camera_manager, calibration_data):
        self.parent = parent
        self.camera_manager = camera_manager
        self.calibration_data = calibration_data
        self.window = None
        self.preview_label = None
        
        # Recording state
        self.recording = False
        self.recording_thread = None
        self.preview_thread = None
        
        # Camera and recording
        self.cap = None
        self.video_writer = None
        self.output_path = None
        
        # Frame queues for threading
        self.frame_queue = queue.Queue(maxsize=60)  # Buffer for frames
        self.preview_queue = queue.Queue(maxsize=5)  # Latest frames for preview
        
        # UI elements
        self.start_button = None
        self.stop_button = None
        self.status_label = None
        self.time_label = None
        
        # Recording stats
        self.frames_recorded = 0
        self.start_time = None
        
        # Completion callback
        self.completion_callback = None
        
    def show_recording_dialog(self):
        """Show dialog to get save location and start recording"""
        # Get save location first
        filename = filedialog.asksaveasfilename(
            title="Save Measurement Video",
            defaultextension=".avi",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        self.output_path = filename
        self.create_recording_window()
        
    def create_recording_window(self):
        """Create the recording interface window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Record Measurement Video")
        self.window.geometry("800x700")
        self.window.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Measurement Video Recording", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Recording Information", padding="10")
        info_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(info_frame, text=f"Save to: {self.output_path}").pack(anchor='w')
        
        # Conditional info text based on calibration availability
        has_lens_correction = self.calibration_data and self.calibration_data.is_calibrated
        has_perspective_correction = self.calibration_data and self.calibration_data.perspective_corrected
        
        if has_lens_correction or has_perspective_correction:
            corrections = []
            if has_lens_correction:
                corrections.append("lens distortion")
            if has_perspective_correction:
                corrections.append("perspective")
            correction_text = " and ".join(corrections)
            
            info_text = f"This video will be automatically processed with {correction_text} corrections after recording."
        else:
            info_text = "Video will be saved without calibration processing (no calibration data available)."
            
        ttk.Label(info_frame, text=info_text).pack(anchor='w', pady=(5, 0))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Recording Controls", padding="10")
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill='x')
        
        self.start_button = ttk.Button(buttons_frame, text="Start Recording", 
                                      command=self.start_recording, width=15)
        self.start_button.pack(side='left', padx=(0, 10))
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop Recording", 
                                     command=self.stop_recording, width=15, state='disabled')
        self.stop_button.pack(side='left', padx=(0, 10))
        
        close_button = ttk.Button(buttons_frame, text="Close", 
                                 command=self.close_window, width=15)
        close_button.pack(side='right')
        
        # Status frame
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill='x', pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready to record", foreground='blue')
        self.status_label.pack(side='left')
        
        self.time_label = ttk.Label(status_frame, text="", foreground='green')
        self.time_label.pack(side='right')
        
        # Preview frame (fixed height so control buttons remain visible)
        PREVIEW_PANE_HEIGHT = 360
        preview_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding="10")
        preview_frame.pack(fill='both', expand=True)
        preview_frame.configure(height=PREVIEW_PANE_HEIGHT)
        preview_frame.pack_propagate(False)

        self.preview_label = ttk.Label(preview_frame, text="Initializing camera...",
              justify='center', background='black', foreground='white')
        self.preview_label.place(relx=0.5, rely=0.5, anchor='center')

        # Start camera preview
        self.start_camera_preview()
        
    def start_camera_preview(self):
        """Start camera preview (low priority)"""
        camera_id = self.camera_manager.selected_camera_id
        if camera_id is None:
            self.status_label.configure(text="No camera selected", foreground='red')
            return
            
        # Use existing configured camera instead of reconfiguring
        if self.camera_manager.is_camera_configured() and self.camera_manager.selected_camera_id == camera_id:
            self.cap = self.camera_manager.get_current_camera()
            if self.cap:
                # Test that the camera is still working
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.status_label.configure(text="Camera ready", foreground='green')
                    # Start preview thread
                    self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
                    self.preview_thread.start()
                    return
                else:
                    print("Existing camera capture not working, will reconfigure")
        
        # Fallback: Configure camera if no existing capture or it's not working        
        self.cap = self.camera_manager.configure_camera(camera_id)
        if not self.cap:
            self.status_label.configure(text="Failed to configure camera", foreground='red')
            return
            
        # Start preview thread
        self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        self.preview_thread.start()
        
        self.status_label.configure(text="Camera ready", foreground='green')
        
    def preview_loop(self):
        """Low-priority preview loop"""
        try:
            while self.cap and not self.recording:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Update preview at lower frequency when not recording
                    if not self.preview_queue.full():
                        self.preview_queue.put(frame)
                        self.window.after(0, self.update_preview)
                        
                time.sleep(0.1)  # Lower frequency preview (10 FPS)
                
        except Exception as e:
            print(f"Preview loop error: {e}")
            
    def update_preview(self):
        """Update preview display"""
        try:
            if not self.preview_queue.empty():
                frame = self.preview_queue.get()
                
                # Resize for preview to fit fixed preview pane height (no upscaling)
                height, width = frame.shape[:2]
                PREVIEW_PANE_HEIGHT = 360
                scale = min(1.0, PREVIEW_PANE_HEIGHT / float(height))
                new_height = int(height * scale)
                new_width = int(width * scale)

                resized = cv2.resize(frame, (new_width, new_height))
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
                
        except Exception as e:
            print(f"Preview update error: {e}")
            
    def start_recording(self):
        """Start video recording"""
        if not self.cap:
            messagebox.showerror("Error", "Camera not available")
            return
            
        # Setup video writer
        resolution = self.camera_manager.selected_resolution
        framerate = self.camera_manager.selected_framerate
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            framerate, 
            resolution
        )
        
        if not self.video_writer.isOpened():
            messagebox.showerror("Error", "Failed to create video file")
            return
            
        # Start recording
        self.recording = True
        self.frames_recorded = 0
        self.start_time = time.time()
        
        # Update UI
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_label.configure(text="Recording...", foreground='red')
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.recording_thread.start()
        
        # Start time update
        self.update_recording_time()
        
    def recording_loop(self):
        """Main recording loop (high priority)"""
        try:
            while self.recording and self.cap:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Write frame to video file
                    self.video_writer.write(frame)
                    self.frames_recorded += 1
                    
                    # Update preview queue (lower priority)
                    if not self.preview_queue.full():
                        self.preview_queue.put(frame)
                        self.window.after(0, self.update_preview)
                        
                time.sleep(0.001)  # Minimal delay for high-priority recording
                
        except Exception as e:
            print(f"Recording loop error: {e}")
            self.window.after(0, self.stop_recording)
            
    def update_recording_time(self):
        """Update recording time display"""
        if self.recording and self.start_time:
            elapsed = time.time() - self.start_time
            self.time_label.configure(text=f"Recording: {elapsed:.1f}s | Frames: {self.frames_recorded}")
            self.window.after(100, self.update_recording_time)
            
    def stop_recording(self):
        """Stop video recording"""
        self.recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        # Update UI
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.status_label.configure(text="Processing video...", foreground='orange')
        
        # Process the recorded video
        self.process_recorded_video()
        
    def process_recorded_video(self):
        """Process the recorded video with calibration corrections (only if calibration is available)"""
        # Check if any calibration corrections are available
        has_lens_correction = self.calibration_data and self.calibration_data.is_calibrated
        has_perspective_correction = self.calibration_data and self.calibration_data.perspective_corrected
        
        if not has_lens_correction and not has_perspective_correction:
            # No calibration corrections available - skip video processing
            self.status_label.configure(text="Recording complete! (No calibration processing needed)", foreground='green')
            
            messagebox.showinfo("Recording Complete", 
                              f"Video saved: {self.output_path}\n\n"
                              f"No calibration corrections were applied as no calibration data is available.\n"
                              f"Recorded {self.frames_recorded} frames.")
            
            # Call completion callback with original video
            if self.completion_callback:
                self.completion_callback(self.output_path)
                
            return
        
        try:
            # Create output filename. Use uppercase _CALIBRATED suffix so the
            # tracker's pre-correction detection (filename + sidecar) matches
            # the same convention as videos produced by the tracker itself.
            base_name = os.path.splitext(self.output_path)[0]
            corrected_path = f"{base_name}_CALIBRATED.avi"
            metadata_path = f"{base_name}_CALIBRATED_metadata.json"
            
            # Open the recorded video
            input_video = cv2.VideoCapture(self.output_path)
            if not input_video.isOpened():
                raise Exception("Failed to open recorded video")
                
            # Get video properties
            fps = input_video.get(cv2.CAP_PROP_FPS)
            total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Read first frame to determine output size
            ret, first_frame = input_video.read()
            if not ret:
                raise Exception("Failed to read video frames")
                
            # Apply corrections to determine output size
            corrected_frame = self.apply_calibration_corrections(first_frame)
            output_height, output_width = corrected_frame.shape[:2]
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_video = cv2.VideoWriter(corrected_path, fourcc, fps, (output_width, output_height))
            
            if not output_video.isOpened():
                raise Exception("Failed to create corrected video file")
                
            # Reset to beginning
            input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process all frames
            frame_count = 0
            while True:
                ret, frame = input_video.read()
                if not ret:
                    break
                    
                # Apply calibration corrections
                corrected_frame = self.apply_calibration_corrections(frame)
                output_video.write(corrected_frame)
                
                frame_count += 1
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                self.status_label.configure(text=f"Processing: {progress:.1f}%")
                self.window.update_idletasks()
                
            # Cleanup
            input_video.release()
            output_video.release()

            # Sidecar: persist the calibration state used to bake corrections
            # into this video so downstream tools (tracker) can reconstruct
            # metric-space measurements and skip re-applying corrections.
            self._save_calibration_metadata(metadata_path)

            # Update status
            self.status_label.configure(text="Processing complete!", foreground='green')

            # Show completion message
            messagebox.showinfo("Video Processing Complete",
                              f"Original video: {self.output_path}\n"
                              f"Corrected video: {corrected_path}\n\n"
                              f"Processed {frame_count} frames with lens and perspective corrections.")
            
            # Call completion callback
            if self.completion_callback:
                self.completion_callback(corrected_path)
                
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process video: {str(e)}")
            self.status_label.configure(text="Processing failed", foreground='red')
            
    def apply_calibration_corrections(self, frame):
        """Apply lens distortion and perspective corrections to a frame."""
        return apply_corrections(frame, self.calibration_data)

    def _save_calibration_metadata(self, metadata_path):
        """Write a sidecar JSON describing the calibration baked into the
        corrected video. Schema matches tracking_v7._save_calibration_metadata
        so a single loader can read either source."""
        cd = self.calibration_data
        metadata = {
            "version": "1.1",
            "calibration_applied": True,
            "lens_correction": bool(getattr(cd, 'is_calibrated', False)),
            "lens_model": getattr(cd, 'model_type', 'pinhole'),
            "fisheye_balance": float(getattr(cd, 'fisheye_balance', 0.0)),
            "perspective_correction": bool(getattr(cd, 'perspective_corrected', False)),
            "real_world_scale": bool(getattr(cd, 'real_world_scale', False)),
            "square_size_real": getattr(cd, 'square_size_real', None),
            "square_size_pixels": getattr(cd, 'square_size_pixels', None),
            "pixels_per_real_unit": getattr(cd, 'pixels_per_real_unit', 1.0),
            "coordinate_units": cd.get_coordinate_units() if hasattr(cd, 'get_coordinate_units') else "pixels",
            # Pixel position of world (0,0) in the output frame; non-zero when
            # the perspective warp shifted content to expose negative-coord space.
            "perspective_translation_x": float(getattr(cd, 'perspective_translation_x', 0.0)),
            "perspective_translation_y": float(getattr(cd, 'perspective_translation_y', 0.0)),
            # Linear downsample factor applied to the perspective warp output
            # (physical canvas = logical canvas / output_scale).  The
            # pixels_per_real_unit value above already reflects this scale,
            # so the tracker doesn't need to compensate further; recorded
            # for documentation/recovery.
            "output_scale": float(getattr(cd, 'output_scale', 1.0)),
            # User-defined output region in pre-flip world coords at native
            # homography scale: [x_min, y_min, x_max, y_max].  Defines the
            # canvas extent the perspective warp produces.  None if perspective
            # correction wasn't applied to this video.
            "output_bbox_world": (
                list(cd.output_bbox_world)
                if getattr(cd, 'output_bbox_world', None) is not None
                else None
            ),
            # Coordinate-frame orientation state (0–7) — see
            # data_models.FRAME_ORIENTATION_BASES.  Affects the export
            # frame only, not the displayed image.
            "frame_orientation_state": int(
                getattr(cd, 'frame_orientation_state', 0)),
            "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "measurement_recorder",
        }

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Calibration metadata saved: {metadata_path}")
        except Exception as e:
            print(f"Warning: could not save calibration metadata: {e}")

    def close_window(self):
        """Close recording window"""
        if self.recording:
            self.stop_recording()
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        if self.window:
            self.window.destroy()
            
    def on_window_close(self):
        """Handle window close event"""
        if self.recording:
            result = messagebox.askyesno("Recording in Progress", 
                                       "Recording is in progress. Stop recording and close?")
            if not result:
                return
                
        self.close_window()
