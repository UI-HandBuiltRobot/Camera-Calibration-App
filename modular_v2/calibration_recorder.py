"""
Calibration video recording for Added-Mass-Lab GUI
Handles camera calibration video recording with real-time checkerboard detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import cv2
import numpy as np
import os
import subprocess
import platform

from .calibration_processor import CHECKERBOARD_SIZES
from PIL import Image, ImageTk
from .config import (
    VISUAL_CUEING_ENABLED, CUEING_GRID_SIZE, CUEING_POINT_THRESHOLDS,
    CUEING_OVERLAY_ALPHA, CUEING_COLORS
)


class CalibrationRecorder:
    """Handles camera calibration video recording with checkerboard detection"""
    
    def __init__(self, parent, camera_manager):
        self.parent = parent
        self.camera_manager = camera_manager
        self.window = None
        self.preview_label = None
        
        # Recording state
        self.preview_active = False
        self.recording = False
        self.recording_thread = None
        self.preview_thread = None
        self.capture_thread = None
        
        # Camera and recording
        self.cap = None
        self.video_writer = None
        self.output_path = None
        
        # UI elements
        self.start_record_button = None
        self.stop_record_button = None
        self.mirror_var = None  # Checkbox variable for video mirroring
        
        # Display options
        self.mirror_video = False  # Whether to mirror the video display
        self.last_frame = None  # Store last frame for resize events
        
        # Frame queues for threading
        self.frame_queue = queue.Queue(maxsize=60)  # Buffer for frames
        self.preview_queue = queue.Queue(maxsize=5)  # Latest frames for preview
        
        # Checkerboard detection settings.
        # Live preview prefers the standard (9,6) board. The smaller boards
        # in the canonical list (8,6 / 7,5 / 6,4) reliably false-positive
        # on subsets of a real (9,6) board, so we only fall back to the
        # broader list if (9,6) has been missing for several seconds — and
        # once a non-preferred size is detected we lock onto it to avoid
        # thrashing back and forth on subsequent frames. The post-recording
        # processor still considers every size in CHECKERBOARD_SIZES.
        self.default_checkerboard = (9, 6)
        self.fallback_sizes = [s for s in CHECKERBOARD_SIZES
                               if s != self.default_checkerboard]
        self._preferred_grace_seconds = 5.0
        self._lock_release_seconds = 2.0
        self._first_detection_attempt_time = None
        self._locked_size = None
        self._last_lock_detection_time = None
        self.detected_size = None
        self.detection_overlay = None
        
        # Visual cueing system for sector-based point tracking
        self.visual_cueing_enabled = VISUAL_CUEING_ENABLED
        self.grid_size = CUEING_GRID_SIZE
        self.point_thresholds = CUEING_POINT_THRESHOLDS
        self.overlay_alpha = CUEING_OVERLAY_ALPHA
        self.cueing_colors = CUEING_COLORS
        self.sector_point_counts = {}  # Track points per sector
        self.frame_resolution = None  # Will be set when camera initializes
        
        # Completion callback
        self.completion_callback = None
        
        # Automatic recording with countdown
        self.auto_recording_enabled = True
        self.countdown_active = False
        self.countdown_start_time = None
        self.countdown_duration = 3  # 3 seconds
        self.temp_video_path = None
        self.last_detection_time = None
        self.detection_timeout = 0.5  # 0.5 seconds without detection resets countdown
        
    def show_recording_dialog(self):
        """Show dialog to start calibration recording preview"""
        if self.camera_manager.selected_camera_id is None:
            messagebox.showerror("Error", "Please select a camera first.")
            return
            
        # Start with preview mode (no recording yet)
        self.start_preview()
        
    def start_preview(self):
        """Start the calibration preview (camera setup without recording)"""
        # Initialize camera with stored settings
        camera_id = self.camera_manager.selected_camera_id
        resolution = self.camera_manager.selected_resolution
        framerate = self.camera_manager.selected_framerate
        
        # Setup camera
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open camera {camera_id}")
            return
            
        # Set MJPEG codec first
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Set resolution and framerate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, framerate)
        
        # Set MJPEG codec again to ensure it's applied
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Preview setup: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Store actual settings for later recording
        self.actual_resolution = (actual_width, actual_height)
        self.actual_fps = actual_fps
        
        # Create preview window
        self.create_preview_window()
        
        # Start preview threads (no recording yet)
        self.preview_active = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        
        self.capture_thread.start()
        self.preview_thread.start()
        
        print("Calibration preview started")
        
    def start_recording_from_preview(self):
        """Start automatic recording to temporary file (called internally by countdown)"""
        if not self.preview_active:
            return
            
        # Create temporary file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_video_path = os.path.join(script_dir, "Temp_calibration.avi")
        
        # Remove existing temp file if it exists
        if os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
            except Exception as e:
                print(f"Warning: Could not remove existing temp file: {e}")
        
        self.output_path = self.temp_video_path
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.actual_fps, 
            self.actual_resolution
        )
        
        if not self.video_writer.isOpened():
            messagebox.showerror("Error", "Failed to create video writer")
            return
            
        # Initialize visual cueing system for recording
        if self.visual_cueing_enabled:
            self.frame_resolution = self.actual_resolution
            self.initialize_sector_tracking()
            
        # Start recording
        self.recording = True
        self.countdown_active = False  # Reset countdown state
        self.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.recording_thread.start()
        
        # Update UI
        self.update_recording_ui(True)
        
        print("Automatic recording started")
        
    def stop_recording_keep_preview(self):
        """Stop recording and prompt for final filename"""
        if self.recording:
            self.recording = False
            
            # Wait for recording thread to finish
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
                
            # Clean up video writer
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
            # Update UI
            self.update_recording_ui(False)
            
            print(f"Recording stopped. Temp file saved to: {self.output_path}")
            # Stop preview and camera immediately so automatic detection
            # cannot start a new recording while the user is choosing a filename.
            # We call stop_preview() here (recording already set to False) which
            # will stop capture/preview threads, release the camera and close
            # the preview window.
            try:
                self.stop_preview()
            except Exception:
                # If stopping preview fails for any reason, continue to prompt
                # the user for saving the file (best-effort cleanup)
                pass

            # Prompt user for final filename (parent the dialog to the main app)
            final_path = filedialog.asksaveasfilename(
                parent=self.parent,
                title="Save Calibration Video As",
                defaultextension=".avi",
                filetypes=[("AVI files", "*.avi"), ("All files", "*.*")],
                initialfile="calibration_video.avi"
            )
            
            if final_path and self.temp_video_path and os.path.exists(self.temp_video_path):
                try:
                    # Rename temp file to final filename
                    if os.path.exists(final_path):
                        os.remove(final_path)  # Remove existing file if present
                    os.rename(self.temp_video_path, final_path)
                    saved_video_path = final_path
                    print(f"Video renamed to: {final_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to rename video file:\n{str(e)}")
                    saved_video_path = self.temp_video_path  # Fall back to temp path
            else:
                # User cancelled or no temp file exists, keep temp file
                saved_video_path = self.temp_video_path if self.temp_video_path else self.output_path
            
            # Close the preview window
            self.stop_preview()
            
            # Call completion callback if set to continue with calibration processing
            if hasattr(self, 'completion_callback') and self.completion_callback:
                self.completion_callback(saved_video_path)
            else:
                # Fallback - show completion message
                messagebox.showinfo("Recording Complete", f"Calibration video saved successfully!\n\nFile: {saved_video_path}")
                
            
    def open_checkerboard_image(self, event):
        """Open the checkerboard image for printing/viewing"""
        try:
            # Get the path to the checkerboard PDF
            script_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(script_dir, "checkerboard_9x6.pdf")
            
            # Check if file exists
            if not os.path.exists(image_path):
                messagebox.showerror("File Not Found", 
                    f"Checkerboard PDF not found at:\n{image_path}\n\n"
                    "Please ensure the checkerboard_9x6.pdf file is in the same directory as this application.")
                return
            
            # Open the PDF with the default system viewer
            if platform.system() == 'Windows':
                os.startfile(image_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', image_path])
            else:  # Linux and others
                subprocess.run(['xdg-open', image_path])
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open checkerboard image:\n{str(e)}")
    
    def toggle_mirror(self):
        """Toggle video mirroring on/off"""
        self.mirror_video = self.mirror_var.get()
        print(f"Video mirroring {'enabled' if self.mirror_video else 'disabled'}")
            
    def stop_preview(self):
        """Stop preview and close window"""
        # Stop recording if active
        if self.recording:
            self.stop_recording_keep_preview()
            
        # Reset countdown state
        self.countdown_active = False
        self.countdown_start_time = None
        self.last_detection_time = None
            
        # Stop preview
        self.preview_active = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.preview_thread:
            self.preview_thread.join(timeout=2.0)
            
        # Clean up camera
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Close window
        if self.window:
            self.window.destroy()
            self.window = None
            
        print("Preview stopped")
        
    def create_preview_window(self):
        """Create the preview window with start/stop recording controls"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Camera Calibration - Preview & Recording")
        self.window.geometry("950x850")
        self.window.protocol("WM_DELETE_WINDOW", self.stop_preview)
        
        # Bind window resize events for dynamic preview scaling
        self.window.bind('<Configure>', self.on_window_resize)
        
        # Instructions frame
        instruction_frame = ttk.LabelFrame(self.window, text="Instructions", padding="10")
        instruction_frame.pack(fill='x', padx=10, pady=5)
        
        instructions = [
            "1. Position the checkerboard in the camera's field of view",
            "2. Recording will begin automatically when a valid checkerboard is detected",
            "3. Move the checkerboard slowly around the camera's field of view while recording",
            "4. Move the board until all areas of the screen are yellow (good) or green (best)",
            "5. Try different angles and distances for 30-60 seconds",
            "6. Click 'STOP RECORDING' when finished to save the video"
        ]
        
        for instruction in instructions:
            ttk.Label(instruction_frame, text=instruction).pack(anchor='w')
        
        # Add spacing
        ttk.Label(instruction_frame, text="").pack()
        
        # Add hyperlink for checkerboard image
        link_frame = ttk.Frame(instruction_frame)
        link_frame.pack(fill='x')
        
        checkerboard_link = tk.Label(link_frame, text="Click HERE to open/print an image of the recommended checkerboard", 
                                   fg="blue", cursor="hand2", font=('Arial', 9, 'underline'))
        checkerboard_link.pack(anchor='w')
        checkerboard_link.bind("<Button-1>", self.open_checkerboard_image)
        
        # Add video options frame
        options_frame = ttk.LabelFrame(self.window, text="Display Options", padding="10")
        options_frame.pack(fill='x', padx=10, pady=5)
        
        # Mirror video checkbox
        self.mirror_var = tk.BooleanVar()
        self.mirror_checkbox = ttk.Checkbutton(options_frame, text="Mirror video display", 
                                          variable=self.mirror_var, command=self.toggle_mirror)
        self.mirror_checkbox.pack(anchor='w')
        self.mirror_checkbox.configure(state='normal')  # Enable when preview is active
        # Tooltip explaining mirroring option
        try:
            from .tooltip import ToolTip
            ToolTip(self.mirror_checkbox, "Mirror the video preview (useful when the camera is facing you)")
        except Exception:
            # Tooltip helper may be unavailable in some contexts; ignore silently
            pass
            
        # Status frame
        status_frame = ttk.Frame(self.window)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Waiting for checkerboard...", 
                                     font=('Arial', 12, 'bold'), foreground='blue')
        self.status_label.pack(side='left')
        
        self.detection_label = ttk.Label(status_frame, text="No checkerboard detected")
        self.detection_label.pack(side='right')
        
        # Control buttons frame - positioned above preview for visibility
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        # Large, prominent Stop Recording button (initially disabled)
        self.stop_record_button = tk.Button(
            button_frame, 
            text="STOP RECORDING", 
            command=self.stop_recording_keep_preview,
            font=('Arial', 14, 'bold'),
            bg='red',
            fg='white',
            height=2,
            width=20,
            state='disabled'
        )
        self.stop_record_button.pack(side='left', padx=10)
        
        # Close Preview button
        ttk.Button(
            button_frame, 
            text="Close Preview", 
            command=self.stop_preview
        ).pack(side='right', padx=5)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(self.window, text="Camera Preview")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(expand=True)
        
    def update_recording_ui(self, recording_active):
        """Update UI elements based on recording state"""
        if recording_active:
            self.status_label.config(text="Recording...", foreground='red')
            self.stop_record_button.config(state='normal', bg='darkred')
        else:
            self.status_label.config(text="Waiting for checkerboard...", foreground='blue')
            self.stop_record_button.config(state='disabled', bg='lightgray')
    
    def on_window_resize(self, event):
        """Handle window resize events to update preview scaling"""
        # Only respond to resize events for the main window, not child widgets
        if event.widget == self.window:
            # Trigger preview update if we have a last frame
            if hasattr(self, 'last_frame') and self.last_frame is not None:
                self.window.after(10, self.update_preview_scaling)
    
    def get_preview_pane_size(self):
        """Get the current available size of the preview pane"""
        try:
            # Find the preview frame (LabelFrame with "Camera Preview" text)
            for widget in self.window.winfo_children():
                if isinstance(widget, ttk.LabelFrame) and "Camera Preview" in str(widget.cget('text')):
                    # Update the widget to get current size
                    widget.update_idletasks()
                    # Account for labelframe padding and borders
                    available_width = widget.winfo_width() - 20  # margin for borders
                    available_height = widget.winfo_height() - 40  # margin for label and borders
                    return max(100, available_width), max(100, available_height)  # minimum size
        except:
            pass
        # Fallback to default size
        return 800, 500
    
    def update_preview_scaling(self):
        """Update preview image scaling based on current window size"""
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            self.update_preview_gui(self.last_frame, "", "blue")
                  
    def capture_loop(self):
        """Main camera capture loop - maintains 30fps for both preview and recording"""
        frame_interval = 1.0 / 30.0  # Target 30 FPS
        last_time = time.time()
        
        while self.preview_active and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                current_time = time.time()
                
                # Add frame to recording queue only if recording is active
                if self.recording:
                    try:
                        self.frame_queue.put(frame.copy(), timeout=0.1)
                    except queue.Full:
                        # Skip frame if recording can't keep up
                        pass
                    
                # Always add frame to preview queue
                try:
                    self.preview_queue.put_nowait(frame.copy())
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        self.preview_queue.get_nowait()
                        self.preview_queue.put_nowait(frame.copy())
                    except queue.Empty:
                        pass
                
                # Maintain frame rate
                elapsed = current_time - last_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()
            else:
                print("Failed to capture frame")
                break
                
    def recording_loop(self):
        """Recording loop - writes frames to disk"""
        while self.recording:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if self.video_writer and frame is not None:
                    self.video_writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Recording error: {e}")
                break
                
    def preview_loop(self):
        """Preview loop with checkerboard detection - runs continuously during preview"""
        while self.preview_active:
            try:
                frame = self.preview_queue.get(timeout=1.0)
                if frame is not None:
                    # Detect checkerboard
                    overlay_frame, detected = self.detect_checkerboard(frame)
                    
                    # Update detection status
                    if detected:
                        status_text = f"Checkerboard detected: {self.detected_size[0]}x{self.detected_size[1]}"
                        color = 'green'
                    else:
                        status_text = "No checkerboard detected"
                        color = 'orange'
                        
                    # Update GUI in main thread
                    if self.window and self.window.winfo_exists():
                        self.window.after(0, self.update_preview_gui, overlay_frame, status_text, color)
                        
                self.preview_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Preview error: {e}")
                break
    
    def initialize_sector_tracking(self):
        """Initialize the sector tracking system for visual cueing"""
        if not self.frame_resolution:
            return
            
        # Initialize point counts for each sector (4x4 grid = 16 sectors)
        rows, cols = self.grid_size
        self.sector_point_counts = {}
        
        for row in range(rows):
            for col in range(cols):
                sector_key = (row, col)
                self.sector_point_counts[sector_key] = 0
        
        print(f"Visual cueing initialized: {rows}x{cols} grid, thresholds {self.point_thresholds}")
    
    def get_sector_for_point(self, x, y):
        """Determine which sector a point belongs to"""
        if not self.frame_resolution:
            return None
            
        width, height = self.frame_resolution
        rows, cols = self.grid_size
        
        # Calculate sector dimensions
        sector_width = width / cols
        sector_height = height / rows
        
        # Determine sector indices
        col_idx = min(int(x / sector_width), cols - 1)
        row_idx = min(int(y / sector_height), rows - 1)
        
        return (row_idx, col_idx)
    
    def update_sector_counts(self, corners):
        """Update sector point counts based on detected corners"""
        if not self.visual_cueing_enabled or not corners.any():
            return
            
        for corner in corners:
            x, y = corner.ravel()
            sector = self.get_sector_for_point(x, y)
            if sector and sector in self.sector_point_counts:
                self.sector_point_counts[sector] += 1
    
    def draw_sector_overlay(self, frame):
        """Draw the visual cueing overlay on the frame"""
        if not self.visual_cueing_enabled or not self.frame_resolution:
            return frame
            
        overlay_frame = frame.copy()
        width, height = self.frame_resolution
        rows, cols = self.grid_size
        
        # Calculate sector dimensions
        sector_width = width / cols
        sector_height = height / rows
        
        # Draw grid sectors with color coding
        for row in range(rows):
            for col in range(cols):
                sector_key = (row, col)
                point_count = self.sector_point_counts.get(sector_key, 0)
                
                # Determine color based on point count
                if point_count >= self.point_thresholds['green']:
                    color = self.cueing_colors['green']
                elif point_count >= self.point_thresholds['yellow']:
                    color = self.cueing_colors['yellow']
                else:
                    color = self.cueing_colors['red']
                
                # Calculate sector boundaries
                x1 = int(col * sector_width)
                y1 = int(row * sector_height)
                x2 = int((col + 1) * sector_width)
                y2 = int((row + 1) * sector_height)
                
                # Create colored overlay for this sector
                sector_overlay = overlay_frame[y1:y2, x1:x2].copy()
                colored_overlay = np.full_like(sector_overlay, color, dtype=np.uint8)
                
                # Blend with original frame
                blended_sector = cv2.addWeighted(
                    sector_overlay, 1 - self.overlay_alpha,
                    colored_overlay, self.overlay_alpha,
                    0
                )
                
                # Place blended sector back in frame
                overlay_frame[y1:y2, x1:x2] = blended_sector
                
                # Draw colorblind-friendly symbols at center of each sector
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                symbol_size = min(sector_width, sector_height) * 0.3  # Scale symbol to sector size
                
                if point_count >= self.point_thresholds['green']:
                    # Draw ✔+ (double checkmark) for high point count
                    self._draw_double_checkmark(overlay_frame, center_x, center_y, symbol_size)
                elif point_count >= self.point_thresholds['yellow']:
                    # Draw ✔ (single checkmark) for medium point count
                    self._draw_checkmark(overlay_frame, center_x, center_y, symbol_size)
                else:
                    # Draw X for low/no point count
                    self._draw_x_symbol(overlay_frame, center_x, center_y, symbol_size)
                
                # Draw sector boundaries
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                
                # Add point count text (optional - for debugging)
                if point_count > 0:
                    text_x = x1 + 10
                    text_y = y1 + 25
                    cv2.putText(overlay_frame, str(point_count), (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return overlay_frame
    
    def _draw_x_symbol(self, frame, center_x, center_y, size):
        """Draw an X symbol for insufficient calibration points"""
        # Use white color with black outline for high visibility
        color = (255, 255, 255)  # White
        outline_color = (0, 0, 0)  # Black outline
        thickness = max(2, int(size / 10))  # Thickness scales with size
        outline_thickness = thickness + 1
        
        # Calculate X endpoints
        half_size = int(size / 2)
        
        # Draw black outline first
        cv2.line(frame, 
                (center_x - half_size, center_y - half_size),
                (center_x + half_size, center_y + half_size),
                outline_color, outline_thickness)
        cv2.line(frame,
                (center_x - half_size, center_y + half_size),
                (center_x + half_size, center_y - half_size),
                outline_color, outline_thickness)
        
        # Draw white X on top
        cv2.line(frame,
                (center_x - half_size, center_y - half_size),
                (center_x + half_size, center_y + half_size),
                color, thickness)
        cv2.line(frame,
                (center_x - half_size, center_y + half_size),
                (center_x + half_size, center_y - half_size),
                color, thickness)
    
    def _draw_checkmark(self, frame, center_x, center_y, size):
        """Draw a checkmark symbol for adequate calibration points"""
        # Use white color with black outline for high visibility
        color = (255, 255, 255)  # White
        outline_color = (0, 0, 0)  # Black outline
        thickness = max(2, int(size / 8))
        outline_thickness = thickness + 1
        
        # Calculate checkmark points (a simple checkmark shape)
        quarter_size = int(size / 4)
        half_size = int(size / 2)
        
        # Checkmark consists of two lines forming a "v" shape
        point1 = (center_x - quarter_size, center_y)
        point2 = (center_x - int(quarter_size / 2), center_y + quarter_size)
        point3 = (center_x + half_size, center_y - quarter_size)
        
        # Draw black outline first
        cv2.line(frame, point1, point2, outline_color, outline_thickness)
        cv2.line(frame, point2, point3, outline_color, outline_thickness)
        
        # Draw white checkmark on top
        cv2.line(frame, point1, point2, color, thickness)
        cv2.line(frame, point2, point3, color, thickness)
    
    def _draw_double_checkmark(self, frame, center_x, center_y, size):
        """Draw a double checkmark symbol for optimal calibration points"""
        # Draw two offset checkmarks for "premium" indicator
        offset = int(size / 8)
        
        # Draw first checkmark slightly to the left
        self._draw_checkmark(frame, center_x - offset, center_y, size * 0.9)
        # Draw second checkmark slightly to the right  
        self._draw_checkmark(frame, center_x + offset, center_y, size * 0.9)
                
    def detect_checkerboard(self, frame):
        """Detect checkerboard in frame and handle automatic recording with countdown"""
        # Apply mirroring if enabled (for display purposes only)
        if self.mirror_video:
            frame = cv2.flip(frame, 1)  # Horizontal flip
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overlay_frame = frame.copy()
        detected = False
        corners_found = None
        current_time = time.time()
        
        # Decide which sizes to try this frame.
        # Default behaviour: only try (9,6). Expand to fallbacks only if (9,6)
        # has not detected for self._preferred_grace_seconds. If a fallback
        # size has previously locked on, only that size is tried until the
        # lock expires.
        if self._first_detection_attempt_time is None:
            self._first_detection_attempt_time = current_time

        if self._locked_size is not None:
            # Release the lock if we've lost the locked size for too long.
            if (self._last_lock_detection_time is not None
                and current_time - self._last_lock_detection_time
                > self._lock_release_seconds):
                self._locked_size = None
                self._last_lock_detection_time = None
                sizes_to_try = [self.default_checkerboard]
            else:
                sizes_to_try = [self._locked_size]
        else:
            sizes_to_try = [self.default_checkerboard]
            grace_elapsed = (current_time
                             - self._first_detection_attempt_time)
            no_recent_detection = (
                self.last_detection_time is None
                or (current_time - self.last_detection_time)
                > self._preferred_grace_seconds
            )
            if (grace_elapsed > self._preferred_grace_seconds
                    and no_recent_detection):
                sizes_to_try = sizes_to_try + self.fallback_sizes

        for size in sizes_to_try:
            ret, corners = cv2.findChessboardCorners(gray, size, None)
            
            if ret:
                self.detected_size = size
                detected = True
                corners_found = corners
                self.last_detection_time = current_time

                # Lock onto non-preferred sizes so we stop trying the
                # preferred (9,6) on every frame; release any lock once
                # the preferred size detects again.
                if size != self.default_checkerboard:
                    self._locked_size = size
                    self._last_lock_detection_time = current_time
                else:
                    self._locked_size = None
                    self._last_lock_detection_time = None
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Update sector counts if recording (visual cueing only during recording)
                if self.recording and self.visual_cueing_enabled:
                    self.update_sector_counts(corners)
                
                # Draw corners
                cv2.drawChessboardCorners(overlay_frame, size, corners, ret)
                # Enhance visibility: draw enlarged, colorful circles at each corner
                # and draw heavier grid lines between adjacent corners.
                try:
                    corner_points = corners.reshape(-1, 2)
                    # Make points roughly 3x larger and lines heavier
                    radius = 9
                    circle_color = (0, 255, 255)  # Yellow-ish for corners
                    line_color = (0, 255, 0)      # Green for connecting lines
                    circle_thickness = 3
                    line_thickness = 3

                    # Draw circles at each corner
                    for pt in corner_points:
                        x, y = int(round(pt[0])), int(round(pt[1]))
                        cv2.circle(overlay_frame, (x, y), radius, circle_color, circle_thickness)

                    # Draw grid lines between adjacent corners if the expected grid size matches
                    rows, cols = size
                    if corner_points.shape[0] == rows * cols:
                        for r in range(rows):
                            for c in range(cols):
                                idx = r * cols + c
                                x1, y1 = map(int, corner_points[idx])
                                # horizontal neighbor
                                if c < cols - 1:
                                    x2, y2 = map(int, corner_points[idx + 1])
                                    cv2.line(overlay_frame, (x1, y1), (x2, y2), line_color, line_thickness)
                                # vertical neighbor
                                if r < rows - 1:
                                    x2, y2 = map(int, corner_points[idx + cols])
                                    cv2.line(overlay_frame, (x1, y1), (x2, y2), line_color, line_thickness)
                except Exception as e:
                    print(f"Overlay draw error: {e}")
                break
        
        # Handle automatic recording logic if not already recording
        if not self.recording and self.auto_recording_enabled:
            if detected:
                # Checkerboard detected - handle countdown
                if not self.countdown_active:
                    # Start countdown
                    self.countdown_active = True
                    self.countdown_start_time = current_time
                    print("Checkerboard detected - starting 3-second countdown")
                else:
                    # Countdown in progress - check if it should complete
                    countdown_elapsed = current_time - self.countdown_start_time
                    if countdown_elapsed >= self.countdown_duration:
                        # Countdown complete - start recording
                        print("Countdown complete - starting automatic recording")
                        self.start_recording_from_preview()
                        
            else:
                # No checkerboard detected - check if we should reset countdown
                if self.countdown_active and self.last_detection_time:
                    time_since_detection = current_time - self.last_detection_time
                    if time_since_detection > self.detection_timeout:
                        # Lost tracking - reset countdown
                        self.countdown_active = False
                        self.countdown_start_time = None
                        print("Tracking lost - resetting countdown")
        
        # Add appropriate text overlays based on state
        self.add_status_overlays(overlay_frame, detected, current_time)
        
        # Apply visual cueing overlay if recording
        if self.recording and self.visual_cueing_enabled:
            overlay_frame = self.draw_sector_overlay(overlay_frame)
            
            # Add cueing legend
            self.draw_cueing_legend(overlay_frame)
                
        return overlay_frame, detected
        
    def add_status_overlays(self, frame, detected, current_time):
        """Add status text overlays to the frame"""
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if self.recording:
            # Recording active
            text = "RECORDING..."
            font_scale = 1.5
            color = (0, 0, 255)  # Red
            thickness = 3
            
            # Get text size and center it
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = 60
            
            # Add background rectangle
            cv2.rectangle(frame, (text_x - 20, text_y - 35), (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
            
        elif self.countdown_active and detected:
            # Countdown active
            countdown_elapsed = current_time - self.countdown_start_time
            remaining = max(0, self.countdown_duration - countdown_elapsed)
            countdown_text = f"Recording in: {remaining:.1f}s"
            
            font_scale = 2.0  # Increased from 1.2 to 2.0
            color = (0, 255, 255)  # Yellow
            thickness = 3  # Increased from 2 to 3
            
            # Get text size and center it
            text_size = cv2.getTextSize(countdown_text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = 80  # Moved down slightly to accommodate larger text
            
            # Add background rectangle
            cv2.rectangle(frame, (text_x - 20, text_y - 40), (text_x + text_size[0] + 20, text_y + 15), (0, 0, 0), -1)
            cv2.putText(frame, countdown_text, (text_x, text_y), font, font_scale, color, thickness)
            
        else:
            # Waiting for checkerboard
            text = "Recording will begin automatically when a valid checkerboard is detected"
            font_scale = 1.2  # Increased from 0.8 to 1.2
            color = (255, 255, 255)  # White
            thickness = 3  # Increased from 2 to 3
            
            # Split text into multiple lines if too long
            max_width = width - 40
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = " ".join(current_line + [word])
                text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
                if text_size[0] > max_width and current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # Draw each line
            line_height = 40  # Increased from 30 to 40 for larger font
            start_y = 50  # Increased from 40 to 50 for better positioning
            
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = start_y + (i * line_height)
                
                # Add background rectangle (larger for bigger text)
                cv2.rectangle(frame, (text_x - 15, text_y - 30), (text_x + text_size[0] + 15, text_y + 10), (0, 0, 0), -1)
                cv2.putText(frame, line, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add checkerboard detection status
        if detected:
            status_text = f"Checkerboard: {self.detected_size[0]}x{self.detected_size[1]}"
            cv2.putText(frame, status_text, (10, height - 30), font, 0.7, (0, 255, 0), 2)
        
    def draw_cueing_legend(self, frame):
        """Draw legend for visual cueing system"""
        height, width = frame.shape[:2]
        legend_y_start = height - 120
        
        # Legend background (semi-transparent)
        legend_overlay = frame.copy()
        cv2.rectangle(legend_overlay, (width - 280, legend_y_start - 10), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(legend_overlay, 0.7, frame, 0.3, 0, frame)
        
        # Legend text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Title
        cv2.putText(frame, "Coverage Guide:", (width - 270, legend_y_start + 15), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Color indicators
        legend_items = [
            ("Red: < 150 points", (0, 0, 255)),
            ("Yellow: 150-299 points", (0, 255, 255)),
            ("Green: 300+ points", (0, 255, 0))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y_start + 35 + (i * 20)
            # Color square
            cv2.rectangle(frame, (width - 270, y_pos - 8), (width - 250, y_pos + 5), color, -1)
            # Text
            cv2.putText(frame, text, (width - 245, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        # Total coverage info
        total_corners = sum(self.sector_point_counts.values())
        coverage_text = f"Total: {total_corners} corners"
        cv2.putText(frame, coverage_text, (width - 270, legend_y_start + 100), 
                   font, font_scale, (255, 255, 255), thickness)
        
    def update_preview_gui(self, frame, status_text, status_color):
        """Update preview GUI in main thread with dynamic sizing"""
        try:
            if self.preview_label:
                # Store last frame for resize events
                self.last_frame = frame.copy()
                
                # Get current available space in preview pane
                available_width, available_height = self.get_preview_pane_size()
                
                # Get original frame dimensions
                height, width = frame.shape[:2]
                
                # Calculate scale to fit both width and height within available space
                width_scale = available_width / float(width)
                height_scale = available_height / float(height)
                scale = min(width_scale, height_scale, 1.0)  # Don't upscale
                
                # Calculate new dimensions
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                frame_resized = cv2.resize(frame, (new_width, new_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update preview
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo
                
                # Update status
                if hasattr(self, 'detection_label'):
                    self.detection_label.configure(text=status_text, foreground=status_color)
                    
        except Exception as e:
            print(f"Error updating preview GUI: {e}")
            
    def cleanup(self):
        """Clean up resources"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.preview_queue.empty():
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                break
