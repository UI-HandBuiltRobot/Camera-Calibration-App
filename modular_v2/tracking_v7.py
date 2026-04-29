"""
tracking_v7.py - Compact high-performance tracking

Streamlined version of v6 with identical functionality:
- Native resolution ROI selection and tracking display
- Sequential frame reading for v4-level performance  
- 4:3 aspect ratio GUI display
- All v6 tracking parameters preserved
"""

import cv2
import numpy as np
import imutils
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import json
from PIL import Image, ImageTk

from .corrections import apply_corrections


class VideoTracker:
    """Compact high-performance video tracker"""
    
    def __init__(self, parent, calibration_data=None):
        self.parent = parent
        self.calibration_data = calibration_data  # Store calibration data
        self.window = None
        
        # Core parameters (identical to v4/v6)
        self.global_width = 600
        self.gui_display_width = 640  # Will be dynamically calculated
        self.max_canvas_width = 600   # Initial maximum canvas width for scaling
        self.max_canvas_height = 400  # Initial maximum canvas height for scaling
        
        # Video state
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.current_frame = None
        
        # Processing
        self.transform_matrix = None
        self.output_size = None

        # Per-video pre-correction state. True means the loaded video already
        # has that correction baked into its pixels, so applying it again
        # during tracking would distort an already-correct frame.
        self.video_lens_corrected = False
        self.video_perspective_corrected = False
        
        # Coordinate origin system (draggable reference frame)
        self.origin_x = 0  # Origin position in full-resolution coordinates
        self.origin_y = 0
        self.origin_dragging = False
        self.origin_drag_offset_x = 0
        self.origin_drag_offset_y = 0
        
        # Tracking - Updated for multi-object support
        self.trackers = []  # List of trackers for multiple objects
        self.tracking_active = False
        self.bboxes = []  # List of bounding boxes for multiple objects
        self.native_bboxes = []  # List of native resolution bounding boxes
        self.tracking_data = []  # Will store list of data for each object
        self.data_save_path = None
        self.data_saved = True  # Track whether current data has been saved
        self.tracking_start_frame = 0  # Frame index where tracking started
        self.object_colors = ['blue', 'red', 'grey', 'black', 'green', 'magenta', 'cyan', 'brown']
        
        # GUI
        self.video_canvas = None
        self.frame_slider = None
        self.play_button = None
        self.status_label = None
        self.progress_label = None
        self.playing = False
        
        # Plot elements
        self.plot_canvas = None
        self.plot_width = 600  # Increased width by 20% (500 * 1.2 = 600)
        self.plot_height = 500  # Increased for taller display area
        
        # Performance optimization variables
        self.last_plot_update_frame = -1
        self.plot_update_interval = 2  # Update plot every N frames instead of every frame
        self.gui_update_counter = 0     # Counter for batching GUI updates
        
        # Plot enable/disable checkbox variable
        self.plot_enabled = tk.BooleanVar(value=True)  # Default to enabled

    def show_tracking_window(self):
        """Main entry point"""
        # Get video and save path
        self.video_path = filedialog.askopenfilename(
            parent=self.parent, title="Select experiment video",
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")])
        if not self.video_path:
            return

        # --- Calibration check logic ---
        base, ext = os.path.splitext(self.video_path)
        calibrated_path = f"{base}_CALIBRATED{ext}"
        is_calibrated = self.video_path.upper().endswith("_CALIBRATED" + ext.upper())
        calibration_available = self.calibration_data is not None and getattr(self.calibration_data, "is_calibrated", False)

        if not is_calibrated:
            # Check if calibrated video exists
            if os.path.exists(calibrated_path):
                result = messagebox.askyesno(
                    "Calibrated Video Available",
                    f"It looks like you are processing a video without lens distortion or perspective correction, which may result in inaccurate locations.\n\nA calibrated version exists:\n{os.path.basename(calibrated_path)}\n\nWould you like to use the calibrated video instead?")
                if result:
                    self.video_path = calibrated_path
            else:
                # No calibrated video, but calibration data exists
                if calibration_available:
                    result = messagebox.askyesno(
                        "Calibration Correction Recommended",
                        f"It looks like you are processing a video without lens distortion or perspective correction, which may result in inaccurate locations.\n\nNo calibrated video exists, but calibration data is available.\n\nWould you like to automatically correct the video before tracking?")
                    if result:
                        # Perform correction using same method as measurement_recorder
                        try:
                            corrected_path = self._apply_calibration_corrections_to_video()
                            self.video_path = corrected_path
                        except Exception as e:
                            messagebox.showerror("Correction Error", f"Failed to correct video: {e}")
                            return
                else:
                    messagebox.showwarning(
                        "Uncalibrated Video",
                        "It looks like you are processing a video without lens distortion or perspective correction, which may result in inaccurate locations.\n\nNo calibration data is available, so tracking will proceed without correction.")

        # Determine whether the (possibly-swapped) video is already pre-corrected
        # so _apply_calibration_corrections does not double-apply during tracking.
        self.video_lens_corrected, self.video_perspective_corrected = (
            self._detect_video_pre_correction(self.video_path))
        if self.video_lens_corrected or self.video_perspective_corrected:
            print(f"Video has baked-in corrections "
                  f"(lens={self.video_lens_corrected}, "
                  f"perspective={self.video_perspective_corrected}); "
                  f"skipping those steps during tracking")

        # Try to load calibration metadata for pre-calibrated videos
        self._load_calibration_metadata(self.video_path)

        # Initialize video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not read video.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Setup processing pipeline (full-resolution tracking)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Use full resolution for tracking instead of global_width scaling
            full_height, full_width = frame.shape[:2]
            
            # Set transform matrix and output size for full resolution
            self.transform_matrix = np.eye(3, dtype=np.float32)
            self.output_size = (full_width, full_height)
            
            # Store full resolution dimensions for reference
            self.full_width = full_width
            self.full_height = full_height
            
            # Initialize coordinate origin at lower-left corner (full-resolution coordinates)
            self.origin_x = 0
            self.origin_y = full_height  # Lower-left corner in image coordinates

        self._create_gui()
        self._load_first_frame()
        
        # Schedule scrub to frame 0 after GUI creation
        self.window.after(50, self._post_create_scrub)
        
        # Schedule initial size update after GUI is fully displayed
        self.window.after(100, self._update_video_size_after_resize)

    def _create_gui(self):
        """Create streamlined GUI"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Object Tracking")
        self.window.geometry("1440x1040")  # Width increased by 20% (1200 * 1.2 = 1440), height 30% increase maintained
        self.window.minsize(1000, 700)  # Set minimum window size
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title and info
        ttk.Label(main_frame, text="Object Tracking", 
                  font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        info_frame = ttk.LabelFrame(main_frame, text="Video Information", padding="5")
        info_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(info_frame, text=f"File: {os.path.basename(self.video_path)}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Frames: {self.total_frames} | FPS: {self.fps:.2f}").pack(anchor='w')
        
        # Instructions button
        inst_frame = ttk.Frame(main_frame)
        inst_frame.pack(fill='x', pady=(0, 10))
        ttk.Button(inst_frame, text="📋 Instructions", command=self._show_instructions, width=15).pack(side='left')
        
        # Controls - compact horizontal layout
        ctrl_frame = ttk.LabelFrame(main_frame, text="Frame Controls", padding="5")
        ctrl_frame.pack(fill='x', pady=(0, 10))
        
        # Single horizontal row with all controls
        controls_row = ttk.Frame(ctrl_frame)
        controls_row.pack(fill='x')
        
        # Step buttons
        ttk.Button(controls_row, text="◀", command=self._step_back, width=3).pack(side='left', padx=(0, 5))
        ttk.Button(controls_row, text="▶", command=self._step_forward, width=3).pack(side='left', padx=(0, 5))
        
        # Frame label and slider
        ttk.Label(controls_row, text="Frame:").pack(side='left', padx=(10, 5))
        self.frame_slider = tk.Scale(controls_row, from_=0, to=self.total_frames-1,
                                    orient='horizontal', command=self._on_frame_change, length=200)
        self.frame_slider.pack(side='left', padx=(0, 10))
        
        # Progress display
        self.progress_label = ttk.Label(controls_row, text="0 / 0")
        self.progress_label.pack(side='left')
        
        # Tracking controls
        track_frame = ttk.LabelFrame(main_frame, text="Tracking", padding="10")
        track_frame.pack(fill='x', pady=(0, 10))
        
        track_btn_frame = ttk.Frame(track_frame)
        track_btn_frame.pack(fill='x')
        self.select_button = ttk.Button(track_btn_frame, text="Select Object(s)", 
                                       command=self._select_objects, width=15)
        self.select_button.pack(side='left', padx=(0, 10))
        self.track_button = ttk.Button(track_btn_frame, text="Start Tracking", 
                                      command=self._start_tracking, width=15, state='disabled')
        self.track_button.pack(side='left', padx=(0, 10))
        self.stop_button = ttk.Button(track_btn_frame, text="Stop Tracking", 
                                     command=self._stop_tracking, width=15, state='disabled')
        self.stop_button.pack(side='left', padx=(0, 10))
        self.save_button = ttk.Button(track_btn_frame, text="Save Data", 
                                     command=self._save_data, width=15, state='disabled')
        self.save_button.pack(side='left', padx=(0, 10))
        ttk.Button(track_btn_frame, text="Reset", command=self._reset, width=10).pack(side='right')
        
        self.status_label = ttk.Label(track_frame, text="Ready", foreground='blue')
        self.status_label.pack(pady=(10, 0))
        
        # Create main display area with video and plot side by side
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Video canvas - sized to match display frame
        video_frame = ttk.LabelFrame(display_frame, text="Video Preview")
        video_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Create canvas without fixed dimensions - let it size itself
        self.video_canvas = tk.Canvas(video_frame, bg='black', highlightthickness=0)
        self.video_canvas.pack(expand=True, fill='both', pady=5)  # Allow both expansion and filling
        
        # Bind mouse events for draggable coordinate origin
        self.video_canvas.bind("<Button-1>", self._on_canvas_click)
        self.video_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        
        # Plot canvas for tracking visualization
        plot_frame = ttk.LabelFrame(display_frame, text="Position Plot")
        plot_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Add checkbox to enable/disable plotting
        plot_control_frame = ttk.Frame(plot_frame)
        plot_control_frame.pack(fill='x', padx=5, pady=(5, 0))
        self.plot_checkbox = ttk.Checkbutton(
            plot_control_frame, 
            text="Enable Live Plotting", 
            variable=self.plot_enabled,
            command=self._on_plot_toggle
        )
        self.plot_checkbox.pack(side='left')
        
        self.plot_canvas = tk.Canvas(plot_frame, width=self.plot_width, height=self.plot_height, bg='white')
        self.plot_canvas.pack(expand=True, pady=10)
        
        # Bind window resize event to update video scaling
        self.window.bind('<Configure>', self._on_window_resize)

    def _load_first_frame(self):
        """Load and display first frame"""
        self.current_frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self._resize_canvas_for_video()
            self._display_frame()
            self._update_progress()

    def _calculate_adaptive_scale(self):
        """Calculate adaptive scaling to fit video in available space"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            return 1.0
            
        height, width = self.current_frame.shape[:2]
        
        # Ensure we have valid max dimensions
        if not hasattr(self, 'max_canvas_width') or not hasattr(self, 'max_canvas_height'):
            return 1.0
            
        # Calculate scale factors for both dimensions
        width_scale = self.max_canvas_width / width if width > 0 else 1.0
        height_scale = self.max_canvas_height / height if height > 0 else 1.0
        
        # Use the smaller scale to ensure both dimensions fit
        scale = min(width_scale, height_scale, 1.0)  # Don't scale up beyond original size
        
        return scale

    def _resize_canvas_for_video(self):
        """Resize canvas to match video dimensions with adaptive scaling plus extra space for coordinate origin"""
        if self.current_frame is not None and self.video_canvas is not None:
            height, width = self.current_frame.shape[:2]
            scale = self._calculate_adaptive_scale()
            
            # Update gui_display_width based on adaptive scale
            self.gui_display_width = int(width * scale)
            
            canvas_height = int(height * scale) + 60  # Add extra space for coordinate origin
            canvas_width = int(width * scale) + 60    # Add extra space for coordinate origin
            self.video_canvas.configure(width=canvas_width, height=canvas_height)

    def _create_display_frame(self):
        """Create lightweight display frame with adaptive scaling"""
        if self.current_frame is None:
            return None
        height, width = self.current_frame.shape[:2]
        scale = self._calculate_adaptive_scale()
        new_width = max(1, int(width * scale))  # Ensure minimum size of 1
        new_height = max(1, int(height * scale))  # Ensure minimum size of 1
        
        if new_width == 0 or new_height == 0:
            print(f"Warning: Zero dimension detected - width: {new_width}, height: {new_height}, scale: {scale}")
            return None
            
        return cv2.resize(self.current_frame, (new_width, new_height))

    def _create_processed_frame(self):
        """Create full-resolution processed frame for accurate tracking"""
        if self.current_frame is None:
            return None
        # Apply calibration corrections to full-resolution frame
        return self._apply_calibration_corrections(self.current_frame)

    def _post_create_scrub(self):
        """Scrub to frame 0 after GUI creation to ensure proper sizing"""
        try:
            self.frame_slider.set(0)
            self._seek_frame(0)
        except Exception as e:
            print(f"Error in post-create scrub: {e}")

    def _display_frame(self):
        """Display frame on canvas"""
        display_frame = self._create_display_frame()
        if display_frame is None:
            return
            
        # Add bbox preview if exists (for multi-object compatibility)
        if self.bboxes:
            # Colors for OpenCV (BGR format)
            cv_colors = {
                'blue': (255, 0, 0),
                'red': (0, 0, 255),
                'grey': (128, 128, 128),
                'black': (0, 0, 0),
                'green': (0, 255, 0),
                'magenta': (255, 0, 255),
                'cyan': (255, 255, 0),
                'brown': (42, 42, 165)
            }
            
            # Use adaptive scaling
            scale = self._calculate_adaptive_scale() if self.current_frame is not None else 1.0
            
            for i, bbox in enumerate(self.bboxes):
                if i >= len(self.object_colors):
                    continue
                    
                color_name = self.object_colors[i]
                color = cv_colors.get(color_name, (0, 255, 0))
                
                x, y, w, h = bbox
                x_d, y_d = int(x * scale), int(y * scale)
                w_d, h_d = int(w * scale), int(h * scale)
                cv2.rectangle(display_frame, (x_d, y_d), (x_d + w_d, y_d + h_d), color, 2)
                cv2.circle(display_frame, (x_d + w_d//2, y_d + h_d//2), 3, color, -1)
        
        # Draw coordinate origin
        self._draw_coordinate_origin(display_frame)
        
        # Convert and show
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        self.video_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        
        # Calculate positioning with the 60px margins
        img_w, img_h = display_frame.shape[1], display_frame.shape[0]
        x_pos = (canvas_w - img_w) // 2
        y_pos = (canvas_h - img_h) // 2
        
        # Position image at calculated offset (top-left anchored)
        self.video_canvas.create_image(x_pos, y_pos, anchor='nw', image=photo)
        self.video_canvas.image = photo

    def _update_tracking_display(self):
        """Update video display during tracking with multiple colored bbox and centroid overlays"""
        if self.current_frame is None:
            return
            
        # Create display frame with tracking overlay
        display_frame = self._create_display_frame()
        if display_frame is None:
            return
            
        # Add bboxes and centroids for all objects if tracking is active
        if self.bboxes and self.tracking_active:
            # Scale bboxes to display resolution
            orig_height, orig_width = self.current_frame.shape[:2]
            display_height, display_width = display_frame.shape[:2]
            
            # Colors for OpenCV (BGR format)
            cv_colors = {
                'blue': (255, 0, 0),
                'red': (0, 0, 255),
                'grey': (128, 128, 128),
                'black': (0, 0, 0),
                'green': (0, 255, 0),
                'magenta': (255, 0, 255),
                'cyan': (255, 255, 0),
                'brown': (42, 42, 165)
            }
            
            for i, bbox in enumerate(self.bboxes):
                if i >= len(self.object_colors):
                    continue
                    
                color_name = self.object_colors[i]
                color = cv_colors.get(color_name, (255, 255, 255))
                
                # Scale from full-resolution coordinates to display coordinates
                # Bboxes are now in full resolution (processed frame resolution)
                # Scale directly from processed to display coordinates
                scale_to_display = display_width / orig_width
                x_orig = bbox[0]
                y_orig = bbox[1] 
                w_orig = bbox[2]
                h_orig = bbox[3]
                
                # Scale to display coordinates
                scale_orig_to_display = scale_to_display
                x_d = int(x_orig * scale_orig_to_display)
                y_d = int(y_orig * scale_orig_to_display)
                w_d = int(w_orig * scale_orig_to_display)
                h_d = int(h_orig * scale_orig_to_display)
                
                # Draw tracking bbox with object-specific color
                cv2.rectangle(display_frame, (x_d, y_d), (x_d + w_d, y_d + h_d), color, 3)
                
                # Draw centroid (filled circle with same color)
                center_x = x_d + w_d // 2
                center_y = y_d + h_d // 2
                cv2.circle(display_frame, (center_x, center_y), 8, color, -1)
                cv2.circle(display_frame, (center_x, center_y), 12, color, 2)
                
                # Draw rotation indicator if rotation tracking is enabled
                if (hasattr(self, 'rotation_tracking') and i < len(self.rotation_tracking) 
                    and self.rotation_tracking[i] and hasattr(self, 'rotation_data') 
                    and i < len(self.rotation_data) and self.rotation_data[i]):
                    
                    # Get current rotation angle
                    rotation_angle = self.rotation_data[i][-1] if self.rotation_data[i] else 0.0
                    
                    # Convert angle to radians (0° = up/12 o'clock, clockwise positive)
                    angle_rad = np.radians(rotation_angle - 90)  # Subtract 90° so 0° points up
                    
                    # Calculate line end point (clock hand extending from center)
                    line_length = max(w_d, h_d) * 0.6  # Line length proportional to bbox size
                    end_x = int(center_x + line_length * np.cos(angle_rad))
                    end_y = int(center_y + line_length * np.sin(angle_rad))
                    
                    # Draw rotation indicator line (thicker, darker version of object color)
                    cv2.line(display_frame, (center_x, center_y), (end_x, end_y), color, 4)
                    
                    # Draw arrowhead at the end of the line
                    arrow_length = 8
                    arrow_angle = np.radians(30)  # 30-degree arrow wings
                    
                    # Calculate arrowhead points
                    arrow1_x = int(end_x - arrow_length * np.cos(angle_rad - arrow_angle))
                    arrow1_y = int(end_y - arrow_length * np.sin(angle_rad - arrow_angle))
                    arrow2_x = int(end_x - arrow_length * np.cos(angle_rad + arrow_angle))
                    arrow2_y = int(end_y - arrow_length * np.sin(angle_rad + arrow_angle))
                    
                    # Draw arrowhead
                    cv2.line(display_frame, (end_x, end_y), (arrow1_x, arrow1_y), color, 3)
                    cv2.line(display_frame, (end_x, end_y), (arrow2_x, arrow2_y), color, 3)
                
                # Add object number label
                label = f"Obj{i+1}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_x = max(0, x_d)
                label_y = max(label_size[1] + 5, y_d - 5)
                
                # Background rectangle for label
                cv2.rectangle(display_frame, 
                            (label_x - 2, label_y - label_size[1] - 2),
                            (label_x + label_size[0] + 2, label_y + 2),
                            color, -1)
                
                # White text for label
                cv2.putText(display_frame, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add tracking info overlay
            info_text = f"Frame: {self.current_frame_idx}/{self.total_frames-1}"
            cv2.putText(display_frame, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Update points text for multiple objects
            total_points = sum(len(obj_data) for obj_data in self.tracking_data)
            points_text = f"Objects: {len(self.bboxes)} | Total Points: {total_points}"
            cv2.putText(display_frame, points_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, points_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add tracking status
            status_text = "MULTI-OBJECT TRACKING ACTIVE"
            cv2.putText(display_frame, status_text, (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw coordinate origin
        self._draw_coordinate_origin(display_frame)
        
        # Convert and show
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        self.video_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        
        # Calculate positioning with the 60px margins
        img_w, img_h = display_frame.shape[1], display_frame.shape[0]
        x_pos = (canvas_w - img_w) // 2
        y_pos = (canvas_h - img_h) // 2
        
        # Position image at calculated offset (top-left anchored)
        self.video_canvas.create_image(x_pos, y_pos, anchor='nw', image=photo)
        self.video_canvas.image = photo

    def _on_plot_toggle(self):
        """Callback for plot enable/disable checkbox"""
        if self.plot_enabled.get():
            # If plotting is re-enabled, update the plot with current data
            if len(self.tracking_data) > 0:
                self._update_plot()
        else:
            # If plotting is disabled, clear the plot canvas
            if self.plot_canvas:
                self.plot_canvas.delete("all")
                # Add a message indicating plotting is disabled
                canvas_width = self.plot_canvas.winfo_width()
                canvas_height = self.plot_canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    self.plot_canvas.create_text(
                        canvas_width // 2, canvas_height // 2,
                        text="Live Plotting Disabled\n(Check 'Enable Live Plotting' to re-enable)",
                        font=('Arial', 12),
                        justify='center',
                        fill='gray'
                    )

    def _update_plot(self):
        """Update three-panel position plot with current tracking data for multiple objects"""
        if not self.plot_canvas:
            return
            
        # Check if plotting is enabled
        if not self.plot_enabled.get():
            return
            
        # Performance optimization: skip update if no data
        if not self.tracking_data or all(len(obj_data) == 0 for obj_data in self.tracking_data):
            return
            
        # Clear canvas
        self.plot_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.plot_canvas.winfo_width()
        canvas_height = self.plot_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Define subplot areas - Modified for 4-panel layout
        margin = 30
        # Left plots (stacked): X vs time, Y vs time, Rotation vs time
        left_width = canvas_width * 0.35
        right_width = canvas_width * 0.45
        center_gap = canvas_width * 0.1
        
        # Left plots (X, Y, Rotation vs time, stacked in thirds)
        left_left = margin
        left_right = left_left + left_width
        plot_height = (canvas_height - 4 * margin) / 3  # Split into thirds
        
        x_plot_top = margin
        x_plot_bottom = x_plot_top + plot_height
        
        y_plot_top = x_plot_bottom + margin
        y_plot_bottom = y_plot_top + plot_height
        
        rot_plot_top = y_plot_bottom + margin
        rot_plot_bottom = rot_plot_top + plot_height
        
        # Right plot (X vs Y, full height)
        right_left = left_right + center_gap
        right_right = right_left + right_width
        right_top = margin
        right_bottom = canvas_height - margin
        
        # Draw subplot borders
        self._draw_plot_border(left_left, x_plot_top, left_right, x_plot_bottom, "X vs Time (rel)")
        self._draw_plot_border(left_left, y_plot_top, left_right, y_plot_bottom, "Y vs Time (rel)")
        self._draw_plot_border(left_left, rot_plot_top, left_right, rot_plot_bottom, "Rotation vs Time")
        self._draw_plot_border(right_left, right_top, right_right, right_bottom, "X vs Y Position (rel)")
        
        # Plot colors for canvas (different from OpenCV colors)
        canvas_colors = {
            'blue': 'blue',
            'red': 'red', 
            'grey': 'gray',
            'black': 'black',
            'green': 'green',
            'magenta': 'magenta',
            'cyan': 'cyan',
            'brown': '#8B4513'
        }
        
        # Plot data for each object
        for obj_idx, obj_data in enumerate(self.tracking_data):
            if len(obj_data) == 0 or obj_idx >= len(self.object_colors):
                continue
                
            # Get color for this object
            color_name = self.object_colors[obj_idx]
            color = canvas_colors.get(color_name, 'black')
            
            # For large datasets, use data sampling to maintain performance
            data_length = len(obj_data)
            if data_length > 500:
                sample_step = max(1, data_length // 500)  # Show ~500 points max
                display_data = obj_data[::sample_step]
            else:
                display_data = obj_data
            
            if len(display_data) > 0:
                # Extract data efficiently using list comprehensions on sampled data
                times = [data[0] for data in display_data]      # timestamp
                
                # Apply origin offset to convert to relative coordinates for plotting
                x_coords = []
                y_coords = []
                for data in display_data:
                    rel_x, rel_y = self._apply_origin_offset(data[1], data[2])
                    x_coords.append(rel_x)
                    y_coords.append(rel_y)
                
                if len(times) > 0:
                    # Plot 1: X vs Time (top left)
                    self._draw_time_plot_multi(left_left, x_plot_top, left_right, x_plot_bottom, 
                                             times, x_coords, "Time (s)", "X Position (rel)", color, obj_idx == 0)
                    
                    # Plot 2: Y vs Time (middle left) 
                    self._draw_time_plot_multi(left_left, y_plot_top, left_right, y_plot_bottom, 
                                             times, y_coords, "Time (s)", "Y Position (rel)", color, obj_idx == 0)
                    
                    # Plot 3: Rotation vs Time (bottom left) - only if rotation tracking is enabled
                    if (hasattr(self, 'rotation_data') and obj_idx < len(self.rotation_data) 
                        and hasattr(self, 'rotation_tracking') and obj_idx < len(self.rotation_tracking) 
                        and self.rotation_tracking[obj_idx]):
                        rot_angles = self.rotation_data[obj_idx]
                        if len(rot_angles) >= len(display_data):
                            # Sample rotation data to match display_data length
                            if data_length > 500:
                                rot_display_data = rot_angles[::sample_step]
                            else:
                                rot_display_data = rot_angles[:len(display_data)]
                            
                            self._draw_time_plot_multi(left_left, rot_plot_top, left_right, rot_plot_bottom, 
                                                     times, rot_display_data, "Time (s)", "Rotation (°)", color, obj_idx == 0)
                    
                    # Plot 4: X vs Y (right, full height)
                    self._draw_xy_plot_multi(right_left, right_top, right_right, right_bottom, 
                                           x_coords, y_coords, color, obj_idx == 0)

    def _draw_empty_plots(self):
        """Draw empty plot borders when starting fresh tracking"""
        if not self.plot_canvas or not self.plot_enabled.get():
            return
            
        # Clear canvas
        self.plot_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.plot_canvas.winfo_width()
        canvas_height = self.plot_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Define subplot areas - same layout as _update_plot
        margin = 30
        # Left plots (stacked): X vs time, Y vs time, Rotation vs time
        left_width = canvas_width * 0.35
        right_width = canvas_width * 0.45
        center_gap = canvas_width * 0.1
        
        # Left plots (X, Y, Rotation vs time, stacked in thirds)
        left_left = margin
        left_right = left_left + left_width
        plot_height = (canvas_height - 4 * margin) / 3  # Split into thirds
        
        x_plot_top = margin
        x_plot_bottom = x_plot_top + plot_height
        
        y_plot_top = x_plot_bottom + margin
        y_plot_bottom = y_plot_top + plot_height
        
        rot_plot_top = y_plot_bottom + margin
        rot_plot_bottom = rot_plot_top + plot_height
        
        # Right plot (X vs Y, full height)
        right_left = left_right + center_gap
        right_right = right_left + right_width
        right_top = margin
        right_bottom = canvas_height - margin
        
        # Draw empty subplot borders
        self._draw_plot_border(left_left, x_plot_top, left_right, x_plot_bottom, "X vs Time (rel)")
        self._draw_plot_border(left_left, y_plot_top, left_right, y_plot_bottom, "Y vs Time (rel)")
        self._draw_plot_border(left_left, rot_plot_top, left_right, rot_plot_bottom, "Rotation vs Time")
        self._draw_plot_border(right_left, right_top, right_right, right_bottom, "X vs Y Position (rel)")

    def _draw_plot_border(self, left, top, right, bottom, title):
        """Draw border and title for a subplot"""
        # Border
        self.plot_canvas.create_rectangle(left, top, right, bottom, outline='black', width=1)
        
        # Title
        title_x = (left + right) // 2
        title_y = top - 15
        self.plot_canvas.create_text(title_x, title_y, text=title, font=('Arial', 10, 'bold'))

    def _draw_time_plot(self, left, top, right, bottom, times, values, xlabel, ylabel, color):
        """Draw a time series plot"""
        if len(times) < 2:
            return
            
        # Calculate plot area (leave margin for axes)
        plot_margin = 20
        plot_left = left + plot_margin + 25
        plot_right = right - plot_margin
        plot_top = top + plot_margin
        plot_bottom = bottom - plot_margin - 20
        
        # Get data ranges
        time_min, time_max = min(times), max(times)
        val_min, val_max = min(values), max(values)
        
        # Add padding
        time_range = max(time_max - time_min, 0.1)
        val_range = max(val_max - val_min, 10)
        time_min -= time_range * 0.05
        time_max += time_range * 0.05
        val_min -= val_range * 0.1
        val_max += val_range * 0.1
        
        # Draw axes
        self.plot_canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill='black', width=1)
        self.plot_canvas.create_line(plot_left, plot_top, plot_left, plot_bottom, fill='black', width=1)
        
        # Convert coordinates
        def to_canvas_x(t):
            return plot_left + (t - time_min) / (time_max - time_min) * (plot_right - plot_left)
        
        def to_canvas_y(v):
            return plot_bottom - (v - val_min) / (val_max - val_min) * (plot_bottom - plot_top)
        
        # Draw data line
        for i in range(len(times) - 1):
            x1, y1 = to_canvas_x(times[i]), to_canvas_y(values[i])
            x2, y2 = to_canvas_x(times[i + 1]), to_canvas_y(values[i + 1])
            self.plot_canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
        
        # Draw current point
        if len(times) > 0:
            curr_x = to_canvas_x(times[-1])
            curr_y = to_canvas_y(values[-1])
            self.plot_canvas.create_oval(curr_x - 3, curr_y - 3, curr_x + 3, curr_y + 3,
                                       fill=color, outline='black', width=1)
        
        # Labels
        self.plot_canvas.create_text((plot_left + plot_right) // 2, plot_bottom + 15, 
                                   text=xlabel, font=('Arial', 8))
        self.plot_canvas.create_text(plot_left - 20, (plot_top + plot_bottom) // 2, 
                                   text=ylabel, font=('Arial', 8), angle=90)
        
        # Scale labels
        self.plot_canvas.create_text(plot_left, plot_bottom + 8, text=f"{time_min:.1f}", font=('Arial', 7))
        self.plot_canvas.create_text(plot_right, plot_bottom + 8, text=f"{time_max:.1f}", font=('Arial', 7))
        self.plot_canvas.create_text(plot_left - 15, plot_bottom, text=f"{val_min:.0f}", font=('Arial', 7))
        self.plot_canvas.create_text(plot_left - 15, plot_top, text=f"{val_max:.0f}", font=('Arial', 7))

    def _draw_xy_plot(self, left, top, right, bottom, x_coords, y_coords):
        """Draw X vs Y position plot"""
        if len(x_coords) < 1:
            return
            
        # Calculate plot area
        plot_margin = 20
        plot_left = left + plot_margin + 25
        plot_right = right - plot_margin
        plot_top = top + plot_margin
        plot_bottom = bottom - plot_margin - 20
        
        # Get data ranges with proper padding
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding to ensure all points are visible
        x_range = max(x_max - x_min, 50)  # Minimum range of 50 pixels
        y_range = max(y_max - y_min, 50)  # Minimum range of 50 pixels
        
        # Add 20% padding around the data
        x_padding = x_range * 0.2
        y_padding = y_range * 0.2
        
        x_min = x_min - x_padding
        x_max = x_max + x_padding
        y_min = y_min - y_padding
        y_max = y_max + y_padding
        
        # Draw axes
        self.plot_canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill='black', width=1)
        self.plot_canvas.create_line(plot_left, plot_top, plot_left, plot_bottom, fill='black', width=1)
        
        # Add grid lines
        grid_lines = 5  # Number of grid lines in each direction
        
        # Vertical grid lines
        for i in range(1, grid_lines):
            x_pos = plot_left + i * (plot_right - plot_left) / grid_lines
            self.plot_canvas.create_line(x_pos, plot_top, x_pos, plot_bottom, 
                                       fill='lightgray', width=1, dash=(2, 2))
        
        # Horizontal grid lines
        for i in range(1, grid_lines):
            y_pos = plot_top + i * (plot_bottom - plot_top) / grid_lines
            self.plot_canvas.create_line(plot_left, y_pos, plot_right, y_pos, 
                                       fill='lightgray', width=1, dash=(2, 2))
        
        # Convert coordinates (axes are properly scaled so no clipping needed)
        def to_canvas_x(x):
            if x_max == x_min:  # Avoid division by zero
                return (plot_left + plot_right) / 2
            return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)
        
        def to_canvas_y(y):
            if y_max == y_min:  # Avoid division by zero
                return (plot_top + plot_bottom) / 2
            return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        
        # Draw trajectory (grey line)
        if len(x_coords) > 1:
            for i in range(len(x_coords) - 1):
                x1, y1 = to_canvas_x(x_coords[i]), to_canvas_y(y_coords[i])
                x2, y2 = to_canvas_x(x_coords[i + 1]), to_canvas_y(y_coords[i + 1])
                self.plot_canvas.create_line(x1, y1, x2, y2, fill='grey', width=1)
        
        # Draw current position (large blue circle)
        if len(x_coords) > 0:
            curr_x = to_canvas_x(x_coords[-1])
            curr_y = to_canvas_y(y_coords[-1])
            self.plot_canvas.create_oval(curr_x - 5, curr_y - 5, curr_x + 5, curr_y + 5,
                                       fill='blue', outline='darkblue', width=2)
        
        # Labels
        self.plot_canvas.create_text((plot_left + plot_right) // 2, plot_bottom + 15, 
                                   text="X Position (pixels)", font=('Arial', 8))
        self.plot_canvas.create_text(plot_left - 20, (plot_top + plot_bottom) // 2, 
                                   text="Y Position (pixels)", font=('Arial', 8), angle=90)
        
        # Scale labels
        self.plot_canvas.create_text(plot_left, plot_bottom + 8, text=f"{x_min:.0f}", font=('Arial', 7))
        self.plot_canvas.create_text(plot_right, plot_bottom + 8, text=f"{x_max:.0f}", font=('Arial', 7))
        self.plot_canvas.create_text(plot_left - 15, plot_bottom, text=f"{y_min:.0f}", font=('Arial', 7))
        self.plot_canvas.create_text(plot_left - 15, plot_top, text=f"{y_max:.0f}", font=('Arial', 7))

    def _draw_time_plot_multi(self, left, top, right, bottom, times, values, xlabel, ylabel, color, draw_axes):
        """Draw a time series plot for multi-object tracking"""
        if len(times) < 2:
            return
            
        # Calculate plot area (leave margin for axes)
        plot_margin = 20
        plot_left = left + plot_margin + 25
        plot_right = right - plot_margin
        plot_top = top + plot_margin
        plot_bottom = bottom - plot_margin - 20
        
        # Only draw axes and labels for the first object to avoid overlap
        if draw_axes:
            # Get global data ranges for consistent scaling across all objects
            all_times = []
            all_values = []
            for obj_data in self.tracking_data:
                if len(obj_data) > 0:
                    obj_times = [data[0] for data in obj_data]
                    if xlabel == "Time (s)" and ylabel == "X Position":
                        obj_values = [data[1] for data in obj_data]
                    elif xlabel == "Time (s)" and ylabel == "Y Position":
                        obj_values = [data[2] for data in obj_data]
                    else:
                        obj_values = values
                    all_times.extend(obj_times)
                    all_values.extend(obj_values)
            
            if all_times and all_values:
                self.plot_time_min = min(all_times)
                self.plot_time_max = max(all_times)
                if ylabel == "X Position":
                    self.plot_x_min = min(all_values)
                    self.plot_x_max = max(all_values)
                elif ylabel == "Y Position":
                    self.plot_y_min = min(all_values)
                    self.plot_y_max = max(all_values)
        
        # Use stored ranges
        if ylabel == "X Position":
            val_min, val_max = getattr(self, 'plot_x_min', min(values)), getattr(self, 'plot_x_max', max(values))
        elif ylabel == "Y Position":
            val_min, val_max = getattr(self, 'plot_y_min', min(values)), getattr(self, 'plot_y_max', max(values))
        else:
            val_min, val_max = min(values), max(values)
            
        time_min = getattr(self, 'plot_time_min', min(times))
        time_max = getattr(self, 'plot_time_max', max(times))
        
        # Add padding
        time_range = max(time_max - time_min, 0.1)
        val_range = max(val_max - val_min, 10)
        time_min -= time_range * 0.05
        time_max += time_range * 0.05
        val_min -= val_range * 0.1
        val_max += val_range * 0.1
        
        # Draw axes and labels only once
        if draw_axes:
            self.plot_canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill='black', width=1)
            self.plot_canvas.create_line(plot_left, plot_top, plot_left, plot_bottom, fill='black', width=1)
            
            # Labels
            self.plot_canvas.create_text((plot_left + plot_right) // 2, plot_bottom + 15, 
                                       text=xlabel, font=('Arial', 8))
            self.plot_canvas.create_text(plot_left - 20, (plot_top + plot_bottom) // 2, 
                                       text=ylabel, font=('Arial', 8), angle=90)
            
            # Scale labels
            self.plot_canvas.create_text(plot_left, plot_bottom + 8, text=f"{time_min:.1f}", font=('Arial', 7))
            self.plot_canvas.create_text(plot_right, plot_bottom + 8, text=f"{time_max:.1f}", font=('Arial', 7))
            self.plot_canvas.create_text(plot_left - 15, plot_bottom, text=f"{val_min:.0f}", font=('Arial', 7))
            self.plot_canvas.create_text(plot_left - 15, plot_top, text=f"{val_max:.0f}", font=('Arial', 7))
        
        # Convert coordinates
        def to_canvas_x(t):
            return plot_left + (t - time_min) / (time_max - time_min) * (plot_right - plot_left)
        
        def to_canvas_y(v):
            return plot_bottom - (v - val_min) / (val_max - val_min) * (plot_bottom - plot_top)
        
        # Draw data line
        for i in range(len(times) - 1):
            x1, y1 = to_canvas_x(times[i]), to_canvas_y(values[i])
            x2, y2 = to_canvas_x(times[i + 1]), to_canvas_y(values[i + 1])
            self.plot_canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
        
        # Draw current point
        if len(times) > 0:
            curr_x = to_canvas_x(times[-1])
            curr_y = to_canvas_y(values[-1])
            self.plot_canvas.create_oval(curr_x - 3, curr_y - 3, curr_x + 3, curr_y + 3,
                                       fill=color, outline='black', width=1)

    def _draw_xy_plot_multi(self, left, top, right, bottom, x_coords, y_coords, color, draw_axes):
        """Draw X vs Y position plot for multi-object tracking"""
        if len(x_coords) < 1:
            return
            
        # Calculate plot area
        plot_margin = 20
        plot_left = left + plot_margin + 25
        plot_right = right - plot_margin
        plot_top = top + plot_margin
        plot_bottom = bottom - plot_margin - 20
        
        # Only calculate ranges and draw axes for the first object
        if draw_axes:
            # Get global coordinate ranges for consistent scaling (with origin offset applied)
            all_x_coords = []
            all_y_coords = []
            for obj_data in self.tracking_data:
                if len(obj_data) > 0:
                    # Apply origin offset to get relative coordinates for range calculation
                    for data in obj_data:
                        rel_x, rel_y = self._apply_origin_offset(data[1], data[2])
                        all_x_coords.append(rel_x)
                        all_y_coords.append(rel_y)
            
            if all_x_coords and all_y_coords:
                x_min, x_max = min(all_x_coords), max(all_x_coords)
                y_min, y_max = min(all_y_coords), max(all_y_coords)
                
                # Add padding to ensure all points are visible
                x_range = max(x_max - x_min, 50)  # Minimum range of 50 pixels
                y_range = max(y_max - y_min, 50)  # Minimum range of 50 pixels
                
                # Add 20% padding around the data
                x_padding = x_range * 0.2
                y_padding = y_range * 0.2
                
                self.plot_xy_x_min = x_min - x_padding
                self.plot_xy_x_max = x_max + x_padding  
                self.plot_xy_y_min = y_min - y_padding
                self.plot_xy_y_max = y_max + y_padding
                
                # Draw axes
                self.plot_canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill='black', width=1)
                self.plot_canvas.create_line(plot_left, plot_top, plot_left, plot_bottom, fill='black', width=1)
                
                # Add grid lines
                grid_lines = 5
                for i in range(1, grid_lines):
                    x_pos = plot_left + i * (plot_right - plot_left) / grid_lines
                    self.plot_canvas.create_line(x_pos, plot_top, x_pos, plot_bottom, 
                                               fill='lightgray', width=1, dash=(2, 2))
                for i in range(1, grid_lines):
                    y_pos = plot_top + i * (plot_bottom - plot_top) / grid_lines
                    self.plot_canvas.create_line(plot_left, y_pos, plot_right, y_pos, 
                                               fill='lightgray', width=1, dash=(2, 2))
                
                # Labels
                self.plot_canvas.create_text((plot_left + plot_right) // 2, plot_bottom + 15, 
                                           text="X Position (pixels)", font=('Arial', 8))
                self.plot_canvas.create_text(plot_left - 20, (plot_top + plot_bottom) // 2, 
                                           text="Y Position (pixels)", font=('Arial', 8), angle=90)
                
                # Scale labels
                self.plot_canvas.create_text(plot_left, plot_bottom + 8, text=f"{self.plot_xy_x_min:.0f}", font=('Arial', 7))
                self.plot_canvas.create_text(plot_right, plot_bottom + 8, text=f"{self.plot_xy_x_max:.0f}", font=('Arial', 7))
                self.plot_canvas.create_text(plot_left - 15, plot_bottom, text=f"{self.plot_xy_y_min:.0f}", font=('Arial', 7))
                self.plot_canvas.create_text(plot_left - 15, plot_top, text=f"{self.plot_xy_y_max:.0f}", font=('Arial', 7))
        
        # Use stored ranges
        x_min = getattr(self, 'plot_xy_x_min', min(x_coords) - 10)
        x_max = getattr(self, 'plot_xy_x_max', max(x_coords) + 10)
        y_min = getattr(self, 'plot_xy_y_min', min(y_coords) - 10)
        y_max = getattr(self, 'plot_xy_y_max', max(y_coords) + 10)
        
        # Convert coordinates (axes are properly scaled so no clipping needed)
        def to_canvas_x(x):
            if x_max == x_min:  # Avoid division by zero
                return (plot_left + plot_right) / 2
            return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)
        
        def to_canvas_y(y):
            if y_max == y_min:  # Avoid division by zero
                return (plot_top + plot_bottom) / 2
            return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        
        # Draw trajectory line
        if len(x_coords) > 1:
            for i in range(len(x_coords) - 1):
                x1, y1 = to_canvas_x(x_coords[i]), to_canvas_y(y_coords[i])
                x2, y2 = to_canvas_x(x_coords[i + 1]), to_canvas_y(y_coords[i + 1])
                self.plot_canvas.create_line(x1, y1, x2, y2, fill=color, width=1)
        
        # Draw current position
        if len(x_coords) > 0:
            curr_x = to_canvas_x(x_coords[-1])
            curr_y = to_canvas_y(y_coords[-1])
            self.plot_canvas.create_oval(curr_x - 5, curr_y - 5, curr_x + 5, curr_y + 5,
                                       fill=color, outline='black', width=2)

    def _update_progress(self):
        """Update progress display"""
        self.progress_label.configure(text=f"{self.current_frame_idx} / {self.total_frames-1}")

    def _on_frame_change(self, value):
        """Handle slider change"""
        if not self.tracking_active:
            self._seek_frame(int(value))

    def _seek_frame(self, frame_idx):
        """Seek to specific frame"""
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self._display_frame()
            self._update_progress()

    def _step_forward(self):
        """Step one frame forward"""
        if not self.tracking_active:
            self._seek_frame(self.current_frame_idx + 1)
            self.frame_slider.set(self.current_frame_idx)

    def _step_back(self):
        """Step one frame backward"""
        if not self.tracking_active:
            self._seek_frame(self.current_frame_idx - 1)
            self.frame_slider.set(self.current_frame_idx)

    def _toggle_play(self):
        """Toggle playback"""
        if self.tracking_active:
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            threading.Thread(target=self._playback_loop, daemon=True).start()

    def _playback_loop(self):
        """Playback loop"""
        while self.playing and self.current_frame_idx < self.total_frames - 1:
            self._seek_frame(self.current_frame_idx + 1)
            self.window.after(0, lambda: self.frame_slider.set(self.current_frame_idx))
            time.sleep(1.0 / self.fps)
        self.window.after(0, lambda: self.play_button.configure(text="Play"))
        self.playing = False

    def _select_objects(self):
        """Select multiple objects using tabbed native resolution interface"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame loaded")
            return
        self.status_label.configure(text="Selecting object(s)...", foreground='orange')
        
        # Inform the user about selection controls
        messagebox.showinfo(
            "Select Objects",
            "Select objects by left-clicking and dragging boxes around them on each tab. Use tabs to switch between objects. Use the scroll wheel to zoom in or out. Each tab maintains its own zoom level."
        )
        
        # Use native resolution for selection
        native_frame = self.current_frame.copy()

        # Convert to PIL image for Tkinter display
        pil_orig = Image.fromarray(cv2.cvtColor(native_frame, cv2.COLOR_BGR2RGB))
        orig_w, orig_h = pil_orig.size

        # Determine maximum display size based on screen dimensions
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()

        max_display_w = min(1200, screen_w - 200)
        max_display_h = min(800, screen_h - 200)

        initial_scale = min(1.0, max_display_w / float(orig_w), max_display_h / float(orig_h))
        min_scale = max(0.1, initial_scale * 0.2)
        max_scale = max(1.0, initial_scale * 4.0)

        # Create modal Tk window for selection
        sel_win = tk.Toplevel(self.window)
        sel_win.title("Select Objects - Native Resolution")
        sel_win.transient(self.window)
        sel_win.grab_set()
        
        # Set window to 85% of screen size to ensure controls are visible
        window_w = int(screen_w * 0.85)
        window_h = int(screen_h * 0.85)
        
        # Center the window on screen
        x = max(0, (screen_w - window_w) // 2)
        y = max(0, (screen_h - window_h) // 2)
        
        sel_win.geometry(f"{window_w}x{window_h}+{x}+{y}")

        # Create notebook for tabs
        sel_win.grid_rowconfigure(0, weight=1)
        sel_win.grid_columnconfigure(0, weight=1)
        
        notebook = ttk.Notebook(sel_win)
        notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Controls frame at bottom
        ctrl_frame = ttk.Frame(sel_win)
        ctrl_frame.grid(row=1, column=0, sticky='ew', pady=(0, 8), padx=8)
        ctrl_frame.columnconfigure(1, weight=1)  # Space between left and right buttons

        # Left side buttons (object management)
        left_buttons = ttk.Frame(ctrl_frame)
        left_buttons.grid(row=0, column=0, sticky='w')
        
        add_obj_btn = ttk.Button(left_buttons, text="Add Object", width=12)
        add_obj_btn.grid(row=0, column=0, padx=(0, 5))
        
        remove_obj_btn = ttk.Button(left_buttons, text="Remove Current Object", width=18)
        remove_obj_btn.grid(row=0, column=1, padx=5)

        # Right side buttons (confirm/cancel)
        right_buttons = ttk.Frame(ctrl_frame)
        right_buttons.grid(row=0, column=2, sticky='e')
        
        confirm_btn = ttk.Button(right_buttons, text="Confirm", width=12)
        confirm_btn.grid(row=0, column=0, padx=5)
        
        cancel_btn = ttk.Button(right_buttons, text="Cancel", width=12)
        cancel_btn.grid(row=0, column=1, padx=(5, 0))

        # Storage for object data
        object_data = {}  # Will store data for each tab
        object_counter = [1]  # Use list to allow modification in nested functions

        def create_object_tab(obj_num):
            """Create a new tab for object selection"""
            tab_frame = ttk.Frame(notebook)
            tab_name = f"Object {obj_num}"
            notebook.add(tab_frame, text=tab_name)
            
            # Create main container with controls at top and canvas below
            tab_frame.grid_rowconfigure(1, weight=1)
            tab_frame.grid_columnconfigure(0, weight=1)
            
            # Controls frame at top
            controls_frame = ttk.Frame(tab_frame)
            controls_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
            
            # Rotation tracking checkbox
            track_rotation_var = tk.BooleanVar(value=False)
            rotation_checkbox = ttk.Checkbutton(
                controls_frame, 
                text="Track Rotation", 
                variable=track_rotation_var
            )
            rotation_checkbox.pack(side='left')
            
            # Canvas frame
            canvas_frame = ttk.Frame(tab_frame)
            canvas_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=(0, 5))
            canvas_frame.grid_rowconfigure(0, weight=1)
            canvas_frame.grid_columnconfigure(0, weight=1)

            canvas = tk.Canvas(canvas_frame, bg='black')
            h_scroll = ttk.Scrollbar(canvas_frame, orient='horizontal', command=canvas.xview)
            v_scroll = ttk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
            canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

            canvas.grid(row=0, column=0, sticky='nsew')
            v_scroll.grid(row=0, column=1, sticky='ns')
            h_scroll.grid(row=1, column=0, sticky='ew')

            # State for this tab's zoom and selection
            state = {
                'scale': initial_scale,
                'image_id': None,
                'tk_image': None,
                'rect_id': None,
                'start_x': None,
                'start_y': None,
                'bbox_native': (0, 0, 0, 0),
                'obj_num': obj_num,
                'canvas': canvas,
                'track_rotation': track_rotation_var  # Add rotation tracking state
            }

            object_data[obj_num] = state

            def render_image():
                # Render scaled image and update canvas
                s = state['scale']
                disp_w = max(1, int(orig_w * s))
                disp_h = max(1, int(orig_h * s))
                disp = pil_orig.resize((disp_w, disp_h), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(disp)
                state['tk_image'] = tk_img
                if state['image_id'] is None:
                    state['image_id'] = canvas.create_image(0, 0, anchor='nw', image=tk_img)
                else:
                    canvas.itemconfig(state['image_id'], image=tk_img)
                canvas.config(scrollregion=(0, 0, disp_w, disp_h))

            def to_native_coords(x_canvas, y_canvas):
                s = state['scale']
                x_native = int(round(x_canvas / s))
                y_native = int(round(y_canvas / s))
                return x_native, y_native

            # Mouse event handlers for drawing rectangle
            def on_button_press(event):
                canvas.focus_set()
                x = canvas.canvasx(event.x)
                y = canvas.canvasy(event.y)
                state['start_x'] = x
                state['start_y'] = y
                if state['rect_id']:
                    canvas.delete(state['rect_id'])
                # Use object-specific color for rectangle
                color = self.object_colors[(obj_num - 1) % len(self.object_colors)]
                state['rect_id'] = canvas.create_rectangle(x, y, x, y, outline=color, width=2)

            def on_move(event):
                if state['rect_id'] is None:
                    return
                x = canvas.canvasx(event.x)
                y = canvas.canvasy(event.y)
                canvas.coords(state['rect_id'], state['start_x'], state['start_y'], x, y)

            def on_button_release(event):
                if state['rect_id'] is None:
                    return
                x1, y1, x2, y2 = canvas.coords(state['rect_id'])
                x1n, x2n = sorted([x1, x2])
                y1n, y2n = sorted([y1, y2])
                w = max(0, x2n - x1n)
                h = max(0, y2n - y1n)
                nx1, ny1 = to_native_coords(x1n, y1n)
                state['bbox_native'] = (nx1, ny1, int(round(w / state['scale'])), int(round(h / state['scale'])))

            # Zoom with mouse wheel
            def on_mouse_wheel(event):
                delta = 0
                if hasattr(event, 'delta') and event.delta:
                    delta = event.delta
                elif event.num == 4:
                    delta = 120
                elif event.num == 5:
                    delta = -120
                if delta == 0:
                    return
                factor = 1.15 if delta > 0 else (1.0 / 1.15)
                new_scale = max(min_scale, min(max_scale, state['scale'] * factor))
                if abs(new_scale - state['scale']) < 1e-6:
                    return
                # Preserve canvas view center around cursor
                cx = canvas.canvasx(event.x)
                cy = canvas.canvasy(event.y)
                rx = cx / state['scale']
                ry = cy / state['scale']
                state['scale'] = new_scale
                render_image()
                # Center view at the same native point
                new_cx = rx * state['scale']
                new_cy = ry * state['scale']
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                try:
                    canvas.xview_moveto(max(0, (new_cx - canvas_width/2) / max(1, canvas.bbox('all')[2])))
                    canvas.yview_moveto(max(0, (new_cy - canvas_height/2) / max(1, canvas.bbox('all')[3])))
                except:
                    pass

            # Bind events
            canvas.bind('<ButtonPress-1>', on_button_press)
            canvas.bind('<B1-Motion>', on_move)
            canvas.bind('<ButtonRelease-1>', on_button_release)
            canvas.bind('<MouseWheel>', on_mouse_wheel)
            canvas.bind('<Button-4>', on_mouse_wheel)
            canvas.bind('<Button-5>', on_mouse_wheel)

            # Initial render
            render_image()
            
            return tab_frame

        def add_object():
            """Add a new object tab"""
            if len(object_data) >= 8:  # Limit to 8 objects (we have 8 colors)
                messagebox.showwarning("Limit Reached", "Maximum 8 objects supported")
                return
            object_counter[0] += 1
            create_object_tab(object_counter[0])
            # Select the new tab
            notebook.select(len(notebook.tabs()) - 1)

        def remove_current_object():
            """Remove the currently selected object tab"""
            if len(object_data) <= 1:
                messagebox.showwarning("Cannot Remove", "At least one object must remain")
                return
            current_tab = notebook.select()
            if current_tab:
                tab_index = notebook.index(current_tab)
                tab_text = notebook.tab(tab_index, "text")
                obj_num = int(tab_text.split()[-1])  # Extract number from "Object N"
                
                # Remove from data
                if obj_num in object_data:
                    del object_data[obj_num]
                
                # Remove tab
                notebook.forget(tab_index)

        def update_remove_button_state(*args):
            """Update remove button state based on tab count"""
            if len(notebook.tabs()) <= 1:
                remove_obj_btn.configure(state='disabled')
            else:
                remove_obj_btn.configure(state='normal')

        # Button callbacks
        def on_confirm():
            """Collect all bounding boxes and rotation tracking settings"""
            valid_objects = []
            rotation_tracking = []
            
            for obj_num, state in object_data.items():
                nx, ny, nw, nh = state['bbox_native']
                if nw > 0 and nh > 0:
                    valid_objects.append((obj_num, (nx, ny, nw, nh)))
                    rotation_tracking.append(state['track_rotation'].get())
            
            if valid_objects:
                # Clear existing selections
                self.native_bboxes = []
                self.bboxes = []
                self.rotation_tracking = []  # Store rotation tracking preferences
                
                # Sort by object number to maintain consistent ordering
                valid_objects.sort(key=lambda x: x[0])
                rotation_tracking = [rotation_tracking[i] for i, _ in enumerate(valid_objects)]
                
                # Convert to processed coordinates and store
                # Since tracking is now at full resolution, apply calibration corrections to coordinates
                for obj_num, (nx, ny, nw, nh) in valid_objects:
                    self.native_bboxes.append((nx, ny, nw, nh))
                    
                    # Transform native coordinates to processed (calibrated) coordinates
                    if hasattr(self, '_apply_calibration_corrections'):
                        # For now, assume 1:1 mapping - coordinates should be transformed by perspective matrix
                        # This may need refinement based on the perspective correction implementation
                        processed_bbox = (nx, ny, nw, nh)  # Direct mapping for full-resolution tracking
                    else:
                        processed_bbox = (nx, ny, nw, nh)
                    
                    self.bboxes.append(processed_bbox)
                
                self.rotation_tracking = rotation_tracking
                
                self.status_label.configure(text=f"{len(valid_objects)} object(s) selected", foreground='green')
                self.track_button.configure(state='normal')
                self._display_frame()
            else:
                self.status_label.configure(text="No objects selected", foreground='red')
                self.bboxes = []
                self.native_bboxes = []
                self.rotation_tracking = []
            
            sel_win.destroy()

        def on_cancel():
            """Cancel object selection"""
            self.status_label.configure(text="Selection cancelled", foreground='red')
            self.bboxes = []
            self.native_bboxes = []
            sel_win.destroy()

        # Configure button commands
        add_obj_btn.configure(command=add_object)
        remove_obj_btn.configure(command=remove_current_object)
        confirm_btn.configure(command=on_confirm)
        cancel_btn.configure(command=on_cancel)

        # Monitor tab changes to update remove button state
        notebook.bind("<<NotebookTabChanged>>", update_remove_button_state)

        # Create initial Object 1 tab
        create_object_tab(1)
        
        # Initial button state update
        update_remove_button_state()

        # Show window and wait
        sel_win.update_idletasks()
        sel_win.deiconify()
        sel_win.wait_window()

    def _start_tracking(self):
        """Start tracking multiple objects with v4-compatible parameters"""
        if not self.bboxes:
            messagebox.showwarning("Warning", "Please select object(s) first")
            return
        
        # Check for unsaved tracking data before starting new tracking
        if hasattr(self, 'tracking_data') and self.tracking_data:
            # Check if there's any existing tracking data
            existing_data = any(len(obj_data) > 0 for obj_data in self.tracking_data)
            if existing_data and not self.data_saved:
                result = messagebox.askyesnocancel(
                    "Existing Tracking Data",
                    "You have existing tracking data that hasn't been saved.\n\n"
                    "Do you want to save it before starting new tracking?\n\n"
                    "Yes: Save existing data first\n"
                    "No: Discard existing data and continue\n"
                    "Cancel: Return without starting tracking"
                )
                
                if result is True:  # Yes - save existing data first
                    self._save_data()
                    # Only proceed if data was actually saved (user didn't cancel save dialog)
                    if not self.data_saved:
                        return
                elif result is None:  # Cancel - don't start tracking
                    return
                # If result is False (No), proceed with clearing data below
            
        num_objects = len(self.bboxes)
        print(f"Initializing tracking for {num_objects} object(s)")
        
        # Initialize multiple CSRT trackers
        self.trackers = []
        tracker_type = "CSRT"
        
        for i, bbox in enumerate(self.bboxes):
            tracker_created = False
            tracker = None
            
            try:
                # Try cv2.legacy.TrackerCSRT_create first (OpenCV 4.x with contrib)
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker_created = True
                tracker_type = "CSRT"
            except (AttributeError, ImportError):
                try:
                    # Try direct cv2.TrackerCSRT_create (some OpenCV versions)
                    tracker = cv2.TrackerCSRT_create()
                    tracker_created = True
                    tracker_type = "CSRT"
                except (AttributeError, ImportError):
                    try:
                        # Try alternative CSRT initialization
                        tracker = cv2.createTrackerCSRT()
                        tracker_created = True
                        tracker_type = "CSRT"
                    except (AttributeError, ImportError):
                        # CSRT not available, try KCF as fallback
                        try:
                            tracker = cv2.legacy.TrackerKCF_create()
                            tracker_type = "KCF"
                            tracker_created = True
                            if i == 0:  # Only show message once
                                messagebox.showwarning("Tracker Fallback", 
                                                     "CSRT tracker not available. Using KCF tracker instead.")
                        except (AttributeError, ImportError):
                            try:
                                tracker = cv2.TrackerKCF_create()
                                tracker_type = "KCF"
                                tracker_created = True
                                if i == 0:
                                    messagebox.showwarning("Tracker Fallback", 
                                                         "CSRT tracker not available. Using KCF tracker instead.")
                            except (AttributeError, ImportError):
                                # Try MIL as another fallback (usually available)
                                try:
                                    tracker = cv2.TrackerMIL_create()
                                    tracker_type = "MIL"
                                    tracker_created = True
                                    if i == 0:
                                        messagebox.showwarning("Tracker Fallback", 
                                                             "CSRT/KCF trackers not available. Using MIL tracker instead.")
                                except (AttributeError, ImportError):
                                    # If all attempts fail, provide detailed error
                                    cv_version = cv2.__version__
                                    available_trackers = [attr for attr in dir(cv2) if 'Tracker' in attr and 'create' in attr]
                                    error_msg = (
                                        f"No compatible trackers available in OpenCV {cv_version}.\n\n"
                                        f"Available trackers: {', '.join(available_trackers) if available_trackers else 'None'}\n\n"
                                        "This may be due to:\n"
                                        "• Missing OpenCV contrib modules\n"
                                        "• Incomplete PyInstaller packaging\n"
                                        "• OpenCV version compatibility\n\n"
                                        "Please try using the Python version of the application."
                                    )
                                    messagebox.showerror("Tracker Error", error_msg)
                                    return
            
            if not tracker_created:
                messagebox.showerror("Error", f"Failed to create tracker for object {i+1}")
                return
                
            self.trackers.append(tracker)
        
        print(f"Successfully created {num_objects} {tracker_type} tracker(s)")
        
        # Initialize all trackers with processed frame
        processed_frame = self._create_processed_frame()
        for i, (tracker, bbox) in enumerate(zip(self.trackers, self.bboxes)):
            if not tracker.init(processed_frame, bbox):
                messagebox.showerror("Error", f"Failed to initialize tracker for object {i+1}")
                return
        
        # Update UI
        self.tracking_active = True
        self.track_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.select_button.configure(state='disabled')
        if hasattr(self, 'play_button') and self.play_button:
            try:
                self.play_button.configure(state='disabled')
            except Exception:
                pass
        self.frame_slider.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.status_label.configure(text=f"Tracking {num_objects} object(s)...", foreground='blue')
        
        # Clear data and initialize for multiple objects
        self.tracking_data = [[] for _ in range(num_objects)]  # List of lists for each object
        
        # Clear the plot canvas to start fresh
        if self.plot_canvas:
            self.plot_canvas.delete("all")
            # Optionally redraw empty plot borders
            if self.plot_enabled.get():
                self.window.after(10, self._draw_empty_plots)
        
        # Initialize rotation tracking data structures
        if hasattr(self, 'rotation_tracking'):
            self.rotation_data = [[] for _ in range(num_objects)]  # List of rotation angles for each object
            # Initialize enhanced rotation tracking data structures
            self.prev_patches = [None for _ in range(num_objects)]  # Previous frame patches for incremental ECC
            self.cumulative_transforms = [np.eye(2, 3, dtype=np.float32) for _ in range(num_objects)]  # Cumulative rotation matrices
            self.keyframe_patches = [None for _ in range(num_objects)]  # Keyframe reference patches
            self.keyframe_transforms = [np.eye(2, 3, dtype=np.float32) for _ in range(num_objects)]  # Transform from keyframe
            self.frames_since_keyframe = [0 for _ in range(num_objects)]  # Counter for keyframe refresh
            self.raw_angles = [[] for _ in range(num_objects)]  # Raw angles before smoothing
            self.smoothed_angles = [0.0 for _ in range(num_objects)]  # EMA smoothed angles
        else:
            self.rotation_tracking = [False] * num_objects  # Default to no rotation tracking
            self.rotation_data = [[] for _ in range(num_objects)]
            self.prev_patches = [None for _ in range(num_objects)]
            self.cumulative_transforms = [np.eye(2, 3, dtype=np.float32) for _ in range(num_objects)]
            self.keyframe_patches = [None for _ in range(num_objects)]
            self.keyframe_transforms = [np.eye(2, 3, dtype=np.float32) for _ in range(num_objects)]
            self.frames_since_keyframe = [0 for _ in range(num_objects)]
            self.raw_angles = [[] for _ in range(num_objects)]
            self.smoothed_angles = [0.0 for _ in range(num_objects)]
            
        self.data_saved = True  # Starting fresh, no data to save yet
        
        # Reset plot update counters
        self.last_plot_update_frame = -1
        self.gui_update_counter = 0
        
        threading.Thread(target=self._track_video, daemon=True).start()

    def _track_video(self):
        """Track multiple objects through video (v4-compatible sequential reading)"""
        start_frame = self.current_frame_idx
        self.tracking_start_frame = start_frame  # Store starting frame for time calculation
        num_objects = len(self.trackers)
        
        # Sequential reading (v4 approach for speed)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        
        while frame_idx < self.total_frames and self.tracking_active:
            ret, raw_frame = self.cap.read()
            if not ret:
                break
                
            # Process frame (full-resolution tracking)
            processed = self._apply_calibration_corrections(raw_frame)
            
            # Update all trackers
            successful_tracks = 0
            current_bboxes = []
            
            for i, tracker in enumerate(self.trackers):
                ok, bbox = tracker.update(processed)
                if ok:
                    successful_tracks += 1
                    current_bboxes.append(bbox)
                    
                    # Calculate center coordinates
                    centerx = int(bbox[0] + bbox[2] / 2)
                    centery = int(bbox[1] + bbox[3] / 2)
                    
                    # Add to tracking data for this object (Y-axis conversion handled in _apply_origin_offset)
                    time_stamp = (frame_idx - self.tracking_start_frame) / self.fps
                    self.tracking_data[i].append((time_stamp, centerx, centery))
                    
                    # Add rotation tracking if enabled for this object
                    if self.rotation_tracking[i]:
                        rotation_angle = self._track_rotation(processed, bbox, i)
                        self.rotation_data[i].append(rotation_angle)
                    else:
                        self.rotation_data[i].append(0.0)  # No rotation tracking
                        
                else:
                    # Tracking failed for this object - keep previous bbox
                    current_bboxes.append(self.bboxes[i] if i < len(self.bboxes) else (0, 0, 1, 1))
                    # Add default rotation value for failed tracking
                    self.rotation_data[i].append(self.rotation_data[i][-1] if self.rotation_data[i] else 0.0)
            
            # Update stored bounding boxes
            self.bboxes = current_bboxes
            self.data_saved = False  # Mark data as unsaved when new data is added
            
            # Update current frame for display
            self.current_frame = raw_frame.copy()
            self.current_frame_idx = frame_idx
            
            # Performance optimization: Update GUI less frequently
            self.gui_update_counter += 1
            should_update_display = (self.gui_update_counter % 2 == 0)  # Every 2nd frame
            should_update_plot = (frame_idx - self.last_plot_update_frame) >= self.plot_update_interval
            
            # Always update display for better visual feedback
            if should_update_display:
                self.window.after(0, self._update_tracking_display)
                self.window.after(0, lambda f=frame_idx: self.frame_slider.set(f))
                total_points = sum(len(obj_data) for obj_data in self.tracking_data)
                self.window.after(0, lambda f=frame_idx, tp=total_points: self.progress_label.configure(
                    text=f"{f} / {self.total_frames-1} | Objects: {successful_tracks}/{num_objects} | Points: {tp}"))
            
            # Update plot less frequently to reduce computational overhead
            if should_update_plot:
                self.window.after(0, self._update_plot)
                self.last_plot_update_frame = frame_idx
                
            # Small delay to allow GUI updates
            time.sleep(0.01)
            frame_idx += 1
        
        # Only call completion if tracking wasn't stopped manually
        if self.tracking_active:
            self.window.after(0, self._tracking_complete)
            # Force final plot update to show complete dataset
            self.window.after(100, self._final_plot_update)

    def _update_ui_after_tracking(self):
        """Update UI state after tracking ends (common logic for both stop and complete)"""
        self.tracking_active = False
        self.track_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.select_button.configure(state='normal')
        if hasattr(self, 'play_button') and self.play_button:
            try:
                self.play_button.configure(state='normal')
            except Exception:
                pass
        self.frame_slider.configure(state='normal')
        if self.tracking_data:
            self.save_button.configure(state='normal')

    def _track_rotation(self, frame, bbox, obj_index):
        """
        Enhanced rotation tracking using incremental ECC with keyframe re-anchoring.
        
        Method:
        1. Incremental ECC each frame (fast & stable)
        2. Re-anchor to keyframe to prevent drift
        3. Proper angle unwrapping and EMA smoothing
        
        Returns rotation angle in degrees.
        """
        try:
            # Extract square crop around bbox center
            x, y, w, h = [int(v) for v in bbox]
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate square crop size: S = ceil(√(w²+h²)) * 1.1
            diagonal = np.sqrt(w*w + h*h)
            crop_size = int(np.ceil(diagonal * 1.1))
            # Ensure crop_size is even
            if crop_size % 2 == 1:
                crop_size += 1
                
            # Calculate crop bounds with padding
            half_size = crop_size // 2
            x1 = center_x - half_size
            y1 = center_y - half_size
            x2 = center_x + half_size
            y2 = center_y + half_size
            
            # Handle boundary conditions with reflection padding
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - frame.shape[1])
            pad_bottom = max(0, y2 - frame.shape[0])
            
            # Extract crop with boundary handling
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                # Use reflection padding
                padded_frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, 
                                                cv2.BORDER_REFLECT_101)
                # Adjust coordinates for padded frame
                crop = padded_frame[pad_top + max(0, y1):pad_top + max(0, y1) + crop_size,
                                 pad_left + max(0, x1):pad_left + max(0, x1) + crop_size]
            else:
                crop = frame[y1:y2, x1:x2]
            
            # Ensure we have a valid crop
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                return 0.0
                
            # Convert to grayscale if needed
            if len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Preprocess patch: gray → Sobel magnitude → whiten → Hann window
            current_patch = self._preprocess_patch(crop)
            
            # Initialize tracking on first frame
            if self.prev_patches[obj_index] is None:
                self.prev_patches[obj_index] = current_patch.copy()
                self.keyframe_patches[obj_index] = current_patch.copy()
                self.cumulative_transforms[obj_index] = np.eye(2, 3, dtype=np.float32)
                self.keyframe_transforms[obj_index] = np.eye(2, 3, dtype=np.float32)
                self.frames_since_keyframe[obj_index] = 0
                self.smoothed_angles[obj_index] = 0.0
                return 0.0
            
            # Incremental ECC: current vs previous frame
            incremental_warp = np.eye(2, 3, dtype=np.float32)  # Start with identity for increment
            
            try:
                # ECC criteria: (COUNT=80, EPS=1e-6)
                criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 80, 1e-6)
                
                # Run incremental ECC
                (cc_inc, incremental_warp) = cv2.findTransformECC(
                    self.prev_patches[obj_index], current_patch, incremental_warp, 
                    cv2.MOTION_EUCLIDEAN, criteria
                )
                
                # Compose incremental transform with cumulative transform
                # Convert to homogeneous coordinates for composition
                cum_homo = np.vstack([self.cumulative_transforms[obj_index], [0, 0, 1]])
                inc_homo = np.vstack([incremental_warp, [0, 0, 1]])
                
                # Compose transforms: new_cumulative = cumulative * incremental
                new_cum_homo = cum_homo @ inc_homo
                self.cumulative_transforms[obj_index] = new_cum_homo[:2, :]
                
                # Extract rotation angle from cumulative transform
                cos_theta = self.cumulative_transforms[obj_index][0, 0]
                sin_theta = self.cumulative_transforms[obj_index][1, 0]
                raw_angle_rad = np.arctan2(sin_theta, cos_theta)
                raw_angle_deg = np.degrees(raw_angle_rad)
                
                # Angle unwrapping
                if len(self.raw_angles[obj_index]) > 0:
                    prev_angle = self.raw_angles[obj_index][-1]
                    # Unwrap angle to prevent 360° jumps
                    angle_diff = raw_angle_deg - prev_angle
                    if angle_diff > 180:
                        raw_angle_deg -= 360
                    elif angle_diff < -180:
                        raw_angle_deg += 360
                
                self.raw_angles[obj_index].append(raw_angle_deg)
                
                # Adaptive EMA smoothing - responsive to large changes, stable for small noise
                angle_change = abs(raw_angle_deg - self.smoothed_angles[obj_index])
                if angle_change > 2.0:  # Large change - minimal smoothing for responsiveness
                    alpha = 0.8
                elif angle_change > 0.5:  # Medium change - moderate smoothing
                    alpha = 0.5
                else:  # Small change - more smoothing to filter noise/jitter
                    alpha = 0.2
                
                self.smoothed_angles[obj_index] = (alpha * raw_angle_deg + 
                                                 (1 - alpha) * self.smoothed_angles[obj_index])
                
                # Update previous patch for next incremental step
                self.prev_patches[obj_index] = current_patch.copy()
                self.frames_since_keyframe[obj_index] += 1
                
                # Check if we need keyframe re-anchoring
                need_reanchor = (
                    cc_inc < 0.85 or  # Low correlation
                    self.frames_since_keyframe[obj_index] >= 30 or  # Time-based refresh
                    abs(raw_angle_deg - self.smoothed_angles[obj_index]) > 8.0  # Large angle jump (increased threshold)
                )
                
                if need_reanchor:
                    self._reanchor_keyframe(obj_index, current_patch)
                
                return self.smoothed_angles[obj_index]
                
            except cv2.error as e:
                print(f"Incremental ECC failed for object {obj_index}: {e}")
                # Try keyframe re-anchoring as fallback
                try:
                    return self._reanchor_keyframe(obj_index, current_patch)
                except:
                    return self.smoothed_angles[obj_index] if hasattr(self, 'smoothed_angles') else 0.0
                    
        except Exception as e:
            print(f"Rotation tracking error for object {obj_index}: {e}")
            return 0.0
    
    def _preprocess_patch(self, patch):
        """
        Preprocess patch: gray → Sobel magnitude → whiten → Hann window
        """
        # Compute Sobel magnitude
        grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        
        # Whiten (z-score normalization)
        mean_val = np.mean(gradient)
        std_val = np.std(gradient)
        if std_val > 0:
            gradient = (gradient - mean_val) / std_val
        
        # Apply Hann window
        if gradient.shape[0] > 1 and gradient.shape[1] > 1:
            hann_1d_y = np.hanning(gradient.shape[0])
            hann_1d_x = np.hanning(gradient.shape[1])
            hann_2d = np.outer(hann_1d_y, hann_1d_x)
            gradient = gradient * hann_2d
        
        # Convert back to uint8 for ECC
        gradient = ((gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8) * 255).astype(np.uint8)
        
        return gradient
    
    def _reanchor_keyframe(self, obj_index, current_patch):
        """
        Re-anchor to keyframe to prevent drift accumulation
        """
        try:
            if self.keyframe_patches[obj_index] is None:
                return 0.0
            
            # Run ECC vs keyframe using current cumulative transform as initializer
            keyframe_warp = self.cumulative_transforms[obj_index].copy()
            
            # ECC criteria: higher count for keyframe alignment
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 120, 1e-6)
            
            (cc_key, refined_warp) = cv2.findTransformECC(
                self.keyframe_patches[obj_index], current_patch, keyframe_warp,
                cv2.MOTION_EUCLIDEAN, criteria
            )
            
            # Replace cumulative transform with refined result
            self.cumulative_transforms[obj_index] = refined_warp.copy()
            
            # Reset incremental tracking
            self.prev_patches[obj_index] = current_patch.copy()
            self.frames_since_keyframe[obj_index] = 0
            
            # Optionally update keyframe (EMA blend)
            if cc_key > 0.90:
                alpha_key = 0.1
                self.keyframe_patches[obj_index] = (
                    alpha_key * current_patch + (1 - alpha_key) * self.keyframe_patches[obj_index]
                ).astype(np.uint8)
            
            # Extract angle from refined transform
            cos_theta = refined_warp[0, 0]
            sin_theta = refined_warp[1, 0]
            angle_rad = np.arctan2(sin_theta, cos_theta)
            angle_deg = np.degrees(angle_rad)
            
            # Update smoothed angle with adaptive smoothing
            if len(self.raw_angles[obj_index]) > 0:
                angle_change = abs(angle_deg - self.smoothed_angles[obj_index])
                if angle_change > 2.0:  # Large change - minimal smoothing for responsiveness
                    alpha = 0.8
                elif angle_change > 0.5:  # Medium change - moderate smoothing
                    alpha = 0.5
                else:  # Small change - more smoothing to filter noise
                    alpha = 0.2
                    
                self.smoothed_angles[obj_index] = (alpha * angle_deg + 
                                                 (1 - alpha) * self.smoothed_angles[obj_index])
            else:
                self.smoothed_angles[obj_index] = angle_deg
            
            print(f"Re-anchored object {obj_index}: cc={cc_key:.3f}, angle={angle_deg:.1f}°")
            return self.smoothed_angles[obj_index]
            
        except cv2.error as e:
            print(f"Keyframe re-anchor failed for object {obj_index}: {e}")
            return self.smoothed_angles[obj_index] if hasattr(self, 'smoothed_angles') else 0.0

    def _tracking_complete(self):
        """Handle tracking completion - only shows popup if tracking completed naturally"""
        # If tracking was manually stopped, don't override the stop message
        if not self.tracking_active:
            return
            
        self._update_ui_after_tracking()
        
        # Reset to first frame when tracking completes naturally
        self.window.after(0, lambda: self._seek_frame(0))
        self.window.after(0, lambda: self.frame_slider.set(0))
        
        # Calculate tracking statistics for completion message
        num_objects = len([obj_data for obj_data in self.tracking_data if len(obj_data) > 0])
        max_frames = 0
        if self.tracking_data:
            for obj_data in self.tracking_data:
                if obj_data:
                    max_frames = max(max_frames, len(obj_data))
        
        if num_objects > 0:
            self.status_label.configure(text=f"Complete - {num_objects} objects over {max_frames} frames", foreground='green')
            messagebox.showinfo("Complete", f"Tracking completed!\n{num_objects} objects tracked over {max_frames} frames")
        else:
            self.status_label.configure(text="Complete - No objects tracked", foreground='red')
            messagebox.showinfo("Complete", "Tracking completed but no objects were tracked")

    def _stop_tracking(self):
        """Stop tracking early and preserve collected data"""
        if not self.tracking_active:
            return
            
        self._update_ui_after_tracking()
        
        # Show appropriate status and message for multi-object tracking
        num_objects = len([obj_data for obj_data in self.tracking_data if len(obj_data) > 0]) if self.tracking_data else 0
        max_frames = 0
        if self.tracking_data:
            for obj_data in self.tracking_data:
                if obj_data:
                    max_frames = max(max_frames, len(obj_data))
        
        if num_objects > 0:
            self.status_label.configure(text=f"Stopped - {num_objects} objects over {max_frames} frames", foreground='orange')
            messagebox.showinfo("Tracking Stopped", 
                              f"Multi-object tracking stopped by user.\n{num_objects} objects tracked over {max_frames} frames and ready to save.")
        else:
            self.status_label.configure(text="Stopped - No data", foreground='red')
            messagebox.showinfo("Tracking Stopped", "Tracking stopped. No data collected.")

    def _save_data(self):
        """Save tracking data for multiple objects"""
        if not self.tracking_data or all(len(obj_data) == 0 for obj_data in self.tracking_data):
            messagebox.showwarning("Warning", "No data to save")
            return
        
        # Ask for filename when save is clicked
        self.data_save_path = filedialog.asksaveasfilename(
            parent=self.window, title="Save position data", defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not self.data_save_path:
            return
            
        try:
            with open(self.data_save_path, 'w') as f:
                # Write header with units and origin information
                cd = self.calibration_data
                if cd and cd.real_world_scale and getattr(cd, 'pixels_per_real_unit', 0) > 0:
                    coordinate_units = cd.get_coordinate_units()
                    f.write(f"# Coordinates in {coordinate_units}\n")
                    # Origin marker position in world coords (relative to the
                    # perspective warp's world (0,0) — the checkerboard origin),
                    # in the user-chosen frame orientation.
                    ppu = cd.pixels_per_real_unit
                    tx = getattr(cd, 'perspective_translation_x', 0.0)
                    ty = getattr(cd, 'perspective_translation_y', 0.0)
                    # Canvas pixels are already in the user-chosen frame
                    # (corrections._apply_perspective rotated the warp), so the
                    # offset from world (0,0) maps directly to user-frame
                    # coords — only Y is negated for canvas-Y-down vs frame-Y-up.
                    ox_world = (self.origin_x - tx) / ppu
                    oy_world = (ty - self.origin_y) / ppu
                    state = int(getattr(cd, 'frame_orientation_state', 0))
                    f.write(f"# Frame orientation state: {state}\n")
                    f.write(f"# Origin marker at world ({ox_world:.3f}, {oy_world:.3f}) {coordinate_units}\n")
                    f.write(f"# Origin marker at canvas pixel ({self.origin_x}, {self.origin_y})\n")
                else:
                    f.write("# Coordinates in pixels\n")
                    f.write(f"# Origin at canvas pixel ({self.origin_x}, {self.origin_y})\n")
                f.write("# All coordinates are relative to this origin (X=right, Y=up)\n")
                
                # Build header with columns for each tracked object
                header = "time"
                num_objects = len([obj_data for obj_data in self.tracking_data if len(obj_data) > 0])
                
                for i in range(num_objects):
                    header += f"\tobj_{i+1}_cx\tobj_{i+1}_cy"
                    # Add rotation column if rotation tracking was enabled for this object
                    if hasattr(self, 'rotation_tracking') and i < len(self.rotation_tracking) and self.rotation_tracking[i]:
                        header += f"\tobj_{i+1}_rotation"
                f.write(header + "\n")
                
                # Collect all unique timestamps from all objects
                all_timestamps = set()
                for obj_data in self.tracking_data:
                    for time_stamp, _, _ in obj_data:
                        all_timestamps.add(time_stamp)
                
                # Sort timestamps for consistent ordering
                sorted_timestamps = sorted(all_timestamps)
                
                # For each timestamp, find corresponding data from each object
                for time_stamp in sorted_timestamps:
                    line = f"{time_stamp:.6f}"
                    
                    for obj_idx, obj_data in enumerate(self.tracking_data):
                        if len(obj_data) == 0:
                            continue
                            
                        # Find data point closest to this timestamp for this object
                        closest_point = None
                        min_time_diff = float('inf')
                        
                        for data_point in obj_data:
                            time_diff = abs(data_point[0] - time_stamp)
                            if time_diff < min_time_diff:
                                min_time_diff = time_diff
                                closest_point = data_point
                        
                        if closest_point and min_time_diff < 0.1:  # Within 0.1 second tolerance
                            cx, cy = closest_point[1], closest_point[2]
                            
                            # Apply origin offset to get relative coordinates
                            rel_cx, rel_cy = self._apply_origin_offset(cx, cy)
                            
                            # Convert coordinates if real-world scaling is enabled
                            # Since tracking is now at full resolution, coordinates can be converted directly
                            if self.calibration_data and self.calibration_data.real_world_scale:
                                real_cx, real_cy = self.calibration_data.convert_to_real_world_coordinates(rel_cx, rel_cy)
                                line += f"\t{real_cx:.6f}\t{real_cy:.6f}"
                            else:
                                line += f"\t{rel_cx}\t{rel_cy}"
                            
                            # Add rotation data if rotation tracking was enabled for this object
                            if (hasattr(self, 'rotation_tracking') and obj_idx < len(self.rotation_tracking) 
                                and self.rotation_tracking[obj_idx] and hasattr(self, 'rotation_data') 
                                and obj_idx < len(self.rotation_data)):
                                
                                # Find corresponding rotation data for this timestamp
                                rotation_data_obj = self.rotation_data[obj_idx]
                                if rotation_data_obj:
                                    # Find the index of the closest point in the main tracking data
                                    data_point_idx = obj_data.index(closest_point)
                                    if data_point_idx < len(rotation_data_obj):
                                        rotation_angle = rotation_data_obj[data_point_idx]
                                        line += f"\t{rotation_angle:.2f}"
                                    else:
                                        line += "\t"
                                else:
                                    line += "\t"
                        else:
                            # No data for this object at this timestamp
                            empty_cols = 2  # cx, cy
                            if (hasattr(self, 'rotation_tracking') and obj_idx < len(self.rotation_tracking) 
                                and self.rotation_tracking[obj_idx]):
                                empty_cols += 1  # rotation
                            line += "\t" * empty_cols
                    
                    f.write(line + "\n")
                        
            self.data_saved = True  # Mark data as saved
            
            # Update success message with units and frame count
            units_info = ""
            if self.calibration_data and self.calibration_data.real_world_scale:
                units_info = f" ({self.calibration_data.get_coordinate_units()})"
            
            # Calculate max frames tracked
            max_frames = 0
            for obj_data in self.tracking_data:
                if obj_data:
                    max_frames = max(max_frames, len(obj_data))
            
            messagebox.showinfo("Saved", f"Multi-object data saved to:\n{os.path.basename(self.data_save_path)}\n\n"
                              f"{num_objects} objects tracked over {max_frames} frames{units_info}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    def _check_unsaved_data(self):
        """Check if there's unsaved tracking data and ask user what to do"""
        num_objects = len([obj_data for obj_data in self.tracking_data if len(obj_data) > 0]) if self.tracking_data else 0
        max_frames = 0
        if self.tracking_data:
            for obj_data in self.tracking_data:
                if obj_data:
                    max_frames = max(max_frames, len(obj_data))
        
        if num_objects > 0 and not self.data_saved:
            result = messagebox.askyesnocancel(
                "Unsaved Data", 
                f"You have {num_objects} objects tracked over {max_frames} frames that haven't been saved.\n\n"
                "Do you want to save them before continuing?\n\n"
                "Yes: Save data first\n"
                "No: Discard data\n"
                "Cancel: Return to tracking")
            
            if result is True:  # Yes - save data
                self._save_data()
                return self.data_saved  # Return True if save was successful
            elif result is False:  # No - discard data
                return True
            else:  # Cancel
                return False
        return True  # No unsaved data

    def _reset(self):
        """Reset tracking state with data loss protection"""
        if not self._check_unsaved_data():
            return  # User cancelled
            
        self.tracking_active = False
        # Reset multi-object tracking data
        self.bboxes = []
        self.native_bboxes = []
        self.trackers = []
        self.tracking_data = []
        # Keep old variables for compatibility
        self.bbox = None
        self.native_bbox = None
        self.tracker = None
        
        self.data_saved = True  # No data means nothing to save
        self.track_button.configure(state='disabled')
        self.stop_button.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.select_button.configure(state='normal')
        if hasattr(self, 'play_button') and self.play_button:
            try:
                self.play_button.configure(state='normal')
            except Exception:
                pass
        self.frame_slider.configure(state='normal')
        self.status_label.configure(text="Reset complete", foreground='blue')
        
    def _final_plot_update(self):
        """Final plot update that shows complete dataset with reduced sampling for performance"""
        if not self.plot_canvas:
            return
            
        total_points = sum(len(obj_data) for obj_data in self.tracking_data) if self.tracking_data else 0
        if total_points == 0:
            return
            
        # Check if plotting is enabled
        if not self.plot_enabled.get():
            return
            
        # For final plot, show more data points but still reasonable for display performance
        # Apply sampling to each object's data independently
        sampled_data = []
        for obj_data in self.tracking_data:
            if len(obj_data) > 2000:
                # Sample every Nth point to show ~1000 points for final plot
                sample_step = max(1, len(obj_data) // 1000)
                sampled_obj_data = obj_data[::sample_step]
            else:
                sampled_obj_data = obj_data
            sampled_data.append(sampled_obj_data)
            
        # Force a complete plot update with the final data
        self._update_plot_with_final_data(sampled_data)
        
    def _update_plot_with_final_data(self, display_data):
        """Update plot with final complete data - separate from real-time updates"""
        # This ensures the final plot shows the complete trajectory
        # Temporarily override the current plot data
        old_tracking_data = self.tracking_data
        self.tracking_data = display_data
        self._update_plot()  # Use existing plot method
        self.tracking_data = old_tracking_data  # Restore full data

    def _close(self):
        """Cleanup and close with data loss protection"""
        if not self._check_unsaved_data():
            return  # User cancelled
        self.playing = False
        self.tracking_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.window:
            self.window.destroy()

    def _apply_calibration_corrections_to_video(self):
        """Apply calibration corrections to video using same method as measurement_recorder"""
        base, ext = os.path.splitext(self.video_path)
        calibrated_path = f"{base}_CALIBRATED{ext}"
        metadata_path = f"{base}_CALIBRATED_metadata.json"
        
        # Save calibration metadata for later use
        self._save_calibration_metadata(metadata_path)
        
        # Open the input video
        input_video = cv2.VideoCapture(self.video_path)
        if not input_video.isOpened():
            raise Exception("Failed to open video for correction")
            
        # Get video properties
        fps = input_video.get(cv2.CAP_PROP_FPS)
        total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create progress dialog
        progress_window = self._create_progress_dialog(total_frames)
        
        # Read first frame to determine output size
        ret, first_frame = input_video.read()
        if not ret:
            progress_window.destroy()
            raise Exception("Failed to read video frames")
            
        # Apply corrections to determine output size
        corrected_frame = self._apply_calibration_corrections(first_frame)
        output_height, output_width = corrected_frame.shape[:2]
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_video = cv2.VideoWriter(calibrated_path, fourcc, fps, (output_width, output_height))
        
        if not output_video.isOpened():
            progress_window.destroy()
            raise Exception("Failed to create corrected video file")
            
        # Reset to beginning
        input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process all frames with progress updates
        frame_count = 0
        while True:
            ret, frame = input_video.read()
            if not ret:
                break
                
            # Apply calibration corrections
            corrected_frame = self._apply_calibration_corrections(frame)
            output_video.write(corrected_frame)
            
            frame_count += 1
            
            # Update progress every 10 frames to avoid GUI lag
            if frame_count % 10 == 0 or frame_count == total_frames:
                self._update_progress_dialog(progress_window, frame_count, total_frames)
                progress_window.update()
            
        # Cleanup
        input_video.release()
        output_video.release()
        progress_window.destroy()
        
        messagebox.showinfo("Correction Complete", 
                           f"Video corrected and saved as:\n{os.path.basename(calibrated_path)}\n\n"
                           f"{frame_count} frames processed with lens and perspective corrections.")
        
        return calibrated_path

    def _save_calibration_metadata(self, metadata_path):
        """Save calibration metadata to JSON file for later use with pre-calibrated videos"""
        import json
        
        metadata = {
            "version": "1.1",
            "calibration_applied": True,
            "lens_correction": self.calibration_data.is_calibrated,
            "lens_model": getattr(self.calibration_data, 'model_type', 'pinhole'),
            "fisheye_balance": float(getattr(self.calibration_data, 'fisheye_balance', 0.0)),
            "perspective_correction": self.calibration_data.perspective_corrected,
            "real_world_scale": getattr(self.calibration_data, 'real_world_scale', False),
            "square_size_real": getattr(self.calibration_data, 'square_size_real', None),
            "square_size_pixels": getattr(self.calibration_data, 'square_size_pixels', None),
            "pixels_per_real_unit": getattr(self.calibration_data, 'pixels_per_real_unit', 1.0),
            "coordinate_units": self.calibration_data.get_coordinate_units() if hasattr(self.calibration_data, 'get_coordinate_units') else "pixels",
            # Pixel position of world (0,0) in the output frame; non-zero when
            # the perspective warp shifted content to expose negative-coord space.
            "perspective_translation_x": float(getattr(self.calibration_data, 'perspective_translation_x', 0.0)),
            "perspective_translation_y": float(getattr(self.calibration_data, 'perspective_translation_y', 0.0)),
            # Linear downsample factor applied to the perspective warp output
            # (physical canvas = logical canvas / output_scale).  The
            # pixels_per_real_unit value above already reflects this scale,
            # so the tracker doesn't need to compensate further; recorded
            # for documentation/recovery.
            "output_scale": float(getattr(self.calibration_data, 'output_scale', 1.0)),
            # User-defined output region in pre-flip world coords at native
            # homography scale: [x_min, y_min, x_max, y_max].  Defines the
            # canvas extent the perspective warp produces.
            "output_bbox_world": (
                list(self.calibration_data.output_bbox_world)
                if getattr(self.calibration_data, 'output_bbox_world', None) is not None
                else None
            ),
            # Coordinate-frame orientation state (0–7) — see
            # data_models.FRAME_ORIENTATION_BASES.
            "frame_orientation_state": int(
                getattr(self.calibration_data, 'frame_orientation_state', 0)),
            "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "tracking_v7",
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Calibration metadata saved: {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save calibration metadata: {e}")

    def _detect_video_pre_correction(self, video_path):
        """Return (lens_done, perspective_done) for the given video.

        Prefers explicit booleans in the *_metadata.json sidecar (written by
        both measurement_recorder and the tracker's own auto-correction path).
        Falls back to the _CALIBRATED filename suffix for legacy videos that
        predate sidecar writing.
        """
        base, ext = os.path.splitext(video_path)
        metadata_path = f"{base}_metadata.json"

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                return (bool(meta.get('lens_correction', False)),
                        bool(meta.get('perspective_correction', False)))
            except Exception as e:
                print(f"Warning: could not read {metadata_path}: {e}")

        if video_path.upper().endswith("_CALIBRATED" + ext.upper()):
            return (True, True)

        return (False, False)

    def _load_calibration_metadata(self, video_path):
        """Load calibration metadata for pre-calibrated video"""
        import json
        
        # Try to find metadata file
        base, ext = os.path.splitext(video_path)
        metadata_path = f"{base}_metadata.json"
        
        if not os.path.exists(metadata_path):
            print(f"No metadata found for {os.path.basename(video_path)}")
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            print(f"Loaded calibration metadata: {os.path.basename(metadata_path)}")
            
            # Apply scaling information to current calibration data
            if metadata.get('real_world_scale', False):
                self.calibration_data.real_world_scale = True
                self.calibration_data.square_size_real = metadata.get('square_size_real')
                self.calibration_data.square_size_pixels = metadata.get('square_size_pixels')
                self.calibration_data.pixels_per_real_unit = metadata.get('pixels_per_real_unit', 1.0)
                print(f"Real-world scaling restored: {metadata.get('square_size_real')} units per square")
            else:
                self.calibration_data.real_world_scale = False
                self.calibration_data.pixels_per_real_unit = 1.0
                print("Pixel-based coordinates (no real-world scaling)")

            # Restore the perspective warp translation so coordinate conversion
            # knows where world (0,0) landed in the output frame.
            self.calibration_data.perspective_translation_x = float(
                metadata.get('perspective_translation_x', 0.0))
            self.calibration_data.perspective_translation_y = float(
                metadata.get('perspective_translation_y', 0.0))
            if (self.calibration_data.perspective_translation_x != 0.0 or
                    self.calibration_data.perspective_translation_y != 0.0):
                print(f"Perspective origin offset restored: "
                      f"({self.calibration_data.perspective_translation_x:.1f}, "
                      f"{self.calibration_data.perspective_translation_y:.1f}) px")

            # Restore output bbox (the user-defined output region in
            # pre-flip world coords).  Required by corrections._apply_perspective
            # if the video isn't already perspective-corrected and the user
            # re-applies corrections at runtime.
            bbox = metadata.get('output_bbox_world')
            self.calibration_data.output_bbox_world = (
                tuple(float(v) for v in bbox) if bbox else None)

            # Restore the user-chosen export coordinate frame.
            self.calibration_data.frame_orientation_state = int(
                metadata.get('frame_orientation_state', 0)) % 8

            # Restore downsample factor (informational; pixels_per_real_unit
            # already reflects the post-downsample scale).
            self.calibration_data.output_scale = float(
                metadata.get('output_scale', 1.0))
            if self.calibration_data.output_scale != 1.0:
                print(f"Output downsample factor: "
                      f"{self.calibration_data.output_scale:.2f}×")

            return metadata
            
        except Exception as e:
            print(f"Warning: Could not load calibration metadata: {e}")
            return None

    def _apply_calibration_corrections(self, frame):
        """Apply lens distortion and perspective corrections to a frame.

        Honors per-tracker `video_lens_corrected` / `video_perspective_corrected`
        flags so steps already baked into the source video are not re-applied.
        """
        return apply_corrections(
            frame,
            self.calibration_data,
            skip_lens=self.video_lens_corrected,
            skip_perspective=self.video_perspective_corrected,
        )

    def _create_progress_dialog(self, total_frames):
        """Create a progress dialog for video processing"""
        progress_window = tk.Toplevel(self.window)
        progress_window.title("Processing Video")
        progress_window.geometry("400x150")
        progress_window.transient(self.window)
        progress_window.grab_set()
        
        # Center the dialog
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (progress_window.winfo_width() // 2)
        y = (progress_window.winfo_screenheight() // 2) - (progress_window.winfo_height() // 2)
        progress_window.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(progress_window, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(main_frame, text="Applying Calibration Corrections", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Status label
        progress_window.status_label = ttk.Label(main_frame, text="Processing video frames...")
        progress_window.status_label.pack(pady=(0, 10))
        
        # Progress bar
        progress_window.progress_var = tk.DoubleVar()
        progress_window.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=progress_window.progress_var, 
            maximum=100,
            length=300
        )
        progress_window.progress_bar.pack(pady=(0, 10))
        
        # Progress text
        progress_window.progress_text = ttk.Label(main_frame, text="0%")
        progress_window.progress_text.pack()
        
        # Store total frames for reference
        progress_window.total_frames = total_frames
        
        return progress_window
    
    def _update_progress_dialog(self, progress_window, current_frame, total_frames):
        """Update the progress dialog"""
        if progress_window and progress_window.winfo_exists():
            percentage = (current_frame / total_frames) * 100
            progress_window.progress_var.set(percentage)
            progress_window.progress_text.configure(text=f"{percentage:.1f}% ({current_frame}/{total_frames})")
            progress_window.status_label.configure(text=f"Processing frame {current_frame} of {total_frames}...")

    def _on_canvas_click(self, event):
        """Handle mouse click on canvas - check if clicking near origin"""
        if self.current_frame is None or not self.video_canvas:
            return
            
        # Check if click is near the origin (within 50 pixels in display space for easier grabbing)
        origin_display_x, origin_display_y = self._fullres_to_canvas(self.origin_x, self.origin_y)
        
        click_distance = ((event.x - origin_display_x)**2 + (event.y - origin_display_y)**2)**0.5
        
        if click_distance <= 50:  # 50 pixel radius for easier grabbing
            self.origin_dragging = True
            self.origin_drag_offset_x = event.x - origin_display_x
            self.origin_drag_offset_y = event.y - origin_display_y
            self.video_canvas.configure(cursor="fleur")  # Four-way arrow cursor
            print(f"Origin grab detected at ({event.x}, {event.y}), origin at ({origin_display_x}, {origin_display_y})")
        else:
            self.origin_dragging = False
    
    def _on_canvas_drag(self, event):
        """Handle mouse drag on canvas - move origin if dragging"""
        if self.origin_dragging and self.current_frame is not None and self.video_canvas:
            # Calculate new origin position
            new_display_x = event.x - self.origin_drag_offset_x
            new_display_y = event.y - self.origin_drag_offset_y
            
            # Convert to full-resolution coordinates
            new_full_x, new_full_y = self._canvas_to_fullres(new_display_x, new_display_y)
            
            # Update origin position
            self.origin_x = new_full_x
            self.origin_y = new_full_y
            
            # Refresh display to show new origin position
            self._display_frame()
            
            # Update plots with new relative coordinates
            if hasattr(self, 'tracking_data') and self.tracking_data:
                self._update_plot()
    
    def _on_canvas_release(self, event):
        """Handle mouse release on canvas - stop dragging"""
        if self.origin_dragging:
            self.origin_dragging = False
            self.video_canvas.configure(cursor="")
    
    def _on_window_resize(self, event):
        """Handle window resize - update video canvas scaling"""
        # Only handle resize events for the main window
        if event.widget == self.window:
            # Use a delayed callback to ensure widgets have updated their sizes
            self.window.after(50, self._update_video_size_after_resize)
    
    def _update_video_size_after_resize(self):
        """Update video canvas size after window resize with proper space calculation"""
        try:
            # Get actual window dimensions
            window_width = self.window.winfo_width()
            window_height = self.window.winfo_height()
            
            # Calculate available space more accurately
            # Leave space for: plot panel (~600px), controls (~200px), padding (~50px)
            available_width = max(300, window_width - 650)  # More conservative
            available_height = max(200, window_height - 250)  # Account for controls and padding
            
            # Update max canvas dimensions with some margin
            self.max_canvas_width = available_width - 100  # Leave margin for coordinate origin
            self.max_canvas_height = available_height - 100
            
            print(f"Window resize: {window_width}x{window_height}, Available: {self.max_canvas_width}x{self.max_canvas_height}")
            
            # Resize video canvas if frame is loaded
            if self.current_frame is not None:
                self._resize_canvas_for_video()
                self._display_frame()
        except Exception as e:
            print(f"Error in resize update: {e}")

    def _canvas_to_fullres(self, canvas_x, canvas_y):
        """Convert canvas display coordinates to full-resolution coordinates"""
        if self.current_frame is None:
            return canvas_x, canvas_y
            
        orig_height, orig_width = self.current_frame.shape[:2]
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        # Use adaptive scaling
        display_scale = self._calculate_adaptive_scale()
        display_width = int(orig_width * display_scale)
        display_height = int(orig_height * display_scale)
        
        # Calculate offset for centering (account for 60px margin on each side)
        x_offset = (canvas_width - display_width) // 2
        y_offset = (canvas_height - display_height) // 2
        
        # Convert to full resolution
        full_x = ((canvas_x - x_offset) / display_scale) if display_scale > 0 else canvas_x
        full_y = ((canvas_y - y_offset) / display_scale) if display_scale > 0 else canvas_y
        
        return int(full_x), int(full_y)
    
    def _fullres_to_canvas(self, full_x, full_y):
        """Convert full-resolution coordinates to canvas display coordinates"""
        if self.current_frame is None:
            return full_x, full_y
            
        orig_height, orig_width = self.current_frame.shape[:2]
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        # Use adaptive scaling
        display_scale = self._calculate_adaptive_scale()
        display_width = int(orig_width * display_scale)
        display_height = int(orig_height * display_scale)
        
        # Calculate offset for centering (account for 60px margin on each side)
        x_offset = (canvas_width - display_width) // 2
        y_offset = (canvas_height - display_height) // 2
        
        # Convert to canvas coordinates
        canvas_x = full_x * display_scale + x_offset
        canvas_y = full_y * display_scale + y_offset
        
        return int(canvas_x), int(canvas_y)
    
    def _apply_origin_offset(self, x, y):
        """Convert absolute canvas pixel coords to user-frame coords.

        The perspective stage already rotates/flips the image into the
        user-chosen frame (see corrections.apply_frame_orientation_to_image),
        so the displayed canvas's right-direction is user-frame +X and the
        up-direction is user-frame +Y. We only need to:
          1. Subtract the draggable origin marker.
          2. Negate Y because Tk canvas Y increases downward but the user
             frame's +Y is up on screen.
        No further orientation matrix is needed here — that compensation
        has moved into the warp itself.
        """
        rel_x = x - self.origin_x
        rel_y = self.origin_y - y
        return rel_x, rel_y
    
    def _show_instructions(self):
        """Show instructions popup dialog"""
        instructions_window = tk.Toplevel(self.window)
        instructions_window.title("Tracking Instructions")
        instructions_window.geometry("500x300")
        instructions_window.transient(self.window)
        instructions_window.grab_set()
        
        # Center the window
        instructions_window.update_idletasks()
        x = (instructions_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (instructions_window.winfo_screenheight() // 2) - (300 // 2)
        instructions_window.geometry(f"500x300+{x}+{y}")
        
        # Main frame with padding
        main_frame = ttk.Frame(instructions_window, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(main_frame, text="Object Tracking Instructions", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 20))
        
        # Instructions text
        instructions_text = [
            "1. Navigate Video:",
            "   • Use ◀/▶ buttons or slider to scrub to desired start frame",
            "",
            "2. Set Coordinate Origin:",
            "   • Drag the white coordinate origin (⊕) to your reference point",
            "   • X-axis (red) points right, Y-axis (green) points up",
            "",
            "3. Select Objects:",
            "   • Click 'Select Object(s)' and draw boxes around objects to track",
            "   • Multiple objects supported with different colors",
            "",
            "4. Track Objects:",
            "   • Click 'Start Tracking' to begin automatic tracking",
            "   • Live plots show relative positions from coordinate origin",
            "",
            "5. Save Data:",
            "   • Click 'Save Data' to export coordinates relative to origin",
            "   • Moving origin and re-saving updates coordinates non-destructively"
        ]
        
        # Scrollable text area
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill='both', expand=True)
        
        text_widget = tk.Text(text_frame, wrap='word', height=12, width=60, 
                             font=('Arial', 10), relief='flat', bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Insert instructions
        for line in instructions_text:
            text_widget.insert('end', line + '\n')
        
        text_widget.configure(state='disabled')  # Make read-only
        
        # Close button
        ttk.Button(main_frame, text="Close", command=instructions_window.destroy, 
                  width=10).pack(pady=(20, 0))
    
    def _draw_coordinate_origin(self, display_frame):
        """Draw the draggable coordinate origin on the display frame"""
        if not hasattr(self, 'origin_x') or not hasattr(self, 'origin_y'):
            return
            
        # Convert origin position to display coordinates
        if self.current_frame is not None:
            orig_height, orig_width = self.current_frame.shape[:2]
            display_height, display_width = display_frame.shape[:2]
            scale_x = display_width / orig_width
            scale_y = display_height / orig_height
            
            # Origin position in display frame
            origin_disp_x = int(self.origin_x * scale_x)
            origin_disp_y = int(self.origin_y * scale_y)
            
            # Draw coordinate axes - longer and thicker for better visibility
            axis_length = 60
            line_thickness = 3
            
            # Draw a larger grab area circle (transparent background indicator)
            grab_radius = 25
            cv2.circle(display_frame, (origin_disp_x, origin_disp_y), grab_radius, (100, 100, 100), 1)
            
            # X-axis (red, pointing right)
            cv2.arrowedLine(display_frame, 
                          (origin_disp_x, origin_disp_y), 
                          (origin_disp_x + axis_length, origin_disp_y),
                          (0, 0, 255), line_thickness, tipLength=0.2)
            
            # Y-axis (green, pointing up)
            cv2.arrowedLine(display_frame, 
                          (origin_disp_x, origin_disp_y), 
                          (origin_disp_x, origin_disp_y - axis_length),
                          (0, 255, 0), line_thickness, tipLength=0.2)
            
            # Origin point - larger and more visible
            cv2.circle(display_frame, (origin_disp_x, origin_disp_y), 8, (255, 255, 255), -1)
            cv2.circle(display_frame, (origin_disp_x, origin_disp_y), 8, (0, 0, 0), 2)
            
            # Labels
            cv2.putText(display_frame, "X", 
                       (origin_disp_x + axis_length + 5, origin_disp_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, "Y", 
                       (origin_disp_x - 5, origin_disp_y - axis_length - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Origin coordinates text. The displayed canvas is already in
            # the user-chosen frame (corrections._apply_perspective rotated
            # the warp), so the offset from world (0,0) maps directly to
            # user-frame coords — only Y is negated for canvas-Y-down vs
            # frame-Y-up.
            cd = getattr(self, 'calibration_data', None)
            if (cd and cd.real_world_scale and
                    getattr(cd, 'pixels_per_real_unit', 0) > 0):
                ppu = cd.pixels_per_real_unit
                tx = getattr(cd, 'perspective_translation_x', 0.0)
                ty = getattr(cd, 'perspective_translation_y', 0.0)
                user_x = (self.origin_x - tx) / ppu
                user_y = (ty - self.origin_y) / ppu
                origin_text = f"Origin: ({user_x:.2f}, {user_y:.2f}) {cd.get_coordinate_units()}"
            else:
                origin_text = f"Origin: ({self.origin_x}, {self.origin_y}) px"
            cv2.putText(display_frame, origin_text, 
                       (10, display_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
