"""
Calibration preview window for Added-Mass-Lab GUI
Window for previewing calibration results with live camera feed or imported video
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .tooltip import ToolTip
from .corrections import apply_corrections
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


class CalibrationPreviewWindow:
    """Window for previewing calibration results with live camera feed"""
    
    def __init__(self, parent, camera_manager, calibration_data, camera_id):
        self.parent = parent
        self.camera_manager = camera_manager
        self.calibration_data = calibration_data
        self.camera_id = camera_id
        
        # Preview state
        self.window = None
        self.cap = None
        self.preview_running = False
        self.preview_thread = None
        
        # Video source mode
        self.use_camera = True  # True for live camera, False for imported video
        self.video_path = None
        self.video_cap = None
        
        # Preview settings
        self.show_original = True
        self.show_lens_corrected = True
        self.show_perspective_corrected = True
        
        # UI elements
        self.canvas_original = None
        self.canvas_lens = None
        self.canvas_perspective = None
        
    def show(self):
        """Show the calibration preview window - first ask for source"""
        self.show_source_selection_dialog()
        
    def show_source_selection_dialog(self):
        """Show dialog to select video source (camera or file)"""
        # Create source selection dialog
        source_dialog = tk.Toplevel(self.parent)
        source_dialog.title("Select Video Source")
        source_dialog.geometry("400x320")  # Increased height from 280 to 320
        source_dialog.transient(self.parent)
        source_dialog.grab_set()
        
        # Center the dialog
        source_dialog.update_idletasks()
        x = (source_dialog.winfo_screenwidth() // 2) - (source_dialog.winfo_width() // 2)
        y = (source_dialog.winfo_screenheight() // 2) - (source_dialog.winfo_height() // 2)
        source_dialog.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(source_dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Choose Video Source", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Select how you want to preview the calibration results:",
                              wraplength=350)
        desc_label.pack(pady=(0, 20))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=(0, 20))

        # Camera button - only enable if camera is available
        camera_available = self.camera_id is not None
        camera_btn = ttk.Button(buttons_frame, text="Use Live Camera",
                       command=lambda: self._start_with_camera(source_dialog),
                       style='Accent.TButton' if camera_available else None,
                       state='normal' if camera_available else 'disabled')
        camera_btn.pack(fill='x', pady=(0, 10))
        ToolTip(camera_btn, "Open a live camera preview using the selected camera.")

        if not camera_available:
            # Add explanation if no camera available
            no_camera_label = ttk.Label(buttons_frame, 
                                       text="(No camera selected - use video file instead)",
                                       font=('Arial', 9), foreground='gray')
            no_camera_label.pack(pady=(0, 10))

        # File button
        file_btn = ttk.Button(buttons_frame, text="Import Video File",
                     command=lambda: self._start_with_video(source_dialog))
        file_btn.pack(fill='x', pady=(0, 10))
        ToolTip(file_btn, "Choose a video file to preview calibration corrections.")

        # Cancel button
        cancel_btn = ttk.Button(buttons_frame, text="Cancel",
                       command=source_dialog.destroy)
        cancel_btn.pack(fill='x')
        ToolTip(cancel_btn, "Close this dialog without opening a preview.")
        
    def _start_with_camera(self, dialog):
        """Start preview with live camera"""
        # Check if camera is available
        if self.camera_id is None:
            messagebox.showwarning("No Camera", "No camera is currently selected. Please select a camera first or use a video file.")
            return
            
        dialog.destroy()
        self.use_camera = True
        self._create_and_show_preview()
        
    def _start_with_video(self, dialog):
        """Start preview with imported video file"""
        dialog.destroy()
        
        # Ask user to select video file
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.use_camera = False
            self._create_and_show_preview()
    
    def _create_and_show_preview(self):
        """Create and show the calibration preview window"""
        self.window = tk.Toplevel(self.parent)
        title_text = "Calibration Preview - Live Camera Feed" if self.use_camera else "Calibration Preview - Video File"
        self.window.title(title_text)
        self.window.geometry("1300x700")  # Adjusted for two panels
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        self.create_widgets()
        self.start_preview()
        
    def create_widgets(self):
        """Create the preview window widgets"""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_text = "Calibration Preview - Live Camera Feed" if self.use_camera else "Calibration Preview - Video File"
        title_label = ttk.Label(main_frame, text=title_text, 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Display Options", padding=10)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="", foreground='blue')
        self.status_label.pack()
        
        # Video controls frame (only for video mode)
        if not self.use_camera:
            self.create_video_controls(control_frame)
        
        # Preview frame with two columns (Raw and Final Corrected)
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill='both', expand=True)
        
        # Configure grid weights
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        
        # Column headers
        ttk.Label(preview_frame, text="Raw Camera Feed", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 5))
        ttk.Label(preview_frame, text="Final Corrected (Lens + Perspective)", font=('Arial', 12, 'bold')).grid(
            row=0, column=1, pady=(0, 5))
        
        # Canvas for each view (larger size for better visibility)
        canvas_width, canvas_height = 600, 450
        
        self.canvas_original = tk.Canvas(preview_frame, width=canvas_width, height=canvas_height, 
                                        bg='black', relief='sunken', bd=2)
        self.canvas_original.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        
        self.canvas_corrected = tk.Canvas(preview_frame, width=canvas_width, height=canvas_height, 
                                         bg='black', relief='sunken', bd=2)
        self.canvas_corrected.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        
        # Close button
        close_button = ttk.Button(main_frame, text="Close", command=self.close_window)
        close_button.pack(pady=(10, 0))
        ToolTip(close_button, "Close the preview window and return to the main menu.")
    
    def create_video_controls(self, parent):
        """Create video playback controls for video mode"""
        video_frame = ttk.Frame(parent)
        video_frame.pack(fill='x', pady=(10, 0))
        
        # Playback controls
        controls_frame = ttk.Frame(video_frame)
        controls_frame.pack(fill='x')
        
        self.play_button = ttk.Button(controls_frame, text="⏸️ Pause", 
                                     command=self.toggle_playback)
        self.play_button.pack(side='left', padx=(0, 5))
        
        ttk.Button(controls_frame, text="⏮️ Start", 
                  command=self.goto_start).pack(side='left', padx=(0, 5))
        
        ttk.Button(controls_frame, text="⏭️ End", 
                  command=self.goto_end).pack(side='left', padx=(0, 5))
        
        # Progress info
        self.progress_label = ttk.Label(controls_frame, text="Frame: 0 / 0")
        self.progress_label.pack(side='right')
        
        # Initialize video playback state
        self.video_playing = True
        self.current_frame = 0
        self.total_frames = 0
        
    def toggle_playback(self):
        """Toggle video playback"""
        self.video_playing = not self.video_playing
        self.play_button.configure(text="▶️ Play" if not self.video_playing else "⏸️ Pause")
        
    def goto_start(self):
        """Go to start of video"""
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            
    def goto_end(self):
        """Go to end of video"""
        if self.video_cap and self.total_frames > 0:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames - 1)
            self.current_frame = self.total_frames - 1
        
    def start_preview(self):
        """Start the camera or video preview"""
        if self.use_camera:
            # Use existing configured camera instead of reconfiguring
            if self.camera_manager.is_camera_configured() and self.camera_manager.selected_camera_id == self.camera_id:
                self.cap = self.camera_manager.get_current_camera()
                if self.cap:
                    # Test that the camera is still working
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.preview_running = True
                        resolution = self.camera_manager.selected_resolution
                        fps = self.camera_manager.selected_framerate
                        self.status_label.configure(
                            text=f"Camera preview active: {resolution[0]}x{resolution[1]} @ {fps:.1f}fps", 
                            foreground='green')
                        return
                    else:
                        print("Existing camera capture not working, will reconfigure")
                        
            # Fallback: Configure camera if no existing capture or it's not working
            self.cap = self.camera_manager.configure_camera(self.camera_id)
            if not self.cap:
                self.status_label.configure(text="Failed to configure camera", foreground='red')
                return
                
            self.preview_running = True
            self.status_label.configure(text="Camera preview active", foreground='green')
        else:
            # Configure video file
            try:
                self.video_cap = cv2.VideoCapture(self.video_path)
                if not self.video_cap.isOpened():
                    self.status_label.configure(text="Failed to open video file", foreground='red')
                    return
                    
                # Get video properties
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                
                self.preview_running = True
                self.status_label.configure(text=f"Video loaded: {self.total_frames} frames at {fps:.1f} FPS", 
                                          foreground='green')
                
            except Exception as e:
                self.status_label.configure(text=f"Error loading video: {str(e)}", foreground='red')
                return
        
        # Start preview thread
        self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        self.preview_thread.start()
        
    def preview_loop(self):
        """Main preview loop running in separate thread"""
        try:
            while self.preview_running:
                if self.use_camera and self.cap:
                    ret, frame = self.cap.read()
                elif not self.use_camera and self.video_cap:
                    # Video mode
                    if self.video_playing:
                        ret, frame = self.video_cap.read()
                        if ret:
                            self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                            # Update progress display
                            if hasattr(self, 'progress_label'):
                                self.window.after(0, lambda: self.progress_label.configure(
                                    text=f"Frame: {self.current_frame} / {self.total_frames}"))
                        else:
                            # End of video - loop back to start
                            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.current_frame = 0
                            continue
                    else:
                        # Paused - just read current frame again
                        current_pos = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
                        ret, frame = self.video_cap.read()
                else:
                    ret, frame = False, None
                    
                if not ret or frame is None:
                    if self.use_camera:
                        continue
                    else:
                        time.sleep(0.033)  # Keep loop running even when paused
                        continue
                    
                try:
                    # Original frame (for display)
                    original_display = self.prepare_display_image(frame)
                    
                    # Apply all corrections at native resolution first
                    corrected_frame = self.apply_all_corrections(frame)
                    
                    # Prepare corrected frame for display
                    corrected_display = self.prepare_display_image(corrected_frame) if corrected_frame is not None else None
                    
                    # Update display in main thread
                    if self.preview_running:
                        self.window.after(0, self.update_display, original_display, corrected_display)
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    
                # Adjust timing based on mode
                if self.use_camera:
                    time.sleep(0.033)  # ~30 FPS for camera
                else:
                    # For video, respect original FPS when playing
                    if self.video_playing and self.video_cap:
                        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
                        time.sleep(1.0 / fps)
                    else:
                        time.sleep(0.1)  # Slower refresh when paused
                
        except Exception as e:
            print(f"Error in calibration preview loop: {e}")
            
    def apply_all_corrections(self, frame):
        """Apply lens distortion and perspective corrections at native resolution."""
        return apply_corrections(frame, self.calibration_data)
            
    def prepare_display_image(self, frame):
        """Prepare frame for display by resizing and converting to PhotoImage"""
        if frame is None:
            return None
            
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width, canvas_height = 600, 450  # Updated canvas size
        height, width = frame.shape[:2]
        
        scale = min(canvas_width/width, canvas_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(rgb_frame)
        return ImageTk.PhotoImage(pil_image)
        
    def update_display(self, original_photo, corrected_photo):
        """Update the display canvases with new images"""
        try:
            if not self.preview_running:
                return
                
            # Update original image
            if original_photo:
                self.canvas_original.delete("all")
                self.canvas_original.create_image(300, 225, image=original_photo)  # Center in 600x450 canvas
                self.canvas_original.image = original_photo  # Keep reference
                
            # Update corrected image
            if corrected_photo:
                self.canvas_corrected.delete("all")
                self.canvas_corrected.create_image(300, 225, image=corrected_photo)  # Center in 600x450 canvas
                self.canvas_corrected.image = corrected_photo  # Keep reference
            else:
                # Show status message when no corrections are available
                self.canvas_corrected.delete("all")
                if not self.calibration_data.is_calibrated and not self.calibration_data.perspective_corrected:
                    self.canvas_corrected.create_text(300, 225, 
                                                    text="No Calibration Data\nAvailable", 
                                                    fill='red', font=('Arial', 16), justify='center')
                elif not self.calibration_data.is_calibrated:
                    self.canvas_corrected.create_text(300, 225, 
                                                    text="No Lens\nCalibration", 
                                                    fill='orange', font=('Arial', 16), justify='center')
                elif not self.calibration_data.perspective_corrected:
                    self.canvas_corrected.create_text(300, 225, 
                                                    text="No Perspective\nCorrection", 
                                                    fill='orange', font=('Arial', 16), justify='center')
                
        except Exception as e:
            print(f"Error updating calibration preview display: {e}")
            
    def close_window(self):
        """Close the preview window and cleanup"""
        self.preview_running = False
        
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
            
        if self.window:
            self.window.destroy()
