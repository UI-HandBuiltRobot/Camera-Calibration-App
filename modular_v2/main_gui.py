"""
Main GUI application for Added-Mass-Lab
Central coordinator for all GUI components and workflows
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
from pathlib import Path

from .camera_manager import CameraManager
from .data_models import CalibrationData
from .camera_selection import CameraSelectionWindow
from .calibration_recorder import CalibrationRecorder
from .calibration_processor import CalibrationProcessor
from .perspective_corrector import PerspectiveCorrector
from .calibration_preview import CalibrationPreviewWindow
from .measurement_recorder import MeasurementVideoRecorder
from .tracking_v7 import VideoTracker


class MainGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Camera Measurement Application")
        # Initial size accommodates the worst-case calibration status text
        # (~7 lines once both lens + perspective + real-world scaling are
        # set) without the action buttons getting clipped.  Auto-grows
        # further if content ever exceeds this; see _fit_window_to_content.
        self.root.geometry("800x800")
        self.root.minsize(700, 650)
        
        # Set up window close handler for cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize camera manager
        self.camera_manager = CameraManager()
        
        # Initialize calibration data (global storage)
        self.calibration_data = CalibrationData()
        
        # Initialize calibration recorder
        self.calibration_recorder = CalibrationRecorder(self.root, self.camera_manager)
        
        # Initialize calibration processor
        self.calibration_processor = CalibrationProcessor(self.root, self.calibration_data)
        
        # Initialize perspective corrector
        self.perspective_corrector = PerspectiveCorrector(self.root, self.camera_manager, self.calibration_data)
        
        # Initialize measurement video recorder
        self.measurement_recorder = MeasurementVideoRecorder(self.root, self.camera_manager, self.calibration_data)
        
        # Initialize video tracker
        self.video_tracker = VideoTracker(self.root, self.calibration_data)
        
        # Camera settings (will be set by camera selection)
        self.selected_camera_id = None
        self.selected_resolution = None
        self.selected_framerate = None
        self.camera_selection_window = None  # Track camera selection window instance
        
        # Video source selection (webcam vs prerecorded)
        self.video_source_var = tk.StringVar(value="webcam")
        
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Camera Measurement Application", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Video source selection frame
        source_frame = ttk.LabelFrame(main_frame, text="Video Source", padding="10")
        source_frame.pack(fill='x', pady=(0, 10))
        
        # Radio buttons for video source
        ttk.Radiobutton(source_frame, text="I'm recording with a webcam", 
                       variable=self.video_source_var, value="webcam",
                       command=self.on_video_source_change).pack(anchor='w')
        ttk.Radiobutton(source_frame, text="I'm using pre-recorded videos (e.g. from my phone)", 
                       variable=self.video_source_var, value="prerecorded",
                       command=self.on_video_source_change).pack(anchor='w')
        
        # Camera status frame
        status_frame = ttk.LabelFrame(main_frame, text="Camera Status", padding="10")
        status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="No camera selected", 
                                     foreground='red')
        self.status_label.pack()
        
        # Calibration status frame
        calibration_frame = ttk.LabelFrame(main_frame, text="Calibration Status", padding="10")
        calibration_frame.pack(fill='x', pady=(0, 20))
        
        self.calibration_status_label = ttk.Label(calibration_frame, text="No calibration loaded", 
                                                 foreground='orange')
        self.calibration_status_label.pack()
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='both', expand=True)
        
        # Button configurations
        button_configs = [
            ("Select Camera", self.select_camera),
            ("New Camera Calibration", self.new_calibration),
            ("Calibrate from pre-recorded video", self.load_calibration),
            ("Import Existing Calibration", self.import_calibration),
            ("Export Calibration to File", self.export_calibration),
            ("Set Perspective & Output Region", self.recalibrate_perspective),
            ("Preview Camera Calibration Results", self.preview_calibration),
            ("Record New Tracking Video", self.record_video),
            ("Track Motions from Video", self.track_motions)
        ]

        # Create buttons and store references
        from .tooltip import ToolTip
        self.buttons = {}
        button_tooltips = {
            "Select Camera": "Choose which camera to use and set its resolution and framerate.",
            "New Camera Calibration": "Record a checkerboard video to create a new camera calibration.",
            "Calibrate from pre-recorded video": "Run a fresh calibration using an already-recorded checkerboard video.",
            "Import Existing Calibration": "Load a previously-exported calibration JSON (the same file produced by Export).",
            "Export Calibration to File": "Save camera intrinsics and perspective matrix to a JSON file for use with OpenCV.",
            "Set Perspective & Output Region": "Re-run perspective calibration and output ROI / coordinate-frame selection without repeating the lens calibration step.",
            "Preview Camera Calibration Results": "View how the current calibration corrects the camera image.",
            "Record New Tracking Video": "Record a measurement video (recommended after calibration).",
            "Track Motions from Video": "Open a video and run the motion tracking analysis on it."
        }
        for text, command in button_configs:
            btn = ttk.Button(buttons_frame, text=text, command=command, width=30)
            btn.pack(pady=5)
            self.buttons[text] = btn
            # Add tooltip to each button
            ToolTip(btn, button_tooltips.get(text, ""))
        # Set initial button states
        self.update_button_states(camera_selected=False)

        # After GUI is created, optionally show the startup tutorial
        self.show_tutorial_if_needed()

    def _prefs_path(self):
        """Return path to preferences file used for storing tutorial preference."""
        try:
            p = Path.home() / ".aml_gui_prefs.json"
            return p
        except Exception:
            return Path(os.path.dirname(__file__)) / "aml_gui_prefs.json"

    def load_prefs(self):
        """Load preferences from disk (returns dict)."""
        p = self._prefs_path()
        if not p.exists():
            return {}
        try:
            with p.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_prefs(self, prefs: dict):
        """Save prefs dict to disk."""
        p = self._prefs_path()
        try:
            with p.open('w', encoding='utf-8') as f:
                json.dump(prefs, f, indent=2)
        except Exception:
            # best-effort only
            pass

    def show_tutorial_if_needed(self):
        """Show a small, skippable tutorial dialog on startup based on user preference."""
        prefs = self.load_prefs()
        show = prefs.get('show_tutorial_on_startup', True)
        if not show:
            return

        # Build tutorial dialog
        dlg = tk.Toplevel(self.root)
        dlg.title("Quick Start Tutorial")
        dlg.geometry("560x360")
        dlg.transient(self.root)
        dlg.grab_set()

        frame = ttk.Frame(dlg, padding=12)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text="Welcome to the Camera Measurement Application", font=('Arial', 14, 'bold')).pack(pady=(0, 8))

        tutorial_lines = [
            "1) Select Camera — choose and configure the camera you will use.",
            "2) New Camera Calibration — record a checkerboard video to compute lens and perspective correction.",
            "3) Preview Camera Calibration Results — inspect how correction affects the image.",
            "4) Record New Tracking Video — record measurement videos (recommended after calibration).",
            "5) Track Motions from Video — open a video and run the motion-tracking analysis.",
        ]

        for line in tutorial_lines:
            ttk.Label(frame, text=line, wraplength=520, justify='left').pack(anchor='w', pady=2)

        ttk.Label(frame, text="").pack()

        # Don't show again checkbox
        dont_show_var = tk.BooleanVar(value=not prefs.get('saw_tutorial', False))
        chk = ttk.Checkbutton(frame, text="Show this tutorial on startup", variable=dont_show_var)
        chk.pack(anchor='w')

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=(12, 0))

        def on_close():
            # Save preference
            prefs['show_tutorial_on_startup'] = bool(dont_show_var.get())
            prefs['saw_tutorial'] = True
            self.save_prefs(prefs)
            dlg.grab_release()
            dlg.destroy()

        ttk.Button(btn_frame, text="Got it — Close", command=on_close).pack(side='right')
        ttk.Button(btn_frame, text="Skip", command=on_close).pack(side='right', padx=(0, 8))
        
    def on_video_source_change(self):
        """Handle video source selection change"""
        # Update button states when video source changes
        camera_selected = self.selected_camera_id is not None
        self.update_button_states(camera_selected=camera_selected)
        
    def update_button_states(self, camera_selected):
        """Update button states based on camera selection, calibration, and video source"""
        is_webcam = self.video_source_var.get() == "webcam"
        is_prerecorded = self.video_source_var.get() == "prerecorded"
        
        for button_text, button in self.buttons.items():
            if button_text == "Select Camera":
                # Disabled when using prerecorded videos
                if is_prerecorded:
                    button.configure(state='disabled')
                else:
                    button.configure(state='normal')
            elif button_text == "New Camera Calibration":
                # Enabled for webcam AND camera selected
                if is_webcam and camera_selected:
                    button.configure(state='normal')
                else:
                    button.configure(state='disabled')
            elif button_text == "Record New Tracking Video":
                # Enabled for webcam AND camera selected
                if is_webcam and camera_selected:
                    button.configure(state='normal')
                else:
                    button.configure(state='disabled')
            elif button_text == "Preview Camera Calibration Results":
                # Enabled when calibration is loaded
                if self.calibration_data.is_calibrated:
                    button.configure(state='normal')
                else:
                    button.configure(state='disabled')
            elif button_text == "Export Calibration to File":
                # Enabled when calibration is loaded
                if self.calibration_data.is_calibrated:
                    button.configure(state='normal')
                else:
                    button.configure(state='disabled')
            elif button_text in ["Calibrate from pre-recorded video",
                                 "Import Existing Calibration",
                                 "Track Motions from Video"]:
                # Always enabled
                button.configure(state='normal')
        
    def select_camera(self):
        """Open camera selection window"""
        # Check if camera selection window is already open
        if self.camera_selection_window and hasattr(self.camera_selection_window, 'window') and self.camera_selection_window.window:
            try:
                # Bring existing window to front
                self.camera_selection_window.window.lift()
                self.camera_selection_window.window.focus_force()
                return
            except tk.TclError:
                # Window was destroyed, clear the reference
                self.camera_selection_window = None
        
        # Create new camera selection window
        self.camera_selection_window = CameraSelectionWindow(self.root, self.camera_manager)
        self.camera_selection_window.show(callback=self.on_camera_selected)
        
    def on_camera_selected(self, camera_id, resolution, framerate):
        """Handle camera selection confirmation"""
        print(f"DEBUG: on_camera_selected called with camera_id={camera_id}")
        
        self.selected_camera_id = camera_id
        self.selected_resolution = resolution
        self.selected_framerate = framerate
        
        # IMPORTANT: Update camera manager state as well 
        # This ensures calibration recorder sees the selected camera
        self.camera_manager.selected_camera_id = camera_id
        self.camera_manager.selected_resolution = resolution
        self.camera_manager.selected_framerate = framerate
        
        # Clear the camera selection window reference
        self.camera_selection_window = None
        
        print(f"DEBUG: After setting - self.selected_camera_id = {self.selected_camera_id}")
        print(f"DEBUG: After setting - camera_manager.selected_camera_id = {self.camera_manager.selected_camera_id}")
        
        # Reset calibration data when camera changes
        self.calibration_data.reset()
        
        # Update status
        status_text = f"Camera {camera_id} selected: {resolution[0]}x{resolution[1]} @ {framerate:.1f}fps"
        self.status_label.configure(text=status_text, foreground='green')
        
        # Update calibration status
        self.update_calibration_status()
        
        # Update button states - camera is now selected
        self.update_button_states(camera_selected=True)
                
        print(f"Camera configured: ID={camera_id}, Resolution={resolution}, FPS={framerate}")
        
    def update_calibration_status(self):
        """Update calibration status display and button states"""
        status_text = self.calibration_data.get_calibration_info()
        color = 'green' if self.calibration_data.is_calibrated else 'orange'
        self.calibration_status_label.configure(text=status_text, foreground=color)

        # The calibration status text grows from one line ("No calibration
        # loaded") up to ~7 lines once lens + perspective + real-world scale
        # are all set.  Grow the window if needed so the action buttons
        # below stay visible.
        self._fit_window_to_content()

        # Update button states when calibration status changes
        camera_selected = self.selected_camera_id is not None
        self.update_button_states(camera_selected)

    def _fit_window_to_content(self):
        """Expand the root window if its required size now exceeds the
        current size.  Never shrinks — if the user has manually enlarged
        the window we don't fight them.  Called whenever a status label
        or other dynamic content might have grown.
        """
        self.root.update_idletasks()
        req_w = self.root.winfo_reqwidth()
        req_h = self.root.winfo_reqheight()
        cur_w = self.root.winfo_width()
        cur_h = self.root.winfo_height()
        new_w = max(req_w, cur_w)
        new_h = max(req_h, cur_h)
        if new_w != cur_w or new_h != cur_h:
            self.root.geometry(f"{new_w}x{new_h}")
    
    def show_calibration_instructions(self):
        """Show calibration instructions popup with image"""
        try:
            import os
            from PIL import Image, ImageTk
            
            # Create popup window
            popup = tk.Toplevel(self.root)
            popup.title("Camera Calibration Instructions")
            popup.geometry("600x500")
            popup.resizable(True, True)
            popup.transient(self.root)
            popup.grab_set()
            
            # Center the popup
            popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 50))
            
            # Main frame with padding
            main_frame = ttk.Frame(popup, padding="20")
            main_frame.pack(fill='both', expand=True)
            
            # Instructions text
            instructions_text = (
                "To calibrate your camera lens, move either the camera or the checkerboard.\n\n"
                "Your camera's position during this stage is not important."
            )
            
            instructions_label = ttk.Label(main_frame, text=instructions_text, 
                                         font=('Arial', 12), justify='center', wraplength=550)
            instructions_label.pack(pady=(0, 20))
            
            # Try to load and display the image
            try:
                image_path = os.path.join(os.path.dirname(__file__), "illustration_images", "CameraCalibrationMethods.png")
                if os.path.exists(image_path):
                    # Load and resize image to fit nicely in popup
                    pil_image = Image.open(image_path)
                    # Calculate size to fit within popup while maintaining aspect ratio
                    max_width = 500
                    max_height = 300
                    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Display image
                    image_label = ttk.Label(main_frame, image=photo)
                    image_label.image = photo  # Keep a reference
                    image_label.pack(pady=(0, 20))
                else:
                    # Image not found, show placeholder
                    ttk.Label(main_frame, text="[Calibration illustration not found]", 
                             font=('Arial', 10), foreground='gray').pack(pady=(0, 20))
            except Exception as e:
                print(f"Error loading calibration image: {e}")
                ttk.Label(main_frame, text="[Error loading calibration illustration]", 
                         font=('Arial', 10), foreground='gray').pack(pady=(0, 20))
            
            # OK button to close popup
            ok_button = ttk.Button(main_frame, text="OK", command=popup.destroy)
            ok_button.pack(pady=(20, 0))
            
            # Wait for popup to be closed
            popup.wait_window()
            
        except Exception as e:
            print(f"Error showing calibration instructions: {e}")
            # Fall back to simple message box if popup fails
            messagebox.showinfo("Camera Calibration Instructions",
                              "To calibrate your camera lens, move either the camera or the checkerboard.\n\n"
                              "Your camera's position during this stage is not important.")
        
    def new_calibration(self):
        """Handle new camera calibration"""
        # Debug logging for troubleshooting
        print(f"DEBUG: new_calibration called")
        print(f"DEBUG: self.selected_camera_id = {self.selected_camera_id}")
        print(f"DEBUG: camera_manager.selected_camera_id = {getattr(self.camera_manager, 'selected_camera_id', 'NOT_SET')}")
        
        # Check both our state AND camera manager state for safety (explicitly check for None, not falsy values)
        if self.selected_camera_id is None and getattr(self.camera_manager, 'selected_camera_id', None) is None:
            print("DEBUG: Camera selection check failed - both IDs are None")
            messagebox.showerror("Error", "Please select a camera first.")
            return
        
        print("DEBUG: Camera selection check passed - proceeding with calibration")
        
        # If we have camera manager state but not main GUI state, sync them
        if self.selected_camera_id is None and getattr(self.camera_manager, 'selected_camera_id', None) is not None:
            print("DEBUG: Syncing camera state from camera manager")
            self.selected_camera_id = self.camera_manager.selected_camera_id
            self.selected_resolution = getattr(self.camera_manager, 'selected_resolution', None)
            self.selected_framerate = getattr(self.camera_manager, 'selected_framerate', None)
            
        # Show calibration instructions popup with image
        self.show_calibration_instructions()
        
        print("DEBUG: About to call calibration_recorder.show_recording_dialog()")
        
        # Set up callback to process video after recording
        original_callback = self.calibration_recorder.completion_callback if hasattr(self.calibration_recorder, 'completion_callback') else None
        self.calibration_recorder.completion_callback = self.on_calibration_recorded
        
        self.calibration_recorder.show_recording_dialog()
        
    def on_calibration_recorded(self, video_path):
        """Called when calibration recording is complete"""
        if video_path and os.path.exists(video_path):
            # Automatically process the recorded video
            self.calibration_processor.load_calibration_video(video_path)
            
            # Set up callback to update status after processing
            original_callback = self.calibration_processor.completion_callback if hasattr(self.calibration_processor, 'completion_callback') else None
            self.calibration_processor.completion_callback = self.on_calibration_processed
            
    def on_calibration_processed(self):
        """Called when calibration processing is complete"""
        self.update_calibration_status()
        
        # Start perspective correction process
        self.perspective_corrector.completion_callback = self.on_perspective_correction_complete
        self.perspective_corrector.show_perspective_correction_dialog()
        
    def on_perspective_correction_complete(self):
        """Called when perspective correction is complete or skipped"""
        self.update_calibration_status()

        # Close the calibration processor window if it's still open
        if hasattr(self.calibration_processor, 'window') and self.calibration_processor.window:
            self.calibration_processor.close_window()

        # Show comprehensive completion message including calibration info
        calibration_info = self.calibration_data.get_calibration_info()
        completion_message = (
            "Camera calibration and perspective correction are complete!\n\n"
            f"{calibration_info}\n\n"
            "You can now proceed with experiments."
        )

        messagebox.showinfo("Calibration Setup Complete", completion_message)

    def recalibrate_perspective(self):
        """Launch perspective calibration + bbox / coordinate-frame selection
        without requiring a new lens calibration.  Any existing lens calibration
        (is_calibrated) is preserved and applied; if none is loaded the warp
        step runs on the raw frame."""
        self.perspective_corrector.completion_callback = self.on_perspective_correction_complete
        self.perspective_corrector.show_perspective_correction_dialog()
        
    def load_calibration(self):
        """Handle load existing calibration with resolution check"""
        # First, let user select the calibration video
        from tkinter import filedialog
        video_path = filedialog.askopenfilename(
            title="Load Calibration Video",
            filetypes=[("Video files", "*.avi *.mp4 *.mov"), ("All files", "*.*")]
        )
        
        if not video_path or not os.path.exists(video_path):
            return
            
        # Check if a camera is selected and get its resolution
        if self.selected_camera_id is not None and self.selected_resolution is not None:
            # Get video resolution
            video_resolution = self.calibration_processor.get_video_resolution(video_path)
            
            if video_resolution is not None:
                camera_width, camera_height = self.selected_resolution
                video_width, video_height = video_resolution
                
                # Check if resolutions match (allow small tolerance)
                if (abs(camera_width - video_width) > 10 or 
                    abs(camera_height - video_height) > 10):
                    
                    # Show warning dialog
                    from tkinter import messagebox
                    message = (f"Resolution Mismatch Warning\n\n"
                              f"Camera resolution: {camera_width}x{camera_height}\n"
                              f"Video resolution: {video_width}x{video_height}\n\n"
                              f"The camera resolution does not match the calibration video resolution. "
                              f"This may cause issues with lens distortion correction.\n\n"
                              f"Would you like to:\n"
                              f"• Yes: Reconfigure camera to match video resolution\n"
                              f"• No: Continue with current camera resolution\n"
                              f"• Cancel: Select a different video")
                    
                    result = messagebox.askyesnocancel("Resolution Mismatch", message)
                    
                    if result is None:  # Cancel
                        return
                    elif result:  # Yes - reconfigure camera
                        # Try to reconfigure camera to match video resolution
                        cap = self.camera_manager.configure_camera_resolution(
                            self.selected_camera_id, video_resolution)
                            
                        if cap is not None:
                            # Update our stored resolution
                            self.selected_resolution = self.camera_manager.selected_resolution
                            self.selected_framerate = self.camera_manager.selected_framerate
                            
                            # Update status
                            status_text = (f"Camera {self.selected_camera_id} reconfigured: "
                                         f"{self.selected_resolution[0]}x{self.selected_resolution[1]} "
                                         f"@ {self.selected_framerate:.1f}fps")
                            self.status_label.configure(text=status_text, foreground='green')
                            
                            # Release camera after reconfiguration to free resources
                            # The camera will be reacquired when needed by other modules
                            self.camera_manager.release_camera()
                            print(f"Camera released after resolution reconfiguration")
                            
                            messagebox.showinfo("Success", 
                                              f"Camera successfully reconfigured to {video_resolution[0]}x{video_resolution[1]}")
                        else:
                            messagebox.showwarning("Warning", 
                                                 f"Could not reconfigure camera to {video_resolution[0]}x{video_resolution[1]}. "
                                                 f"Continuing with current camera resolution.")
                    # If No was selected, continue with current resolution
        
        # Proceed with loading the calibration video
        self.calibration_processor.load_calibration_video(video_path)
        
        # Set up callback to update status after processing
        self.calibration_processor.completion_callback = self.on_existing_calibration_loaded
        
    def on_existing_calibration_loaded(self):
        """Called when existing calibration is loaded - now includes perspective correction"""
        self.update_calibration_status()
        
        # Start perspective correction process (same as new calibration workflow)
        self.perspective_corrector.completion_callback = self.on_perspective_correction_complete
        self.perspective_corrector.show_perspective_correction_dialog()
        
    def export_calibration(self):
        """Export calibration data to a JSON file for use with OpenCV"""
        if not self.calibration_data.is_calibrated:
            messagebox.showwarning("Warning", "No calibration data available. Please run or load a camera calibration first.")
            return

        from tkinter import filedialog
        save_path = filedialog.asksaveasfilename(
            title="Export Calibration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="camera_calibration.json"
        )
        if not save_path:
            return

        cal = self.calibration_data

        # Build the export dict with OpenCV-compatible naming
        export = {
            "calibration_info": {
                "checkerboard_size": list(cal.checkerboard_size) if cal.checkerboard_size is not None else None,
                "image_size": list(cal.image_size) if cal.image_size is not None else None,
                "calibration_error_pixels": float(cal.calibration_error) if cal.calibration_error is not None else None
            },
            "intrinsics": {
                "camera_matrix": cal.camera_matrix.tolist() if cal.camera_matrix is not None else None,
                "distortion_coefficients": cal.distortion_coefficients.tolist() if cal.distortion_coefficients is not None else None,
                # Lens distortion model — selects the right cv2 call at runtime.
                # "pinhole"/"rational" → cv2.undistort; "fisheye" → cv2.fisheye.undistortImage;
                # "scaramuzza" → _build_scaramuzza_maps + cv2.remap (no camera_matrix/dist_coeffs).
                "model_type": getattr(cal, 'model_type', 'pinhole'),
                "fisheye_balance": float(getattr(cal, 'fisheye_balance', 0.0)),
                # Scaramuzza / OCamCalib model parameters (model_type=="scaramuzza" only).
                "scaramuzza_params": getattr(cal, 'scaramuzza_params', None),
                "scaramuzza_fov": float(getattr(cal, 'scaramuzza_fov', 180.0)),
            },
            "extrinsics": {
                "perspective_corrected": bool(cal.perspective_corrected),
                "perspective_matrix": cal.perspective_matrix.tolist() if cal.perspective_matrix is not None else None,
                # User-defined output region in pre-flip world coords (native homography scale)
                "output_bbox_world": (
                    list(cal.output_bbox_world)
                    if getattr(cal, 'output_bbox_world', None) is not None
                    else None),
                # Coordinate-frame orientation state (0-7).  Affects how
                # tracker output X/Y are interpreted; the displayed image
                # is unchanged.  See data_models.FRAME_ORIENTATION_BASES.
                "frame_orientation_state": int(
                    getattr(cal, 'frame_orientation_state', 0))
            },
            "scaling": {
                "real_world_scale": bool(cal.real_world_scale),
                "square_size_real": float(cal.square_size_real) if cal.square_size_real is not None else None,
                "square_size_pixels": float(cal.square_size_pixels) if cal.square_size_pixels is not None else None,
                "pixels_per_real_unit": float(cal.pixels_per_real_unit) if cal.pixels_per_real_unit is not None else None,
                # Native (pre-downsample) scale values, used by Import to reconstruct
                # the runtime downsample-idempotency. External consumers that just
                # want the "live" scale should keep using square_size_pixels /
                # pixels_per_real_unit above.
                "square_size_pixels_native": (
                    float(cal.square_size_pixels_native)
                    if getattr(cal, 'square_size_pixels_native', None) is not None
                    else None),
                "pixels_per_real_unit_native": (
                    float(cal.pixels_per_real_unit_native)
                    if getattr(cal, 'pixels_per_real_unit_native', None) is not None
                    else None),
                "output_scale": float(getattr(cal, 'output_scale', 1.0))
            },
        }

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, indent=2)
            messagebox.showinfo("Export Successful",
                                f"Calibration exported to:\n{save_path}\n\n"
                                "The file contains:\n"
                                "  • camera_matrix  — 3×3 intrinsic matrix\n"
                                "  • distortion_coefficients — lens distortion (k1,k2,p1,p2,...)\n"
                                "  • perspective_matrix — 3×3 homography for perspective correction")
        except OSError as e:
            messagebox.showerror("Export Failed", f"Could not write file:\n{e}")

    def import_calibration(self):
        """Import a calibration JSON previously written by export_calibration.

        Reads the same schema as export, populates calibration_data, and
        refreshes the GUI status.  Tolerant of older exports that may be
        missing the additive fields (model_type, _native scaling, output bbox);
        in those cases sensible defaults are filled in and the user is warned
        if perspective correction can't be reconstructed.
        """
        import numpy as np
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Import Calibration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("Import Failed", f"Could not read file:\n{e}")
            return

        try:
            info = data.get("calibration_info", {}) or {}
            intr = data.get("intrinsics", {}) or {}
            extr = data.get("extrinsics", {}) or {}
            scal = data.get("scaling", {}) or {}

            cm = intr.get("camera_matrix")
            dc = intr.get("distortion_coefficients")
            model_type = intr.get("model_type", "rational")
            fisheye_balance = float(intr.get("fisheye_balance", 0.0))
            scaramuzza_params = intr.get("scaramuzza_params", None)
            scaramuzza_fov = float(intr.get("scaramuzza_fov", 180.0))

            # Scaramuzza calibrations have no camera_matrix / dist_coeffs.
            if model_type != "scaramuzza" and (cm is None or dc is None):
                messagebox.showerror(
                    "Import Failed",
                    "JSON is missing camera_matrix or distortion_coefficients.")
                return
            if model_type == "scaramuzza" and scaramuzza_params is None:
                messagebox.showerror(
                    "Import Failed",
                    "JSON model_type is 'scaramuzza' but scaramuzza_params is missing.")
                return

            checkerboard_size = info.get("checkerboard_size")
            image_size = info.get("image_size")
            cal_error = info.get("calibration_error_pixels")

            if image_size is None:
                messagebox.showerror(
                    "Import Failed",
                    "JSON is missing calibration_info.image_size.")
                return

            cd = self.calibration_data
            cd.reset()
            cd.set_calibration(
                np.array(cm, dtype=np.float64) if cm is not None else None,
                np.array(dc, dtype=np.float64) if dc is not None else None,
                tuple(checkerboard_size) if checkerboard_size else None,
                float(cal_error) if cal_error is not None else 0.0,
                tuple(image_size),
                model_type=model_type,
                fisheye_balance=fisheye_balance,
                scaramuzza_params=scaramuzza_params,
                scaramuzza_fov=scaramuzza_fov,
            )

            # Scaling: prefer _native values when present, else fall back to
            # the effective values (older export files won't have _native).
            cd.real_world_scale = bool(scal.get("real_world_scale", False))
            cd.square_size_real = scal.get("square_size_real")
            cd.square_size_pixels = scal.get("square_size_pixels")
            cd.pixels_per_real_unit = float(
                scal.get("pixels_per_real_unit", 1.0))
            cd.square_size_pixels_native = (
                scal.get("square_size_pixels_native")
                if scal.get("square_size_pixels_native") is not None
                else cd.square_size_pixels)
            cd.pixels_per_real_unit_native = float(
                scal.get("pixels_per_real_unit_native",
                         cd.pixels_per_real_unit))
            cd.output_scale = float(scal.get("output_scale", 1.0))

            # Perspective: only commit if both matrix and bbox are present.
            warn_no_perspective = False
            if extr.get("perspective_corrected") and extr.get("perspective_matrix"):
                bbox = extr.get("output_bbox_world")
                if bbox is not None:
                    cd.set_perspective_correction(
                        np.array(extr["perspective_matrix"], dtype=np.float32),
                        tuple(bbox),
                        frame_orientation_state=int(
                            extr.get("frame_orientation_state", 0)))
                else:
                    warn_no_perspective = True

            self.update_calibration_status()

            msg = f"Calibration imported from:\n{path}"
            if warn_no_perspective:
                msg += ("\n\nNote: perspective correction was present in the "
                        "file but the output bbox is missing (older export "
                        "format).  Re-run perspective calibration via "
                        "'Calibrate from pre-recorded video' to set it.")
            messagebox.showinfo("Import Successful", msg)

        except (KeyError, ValueError, TypeError) as e:
            messagebox.showerror("Import Failed",
                                 f"Could not parse calibration JSON:\n{e}")

    def preview_calibration(self):
        """Handle preview camera calibration results"""
        if not self.calibration_data.is_calibrated:
            messagebox.showwarning("Warning", "No calibration data available. Please run camera calibration first.")
            return
            
        # Open calibration preview window (handles both camera and video modes)
        preview_window = CalibrationPreviewWindow(self.root, self.camera_manager, 
                                                self.calibration_data, self.selected_camera_id)
        preview_window.show()
        
    def record_video(self):
        """Handle record new measurement video"""
        # Show explanatory popup about the purpose of measurement recording
        explanation = (
            "Measurement Video Recording\n\n"
            "This feature records a video of the object you want to track.\n\n"
            "Important Notes:\n"
            "• This is NOT live tracking\n"
            "• The video will be saved for later analysis\n"
            "• Object tracking will be performed afterward using the recorded video\n"
            "• Use 'Track Motions from Video' to analyze the recorded footage\n\n"
            "Setup Requirements:\n"
            "• The checkerboard is no longer needed\n"
            "• Only a clear view of the object to be tracked is required\n\n"
            "The recorded video will be calibration-corrected if calibration data is available."
        )
        
        result = messagebox.showinfo("About Measurement Recording", explanation)
        
        # Check if camera is selected
        if self.selected_camera_id is None:
            messagebox.showwarning("Warning", "No camera selected. Please select a camera first to record videos.")
            return
            
        # Check if calibration is available (recommended but not required)
        if not self.calibration_data.is_calibrated and not self.calibration_data.perspective_corrected:
            result = messagebox.askyesno("No Calibration", 
                                       "No calibration data is available. The recorded video will not be corrected.\n\n"
                                       "Do you want to continue recording without corrections?")
            if not result:
                return
        
        # Show recording interface
        self.measurement_recorder.completion_callback = self.on_measurement_video_complete
        self.measurement_recorder.show_recording_dialog()
        
    def on_measurement_video_complete(self, corrected_video_path):
        """Called when measurement video recording and processing is complete"""
        messagebox.showinfo("Recording Complete", 
                           f"Measurement video has been recorded and processed.\n\n"
                           f"Calibration-corrected video saved to:\n{corrected_video_path}")
        
        # Auto-close the measurement recording window
        if hasattr(self, 'measurement_recorder') and self.measurement_recorder.window:
            try:
                if self.measurement_recorder.window.winfo_exists():
                    self.measurement_recorder.window.destroy()
            except tk.TclError:
                pass  # Window already closed
        
    def track_motions(self):
        """Handle track motions from video using tracking_v7"""
        # Destroy any existing tracking window to avoid multiple instances
        if hasattr(self.video_tracker, 'window') and self.video_tracker.window:
            try:
                self.video_tracker.window.destroy()
                self.video_tracker.window = None
            except tk.TclError:
                # Window may already be destroyed
                self.video_tracker.window = None
        
        # Note: Video tracking doesn't require a live camera - works with video files
        # Uses compact v7 tracker with native resolution and v4-compatible performance
        self.video_tracker.show_tracking_window()
        
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window close event"""
        self.cleanup()
        self.root.destroy()
        
    def cleanup(self):
        """Cleanup resources before closing"""
        if hasattr(self, 'camera_manager') and self.camera_manager:
            self.camera_manager.release_camera()
        
    def __del__(self):
        """Cleanup when application closes"""
        self.cleanup()
