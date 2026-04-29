"""
Camera selection window for Added-Mass-Lab GUI
Provides camera selection interface with live preview
"""

import tkinter as tk
from tkinter import ttk, messagebox
from .tooltip import ToolTip
import threading
import time
import cv2
from PIL import Image, ImageTk


class CameraSelectionWindow:
    """Window for camera selection with live preview"""
    
    def __init__(self, parent, camera_manager):
        self.parent = parent
        self.camera_manager = camera_manager
        self.window = None
        self.preview_label = None
        self.preview_thread = None
        self.preview_running = False
        self.selected_camera_var = None
        self.current_preview_cap = None
        self.callback = None
        self.last_frame = None  # Store last frame for dynamic resizing
        
    def show(self, callback=None):
        """Show camera selection window"""
        self.callback = callback
        
        # Show status window during camera detection
        status_window = self._create_status_window()
        
        # Detect cameras in a separate thread to keep UI responsive
        detection_thread = threading.Thread(target=self._detect_and_show_cameras, args=(status_window,))
        detection_thread.daemon = True
        detection_thread.start()
        
    def _create_status_window(self):
        """Create status window for camera detection"""
        status_window = tk.Toplevel(self.parent)
        status_window.title("Camera Detection")
        status_window.geometry("400x150")
        status_window.transient(self.parent)
        status_window.grab_set()
        
        # Center the window
        status_window.geometry("+%d+%d" % (self.parent.winfo_rootx() + 150, self.parent.winfo_rooty() + 100))
        
        frame = ttk.Frame(status_window, padding="20")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Detecting available cameras...", 
                 font=('Arial', 12)).pack(pady=(20, 10))
        
        # Progress bar
        progress = ttk.Progressbar(frame, mode='indeterminate')
        progress.pack(pady=10, fill='x')
        progress.start()
        
        status_window.progress = progress  # Store reference for cleanup
        return status_window
        
    def _detect_and_show_cameras(self, status_window):
        """Detect cameras and show selection window"""
        try:
            # Detect cameras
            cameras = self.camera_manager.detect_cameras()
            
            # Close status window
            self.parent.after(0, lambda: self._close_status_and_show_selection(status_window, cameras))
        except Exception as e:
            # Handle errors
            self.parent.after(0, lambda: self._close_status_and_show_error(status_window, str(e)))
            
    def _close_status_and_show_selection(self, status_window, cameras):
        """Close status window and show camera selection"""
        # Clean up status window
        if status_window and status_window.winfo_exists():
            status_window.progress.stop()
            status_window.destroy()
            
        if not cameras:
            messagebox.showerror("No Cameras", "No valid cameras found on the system.")
            return
            
        self._create_selection_window(cameras)
        
    def _close_status_and_show_error(self, status_window, error_msg):
        """Close status window and show error"""
        if status_window and status_window.winfo_exists():
            status_window.progress.stop()
            status_window.destroy()
            
        messagebox.showerror("Camera Detection Error", f"Failed to detect cameras: {error_msg}")
        
    def _create_selection_window(self, cameras):
        """Create the camera selection window"""
            
        # Create selection window
        self.window = tk.Toplevel(self.parent)
        self.window.title("Select Camera")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Camera selection frame
        selection_frame = ttk.Frame(self.window)
        selection_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(selection_frame, text="Available Cameras:", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        # Radio buttons for camera selection
        # Initialize with -1 to ensure no camera is pre-selected
        self.selected_camera_var = tk.IntVar(value=-1)
        
        for camera in cameras:
            radio = ttk.Radiobutton(
                selection_frame,
                text=f"{camera['name']}",
                variable=self.selected_camera_var,
                value=camera['id'],
                command=self.on_camera_selection_change
            )
            radio.pack(anchor='w', pady=2)
        # Preview frame (dynamically resizable)
        preview_frame = ttk.LabelFrame(self.window, text="Camera Preview")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # The preview label will display the image; keep it centered
        self.preview_label = ttk.Label(preview_frame, text="Select a camera to see preview")
        self.preview_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Bind resize events to update preview scaling
        self.window.bind('<Configure>', self.on_window_resize)

        # Buttons frame
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill='x', padx=10, pady=10)

        # Confirm and Cancel buttons with tooltips
        confirm_btn = ttk.Button(button_frame, text="Confirm Selection", command=self.confirm_selection)
        confirm_btn.pack(side='right', padx=5)
        ToolTip(confirm_btn, "Confirm and use the selected camera for previews and recordings.")

        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.on_close)
        cancel_btn.pack(side='right')
        ToolTip(cancel_btn, "Cancel and close the camera selection window.")

        # Don't auto-select first camera - let user choose explicitly
        # Note: The IntVar is initialized with -1, so no camera is pre-selected
            
    def on_camera_selection_change(self):
        """Handle camera selection change for preview only"""
        camera_id = self.selected_camera_var.get()
        if camera_id != -1:  # Only start preview if a valid camera is selected
            print(f"Camera selection changed to ID: {camera_id} (preview only)")
            self.start_preview(camera_id)
        else:
            # Stop any existing preview if no camera selected
            self.stop_preview()
            self.preview_label.configure(image="", text="Select a camera to see preview")
        
    def start_preview(self, camera_id):
        """Start camera preview for selected camera (preview only, doesn't affect camera manager state)"""
        print(f"Starting preview for camera {camera_id} (preview only)")
        self.stop_preview()
        
        # Configure camera for preview only - don't use camera_manager.configure_camera
        # which would modify the persistent camera manager state
        cap = self._configure_camera_for_preview(camera_id)
        if cap:
            print(f"Camera {camera_id} configured successfully for preview (no state changes)")
            self.current_preview_cap = cap
            self.preview_running = True
            self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
            self.preview_thread.start()
        else:
            print(f"Failed to configure camera {camera_id} for preview")
            self.preview_label.configure(image="", text="Failed to start camera preview")
    
    def _configure_camera_for_preview(self, camera_id):
        """Configure camera for preview only - doesn't modify camera manager state"""
        try:
            import cv2
            from .config import DEFAULT_CAMERA_BACKEND
            
            # Create a temporary camera capture for preview
            cap = cv2.VideoCapture(camera_id, DEFAULT_CAMERA_BACKEND)
            
            if not cap.isOpened():
                print(f"Failed to open camera {camera_id} for preview")
                return None
                
            # Set basic configuration for preview
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Use a reasonable preview resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test if we can get frames
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Preview camera {camera_id} ready: {frame.shape[1]}x{frame.shape[0]}")
                return cap
            else:
                cap.release()
                return None
                
        except Exception as e:
            print(f"Error configuring camera {camera_id} for preview: {e}")
            if 'cap' in locals():
                try:
                    cap.release()
                except:
                    pass
            return None
    
    def on_window_resize(self, event):
        """Handle window resize events to update preview scaling"""
        # Only respond to resize events for the main window, not child widgets
        if event.widget == self.window:
            # Trigger preview update if running
            if self.preview_running and hasattr(self, 'last_frame'):
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
        return 640, 360
    
    def update_preview_scaling(self):
        """Update preview image scaling based on current window size"""
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            self.update_frame_display(self.last_frame)
            
    def preview_loop(self):
        """Main preview loop running in separate thread"""
        try:
            while self.preview_running and self.current_preview_cap:
                ret, frame = self.current_preview_cap.read()
                if ret and frame is not None:
                    # Store the frame for potential rescaling
                    self.last_frame = frame
                    
                    # Update display in main thread
                    if self.preview_label and self.preview_running:
                        self.window.after(0, self.update_frame_display, frame)
                
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Error in preview loop: {e}")
    
    def update_frame_display(self, frame):
        """Update preview image display with dynamic scaling"""
        try:
            if not self.preview_label or not self.preview_running:
                return
                
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
            
            # Resize frame
            frame_resized = cv2.resize(frame, (new_width, new_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update preview label
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error updating frame display: {e}")
            
    def stop_preview(self):
        """Stop camera preview"""
        self.preview_running = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            
        if self.current_preview_cap:
            self.current_preview_cap.release()
            self.current_preview_cap = None
            
    def confirm_selection(self):
        """Confirm camera selection and close window"""
        camera_id = self.selected_camera_var.get()
        
        # Check if a camera was actually selected (not the default -1)
        if camera_id == -1:
            messagebox.showwarning("No Camera Selected", 
                                 "Please select a camera before confirming.")
            return
        
        print(f"Confirming camera selection: ID={camera_id}")
        
        # Now configure the camera manager for real use (not just preview)
        cap = self.camera_manager.configure_camera(camera_id)
        if not cap:
            messagebox.showerror("Camera Configuration Failed", 
                               f"Failed to configure camera {camera_id}. Please try another camera.")
            return
        
        if self.callback:
            # Get the actual configured resolution and framerate from camera manager
            resolution = (self.camera_manager.selected_resolution[0], self.camera_manager.selected_resolution[1])
            framerate = self.camera_manager.selected_framerate
            
            # Debug logging
            print(f"Camera selection confirmed: ID={camera_id}, Resolution={resolution}, Framerate={framerate}")
            
            self.callback(camera_id, resolution, framerate)
            
        self.on_close()
        
    def on_close(self):
        """Handle window close"""
        self.stop_preview()
        if self.window:
            self.window.destroy()
