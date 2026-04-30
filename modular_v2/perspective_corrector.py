"""
Perspective correction for Added-Mass-Lab GUI
Handles perspective correction using checkerboard in measurement plane
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from .corrections import apply_corrections
from .calibration_processor import CHECKERBOARD_SIZES

# Perspective-correction checkerboard auto-detection.
# We try sizes from largest to smallest by total inner-corner count, because
# a smaller pattern can falsely match a sub-region of a larger physical board
# but a larger pattern cannot match a smaller board. This lets the user pick
# any board they like for perspective (often a larger one than the camera-
# calibration board) without the system mis-identifying it.
PERSPECTIVE_CHECKERBOARD_SIZES = sorted(
    CHECKERBOARD_SIZES, key=lambda s: -(s[0] * s[1])
)


class VideoFrameSelector:
    """Helper class for selecting a frame from a video"""
    
    def __init__(self, parent_window, video_path, video_label, play_button, 
                 progress_slider, progress_var, frame_info_label, perspective_corrector):
        self.parent_window = parent_window
        self.video_path = video_path
        self.video_label = video_label
        self.play_button = play_button
        self.progress_slider = progress_slider
        self.progress_var = progress_var
        self.frame_info_label = frame_info_label
        self.perspective_corrector = perspective_corrector
        
        # Video state
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.current_image = None
        self.playback_thread = None
        
        # Initialize video
        self.initialize_video()
        
        # Set up controls
        self.setup_controls()
        
    def initialize_video(self):
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open video: {self.video_path}")
                return
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_slider.configure(to=self.total_frames - 1)
            
            # Load first frame
            self.goto_frame(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error initializing video: {str(e)}")
            
    def setup_controls(self):
        """Setup control button commands"""
        self.play_button.configure(command=self.toggle_playback)
        
        # Find and configure navigation buttons
        playback_frame = self.play_button.master
        for widget in playback_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                text = widget['text']
                if text == "⏮️ Start":
                    widget.configure(command=self.goto_start)
                elif text == "⏭️ End":
                    widget.configure(command=self.goto_end)
        
        # Find and configure frame navigation buttons
        frame_nav_frame = playback_frame.master.winfo_children()[1]  # Second child should be frame_nav_frame
        for widget in frame_nav_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                text = widget['text']
                if text == "◀◀ -10":
                    widget.configure(command=lambda: self.skip_frames(-10))
                elif text == "◀ -1":
                    widget.configure(command=lambda: self.skip_frames(-1))
                elif text == "▶ +1":
                    widget.configure(command=lambda: self.skip_frames(1))
                elif text == "▶▶ +10":
                    widget.configure(command=lambda: self.skip_frames(10))
        
        # Bind slider
        self.progress_slider.configure(command=self.on_slider_change)
        
    def toggle_playback(self):
        """Toggle video playback"""
        self.is_playing = not self.is_playing
        self.play_button.configure(text="▶️ Play" if not self.is_playing else "⏸️ Pause")
        
        if self.is_playing and not self.playback_thread:
            self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
            self.playback_thread.start()
            
    def playback_loop(self):
        """Video playback loop"""
        while self.is_playing and self.cap:
            if self.current_frame < self.total_frames - 1:
                self.goto_frame(self.current_frame + 1)
                # Get FPS for proper timing
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                time.sleep(1.0 / fps)
            else:
                # End of video
                self.is_playing = False
                self.parent_window.after(0, lambda: self.play_button.configure(text="▶️ Play"))
                break
                
        self.playback_thread = None
        
    def goto_start(self):
        """Go to start of video"""
        self.goto_frame(0)
        
    def goto_end(self):
        """Go to end of video"""
        self.goto_frame(self.total_frames - 1)
        
    def skip_frames(self, delta):
        """Skip frames forward or backward"""
        new_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.goto_frame(new_frame)
        
    def goto_frame(self, frame_number):
        """Go to specific frame"""
        if not self.cap:
            return
            
        self.current_frame = max(0, min(self.total_frames - 1, frame_number))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_image = frame.copy()
            self.update_display(frame)
            
        # Update UI
        self.progress_var.set(self.current_frame)
        self.frame_info_label.configure(text=f"Frame: {self.current_frame + 1} / {self.total_frames}")
        
    def on_slider_change(self, value):
        """Handle slider position change"""
        frame_number = int(float(value))
        if frame_number != self.current_frame:
            self.goto_frame(frame_number)
            
    def update_display(self, frame):
        """Update video display"""
        try:
            # Resize frame for display
            height, width = frame.shape[:2]
            max_width, max_height = 800, 400
            
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            frame_resized = cv2.resize(frame, (new_width, new_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            
        except Exception as e:
            print(f"Display update error: {e}")
            
    def use_current_frame(self):
        """Use the current frame for perspective correction"""
        if self.current_image is not None:
            # Set the loaded image in the perspective corrector
            self.perspective_corrector.loaded_image = self.current_image.copy()
            self.perspective_corrector.use_camera = False
            
            # Close the frame selection dialog
            self.cleanup()
            self.parent_window.destroy()
            
            # Start perspective correction with square size dialog first
            self.perspective_corrector._get_square_size_and_start()
        else:
            messagebox.showerror("Error", "No frame available")
            
    def cleanup(self):
        """Cleanup video resources"""
        self.is_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None


class PerspectiveCorrector:
    """Handles perspective correction using checkerboard in measurement plane"""
    
    def __init__(self, parent, camera_manager, calibration_data):
        self.parent = parent
        self.camera_manager = camera_manager
        self.calibration_data = calibration_data
        self.window = None
        self.preview_label = None
        
        # Camera and preview
        self.cap = None
        self.preview_running = False
        self.preview_thread = None
        
        # Image source mode
        self.use_camera = True  # True for live camera, False for loaded image
        self.loaded_image = None
        
        # Real-world scaling
        self.real_world_square_size = None  # None for pixel units, float for real units

        # Perspective-step checkerboard detection state. The user may pick a
        # different (often larger) board for perspective than the one used for
        # camera calibration. We auto-detect from PERSPECTIVE_CHECKERBOARD_SIZES
        # and lock onto whichever pattern matches first so the live preview
        # doesn't have to retry every size on every frame.
        self._perspective_locked_size = None
        self._perspective_last_detection_time = None
        self._perspective_lock_release_seconds = 2.0

        # Fisheye balance tuning runs once per perspective session, on the first
        # successful frame capture (so the preview shows the actual scene).
        # Reset by start_perspective_correction.
        self._fisheye_balance_tuned = False

        # Completion callback
        self.completion_callback = None

    def _detect_perspective_checkerboard(self, gray, allow_lock=True):
        """Find a checkerboard in `gray` using the perspective sizes list.

        Tries the largest pattern first; on success returns the (size, corners)
        pair. When allow_lock is True, a previously-locked size is tried alone
        until self._perspective_lock_release_seconds elapses without detection,
        then the full largest-first sweep is attempted again.
        """
        current_time = time.time()
        sizes_to_try = list(PERSPECTIVE_CHECKERBOARD_SIZES)

        if allow_lock and self._perspective_locked_size is not None:
            stale = (
                self._perspective_last_detection_time is not None
                and (current_time - self._perspective_last_detection_time)
                > self._perspective_lock_release_seconds
            )
            if stale:
                self._perspective_locked_size = None
                self._perspective_last_detection_time = None
            else:
                sizes_to_try = [self._perspective_locked_size]

        for size in sizes_to_try:
            ret, corners = cv2.findChessboardCorners(gray, size, None)
            if ret:
                if allow_lock:
                    self._perspective_locked_size = size
                    self._perspective_last_detection_time = current_time
                return size, corners

        return None, None
        
    def show_perspective_correction_dialog(self):
        """Show dialog asking if user wants perspective correction and source"""
        # First ask if they want perspective correction
        result = messagebox.askyesno(
            "Perspective Correction",
            "Do you want to correct for perspective distortion?\n\n"
            "This step will help correct for camera angle relative to the measurement plane.\n"
            "You will need to place the checkerboard flat in the measurement plane."
        )
        
        if result:
            # Show perspective correction instructions popup with image
            self.show_perspective_instructions()
            # Ask for source type
            self.show_source_selection_dialog()
        else:
            # Keep identity matrix (no correction)
            if self.completion_callback:
                self.completion_callback()
    
    def show_perspective_instructions(self):
        """Show perspective correction instructions with two sequential popups"""
        try:
            # Show first instruction popup
            show_second = self._show_perspective_instruction_1()
            
            # Show second instruction popup if user clicked "Next"
            if show_second:
                self._show_perspective_instruction_2()
                
        except Exception as e:
            print(f"Error showing perspective instructions: {e}")
            # Continue even if popup fails
            pass
    
    def _show_perspective_instruction_1(self):
        """Show first perspective correction instruction popup"""
        import os
        from PIL import Image, ImageTk
        
        # Variable to track user choice
        show_second = [False]  # Using list to modify from inner function
        
        # Create popup window
        popup = tk.Toplevel(self.parent)
        popup.title("Perspective Correction Instructions (1/2)")
        popup.geometry("750x650")
        popup.resizable(True, True)
        popup.transient(self.parent)
        popup.grab_set()
        
        # Center the popup
        popup.geometry("+%d+%d" % (self.parent.winfo_rootx() + 100, self.parent.winfo_rooty() + 50))
        
        # Main frame with padding
        main_frame = ttk.Frame(popup, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Try to load and display the first image
        try:
            image_path = os.path.join(os.path.dirname(__file__), "illustration_images", "PerspectiveCorrection_1.png")
            if os.path.exists(image_path):
                # Load and resize image to fit nicely in popup
                pil_image = Image.open(image_path)
                # Calculate size to fit within popup while maintaining aspect ratio
                max_width = 625
                max_height = 375
                pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # Display image
                image_label = ttk.Label(main_frame, image=photo)
                image_label.image = photo  # Keep a reference
                image_label.pack(pady=(20, 20))
            else:
                # Image not found, show fallback text
                instructions_text = (
                    "Perspective Correction Instructions (Part 1)\n\n"
                    "For perspective correction, place the checkerboard flat in the measurement plane.\n\n"
                    "[Perspective correction illustration 1 not found]"
                )
                instructions_label = ttk.Label(main_frame, text=instructions_text, 
                                             font=('Arial', 12), justify='center', wraplength=700)
                instructions_label.pack(pady=(20, 20))
                
        except Exception as e:
            print(f"Error loading perspective correction image 1: {e}")
            # Show fallback text on error
            instructions_text = (
                "Perspective Correction Instructions (Part 1)\n\n"
                "For perspective correction, place the checkerboard flat in the measurement plane.\n\n"
                "[Error loading perspective correction illustration 1]"
            )
            instructions_label = ttk.Label(main_frame, text=instructions_text, 
                                         font=('Arial', 12), justify='center', wraplength=700)
            instructions_label.pack(pady=(20, 20))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(20, 0))
        
        # Next and Skip buttons
        def on_next():
            show_second[0] = True
            popup.destroy()
            
        def on_skip():
            show_second[0] = False
            popup.destroy()
        
        next_button = ttk.Button(button_frame, text="Next", command=on_next)
        next_button.pack(side='left', padx=(0, 10))
        
        skip_button = ttk.Button(button_frame, text="Skip", command=on_skip)
        skip_button.pack(side='left')
        
        # Wait for popup to be closed
        popup.wait_window()
        
        return show_second[0]
    
    def _show_perspective_instruction_2(self):
        """Show second perspective correction instruction popup"""
        import os
        from PIL import Image, ImageTk
        
        # Create popup window
        popup = tk.Toplevel(self.parent)
        popup.title("Perspective Correction Instructions (2/2)")
        popup.geometry("750x650")
        popup.resizable(True, True)
        popup.transient(self.parent)
        popup.grab_set()
        
        # Center the popup
        popup.geometry("+%d+%d" % (self.parent.winfo_rootx() + 100, self.parent.winfo_rooty() + 50))
        
        # Main frame with padding
        main_frame = ttk.Frame(popup, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Try to load and display the second image
        try:
            image_path = os.path.join(os.path.dirname(__file__), "illustration_images", "PerspectiveCorrection_2.png")
            if os.path.exists(image_path):
                # Load and resize image to fit nicely in popup
                pil_image = Image.open(image_path)
                # Calculate size to fit within popup while maintaining aspect ratio
                max_width = 625
                max_height = 375
                pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # Display image
                image_label = ttk.Label(main_frame, image=photo)
                image_label.image = photo  # Keep a reference
                image_label.pack(pady=(20, 20))
            else:
                # Image not found, show fallback text
                instructions_text = (
                    "Perspective Correction Instructions (Part 2)\n\n"
                    "The checkerboard should be parallel to the surface where you will be measuring.\n\n"
                    "[Perspective correction illustration 2 not found]"
                )
                instructions_label = ttk.Label(main_frame, text=instructions_text, 
                                             font=('Arial', 12), justify='center', wraplength=700)
                instructions_label.pack(pady=(20, 20))
                
        except Exception as e:
            print(f"Error loading perspective correction image 2: {e}")
            # Show fallback text on error
            instructions_text = (
                "Perspective Correction Instructions (Part 2)\n\n"
                "The checkerboard should be parallel to the surface where you will be measuring.\n\n"
                "[Error loading perspective correction illustration 2]"
            )
            instructions_label = ttk.Label(main_frame, text=instructions_text, 
                                         font=('Arial', 12), justify='center', wraplength=700)
            instructions_label.pack(pady=(20, 20))
        
        # OK button to close popup
        ok_button = ttk.Button(main_frame, text="OK", command=popup.destroy)
        ok_button.pack(pady=(20, 0))
        
        # Wait for popup to be closed
        popup.wait_window()

    def show_source_selection_dialog(self):
        """Show dialog to select image source (camera or file)"""
        # Create source selection dialog
        source_dialog = tk.Toplevel(self.parent)
        source_dialog.title("Select Image Source")
        source_dialog.geometry("400x340")  # Increased height from 300 to 340
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
        title_label = ttk.Label(main_frame, text="Choose Image Source", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Select how you want to provide the checkerboard image:",
                              wraplength=350)
        desc_label.pack(pady=(0, 20))
        
        # Additional info
        info_label = ttk.Label(main_frame, 
                              text="Note: If you select a video file, you'll be able to scrub through and select the best frame.",
                              wraplength=350, font=('Arial', 9), foreground='gray')
        info_label.pack(pady=(0, 15))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=(0, 20))
        
        # Camera button
        camera_btn = ttk.Button(buttons_frame, text="Use Live Camera",
                               command=lambda: self._start_with_camera(source_dialog),
                               style='Accent.TButton')
        camera_btn.pack(fill='x', pady=(0, 10))
        
        # Disable camera button if no camera is selected
        if self.camera_manager.selected_camera_id is None:
            camera_btn.configure(state='disabled')
            # Add a note about camera selection
            camera_note = ttk.Label(buttons_frame, 
                                   text="(Camera option disabled - no camera selected)",
                                   font=('Arial', 8), foreground='gray')
            camera_note.pack(pady=(0, 5))
        
        # File button
        file_btn = ttk.Button(buttons_frame, text="Load Image/Video File",
                             command=lambda: self._start_with_file(source_dialog))
        file_btn.pack(fill='x', pady=(0, 10))
        
        # Cancel button
        cancel_btn = ttk.Button(buttons_frame, text="Cancel",
                               command=source_dialog.destroy)
        cancel_btn.pack(fill='x')
        
    def _start_with_camera(self, dialog):
        """Start perspective correction with live camera"""
        dialog.destroy()
        self.use_camera = True
        self._get_square_size_and_start()
        
    def _start_with_file(self, dialog):
        """Start perspective correction with loaded image or video file"""
        dialog.destroy()
        
        # Ask user to select image or video file
        file_path = filedialog.askopenfilename(
            title="Select Checkerboard Image or Video",
            filetypes=[
                ("All supported", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Check if it's a video file
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in video_extensions:
                # Video file - show frame selection window
                self._select_frame_from_video(file_path)
            else:
                # Image file - load directly
                try:
                    self.loaded_image = cv2.imread(file_path)
                    if self.loaded_image is None:
                        messagebox.showerror("Error", f"Failed to load image: {file_path}")
                        return
                        
                    self.use_camera = False
                    self._get_square_size_and_start()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Error loading image: {str(e)}")
        else:
            # User cancelled file selection - ask what to do
            if self.completion_callback:
                self.completion_callback()

    def _get_square_size_and_start(self):
        """Get optional checkerboard square size from user and start perspective correction"""
        # Create square size input dialog
        square_dialog = tk.Toplevel(self.parent)
        square_dialog.title("Checkerboard Square Size")
        square_dialog.geometry("400x300")
        square_dialog.transient(self.parent)
        square_dialog.grab_set()
        square_dialog.resizable(False, False)
        
        # Center the dialog
        square_dialog.update_idletasks()
        x = (square_dialog.winfo_screenwidth() // 2) - (square_dialog.winfo_width() // 2)
        y = (square_dialog.winfo_screenheight() // 2) - (square_dialog.winfo_height() // 2)
        square_dialog.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(square_dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Real-World Scaling (Optional)", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Description
        desc_text = ("If you know the physical size of your checkerboard squares, "
                    "enter it below to enable real-world measurements.\n\n"
                    "Leave blank to use pixel units (current behavior).")
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=350, justify='left')
        desc_label.pack(pady=(0, 20))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(input_frame, text="Square size:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        
        square_size_var = tk.StringVar()
        square_size_entry = ttk.Entry(input_frame, textvariable=square_size_var, width=10)
        square_size_entry.grid(row=0, column=1, padx=(0, 10))
        square_size_entry.focus()
        
        ttk.Label(input_frame, text="(units: mm, cm, inches, etc.)").grid(row=0, column=2, sticky='w')
        
        # Example
        example_label = ttk.Label(main_frame, 
                                 text="Examples: 25.4 (for 1 inch squares), 20 (for 20mm squares)",
                                 font=('Arial', 9), foreground='gray')
        example_label.pack(pady=(0, 20))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=(0, 10))
        
        def on_continue():
            try:
                square_size_text = square_size_var.get().strip()
                if square_size_text:
                    # User entered a value - validate it
                    self.real_world_square_size = float(square_size_text)
                    if self.real_world_square_size <= 0:
                        messagebox.showerror("Error", "Square size must be positive")
                        return
                else:
                    # User left blank - use pixel units
                    self.real_world_square_size = None
                    
                square_dialog.destroy()
                self.start_perspective_correction()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number or leave blank")
        
        def on_skip():
            self.real_world_square_size = None
            square_dialog.destroy()
            self.start_perspective_correction()
        
        continue_btn = ttk.Button(buttons_frame, text="Continue", command=on_continue, style='Accent.TButton')
        continue_btn.pack(side='left', padx=(0, 10))
        
        skip_btn = ttk.Button(buttons_frame, text="Skip (Use Pixels)", command=on_skip)
        skip_btn.pack(side='left')
        
        # Bind Enter key to continue
        square_dialog.bind('<Return>', lambda e: on_continue())
    
    def _select_frame_from_video(self, video_path):
        """Show video frame selection window"""
        # Create video frame selection dialog
        frame_dialog = tk.Toplevel(self.parent)
        frame_dialog.title("Select Frame from Video")
        frame_dialog.geometry("900x700")
        frame_dialog.transient(self.parent)
        frame_dialog.grab_set()
        
        # Center the dialog
        frame_dialog.update_idletasks()
        x = (frame_dialog.winfo_screenwidth() // 2) - (frame_dialog.winfo_width() // 2)
        y = (frame_dialog.winfo_screenheight() // 2) - (frame_dialog.winfo_height() // 2)
        frame_dialog.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(frame_dialog, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Select Video Frame", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Instructions
        instructions_label = ttk.Label(main_frame, 
                                     text="Use the controls below to navigate through the video and select a frame with the checkerboard.",
                                     wraplength=800)
        instructions_label.pack(pady=(0, 10))
        
        # Video display frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Preview")
        video_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        video_label = ttk.Label(video_frame, text="Loading video...")
        video_label.pack(expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Video playback controls
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(fill='x', pady=(0, 5))
        
        play_button = ttk.Button(playback_frame, text="⏸️ Pause")
        play_button.pack(side='left', padx=(0, 5))
        
        ttk.Button(playback_frame, text="⏮️ Start").pack(side='left', padx=(0, 5))
        ttk.Button(playback_frame, text="⏭️ End").pack(side='left', padx=(0, 5))
        
        # Frame navigation
        frame_nav_frame = ttk.Frame(controls_frame)
        frame_nav_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Button(frame_nav_frame, text="◀◀ -10").pack(side='left', padx=(0, 2))
        ttk.Button(frame_nav_frame, text="◀ -1").pack(side='left', padx=(0, 2))
        ttk.Button(frame_nav_frame, text="▶ +1").pack(side='left', padx=(0, 2))
        ttk.Button(frame_nav_frame, text="▶▶ +10").pack(side='left', padx=(0, 2))
        
        # Progress slider
        slider_frame = ttk.Frame(controls_frame)
        slider_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(slider_frame, text="Position:").pack(side='left')
        progress_var = tk.DoubleVar()
        progress_slider = ttk.Scale(slider_frame, from_=0, to=100, 
                                   orient='horizontal', variable=progress_var)
        progress_slider.pack(side='left', fill='x', expand=True, padx=(5, 5))
        
        # Frame info
        frame_info_label = ttk.Label(slider_frame, text="Frame: 0 / 0")
        frame_info_label.pack(side='right')
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x')
        
        ttk.Button(buttons_frame, text="Use This Frame", 
                  command=None,  # Will be set later
                  style='Accent.TButton').pack(side='right', padx=(5, 0))
        
        ttk.Button(buttons_frame, text="Cancel", 
                  command=frame_dialog.destroy).pack(side='right', padx=(5, 0))
        
        # Initialize video frame selector
        selector = VideoFrameSelector(frame_dialog, video_path, video_label, 
                                    play_button, progress_slider, progress_var,
                                    frame_info_label, self)
        
        # Connect the "Use This Frame" button
        for widget in buttons_frame.winfo_children():
            if isinstance(widget, ttk.Button) and widget['text'] == 'Use This Frame':
                widget.configure(command=selector.use_current_frame)
                break
    
    def start_perspective_correction(self):
        """Start the perspective correction process"""
        # Re-tune fisheye balance once per session (first frame capture).
        self._fisheye_balance_tuned = False

        if self.use_camera:
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
        
        self.create_perspective_window()
        
        if self.use_camera:
            # Start preview for live camera
            self.preview_running = True
            self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
            self.preview_thread.start()
        else:
            # Display static image
            self.display_static_image()
        
    def create_perspective_window(self):
        """Create the perspective correction window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Perspective Correction")
        self.window.geometry("900x800")
        self.window.protocol("WM_DELETE_WINDOW", self.cancel_correction)
        
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Perspective Correction Setup", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Instructions frame
        instruction_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instruction_frame.pack(fill='x', pady=(0, 10))
        
        if self.use_camera:
            instructions = [
                "1. Place the checkerboard flat in the measurement plane",
                "2. Ensure the checkerboard is fully visible and well-lit",
                "3. The preview shows the lens-corrected view",
                "4. Click 'Record Image' when the checkerboard is positioned correctly",
                "5. The system will calculate perspective correction automatically"
            ]
        else:
            instructions = [
                "1. Review the loaded checkerboard image/frame",
                "2. Ensure the checkerboard is flat in the measurement plane",
                "3. The preview shows the lens-corrected view",
                "4. Click 'Process Image' to calculate perspective correction",
                "5. The system will detect the checkerboard automatically"
            ]
        
        for instruction in instructions:
            ttk.Label(instruction_frame, text=instruction).pack(anchor='w', pady=1)
            
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x', pady=(0, 10))
        
        initial_text = "Position checkerboard in measurement plane" if self.use_camera else "Loaded image/frame ready for processing"
        self.status_label = ttk.Label(status_frame, text=initial_text, 
                                     font=('Arial', 12), foreground='blue')
        self.status_label.pack()
        
        # Preview frame
        preview_title = "Lens-Corrected Camera Preview" if self.use_camera else "Lens-Corrected Image/Frame Preview"
        preview_frame = ttk.LabelFrame(main_frame, text=preview_title)
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.preview_label = ttk.Label(preview_frame, text="Initializing..." if self.use_camera else "Loading image...")
        self.preview_label.pack(expand=True)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        button_text = "Record Image" if self.use_camera else "Process Image"
        self.record_button = ttk.Button(button_frame, text=button_text, 
                                       command=self.record_perspective_image,
                                       style='Accent.TButton')
        self.record_button.pack(side='right', padx=(5, 0))
        
        ttk.Button(button_frame, text="Skip Correction", 
                  command=self.skip_correction).pack(side='right', padx=(5, 0))
        
        ttk.Button(button_frame, text="Cancel", 
                  command=self.cancel_correction).pack(side='right', padx=(5, 0))
        
    def preview_loop(self):
        """Preview loop showing lens-corrected camera feed"""
        while self.preview_running and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Fire the fisheye/Scaramuzza FOV tuner once on the first
                # frame, before rendering any undistortion. Otherwise a bad
                # default FOV makes the preview unreadable and the user
                # can't position the board. Tuner is modal+main-thread, so
                # marshal via window.after and pause this loop until OK.
                _model = getattr(self.calibration_data, 'model_type', None)
                if _model in ('fisheye', 'scaramuzza') and not self._fisheye_balance_tuned:
                    self._fisheye_balance_tuned = True   # set first to avoid re-entry
                    done = threading.Event()

                    def _run_tuner(raw=frame.copy()):
                        try:
                            if _model == 'fisheye':
                                self._show_fisheye_balance_tuner(raw)
                            else:
                                self._show_scaramuzza_fov_tuner(raw)
                        finally:
                            done.set()

                    self.window.after(0, _run_tuner)
                    done.wait()
                    continue   # skip this frame; next iteration uses new FOV

                # Lens-only correction via the shared pipeline (handles
                # pinhole/rational/fisheye branching and the canvas
                # expansion in corrections._apply_lens).  skip_perspective
                # because the perspective homography hasn't been calibrated
                # yet — that's what this UI is collecting a frame for.
                undistorted = apply_corrections(
                    frame, self.calibration_data, skip_perspective=True)
                
                # Auto-detect any of the canonical board sizes — the
                # perspective board may differ from the camera-calibration one.
                gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                detected_size, corners = self._detect_perspective_checkerboard(gray)

                if detected_size is not None:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(undistorted, detected_size, corners, True)

                    status_text = (f"Checkerboard {detected_size[0]}x{detected_size[1]} "
                                   "detected - Ready to record")
                    self.window.after(0, lambda t=status_text: self.status_label.configure(
                        text=t, foreground='green'))
                else:
                    self.window.after(0, lambda: self.status_label.configure(
                        text="Position checkerboard in measurement plane", foreground='blue'))

                # Update preview
                self.update_preview(undistorted)
                
            time.sleep(0.033)  # ~30 FPS
    
    def display_static_image(self):
        """Display the loaded static image with checkerboard detection"""
        if self.loaded_image is None:
            return

        # Fire the fisheye/Scaramuzza tuner before any undistortion happens.
        # If we render first, a poorly-defaulted FOV produces an unreadable
        # preview and the user can't reach 'Process Image' to trigger the
        # tuner — they get stuck on "Checkerboard not detected".
        _model = getattr(self.calibration_data, 'model_type', None)
        if _model in ('fisheye', 'scaramuzza') and not self._fisheye_balance_tuned:
            if _model == 'fisheye':
                self._show_fisheye_balance_tuner(self.loaded_image)
            else:
                self._show_scaramuzza_fov_tuner(self.loaded_image)
            self._fisheye_balance_tuned = True

        # Lens-only correction via the shared pipeline (see live-preview
        # site above for rationale).
        undistorted = apply_corrections(
            self.loaded_image, self.calibration_data, skip_perspective=True)
        
        # Auto-detect from the canonical sizes list. Static images don't need
        # the lock — the image isn't changing — so disable it for clarity.
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        detected_size, corners = self._detect_perspective_checkerboard(
            gray, allow_lock=False)

        if detected_size is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(undistorted, detected_size, corners, True)
            self.status_label.configure(
                text=(f"Checkerboard {detected_size[0]}x{detected_size[1]} "
                      "detected - Ready to process"),
                foreground='green')
        else:
            self.status_label.configure(
                text="Checkerboard not detected in image", foreground='red')
        
        # Update preview
        self.update_preview(undistorted)
            
    def update_preview(self, frame):
        """Update preview image"""
        try:
            if self.preview_label and (self.preview_running or not self.use_camera):
                # Resize frame for preview
                height, width = frame.shape[:2]
                max_width, max_height = 800, 500
                
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                frame_resized = cv2.resize(frame, (new_width, new_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update preview in main thread
                if self.use_camera:
                    self.window.after(0, self._update_preview_image, photo)
                else:
                    # Direct update for static image
                    self._update_preview_image(photo)
                
        except Exception as e:
            print(f"Preview update error: {e}")
            
    def _update_preview_image(self, photo):
        """Helper to update preview image in main thread"""
        try:
            if self.preview_label and (self.preview_running or not self.use_camera):
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
        except Exception as e:
            print(f"Preview image update error: {e}")
            
    # ------------------------------------------------------------------
    # Fisheye balance tuning (model_type=="fisheye" only).
    # Lives here, not in calibration_processor, because the meaningful
    # preview / auto-tune signal needs the captured perspective frame.
    # ------------------------------------------------------------------

    def _render_fisheye_lens_corrected(self, frame, balance):
        """Apply cv2.fisheye undistortion at the given balance, no canvas
        expansion.  Mirrors what corrections._apply_lens does for fisheye
        except for the canvas expansion (which would push the scene into a
        small island of mostly-black pixels and make balance hard to judge).
        Scales K for the frame resolution the same way runtime does.
        """
        h, w = frame.shape[:2]
        K = self.calibration_data.get_scaled_camera_matrix((w, h))
        D = self.calibration_data.distortion_coefficients
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=float(balance))
        return cv2.fisheye.undistortImage(
            frame, K, D=D, Knew=Knew, new_size=(w, h))

    def _draw_grid_overlay(self, image, divisions=10, color=(0, 255, 255)):
        """Overlay a divisions×divisions grid of straight lines as a visual
        straightness reference.  Lines that cut across straight scene features
        (wall edges, table edges) reveal residual fisheye curvature when the
        balance is mis-tuned.  Returns a new image; does not mutate input.
        """
        out = image.copy()
        h, w = out.shape[:2]
        for i in range(1, divisions):
            x = int(round(i * w / divisions))
            cv2.line(out, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
            y = int(round(i * h / divisions))
            cv2.line(out, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
        return out

    def _detect_corners_for_autotune(self, raw_frame):
        """Detect checkerboard corners in the *raw* (pre-lens-correction)
        frame so the auto-tune loss can re-project them through the fisheye
        model at any candidate balance without re-detecting.

        Returns (corners_Nx2, (cols, rows)) on success, or (None, None)
        if no checkerboard pattern is detected.  cv2.findChessboardCorners
        is generally robust to fisheye distortion as long as the corners
        are visible.
        """
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        size, corners = self._detect_perspective_checkerboard(
            gray, allow_lock=False)
        if size is None or corners is None:
            return None, None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        return corners.reshape(-1, 2).astype(np.float64), size

    def _straightness_loss(self, source_corners, board_size, image_size_wh, balance):
        """Sum-of-squared perpendicular residuals when each row and each
        column of the (lens-corrected) checkerboard corners is fitted with
        a best-fit line.  Lower = the rows/columns are closer to actually
        being straight in the corrected image, which is what we want.

        Source corners are undistorted via cv2.fisheye.undistortPoints with
        Knew computed at the candidate balance — same operation
        corrections._apply_lens applies to the image at runtime.

        image_size_wh is the (W, H) of the frame the source corners came
        from; we scale K for it so this matches runtime behavior.
        """
        K = self.calibration_data.get_scaled_camera_matrix(image_size_wh)
        D = self.calibration_data.distortion_coefficients
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, image_size_wh, np.eye(3), balance=float(balance))
        pts = source_corners.reshape(-1, 1, 2)
        undist = cv2.fisheye.undistortPoints(
            pts, K, D, P=Knew).reshape(-1, 2)
        cols, rows = board_size  # board_size is (cols, rows) per OpenCV convention
        if undist.shape[0] != cols * rows:
            return float('inf')
        grid = undist.reshape(rows, cols, 2)

        def line_residuals(points):
            # SVD-based total-least-squares perpendicular-distance fit.
            if points.shape[0] < 2:
                return 0.0
            centered = points - points.mean(axis=0)
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                return float('inf')
            normal = vt[-1]
            return float(np.sum((centered @ normal) ** 2))

        total = 0.0
        for r in range(rows):
            total += line_residuals(grid[r, :, :])
        for c in range(cols):
            total += line_residuals(grid[:, c, :])
        return total

    def _auto_tune_fisheye_balance(self, source_corners, board_size, image_size_wh):
        """Golden-section search over balance ∈ [0, 1] minimising the
        row/column straightness loss.  Returns the optimal balance.

        Uses golden section (no derivatives, no scipy dependency, ~20
        evaluations to 1e-3 precision on a unimodal function).  The loss
        is generally well-behaved here because higher balance retains more
        FOV but extrapolates the model into the vignette where corners
        weren't trained, so straightness degrades smoothly with balance
        once you're outside the calibrated region.
        """
        phi = (1 + 5 ** 0.5) / 2
        invphi = 1.0 / phi
        a, b = 0.0, 1.0
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        fc = self._straightness_loss(source_corners, board_size, image_size_wh, c)
        fd = self._straightness_loss(source_corners, board_size, image_size_wh, d)
        # ~20 iterations brings the bracket to ~1e-4 of the original interval.
        for _ in range(20):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b - a) * invphi
                fc = self._straightness_loss(source_corners, board_size, image_size_wh, c)
            else:
                a, c, fc = c, d, fd
                d = a + (b - a) * invphi
                fd = self._straightness_loss(source_corners, board_size, image_size_wh, d)
        return (a + b) / 2.0

    def _show_fisheye_balance_tuner(self, raw_frame):
        """Modal dialog: lens-corrected preview + grid overlay + balance
        slider + auto-tune.  Writes the chosen balance to
        self.calibration_data.fisheye_balance on OK.

        Caller is responsible for the once-per-session gating; this method
        always shows the dialog when invoked.  Cancel keeps whatever the
        previously-stored balance was (default 1.0 from set_calibration).
        """
        # Try to pre-detect corners on the raw frame so auto-tune is enabled.
        # Detection failure is non-fatal — manual slider still works.
        source_corners, board_size = self._detect_corners_for_autotune(raw_frame)
        autotune_available = source_corners is not None
        raw_h, raw_w = raw_frame.shape[:2]
        image_size_wh = (raw_w, raw_h)

        dialog = tk.Toplevel(self.window if self.window else self.parent)
        dialog.title("Tune Fisheye Balance")
        dialog.geometry("980x800")
        dialog.transient(self.window if self.window else self.parent)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding="12")
        main_frame.pack(fill='both', expand=True)

        ttk.Label(
            main_frame,
            text=("Since you picked the fisheye lens model, the undistortion "
                  "balance needs tuning against your scene."),
            font=('Arial', 11, 'bold'),
        ).pack(anchor='w', pady=(0, 4))
        ttk.Label(
            main_frame,
            text=("A 10×10 reference grid is overlaid in yellow.  Adjust "
                  "balance until known-straight scene features (walls, table "
                  "edges, the checkerboard rows) run parallel to the grid in "
                  "the region you'll be tracking in.  Some black border at "
                  "the corners is healthy — pushing balance higher to "
                  "eliminate it forces the model to extrapolate into the "
                  "vignette and produces balloon distortion."),
            font=('Arial', 9), foreground='gray', wraplength=940, justify='left',
        ).pack(anchor='w', pady=(0, 8))

        preview_label = ttk.Label(main_frame)
        preview_label.pack(fill='both', expand=True, pady=(0, 8))

        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill='x', pady=(0, 8))

        ttk.Label(slider_frame, text="Balance:").pack(side='left', padx=(0, 6))
        balance_var = tk.DoubleVar(
            value=float(self.calibration_data.fisheye_balance))
        balance_label = ttk.Label(
            slider_frame, text=f"{balance_var.get():.2f}", width=5)
        balance_label.pack(side='right')

        update_state = {'pending': None, 'last_value': balance_var.get()}

        def render(b):
            try:
                img = self._render_fisheye_lens_corrected(raw_frame, float(b))
                img = self._draw_grid_overlay(img, divisions=10)
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pil.thumbnail((920, 540), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                preview_label.configure(image=photo, text="")
                preview_label.image = photo
            except Exception as e:
                print(f"Balance preview render error: {e}")

        def on_slider(_=None):
            b = balance_var.get()
            balance_label.configure(text=f"{b:.2f}")
            update_state['last_value'] = b
            if update_state['pending'] is not None:
                dialog.after_cancel(update_state['pending'])
            update_state['pending'] = dialog.after(
                80, lambda: render(update_state['last_value']))

        ttk.Scale(
            slider_frame, variable=balance_var, from_=0.0, to=1.0,
            command=on_slider,
        ).pack(side='left', fill='x', expand=True, padx=(0, 6))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')

        result = {'ok': False}

        def on_ok():
            self.calibration_data.fisheye_balance = float(balance_var.get())
            result['ok'] = True
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        def on_autotune():
            if not autotune_available:
                return
            try:
                opt = self._auto_tune_fisheye_balance(
                    source_corners, board_size, image_size_wh)
                balance_var.set(float(opt))
                balance_label.configure(text=f"{opt:.2f}")
                render(opt)
            except Exception as e:
                messagebox.showwarning(
                    "Auto-tune failed",
                    f"Could not auto-tune balance: {e}")

        ttk.Button(button_frame, text="OK", command=on_ok).pack(side='right')
        ttk.Button(
            button_frame, text="Cancel", command=on_cancel,
        ).pack(side='right', padx=(0, 6))
        autotune_btn = ttk.Button(
            button_frame, text="Auto-tune", command=on_autotune)
        autotune_btn.pack(side='left')
        if not autotune_available:
            autotune_btn.configure(state='disabled')
            ttk.Label(
                button_frame,
                text=("(checkerboard not detected in the captured frame — "
                      "auto-tune disabled, manual slider only)"),
                font=('Arial', 9), foreground='gray',
            ).pack(side='left', padx=(8, 0))

        render(balance_var.get())
        dialog.wait_window()

    # ------------------------------------------------------------------
    # Scaramuzza FOV tuning (model_type=="scaramuzza" only).
    # ------------------------------------------------------------------

    def _render_scaramuzza_lens_corrected(self, frame, fov_deg):
        """Apply Scaramuzza undistortion at the given FOV for the tuner preview.
        Output is the same size as the input (no canvas expansion) so the
        FOV effect is easy to judge in the dialog.
        """
        from .corrections import _build_scaramuzza_maps
        h, w = frame.shape[:2]
        params = self.calibration_data.scaramuzza_params
        mapx, mapy = _build_scaramuzza_maps(params, float(fov_deg), w, h)
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _show_scaramuzza_fov_tuner(self, raw_frame):
        """Modal dialog: Scaramuzza-undistorted preview + grid overlay + FOV
        slider.  Writes the chosen FOV to self.calibration_data.scaramuzza_fov
        on OK; Cancel keeps the previously-stored value (default 180°).

        FOV is the horizontal field-of-view of the virtual perspective camera.
        Lower values crop tighter (less black border, smaller apparent FOV);
        higher values include more of the fisheye circle.  A 10×10 reference
        grid is overlaid in yellow — adjust until known-straight scene features
        run parallel to the grid in the region you will be tracking in.
        """
        dialog = tk.Toplevel(self.window if self.window else self.parent)
        dialog.title("Tune Scaramuzza Undistortion FOV")
        dialog.geometry("980x800")
        dialog.transient(self.window if self.window else self.parent)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding="12")
        main_frame.pack(fill='both', expand=True)

        ttk.Label(
            main_frame,
            text="Scaramuzza (OCamCalib) lens model — tune the undistortion FOV.",
            font=('Arial', 11, 'bold'),
        ).pack(anchor='w', pady=(0, 4))
        ttk.Label(
            main_frame,
            text=("A 10×10 reference grid is overlaid in yellow.  Adjust the FOV "
                  "slider until known-straight scene features (walls, table edges, "
                  "checkerboard rows) run parallel to the grid in the region you "
                  "will be tracking.  Lower FOV = tighter crop, less black border. "
                  "Higher FOV = wider view, more black border at edges."),
            font=('Arial', 9), foreground='gray', wraplength=940, justify='left',
        ).pack(anchor='w', pady=(0, 8))

        preview_label = ttk.Label(main_frame)
        preview_label.pack(fill='both', expand=True, pady=(0, 8))

        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill='x', pady=(0, 8))

        ttk.Label(slider_frame, text="FOV (°):").pack(side='left', padx=(0, 6))
        fov_var = tk.DoubleVar(
            value=float(self.calibration_data.scaramuzza_fov))
        fov_label = ttk.Label(slider_frame, text=f"{fov_var.get():.0f}°", width=5)
        fov_label.pack(side='right')

        update_state = {'pending': None, 'last_value': fov_var.get()}

        def render(fov):
            try:
                img = self._render_scaramuzza_lens_corrected(raw_frame, float(fov))
                img = self._draw_grid_overlay(img, divisions=10)
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pil.thumbnail((920, 540), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                preview_label.configure(image=photo, text="")
                preview_label.image = photo
            except Exception as exc:
                print(f"Scaramuzza FOV preview render error: {exc}")

        def on_slider(_=None):
            fov = fov_var.get()
            fov_label.configure(text=f"{fov:.0f}°")
            update_state['last_value'] = fov
            if update_state['pending'] is not None:
                dialog.after_cancel(update_state['pending'])
            update_state['pending'] = dialog.after(
                80, lambda: render(update_state['last_value']))

        ttk.Scale(
            slider_frame, variable=fov_var, from_=10.0, to=170.0,
            command=on_slider,
        ).pack(side='left', fill='x', expand=True, padx=(0, 6))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')

        def on_ok():
            self.calibration_data.scaramuzza_fov = float(fov_var.get())
            # Invalidate the remap cache so corrections rebuilds at runtime.
            if hasattr(self.calibration_data, '_scaramuzza_remap_cache'):
                del self.calibration_data._scaramuzza_remap_cache
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="OK", command=on_ok).pack(side='right')
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side='right', padx=(0, 6))

        render(fov_var.get())
        dialog.wait_window()

    def record_perspective_image(self):
        """Record image for perspective correction calculation"""
        if self.use_camera:
            if not self.cap:
                return

            ret, frame = self.cap.read()
            if not ret or frame is None:
                messagebox.showerror("Error", "Failed to capture image")
                return
        else:
            # Use loaded image
            if self.loaded_image is None:
                messagebox.showerror("Error", "No image loaded")
                return
            frame = self.loaded_image.copy()

        # Balance / FOV tuning: only judgeable against a real scene, so we
        # run it here against the captured perspective frame.  Runs once per
        # perspective session (_fisheye_balance_tuned reused for both models).
        _model = getattr(self.calibration_data, 'model_type', None)
        if _model in ('fisheye', 'scaramuzza') and not self._fisheye_balance_tuned:
            if _model == 'fisheye':
                self._show_fisheye_balance_tuner(frame)
            else:
                self._show_scaramuzza_fov_tuner(frame)
            self._fisheye_balance_tuned = True

        # Lens-only correction via the shared pipeline.  The detected
        # checkerboard corners (below) will be in the *expanded* lens-
        # corrected coordinate space, and the homography calibrated from
        # them will match the same expanded space at runtime.
        undistorted = apply_corrections(
            frame, self.calibration_data, skip_perspective=True)
            
        # Detect checkerboard, auto-selecting the size. Disable lock here so
        # the one-shot record path is independent of any preview lock state
        # — we always do a fresh largest-first sweep on the captured frame.
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        checkerboard_size, corners = self._detect_perspective_checkerboard(
            gray, allow_lock=False)

        if checkerboard_size is None:
            messagebox.showerror(
                "Error",
                "Checkerboard not detected. Please position it clearly and try again.")
            return

        print(f"Perspective correction using {checkerboard_size[0]}x{checkerboard_size[1]} board")

        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Calculate perspective correction (tristate: True/False/None)
        result = self.calculate_perspective_correction(
            corners, checkerboard_size, undistorted.shape[:2], undistorted)

        if result is True:
            self.status_label.configure(
                text="Perspective correction calculated successfully!",
                foreground='green')
            self.record_button.configure(state='disabled')
            # Show completion message and close
            self.window.after(1000, self.complete_correction)
        elif result is None:
            # User cancelled the output-region dialog — leave the perspective
            # setup window open so they can try again or pick a different frame.
            self.status_label.configure(
                text="Output region cancelled — pick a frame or try again.",
                foreground='orange')
        else:
            messagebox.showerror("Error", "Failed to calculate perspective correction")
            
    def calculate_perspective_correction(self, corners, checkerboard_size, image_shape, image):
        """Calculate perspective correction matrix from checkerboard"""
        try:
            # Reshape corners to 2D array for easier processing
            corners_2d = corners.reshape(-1, 2)
            
            # Determine square size for perspective transformation
            # ALWAYS use pixel-based coordinates for the homography to preserve resolution
            
            # STEP 1: Create initial perspective correction using rough square size estimate
            # Use simple min/max method for initial homography calculation
            print(f"Step 1: Creating initial perspective correction from {len(corners_2d)} corners...")
            
            min_x = np.min(corners_2d[:, 0])
            max_x = np.max(corners_2d[:, 0])
            x_checkerboard_extents = max_x - min_x
            initial_square_size = round(x_checkerboard_extents / (checkerboard_size[0] - 1))
            initial_square_size = max(initial_square_size, 10.0)  # Ensure minimum size
            
            print(f"Initial square size estimate: {initial_square_size:.1f} pixels")
            
            # Create initial ideal checkerboard points
            initial_ideal_points = []
            for j in range(checkerboard_size[1]):
                for i in range(checkerboard_size[0]):
                    initial_ideal_points.append([i * initial_square_size, j * initial_square_size])
            initial_ideal_points = np.array(initial_ideal_points, dtype=np.float32)
            
            # Calculate initial homography
            initial_homography, _ = cv2.findHomography(corners_2d, initial_ideal_points, cv2.RANSAC)
            
            if initial_homography is None:
                print("Failed to calculate initial homography")
                return False
            
            # STEP 2: Apply initial correction and re-detect corners for accurate measurement
            print("Step 2: Applying initial correction and measuring true square size...")
            
            # Apply perspective correction to the image
            corrected_height = int((checkerboard_size[1] - 1) * initial_square_size * 1.2)  # Add margin
            corrected_width = int((checkerboard_size[0] - 1) * initial_square_size * 1.2)
            corrected_image = cv2.warpPerspective(image, initial_homography, (corrected_width, corrected_height))
            
            # Convert to grayscale if needed
            if len(corrected_image.shape) == 3:
                corrected_gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
            else:
                corrected_gray = corrected_image
            
            # Detect corners in the corrected image
            ret_corrected, corrected_corners = cv2.findChessboardCorners(corrected_gray, checkerboard_size, None)
            
            if ret_corrected and corrected_corners is not None:
                # Refine corners in corrected image
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corrected_corners = cv2.cornerSubPix(corrected_gray, corrected_corners, (11, 11), (-1, -1), criteria)
                corrected_corners_2d = corrected_corners.reshape(-1, 2)
                
                # Now measure the true square size in the corrected image
                x_distances = []
                y_distances = []
                
                # Calculate horizontal distances
                for j in range(checkerboard_size[1]):
                    for i in range(checkerboard_size[0] - 1):
                        idx1 = j * checkerboard_size[0] + i
                        idx2 = j * checkerboard_size[0] + (i + 1)
                        if idx2 < len(corrected_corners_2d):
                            x_dist = abs(corrected_corners_2d[idx2][0] - corrected_corners_2d[idx1][0])
                            x_distances.append(x_dist)
                
                # Calculate vertical distances
                for j in range(checkerboard_size[1] - 1):
                    for i in range(checkerboard_size[0]):
                        idx1 = j * checkerboard_size[0] + i
                        idx2 = (j + 1) * checkerboard_size[0] + i
                        if idx2 < len(corrected_corners_2d):
                            y_dist = abs(corrected_corners_2d[idx2][1] - corrected_corners_2d[idx1][1])
                            y_distances.append(y_dist)
                
                if x_distances and y_distances:
                    median_x = np.median(x_distances)
                    median_y = np.median(y_distances)
                    calculated_square_size = round((median_x + median_y) / 2.0)
                    
                    print(f"Corrected image square measurements:")
                    print(f"  Horizontal: {median_x:.1f} pixels (range: {min(x_distances):.1f}-{max(x_distances):.1f})")
                    print(f"  Vertical: {median_y:.1f} pixels (range: {min(y_distances):.1f}-{max(y_distances):.1f})")
                    print(f"  Final square size: {calculated_square_size:.1f} pixels")
                    
                    # Check measurement consistency (should be very good in corrected image)
                    x_variation = (max(x_distances) - min(x_distances)) / median_x if median_x > 0 else 0
                    y_variation = (max(y_distances) - min(y_distances)) / median_y if median_y > 0 else 0
                    
                    if x_variation < 0.05 and y_variation < 0.05:
                        print("✓ Excellent measurement consistency in corrected image")
                    else:
                        print(f"⚠️  Some variation still present: X={x_variation:.1%}, Y={y_variation:.1%}")
                else:
                    print("Failed to measure distances in corrected image, using initial estimate")
                    calculated_square_size = initial_square_size
            else:
                print("Failed to detect corners in corrected image, using initial estimate")
                calculated_square_size = initial_square_size
            
            # Ensure minimum square size for numerical stability
            pixel_square_size = max(calculated_square_size, 10.0)
            
            if self.real_world_square_size is not None:
                # User provided real-world square size
                print(f"Real-world scaling mode:")
                print(f"  Real-world square size: {self.real_world_square_size:.3f} units")
                print(f"  Pixel square size for transform: {pixel_square_size:.1f} pixels")
                
                # Calculate conversion factor from pixels to real-world units
                pixels_per_real_unit = pixel_square_size / self.real_world_square_size
                print(f"  Conversion factor: {pixels_per_real_unit:.3f} pixels per real unit")
                
                # Store scaling info in calibration data for coordinate conversion.
                # Both effective and _native start at the homography scale; the
                # effective values get updated when corrections._apply_perspective
                # downsamples the output canvas.
                self.calibration_data.real_world_scale = True
                self.calibration_data.square_size_real = self.real_world_square_size
                self.calibration_data.square_size_pixels = pixel_square_size
                self.calibration_data.square_size_pixels_native = pixel_square_size
                self.calibration_data.pixels_per_real_unit = pixels_per_real_unit
                self.calibration_data.pixels_per_real_unit_native = pixels_per_real_unit
            else:
                # Pixel units mode
                print(f"Pixel-based mode:")
                print(f"  X extents: {x_checkerboard_extents:.1f} pixels")
                print(f"  Corners along X: {checkerboard_size[0]}")
                print(f"  Calculated square size: {calculated_square_size:.1f}")
                print(f"  Final square size: {pixel_square_size:.1f}")

                # Store scaling info in calibration data for reference
                self.calibration_data.real_world_scale = False
                self.calibration_data.square_size_pixels = pixel_square_size
                self.calibration_data.square_size_pixels_native = pixel_square_size
                self.calibration_data.pixels_per_real_unit = 1.0
                self.calibration_data.pixels_per_real_unit_native = 1.0
            
            # ALWAYS use pixel-based square size for the homography transformation
            # This preserves video resolution while allowing coordinate conversion later
            square_size = pixel_square_size
            
            # Create ideal checkerboard points (rectangular grid)
            ideal_points = []
            
            for j in range(checkerboard_size[1]):
                for i in range(checkerboard_size[0]):
                    ideal_points.append([i * square_size, j * square_size])
                    
            ideal_points = np.array(ideal_points, dtype=np.float32)
            
            # Calculate homography from detected corners to ideal grid
            homography, _ = cv2.findHomography(corners_2d, ideal_points, cv2.RANSAC)

            if homography is None:
                return False

            # Show the bbox-selector dialog so the user defines the runtime
            # output region AND the export coordinate frame.  Homography,
            # bbox, and orientation are committed together — if the user
            # cancels here, the calibration is not accepted (atomic commit).
            picked = self._pick_output_bbox(
                homography, image, checkerboard_size, pixel_square_size)
            if picked is None:
                print("Perspective correction cancelled by user (bbox dialog)")
                # Tristate: None = user cancelled, distinct from hard failure
                return None
            bbox, frame_orientation_state = picked

            self.calibration_data.set_perspective_correction(
                homography, bbox,
                frame_orientation_state=frame_orientation_state)
            return True

        except Exception as e:
            print(f"Perspective correction calculation error: {e}")
            return False

    def _pick_output_bbox(self, homography, lens_corrected_frame,
                          checkerboard_size, pixel_square_size):
        """Render a 12 MP perspective-corrected preview, show the
        bbox-selector dialog, return the user-confirmed bbox in world
        coords or None if cancelled.

        The preview passed to BboxSelectorDialog is the RAW perspective
        warp with no orientation applied.  BboxSelectorDialog applies
        apply_frame_orientation_to_image(warped, state) itself so the
        displayed image matches what the tracker will show for the chosen
        orientation state.
        """
        from .bbox_selector import BboxSelectorDialog

        h, w = lens_corrected_frame.shape[:2]

        # Project source-frame boundary through the homography to find the
        # full world extent (corners + edge midpoints densify the sample).
        src_pts = np.array([
            [0, 0], [w-1, 0], [w-1, h-1], [0, h-1],
            [w/2, 0], [w-1, h/2], [w/2, h-1], [0, h/2],
        ], dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(src_pts, homography).reshape(-1, 2)
        proj_min_x = float(proj[:, 0].min())
        proj_min_y = float(proj[:, 1].min())
        proj_max_x = float(proj[:, 0].max())
        proj_max_y = float(proj[:, 1].max())

        extent_w = proj_max_x - proj_min_x
        extent_h = proj_max_y - proj_min_y
        if extent_w <= 0 or extent_h <= 0:
            print("Cannot render preview: degenerate projected extent")
            return None

        # Cap preview at 12 MP — big enough to zoom into without losing
        # detail, small enough to stay responsive.  preview_scale is in
        # preview-pixels per native-world-pixel.
        PREVIEW_MAX_PIXELS = 12_000_000
        extent_area = extent_w * extent_h
        if extent_area > PREVIEW_MAX_PIXELS:
            preview_scale = float(np.sqrt(PREVIEW_MAX_PIXELS / extent_area))
        else:
            preview_scale = 1.0
        preview_w = max(1, int(np.ceil(extent_w * preview_scale)))
        preview_h = max(1, int(np.ceil(extent_h * preview_scale)))

        # Combined warp matrix: scale + translate + homography.
        M_preview = np.array(
            [[preview_scale, 0, -proj_min_x * preview_scale],
             [0, preview_scale, -proj_min_y * preview_scale],
             [0, 0, 1]], dtype=np.float64,
        ) @ homography.astype(np.float64)
        preview_image = cv2.warpPerspective(
            lens_corrected_frame, M_preview, (preview_w, preview_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        # No vertical flip: the preview preserves the source frame's
        # top/bottom/left/right orientation. The user picks the coordinate
        # frame separately via the rotate-frame button.

        # Quick-select bboxes in world coords at native scale.
        full_frame_bbox = (proj_min_x, proj_min_y, proj_max_x, proj_max_y)
        board_extents_bbox = (0.0, 0.0,
                              (checkerboard_size[0] - 1) * pixel_square_size,
                              (checkerboard_size[1] - 1) * pixel_square_size)

        # HUD: show bbox dimensions in real-world units when available.
        if (getattr(self, 'real_world_square_size', None)
                and pixel_square_size > 0):
            world_units_per_pixel = (self.real_world_square_size
                                     / pixel_square_size)
            units_label = "units"
        else:
            world_units_per_pixel = None
            units_label = "px"

        # Preserve any previously-chosen frame orientation across re-runs.
        prior_orientation = int(getattr(
            self.calibration_data, 'frame_orientation_state', 4))

        dialog = BboxSelectorDialog(
            parent=self.window,
            preview_image_bgr=preview_image,
            proj_min_x=proj_min_x,
            proj_min_y=proj_min_y,
            preview_scale=preview_scale,
            initial_bbox=board_extents_bbox,
            full_frame_bbox=full_frame_bbox,
            board_extents_bbox=board_extents_bbox,
            initial_frame_orientation=prior_orientation,
            world_units_per_pixel=world_units_per_pixel,
            units_label=units_label,
        )
        return dialog.show()
            
    def complete_correction(self):
        """Complete perspective correction and close window"""
        self.stop_preview()
        
        if self.completion_callback:
            self.completion_callback()
            
        if self.window:
            self.window.destroy()
            
        messagebox.showinfo("Perspective Correction", 
                           "Perspective correction has been applied successfully!")
        
    def skip_correction(self):
        """Skip perspective correction"""
        self.stop_preview()
        
        if self.completion_callback:
            self.completion_callback()
            
        if self.window:
            self.window.destroy()
            
    def cancel_correction(self):
        """Cancel perspective correction"""
        self.stop_preview()
        
        if self.window:
            self.window.destroy()
            
    def stop_preview(self):
        """Stop camera preview and cleanup"""
        self.preview_running = False
        
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Clear loaded image when done
        self.loaded_image = None
