"""
Data models for Added-Mass-Lab GUI
Contains the CalibrationData class for storing calibration information
"""

import numpy as np


# Coordinate-frame orientation lookup.  State 0 = identity (X right, Y up,
# Z out of page).  States 1–3 are successive 90° CCW rotations about Z.
# States 4–7 are states 0–3 with Y inverted (equivalent to a 180° rotation
# about X — Z flips into the page, making the frame left-handed).  The
# basis vectors are expressed in the displayed-image cartesian frame
# (where +X is right and +Y is up after the perspective stage's flip).
FRAME_ORIENTATION_BASES = {
    # state: ((user_X_basis), (user_Y_basis), label)
    0: ((+1,  0), ( 0, +1), "X right, Y up (Z out)"),
    1: (( 0, +1), (-1,  0), "X up, Y left (Z out)"),
    2: ((-1,  0), ( 0, -1), "X left, Y down (Z out)"),
    3: (( 0, -1), (+1,  0), "X down, Y right (Z out)"),
    4: ((+1,  0), ( 0, -1), "X right, Y down (Z in)"),
    5: (( 0, +1), (+1,  0), "X up, Y right (Z in)"),
    6: ((-1,  0), ( 0, +1), "X left, Y up (Z in)"),
    7: (( 0, -1), (-1,  0), "X down, Y left (Z in)"),
}

# Conversion matrices: (x_user, y_user) = M @ (x_image, y_image).
# Equivalent to the inverse of the basis matrix [user_X_basis | user_Y_basis].
# Pre-computed because the bases above are all ±1/0 entries; the inverse
# is just a transpose for the right-handed states and a transpose-with-
# row-flip for the left-handed states.
_FRAME_ORIENTATION_MATRICES = {
    0: ((+1,  0), ( 0, +1)),
    1: (( 0, +1), (-1,  0)),
    2: ((-1,  0), ( 0, -1)),
    3: (( 0, -1), (+1,  0)),
    4: ((+1,  0), ( 0, -1)),
    5: (( 0, +1), (+1,  0)),
    6: ((-1,  0), ( 0, +1)),
    7: (( 0, -1), (-1,  0)),
}


class CalibrationData:
    """Global storage for camera calibration data"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset calibration to no distortion correction"""
        self.is_calibrated = False
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.checkerboard_size = None
        self.calibration_error = None
        self.image_size = None

        # Lens distortion model. The GUI only produces "rational"
        # calibrations now (a strict superset of the 5-coef Brown-Conrady
        # "pinhole" fit — extra k4..k6 coefficients converge near zero for
        # low-distortion lenses), so the default reflects what new
        # calibrations use. "pinhole" and "fisheye" remain valid values so
        # JSONs saved before this change still load:
        #   "pinhole"  - 5-coef Brown-Conrady (legacy)
        #   "rational" - 8-coef Brown-Conrady (k1..k6, p1, p2)
        #   "fisheye"  - cv2.fisheye equidistant (fitting path retained but
        #                hidden in the GUI; runtime apply still works for
        #                old calibrations)
        # Runtime apply: pinhole/rational share cv2.undistort (auto-handles
        # either D length); fisheye uses cv2.fisheye.undistortImage.
        self.model_type = "rational"
        # Fisheye-only: controls how much of the source FOV is preserved in
        # the undistorted output. 0 = max crop (no black corners), 1 = no
        # zoom (full computed FOV, large black corners). For lenses with
        # significant vignetting (image circle smaller than the sensor),
        # balance=0 forces the optimizer to extrapolate the distortion
        # model into the vignette region, producing balloon-shaped output.
        # Default 1.0 is the safe choice; users with full-coverage lenses
        # can lower it for less black border.
        self.fisheye_balance = 1.0

        # Initialize perspective correction as identity matrix (no correction)
        self.perspective_matrix = np.eye(3, dtype=np.float32)
        self.perspective_corrected = False
        # Translation (tx, ty) baked into the perspective warp so that content
        # in the negative-coordinate quadrant of world space is shifted into the
        # visible canvas. Pixel (tx, ty) in the output frame corresponds to
        # world origin (0, 0). Updated by corrections._apply_perspective on first
        # warp call and written to the metadata sidecar for downstream tools.
        self.perspective_translation_x = 0.0
        self.perspective_translation_y = 0.0
        # User-defined output region in pre-flip world coordinates at native
        # homography scale: (x_min, y_min, x_max, y_max). Set by the bbox
        # selector dialog at the end of perspective calibration; used by
        # corrections._apply_perspective to size the runtime output canvas.
        # Invalidated whenever the homography changes (set_perspective_correction).
        self.output_bbox_world = None
        # Coordinate-frame orientation (0–7) — see FRAME_ORIENTATION_BASES.
        # Default 4 = no rotation, no flip (raw warp as-is, Y-down screen
        # convention).  Picked by the bbox-selector dialog; affects both the
        # displayed image and how exported tracker coords map to user X/Y.
        self.frame_orientation_state = 4

        # Real-world scaling information.
        # The "_native" values are the homography's intrinsic pixel scale (set
        # by perspective_corrector at calibration time). The non-native values
        # are the *effective* scale after corrections._apply_perspective
        # downsamples the output canvas to keep memory in check. Tracker
        # coordinate conversion uses the effective values directly; output_scale
        # is recorded in the sidecar for documentation/recovery.
        self.real_world_scale = False
        self.square_size_real = None
        self.square_size_pixels = None
        self.square_size_pixels_native = None
        self.pixels_per_real_unit = 1.0
        self.pixels_per_real_unit_native = 1.0
        self.output_scale = 1.0  # downsample factor: physical = logical / output_scale

    def set_calibration(self, camera_matrix, dist_coeffs, checkerboard_size,
                        error, image_size, model_type="pinhole",
                        fisheye_balance=0.0):
        """Set calibration parameters"""
        self.is_calibrated = True
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = dist_coeffs
        self.checkerboard_size = checkerboard_size
        self.calibration_error = error
        self.image_size = image_size
        self.model_type = model_type
        self.fisheye_balance = fisheye_balance

        # Initialize perspective correction matrix as identity (no correction)
        self.perspective_matrix = np.eye(3, dtype=np.float32)
        self.perspective_corrected = False
        self.perspective_translation_x = 0.0
        self.perspective_translation_y = 0.0
        self.output_bbox_world = None
        self.frame_orientation_state = 4

        # Reset scaling information
        self.real_world_scale = False
        self.square_size_real = None
        self.square_size_pixels = None
        self.square_size_pixels_native = None
        self.pixels_per_real_unit = 1.0
        self.pixels_per_real_unit_native = 1.0
        self.output_scale = 1.0

    def get_calibration_info(self):
        """Get human-readable calibration info"""
        if not self.is_calibrated:
            return "No calibration loaded"
        
        model_label = {
            "pinhole":  "Standard (pinhole, 5 coef)",
            "rational": "Wide-angle (rational, 8 coef)",
            "fisheye":  "Fisheye (equidistant, 4 coef)",
        }.get(self.model_type, f"Unknown ({self.model_type})")
        base_info = (f"Lens model: {model_label}\n"
                    f"Calibrated with {self.checkerboard_size[0]}x{self.checkerboard_size[1]} checkerboard\n"
                    f"Image size: {self.image_size[0]}x{self.image_size[1]}\n"
                    f"Calibration error: {self.calibration_error:.3f} pixels")
        
        if self.perspective_corrected:
            perspective_info = "\nPerspective correction: Applied"
            if self.real_world_scale and self.square_size_real:
                perspective_info += f"\nMeasurement units: Real-world ({self.square_size_real:.3f} units per square)"
                perspective_info += f"\nResolution maintained: {self.square_size_pixels:.1f} pixels per square in video"
            else:
                perspective_info += f"\nMeasurement units: Pixels"
            return base_info + perspective_info
        else:
            return base_info + "\nPerspective correction: Not applied"
            
    def set_perspective_correction(self, perspective_matrix, output_bbox_world,
                                   frame_orientation_state=4):
        """Commit a perspective calibration.

        output_bbox_world is required: the user-defined output region in
        pre-flip world coords at native homography scale, as
        (x_min, y_min, x_max, y_max). The runtime warp uses this bbox to
        size the output canvas, so a homography is only useful in tandem
        with one — they're set together to keep that invariant explicit.

        frame_orientation_state (0–7) chooses the export coordinate frame:
        states 0–3 are right-handed Z-up rotations (90° each); states 4–7
        are the same rotations with Y inverted (Z into the page).  The
        displayed image and bbox are unchanged — this only affects how
        tracker rel_x/rel_y values map onto the user's reported X/Y axes.
        See FRAME_ORIENTATION_BASES.
        """
        self.perspective_matrix = perspective_matrix
        self.perspective_corrected = True
        self.output_bbox_world = tuple(float(v) for v in output_bbox_world)
        self.frame_orientation_state = int(frame_orientation_state) % 8
        # Reset translation and downsample — recomputed on first warp call
        self.perspective_translation_x = 0.0
        self.perspective_translation_y = 0.0
        self.output_scale = 1.0

    def apply_frame_orientation(self, x, y):
        """Convert (x, y) from the displayed-image cartesian frame
        (state 0: X right, Y up) into the user-chosen coordinate frame
        defined by frame_orientation_state.

        See FRAME_ORIENTATION_BASES for the basis vectors.  The conversion
        is just a 2×2 matrix-vector multiply, but indirected through this
        method so callers don't have to care which state is set.
        """
        state = int(getattr(self, 'frame_orientation_state', 0)) % 8
        m = _FRAME_ORIENTATION_MATRICES[state]
        return (m[0][0] * x + m[0][1] * y,
                m[1][0] * x + m[1][1] * y)
        
    def get_scaled_camera_matrix(self, current_resolution):
        """Get camera matrix scaled for different resolution"""
        if not self.is_calibrated or self.image_size is None:
            return self.camera_matrix
            
        # If resolutions match, return original matrix
        calib_width, calib_height = self.image_size
        current_width, current_height = current_resolution
        
        if (abs(calib_width - current_width) <= 10 and 
            abs(calib_height - current_height) <= 10):
            return self.camera_matrix
            
        # Scale the camera matrix for different resolution
        scale_x = current_width / calib_width
        scale_y = current_height / calib_height
        
        # Scale camera matrix parameters
        scaled_matrix = self.camera_matrix.copy()
        scaled_matrix[0, 0] *= scale_x  # fx
        scaled_matrix[1, 1] *= scale_y  # fy
        scaled_matrix[0, 2] *= scale_x  # cx
        scaled_matrix[1, 2] *= scale_y  # cy
        
        print(f"Scaling camera matrix from {calib_width}x{calib_height} to {current_width}x{current_height}")
        print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        return scaled_matrix

    def convert_to_real_world_coordinates(self, pixel_x, pixel_y):
        """Convert pixel coordinates to real-world coordinates"""
        if self.real_world_scale:
            real_x = pixel_x / self.pixels_per_real_unit
            real_y = pixel_y / self.pixels_per_real_unit
            return real_x, real_y
        else:
            # No conversion needed for pixel coordinates
            return pixel_x, pixel_y
            
    def get_coordinate_units(self):
        """Get a description of coordinate units"""
        if self.real_world_scale:
            return f"real-world units ({self.square_size_real} units per square)"
        else:
            return "pixels"
