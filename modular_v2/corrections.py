"""
Runtime application of lens + perspective corrections.

Single source of truth for the per-frame correction pipeline. Three GUI
modules (calibration_preview, measurement_recorder, tracking_v7) call into
here so the math can't drift between them.

Lens step branches on calibration_data.model_type:
  - "pinhole" / "rational":  cv2.initUndistortRectifyMap + cv2.remap
  - "fisheye":               cv2.fisheye.undistortImage
The output canvas is expanded by _LENS_EXPAND_FACTOR with the principal
point shifted to centre source content, so pixels pushed beyond the
original frame extent by aggressive undistortion are retained instead
of clipped.  Empty regions are filled with black.

Perspective step uses the user-defined output bbox stored on
calibration_data.output_bbox_world (set by the bbox-selector dialog at
the end of perspective calibration). The bbox is in pre-flip world
coords at native homography scale; the warp translates so its (x_min,
y_min) corner lands at canvas (0, 0), then downsamples to keep the
physical buffer ≤ _MAX_PHYSICAL_PIXELS. The downsample factor is folded
into the homography and pushed back to calibration_data so
pixels_per_real_unit reflects the physical pixel grid (tracker metric
output stays correct without further compensation). A vertical flip
follows so the output obeys Cartesian convention.
"""

import cv2
import numpy as np


def apply_corrections(frame, calibration_data,
                      skip_lens=False, skip_perspective=False):
    """Apply lens distortion + perspective corrections to a single frame.

    Args:
        frame: BGR ndarray.
        calibration_data: CalibrationData instance (see data_models.py).
        skip_lens: True when the source frame already has lens correction
            baked in (avoids double-application).
        skip_perspective: True when the source frame already has perspective
            correction baked in.

    Returns the corrected frame. On any cv2 error returns the original
    frame unchanged and prints a diagnostic; callers may surface a UI
    warning if they want a stronger signal.
    """
    if frame is None:
        return None

    corrected = frame.copy()

    try:
        if calibration_data.is_calibrated and not skip_lens:
            corrected = _apply_lens(corrected, calibration_data)

        if calibration_data.perspective_corrected and not skip_perspective:
            corrected = _apply_perspective(corrected, calibration_data)

        return corrected

    except Exception as e:
        print(f"Error applying corrections: {e}")
        return frame


# Lens-correction output canvas multiplier. Output dims = source dims ×
# this factor; the principal point is shifted by half the expansion so
# source content stays centred. Values > 1 retain pixels that aggressive
# undistortion would otherwise push beyond the original frame extent
# (visible as a hard horizontal/vertical clip line in the corrected
# preview). The trade-off is more memory and downstream compute, plus a
# perspective homography that is coordinated with this factor — changing
# this constant invalidates existing perspective calibrations.
_LENS_EXPAND_FACTOR = 2.0


def _scaramuzza_world2cam(world_points, inverse_poly, distortion_center, stretch_matrix):
    """3-D world rays -> fisheye image pixels (Scaramuzza forward projection).

    Hand-rolled port of py-OCamCalib's Camera.world2cam_fast so the GUI and
    the ROS service share an implementation we can verify against each other.
    Math is identical except for an arccos clip that hardens against float
    drift at z/||p|| = ±1 (py-OCamCalib would NaN there).
    """
    norms = np.linalg.norm(world_points, axis=1)
    norms_safe = np.where(norms == 0, 1.0, norms)
    z_norm = np.clip(world_points[:, 2] / norms_safe, -1.0, 1.0)
    theta = np.arccos(z_norm)
    rho = np.polyval(inverse_poly, theta)

    pr = np.sqrt(world_points[:, 0] ** 2 + world_points[:, 1] ** 2)
    pr_safe = np.where(pr == 0, np.finfo(float).eps, pr)
    px = (world_points[:, 0] / pr_safe) * rho
    py = (world_points[:, 1] / pr_safe) * rho

    # Replicates py-OCamCalib's stretch-matrix ordering: the second line
    # uses the already-modified px_new (equivalent to their Camera.world2cam_fast).
    px_new = px * stretch_matrix[0, 0] + py * stretch_matrix[0, 1]
    py_new = px_new * stretch_matrix[1, 0] + py
    return np.column_stack((px_new + distortion_center[0],
                            py_new + distortion_center[1]))


def _build_scaramuzza_maps(params, fov_deg, out_w, out_h):
    """Build cv2.remap source maps for Scaramuzza omnidirectional undistortion.

    For each pixel (u, v) in an out_w × out_h perspective output image, computes
    the corresponding source pixel in the raw fisheye frame using the Scaramuzza
    inverse polynomial model.  Returns (mapx, mapy) as float32 arrays of shape
    (out_h, out_w), ready for cv2.remap.

    fov_deg is the horizontal field-of-view of the virtual perspective camera.
    Smaller values crop the fisheye circle tighter; larger values include more
    of the FOV at the cost of a wider black border.

    params dict keys (py-OCamCalib JSON):
      inverse_poly      — descending-order polynomial (np.polyfit output)
      distortion_center — [xc, yc] principal point in the fisheye image
      stretch_matrix    — 2×2 affine
    """
    inverse_poly = np.asarray(params['inverse_poly'], dtype=np.float64)
    dc = params['distortion_center']
    stretch = np.asarray(params['stretch_matrix'], dtype=np.float64)

    # Clamp away from ≥180° where tan(90°) → ∞ makes f → 0 and collapses
    # all output pixels to the same source point (black image).
    fov_deg = min(float(fov_deg), 179.0)
    f = max(out_w, out_h) / (2.0 * np.tan(np.radians(fov_deg / 2.0)))

    u_out = np.arange(out_w, dtype=np.float64) - out_w / 2.0
    v_out = np.arange(out_h, dtype=np.float64) - out_h / 2.0
    X, Y = np.meshgrid(u_out, v_out)
    world_points = np.column_stack((X.ravel(), Y.ravel(),
                                    np.full(X.size, f, dtype=np.float64)))

    img_pts = _scaramuzza_world2cam(world_points, inverse_poly, dc, stretch)
    mapx = img_pts[:, 0].reshape(out_h, out_w).astype(np.float32)
    mapy = img_pts[:, 1].reshape(out_h, out_w).astype(np.float32)
    return mapx, mapy


def _apply_lens(frame, calibration_data):
    """Apply lens distortion correction with an expanded output canvas.

    Branches on model_type. Output is _LENS_EXPAND_FACTOR × source in
    each dim, with the principal point shifted to centre source content
    in the expanded canvas. Areas outside the source's projected extent
    are filled with black (BORDER_CONSTANT).
    """
    h, w = frame.shape[:2]
    current_resolution = (w, h)
    K = calibration_data.get_scaled_camera_matrix(current_resolution)
    D = calibration_data.distortion_coefficients

    new_w = int(w * _LENS_EXPAND_FACTOR)
    new_h = int(h * _LENS_EXPAND_FACTOR)
    dx = (new_w - w) // 2
    dy = (new_h - h) // 2

    if getattr(calibration_data, 'model_type', 'pinhole') == 'scaramuzza':
        params = getattr(calibration_data, 'scaramuzza_params', None)
        if params is None:
            return frame
        fov = float(getattr(calibration_data, 'scaramuzza_fov', 180.0))
        # Cache remap maps; rebuild only when resolution or FOV changes.
        cache_key = (fov, new_w, new_h)
        cached = getattr(calibration_data, '_scaramuzza_remap_cache', None)
        if cached is None or cached[0] != cache_key:
            mapx, mapy = _build_scaramuzza_maps(params, fov, new_w, new_h)
            calibration_data._scaramuzza_remap_cache = (cache_key, mapx, mapy)
        else:
            _, mapx, mapy = cached
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if getattr(calibration_data, 'model_type', 'pinhole') == 'fisheye':
        balance = float(getattr(calibration_data, 'fisheye_balance', 0.0))
        # Compute Knew sized for the original frame, then shift cx/cy so
        # the same content is centred in the expanded canvas.
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, current_resolution, np.eye(3), balance=balance)
        Knew = Knew.copy().astype(np.float64)
        Knew[0, 2] += dx
        Knew[1, 2] += dy
        return cv2.fisheye.undistortImage(
            frame, K, D=D, Knew=Knew, new_size=(new_w, new_h))

    # Pinhole / Brown-Conrady (the rational model uses the same path —
    # cv2.undistort/initUndistortRectifyMap auto-handle the longer D).
    Knew = K.copy().astype(np.float64)
    Knew[0, 2] += dx
    Knew[1, 2] += dy
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, D, None, Knew, (new_w, new_h), cv2.CV_32FC1)
    return cv2.remap(
        frame, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# Maximum pixels in the physical output buffer. The logical canvas (the
# user's chosen output bbox in world coords) may be much larger; when it
# exceeds this budget, the warp is downsampled so the buffer stays under
# the cap. 8 MP ≈ 24 MB BGR per frame.
_MAX_PHYSICAL_PIXELS = 8_000_000


# Image transformation per frame_orientation_state. Each entry is
# (rotation_op, vflip) where rotation_op ∈ {None, 'cw', 'ccw', '180'} and
# vflip flips the result vertically afterward. After the transform, user-frame
# +X points to canvas-right and user-frame +Y points to canvas-up (Cartesian
# screen convention), where user directions are defined relative to the raw
# warp image the user sees in the bbox selector (small y_raw = top of display).
# States 0-3 are right-handed (Z out); states 4-7 are left-handed (Z in).
_FRAME_DISPLAY_OPS = {
    0: (None,  False),  # X right, Y up    (Z out)
    1: ('cw',  False),  # X up,    Y left  (Z out)
    2: ('180', False),  # X left,  Y down  (Z out)
    3: ('ccw', False),  # X down,  Y right (Z out)
    4: (None,  True),   # X right, Y down  (Z in)
    5: ('cw',  True),   # X up,    Y right (Z in)
    6: ('180', True),   # X left,  Y up    (Z in)
    7: ('ccw', True),   # X down,  Y left  (Z in)
}


def apply_frame_orientation_to_image(image, state):
    """Rotate/flip an image so user-frame axes align with screen axes.

    After the transform, the displayed image's right-direction is the
    user-frame +X and the up-direction is the user-frame +Y. Together with
    the perspective warp this means tracker pixel coords on the displayed
    canvas can be read directly as user-frame coords (after subtracting
    the origin and negating Y for canvas-vs-screen Y direction).
    """
    rot, vflip = _FRAME_DISPLAY_OPS.get(int(state), (None, False))
    if rot == 'cw':
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 'ccw':
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot == '180':
        image = cv2.rotate(image, cv2.ROTATE_180)
    if vflip:
        image = cv2.flip(image, 0)
    return image


def _transform_pixel(px, py, w, h, state):
    """Map a (px, py) pixel position from the warped (pre-orientation)
    canvas of size (w, h) to its post-orientation pixel position. Used to
    reproject the world origin marker after the orientation transform.
    """
    rot, vflip = _FRAME_DISPLAY_OPS.get(int(state), (None, False))
    if rot == 'cw':
        # cv2.ROTATE_90_CLOCKWISE: (x, y) → (h - 1 - y, x), new size (h, w)
        new_w, new_h = h, w
        px, py = (h - 1 - py), px
    elif rot == 'ccw':
        # cv2.ROTATE_90_COUNTERCLOCKWISE: (x, y) → (y, w - 1 - x), new size (h, w)
        new_w, new_h = h, w
        px, py = py, (w - 1 - px)
    elif rot == '180':
        # cv2.ROTATE_180: (x, y) → (w - 1 - x, h - 1 - y), size unchanged
        new_w, new_h = w, h
        px, py = (w - 1 - px), (h - 1 - py)
    else:
        new_w, new_h = w, h
    if vflip:
        py = new_h - 1 - py
    return px, py, new_w, new_h


def _inverse_transform_pixel(px, py, raw_w, raw_h, state):
    """Inverse of _transform_pixel.

    Maps a pixel (px, py) in the post-orientation canvas back to the
    corresponding pixel in the raw (pre-orientation) warp canvas of size
    (raw_w, raw_h).  Used by the bbox selector to convert oriented preview
    pixel coords to world coords.
    """
    s = int(state) % 8
    if s == 0:   # vflip only
        return px, raw_h - 1 - py
    elif s == 1: # ccw + vflip  → (px0,py0)=(py,px)
        return py, px
    elif s == 2: # 180 + vflip  → hflip only
        return raw_w - 1 - px, py
    elif s == 3: # cw  + vflip  → (px0,py0)=(raw_w-1-py, raw_h-1-px)
        return raw_w - 1 - py, raw_h - 1 - px
    elif s == 4: # identity
        return px, py
    elif s == 5: # ccw only     → (px0,py0)=(raw_w-1-py, px)
        return raw_w - 1 - py, px
    elif s == 6: # 180 only
        return raw_w - 1 - px, raw_h - 1 - py
    else:        # s == 7: cw only → (px0,py0)=(py, raw_h-1-px)
        return py, raw_h - 1 - px


def _apply_perspective(frame, calibration_data):
    """Apply perspective warp + frame-orientation transform.

    The bbox on calibration_data.output_bbox_world (in world coords at
    native homography scale) crops the warp to the user-chosen region and
    is downsampled to fit the _MAX_PHYSICAL_PIXELS budget. The result is
    then rotated/flipped per calibration_data.frame_orientation_state so
    the displayed image is in the user's chosen coord frame: canvas-right
    is user-frame +X, canvas-up is user-frame +Y. Tracker pixel coords on
    the final canvas can therefore be read directly as user-frame coords
    (after subtracting the origin and negating Y for canvas Y direction).

    perspective_translation_x/y is set to the canvas pixel of world (0, 0)
    in the FINAL transformed canvas, so callers that draw the world-origin
    marker get the right location regardless of orientation.

    If the bbox isn't set, returns the frame unchanged.
    """
    bbox = getattr(calibration_data, 'output_bbox_world', None)
    if bbox is None:
        return frame
    x_min, y_min, x_max, y_max = bbox

    M = calibration_data.perspective_matrix
    state = int(getattr(calibration_data, 'frame_orientation_state', 0))

    logical_w = max(1, int(np.ceil(x_max - x_min)))
    logical_h = max(1, int(np.ceil(y_max - y_min)))

    # Downsample for the physical pixel-budget cap.
    pixel_budget_ratio = (logical_w * logical_h) / _MAX_PHYSICAL_PIXELS
    output_scale = max(1.0, float(np.sqrt(pixel_budget_ratio)))
    out_w = max(1, int(np.ceil(logical_w / output_scale)))
    out_h = max(1, int(np.ceil(logical_h / output_scale)))

    # Combined matrix: scale (1/output_scale) folded with translation that
    # shifts world (x_min, y_min) to canvas (0, 0).  Source pixel p →
    # world coord (M @ p), then translated and downsampled so the chosen
    # bbox fills the physical canvas.
    s_inv = 1.0 / output_scale
    M_combined = np.array(
        [[s_inv, 0, -x_min * s_inv],
         [0, s_inv, -y_min * s_inv],
         [0, 0, 1]],
        dtype=np.float64,
    ) @ M.astype(np.float64)

    warped = cv2.warpPerspective(
        frame,
        M_combined,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Apply user-frame orientation (rotation + optional vflip).
    result = apply_frame_orientation_to_image(warped, state)

    # Push effective scale back to calibration_data. _native values are
    # left untouched (set once by perspective_corrector); effective values
    # are recomputed each call from native / output_scale, so the operation
    # is idempotent across repeated frame calls.
    sq_native = getattr(calibration_data, 'square_size_pixels_native', None)
    pp_native = getattr(calibration_data, 'pixels_per_real_unit_native', None)
    calibration_data.output_scale = output_scale
    if pp_native is not None:
        calibration_data.pixels_per_real_unit = pp_native / output_scale
    if sq_native is not None:
        calibration_data.square_size_pixels = sq_native / output_scale

    # World origin (0, 0) lands at warped pixel (-x_min/s, -y_min/s); push
    # that through the orientation transform to get the post-transform pixel.
    pre_tx = -x_min * s_inv
    pre_ty = -y_min * s_inv
    final_tx, final_ty, _, _ = _transform_pixel(
        pre_tx, pre_ty, out_w, out_h, state)
    calibration_data.perspective_translation_x = float(final_tx)
    calibration_data.perspective_translation_y = float(final_ty)

    return result
