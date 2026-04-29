"""
Modal dialog with a zoomable, pannable preview canvas + draggable bbox.

Used at the end of perspective calibration to let the user define the
output region the runtime warp will produce.  The bbox is stored in
pre-orientation world coordinates at native homography scale, so it plugs
directly into calibration_data.output_bbox_world.

The preview always shows the scene in the chosen frame orientation — i.e.,
after apply_frame_orientation_to_image(warped, state) — so what the user
sees matches what the tracker will display.  Bbox world coordinates are
frame-orientation-independent; the same bbox is valid regardless of which
rotation the user picks.

Inputs (from the caller — typically perspective_corrector):
  preview_image_bgr : raw perspective-warped image, NO orientation applied
  proj_min_x, proj_min_y, preview_scale : conversion params so the dialog
                       can map screen ↔ world coords
  initial_bbox, full_frame_bbox, board_extents_bbox : in world coords
  world_units_per_pixel, units_label : optional, drives the dimensions HUD

Output:
  show() returns (bbox_world, frame_orientation_state) on OK, or None
  if cancelled.  frame_orientation_state is an integer 0–7 chosen by the
  "Rotate frame ↻" button — see data_models.FRAME_ORIENTATION_BASES.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

from .data_models import FRAME_ORIENTATION_BASES
from .corrections import apply_frame_orientation_to_image


# Handle ownership per frame_orientation_state.  For each state, the
# compass-named handle sits at the screen position controlled by these world
# coordinates.  Derived from the inverse of _transform_pixel: for a given
# state, "north" (smallest screen-Y) corresponds to whichever world axis
# maps to small screen-Y in that orientation.
_HANDLE_OWNERSHIP_BY_STATE = {
    # (xname_or_None, yname_or_None) for each handle
    0: {"nw": ("x_min", "y_max"), "ne": ("x_max", "y_max"),
        "sw": ("x_min", "y_min"), "se": ("x_max", "y_min"),
        "n":  (None,    "y_max"), "s":  (None,    "y_min"),
        "e":  ("x_max", None),    "w":  ("x_min", None)},
    1: {"nw": ("x_min", "y_min"), "ne": ("x_min", "y_max"),
        "sw": ("x_max", "y_min"), "se": ("x_max", "y_max"),
        "n":  ("x_min", None),    "s":  ("x_max", None),
        "e":  (None,    "y_max"), "w":  (None,    "y_min")},
    2: {"nw": ("x_max", "y_min"), "ne": ("x_min", "y_min"),
        "sw": ("x_max", "y_max"), "se": ("x_min", "y_max"),
        "n":  (None,    "y_min"), "s":  (None,    "y_max"),
        "e":  ("x_min", None),    "w":  ("x_max", None)},
    3: {"nw": ("x_max", "y_max"), "ne": ("x_max", "y_min"),
        "sw": ("x_min", "y_max"), "se": ("x_min", "y_min"),
        "n":  ("x_max", None),    "s":  ("x_min", None),
        "e":  (None,    "y_min"), "w":  (None,    "y_max")},
    4: {"nw": ("x_min", "y_min"), "ne": ("x_max", "y_min"),
        "sw": ("x_min", "y_max"), "se": ("x_max", "y_max"),
        "n":  (None,    "y_min"), "s":  (None,    "y_max"),
        "e":  ("x_max", None),    "w":  ("x_min", None)},
    5: {"nw": ("x_max", "y_min"), "ne": ("x_max", "y_max"),
        "sw": ("x_min", "y_min"), "se": ("x_min", "y_max"),
        "n":  ("x_max", None),    "s":  ("x_min", None),
        "e":  (None,    "y_max"), "w":  (None,    "y_min")},
    6: {"nw": ("x_max", "y_max"), "ne": ("x_min", "y_max"),
        "sw": ("x_max", "y_min"), "se": ("x_min", "y_min"),
        "n":  (None,    "y_max"), "s":  (None,    "y_min"),
        "e":  ("x_min", None),    "w":  ("x_max", None)},
    7: {"nw": ("x_min", "y_max"), "ne": ("x_min", "y_min"),
        "sw": ("x_max", "y_max"), "se": ("x_max", "y_min"),
        "n":  ("x_min", None),    "s":  ("x_max", None),
        "e":  (None,    "y_min"), "w":  (None,    "y_max")},
}

_HANDLE_SIZE = 6  # half-width of handle square in screen pixels


class BboxSelectorDialog:
    """Modal dialog that asks the user to draw an output bbox."""

    def __init__(self, parent, preview_image_bgr,
                 proj_min_x, proj_min_y, preview_scale,
                 initial_bbox, full_frame_bbox, board_extents_bbox,
                 world_units_per_pixel=None, units_label="px",
                 initial_frame_orientation=0,
                 title="Define Output Region"):
        self.parent = parent

        # Raw warp image (no orientation applied) and its dimensions.
        self._raw_warped_bgr = preview_image_bgr
        self._raw_h, self._raw_w = preview_image_bgr.shape[:2]

        # World-to-raw-warp-pixel conversion parameters.
        # Raw warp pixel (px0,py0) = ((wx-proj_min_x)*ps, (wy-proj_min_y)*ps)
        self.proj_min_x = float(proj_min_x)
        self.proj_min_y = float(proj_min_y)
        self.preview_scale = float(preview_scale)

        # HUD parameters
        self.world_units_per_pixel = world_units_per_pixel  # may be None
        self.units_label = units_label

        # Bbox state in world coords (frame-orientation-independent).
        self.bbox_world = tuple(float(v) for v in initial_bbox)
        self.full_frame_bbox = tuple(float(v) for v in full_frame_bbox)
        self.board_extents_bbox = tuple(float(v) for v in board_extents_bbox)

        # Frame orientation state (0–7).  Cycled by "Rotate frame" button.
        # Determines both the rendered preview and the handle/coord mapping.
        self.frame_orientation_state = int(initial_frame_orientation) % 8

        # Returned via show() — (bbox_world, frame_orientation_state)
        self.result = None

        # Display state — zoom multiplier, pan offset in screen pixels
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Drag state
        self._drag_mode = None
        self._drag_start_screen = None
        self._drag_initial_bbox = None
        self._pan_start = None

        # Oriented preview — rebuilt whenever frame_orientation_state changes.
        self._preview_pil = None
        self._preview_photo = None
        self.preview_w = 0
        self.preview_h = 0
        self._rebuild_oriented_preview()

        self._build_dialog(title)
        self.dialog.after(50, self._fit_to_window)

    # ------------------------------------------------ oriented-preview rebuild

    def _rebuild_oriented_preview(self):
        """Convert the raw warp to a PIL image.

        The preview is always the raw warp with no rotation or flip applied —
        the frame orientation only affects the XY axis overlay, not the image.
        """
        rgb = cv2.cvtColor(self._raw_warped_bgr, cv2.COLOR_BGR2RGB)
        self._preview_pil = Image.fromarray(rgb)
        self.preview_w, self.preview_h = self._preview_pil.size

    # ------------------------------------------------------------------ UI

    def _build_dialog(self, title):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(title)
        self.dialog.geometry("960x780")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)

        top = ttk.Frame(self.dialog, padding=8)
        top.pack(side='top', fill='x')
        ttk.Label(
            top,
            text=("Drag in empty space to draw, drag interior to move, drag "
                  "handles to resize. Mouse wheel = zoom, middle-click drag "
                  "= pan."),
            wraplength=520,
        ).pack(side='left')
        ttk.Button(top, text="Board extents",
                   command=self._on_board_extents).pack(side='right', padx=2)
        ttk.Button(top, text="Full frame",
                   command=self._on_full_frame).pack(side='right', padx=2)
        ttk.Button(top, text="Fit to window",
                   command=self._fit_to_window).pack(side='right', padx=2)
        ttk.Button(top, text="Rotate frame ↻",
                   command=self._on_rotate_frame).pack(side='right', padx=2)

        bot = ttk.Frame(self.dialog, padding=8)
        bot.pack(side='bottom', fill='x')
        ttk.Button(bot, text="OK", command=self._on_ok).pack(side='right', padx=2)
        ttk.Button(bot, text="Cancel",
                   command=self._on_cancel).pack(side='right', padx=2)
        self.hud_label = ttk.Label(bot, text="", font=('Consolas', 10))
        self.hud_label.pack(side='left')

        canvas_frame = ttk.Frame(self.dialog)
        canvas_frame.pack(side='top', fill='both', expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg='black',
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill='both', expand=True)

        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<ButtonPress-1>",   self._on_left_press)
        self.canvas.bind("<B1-Motion>",       self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-2>",   self._on_middle_press)
        self.canvas.bind("<B2-Motion>",       self._on_middle_drag)
        self.canvas.bind("<MouseWheel>",      self._on_mouse_wheel)
        self.canvas.bind("<Button-4>",        self._on_mouse_wheel)
        self.canvas.bind("<Button-5>",        self._on_mouse_wheel)
        self.canvas.bind("<Motion>",          self._on_motion)

    def show(self):
        """Block until OK or Cancel.  Returns (bbox_world, state) or None."""
        self.dialog.wait_window()
        return self.result

    # --------------------------------------------------- coord conversions

    def _screen_to_preview(self, sx, sy):
        return ((sx - self.pan_x) / self.zoom,
                (sy - self.pan_y) / self.zoom)

    def _preview_to_screen(self, px, py):
        return (px * self.zoom + self.pan_x,
                py * self.zoom + self.pan_y)

    def _world_to_preview(self, wx, wy):
        """World coord → preview pixel (simple linear; image is never rotated)."""
        return ((wx - self.proj_min_x) * self.preview_scale,
                (wy - self.proj_min_y) * self.preview_scale)

    def _preview_to_world(self, px, py):
        """Preview pixel → world coord (inverse of _world_to_preview)."""
        return (px / self.preview_scale + self.proj_min_x,
                py / self.preview_scale + self.proj_min_y)

    def _world_to_screen(self, wx, wy):
        return self._preview_to_screen(*self._world_to_preview(wx, wy))

    def _screen_to_world(self, sx, sy):
        return self._preview_to_world(*self._screen_to_preview(sx, sy))

    # --------------------------------------------------------- redrawing

    def _fit_to_window(self):
        self.dialog.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.dialog.after(50, self._fit_to_window)
            return
        zoom = min(cw / self.preview_w, ch / self.preview_h)
        self.zoom = max(0.01, zoom)
        self.pan_x = (cw - self.preview_w * self.zoom) / 2
        self.pan_y = (ch - self.preview_h * self.zoom) / 2
        self._redraw()

    def _redraw(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        vis_x0 = max(0, int(-self.pan_x / self.zoom))
        vis_y0 = max(0, int(-self.pan_y / self.zoom))
        vis_x1 = min(self.preview_w, int((cw - self.pan_x) / self.zoom) + 1)
        vis_y1 = min(self.preview_h, int((ch - self.pan_y) / self.zoom) + 1)

        self.canvas.delete("preview_image")
        if vis_x1 > vis_x0 and vis_y1 > vis_y0:
            cropped = self._preview_pil.crop((vis_x0, vis_y0, vis_x1, vis_y1))
            disp_w = max(1, int((vis_x1 - vis_x0) * self.zoom))
            disp_h = max(1, int((vis_y1 - vis_y0) * self.zoom))
            resampling = (Image.LANCZOS if self.zoom < 1.0 else Image.NEAREST)
            resized = cropped.resize((disp_w, disp_h), resampling)
            self._preview_photo = ImageTk.PhotoImage(resized)
            screen_x = vis_x0 * self.zoom + self.pan_x
            screen_y = vis_y0 * self.zoom + self.pan_y
            self.canvas.create_image(screen_x, screen_y, anchor='nw',
                                     image=self._preview_photo,
                                     tags="preview_image")
        self._draw_bbox_overlay()
        self._draw_axes_overlay()
        self._update_hud()

    def _draw_axes_overlay(self):
        """Draw user-frame axes at world origin.

        The preview image is always the raw warp (no rotation/flip).  The axis
        arrows are drawn in the screen directions implied by the chosen
        frame_orientation_state.  FRAME_ORIENTATION_BASES gives each axis as a
        2-vector in the after-vflip cartesian frame (Y-up).  The preview uses
        Y-down (screen), so the screen direction is (bx, -by).
        """
        self.canvas.delete("axes")
        origin_sx, origin_sy = self._world_to_screen(0.0, 0.0)
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        margin = 60
        if (origin_sx < -margin or origin_sx > cw + margin
                or origin_sy < -margin or origin_sy > ch + margin):
            return

        x_basis, y_basis, label = FRAME_ORIENTATION_BASES[self.frame_orientation_state]
        L = 60
        # (bx, by) in Y-up frame → screen direction (bx, -by) in Y-down frame
        x_end = (origin_sx + L * x_basis[0], origin_sy - L * x_basis[1])
        y_end = (origin_sx + L * y_basis[0], origin_sy - L * y_basis[1])

        self.canvas.create_oval(origin_sx - 3, origin_sy - 3,
                                origin_sx + 3, origin_sy + 3,
                                fill='yellow', outline='black', tags="axes")
        self._arrow(origin_sx, origin_sy, *x_end, "red",  "X")
        self._arrow(origin_sx, origin_sy, *y_end, "lime", "Y")
        self.canvas.create_text(origin_sx, origin_sy + L + 22,
                                text=label, fill='yellow',
                                font=('Consolas', 9, 'bold'), tags="axes")

    def _arrow(self, x0, y0, x1, y1, colour, label):
        self.canvas.create_line(x0, y0, x1, y1, fill=colour, width=2,
                                arrow=tk.LAST, arrowshape=(10, 12, 4),
                                tags="axes")
        self.canvas.create_text(x1, y1, text=label, fill=colour,
                                font=('Consolas', 9, 'bold'), tags="axes")

    def _draw_bbox_overlay(self):
        self.canvas.delete("bbox")
        if self.bbox_world is None:
            return
        x_min, y_min, x_max, y_max = self.bbox_world

        # Rectangle: warpPerspective + _world_to_screen maps any two opposite
        # world corners to opposite screen corners.  (x_min,y_min)–(x_max,y_max)
        # are always opposite in world space, so this works for all states.
        sx1, sy1 = self._world_to_screen(x_min, y_min)
        sx2, sy2 = self._world_to_screen(x_max, y_max)
        self.canvas.create_rectangle(sx1, sy1, sx2, sy2,
                                     outline='cyan', width=2, tags="bbox")
        for sx, sy, name in self._iter_handles():
            h = _HANDLE_SIZE
            self.canvas.create_rectangle(
                sx - h, sy - h, sx + h, sy + h,
                fill='cyan', outline='white', tags="bbox")

    def _iter_handles(self):
        if self.bbox_world is None:
            return
        x_min, y_min, x_max, y_max = self.bbox_world
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        ownership = _HANDLE_OWNERSHIP_BY_STATE[4]  # image never rotated
        for name, (xname, yname) in ownership.items():
            wx = (x_min if xname == "x_min" else
                  x_max if xname == "x_max" else x_mid)
            wy = (y_min if yname == "y_min" else
                  y_max if yname == "y_max" else y_mid)
            sx, sy = self._world_to_screen(wx, wy)
            yield (sx, sy, name)

    def _update_hud(self):
        if self.bbox_world is None:
            self.hud_label.configure(text="No region selected")
            return
        x_min, y_min, x_max, y_max = self.bbox_world
        w_world = x_max - x_min
        h_world = y_max - y_min
        if self.world_units_per_pixel and self.world_units_per_pixel > 0:
            w_real = w_world / self.world_units_per_pixel
            h_real = h_world / self.world_units_per_pixel
            raw_w = int(round(w_world))
            raw_h = int(round(h_world))
            _, _, label = FRAME_ORIENTATION_BASES[self.frame_orientation_state]
            self.hud_label.configure(
                text=(f"BBox: {w_real:.2f} × {h_real:.2f} {self.units_label}"
                      f"  (native px: {raw_w} × {raw_h})"
                      f"  |  Frame: {label}"))
        else:
            _, _, label = FRAME_ORIENTATION_BASES[self.frame_orientation_state]
            self.hud_label.configure(
                text=(f"BBox: {int(w_world)} × {int(h_world)} px"
                      f"  |  Frame: {label}"))

    # -------------------------------------------------------- hit-testing

    def _hit_test(self, sx, sy):
        h = _HANDLE_SIZE + 2
        for handle_sx, handle_sy, name in self._iter_handles():
            if abs(sx - handle_sx) <= h and abs(sy - handle_sy) <= h:
                return name
        if self.bbox_world is not None:
            x_min, y_min, x_max, y_max = self.bbox_world
            sx1, sy1 = self._world_to_screen(x_min, y_min)
            sx2, sy2 = self._world_to_screen(x_max, y_max)
            if (min(sx1, sx2) <= sx <= max(sx1, sx2)
                    and min(sy1, sy2) <= sy <= max(sy1, sy2)):
                return "move"
        return None

    # -------------------------------------------------------- mouse handlers

    def _on_left_press(self, event):
        sx, sy = event.x, event.y
        hit = self._hit_test(sx, sy)
        self._drag_start_screen = (sx, sy)
        self._drag_initial_bbox = self.bbox_world
        self._drag_mode = hit if hit else "draw"

    def _on_left_drag(self, event):
        if self._drag_mode is None:
            return
        sx, sy = event.x, event.y
        start_sx, start_sy = self._drag_start_screen

        if self._drag_mode == "draw":
            wx_start, wy_start = self._screen_to_world(start_sx, start_sy)
            wx_cur,   wy_cur   = self._screen_to_world(sx, sy)
            self.bbox_world = (
                min(wx_start, wx_cur), min(wy_start, wy_cur),
                max(wx_start, wx_cur), max(wy_start, wy_cur),
            )
        else:
            # Use _screen_to_world to compute world delta — this is
            # coordinate-system agnostic and handles all 8 orientation states
            # correctly (including 90° rotations where screen X ↔ world Y).
            wx_start, wy_start = self._screen_to_world(start_sx, start_sy)
            wx_cur,   wy_cur   = self._screen_to_world(sx, sy)
            dwx = wx_cur - wx_start
            dwy = wy_cur - wy_start

            x_min, y_min, x_max, y_max = self._drag_initial_bbox
            if self._drag_mode == "move":
                x_min += dwx; x_max += dwx
                y_min += dwy; y_max += dwy
            else:
                ownership = _HANDLE_OWNERSHIP_BY_STATE[4]  # image never rotated
                xname, yname = ownership[self._drag_mode]
                if xname == "x_min": x_min += dwx
                if xname == "x_max": x_max += dwx
                if yname == "y_min": y_min += dwy
                if yname == "y_max": y_max += dwy
            self.bbox_world = (
                min(x_min, x_max), min(y_min, y_max),
                max(x_min, x_max), max(y_min, y_max),
            )

        self._draw_bbox_overlay()
        self._update_hud()

    def _on_left_release(self, _event):
        self._drag_mode = None
        self._drag_start_screen = None
        self._drag_initial_bbox = None

    def _on_middle_press(self, event):
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)
        self.canvas.configure(cursor="fleur")

    def _on_middle_drag(self, event):
        if self._pan_start is None:
            return
        sx0, sy0, px0, py0 = self._pan_start
        self.pan_x = px0 + (event.x - sx0)
        self.pan_y = py0 + (event.y - sy0)
        self._redraw()

    def _on_mouse_wheel(self, event):
        if hasattr(event, 'delta') and event.delta:
            zoom_in = event.delta > 0
        elif getattr(event, 'num', None) == 4:
            zoom_in = True
        elif getattr(event, 'num', None) == 5:
            zoom_in = False
        else:
            return
        factor = 1.25 if zoom_in else 1.0 / 1.25

        sx, sy = event.x, event.y
        px_before = (sx - self.pan_x) / self.zoom
        py_before = (sy - self.pan_y) / self.zoom

        new_zoom = max(0.05, min(self.zoom * factor, 50.0))
        self.zoom = new_zoom
        self.pan_x = sx - px_before * self.zoom
        self.pan_y = sy - py_before * self.zoom
        self._redraw()

    def _on_motion(self, event):
        if self._drag_mode is not None:
            return
        hit = self._hit_test(event.x, event.y)
        cursor_for = {
            "nw": "size_nw_se", "se": "size_nw_se",
            "ne": "size_ne_sw", "sw": "size_ne_sw",
            "n":  "size_ns",    "s":  "size_ns",
            "e":  "size_we",    "w":  "size_we",
            "move": "fleur",
        }
        self.canvas.configure(cursor=cursor_for.get(hit, "crosshair"))

    # ------------------------------------------------------ button actions

    def _on_full_frame(self):
        self.bbox_world = self.full_frame_bbox
        self._draw_bbox_overlay()
        self._update_hud()

    def _on_board_extents(self):
        self.bbox_world = self.board_extents_bbox
        self._draw_bbox_overlay()
        self._update_hud()

    def _on_rotate_frame(self):
        self.frame_orientation_state = (self.frame_orientation_state + 1) % 8
        self._rebuild_oriented_preview()
        self._fit_to_window()

    def _on_ok(self):
        x_min, y_min, x_max, y_max = self.bbox_world
        if x_max - x_min < 1 or y_max - y_min < 1:
            messagebox.showerror(
                "Invalid bbox",
                "The selected region is degenerate. Pick a non-zero area.",
                parent=self.dialog)
            return
        self.result = (self.bbox_world, self.frame_orientation_state)
        self.dialog.destroy()

    def _on_cancel(self):
        self.result = None
        self.dialog.destroy()
