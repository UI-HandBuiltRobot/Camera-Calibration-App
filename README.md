# Camera Measurement Application — Student Guide

This application turns ordinary video into quantitative motion data. You point a
camera (or your phone) at a moving object, calibrate so the software knows how
your lens distorts the world, and then track the object frame-by-frame to get a
time series of position in real-world units.

It is the measurement front-end for the SDOF added-mass experiment, but the
workflow is general: anything you can record and see can be tracked.

---

## What the app actually computes

Three pieces of math sit underneath the GUI. You don't need to derive them, but
knowing what each step *does* makes the buttons less mysterious.

1. **Lens (intrinsic) calibration** — fits a pinhole camera model with a
   *rational* Brown–Conrady distortion polynomial (8 distortion coefficients,
   `k1, k2, p1, p2, k3, k4, k5, k6`). The fit produces a 3×3 camera matrix `K`
   (focal lengths `fx, fy` and principal point `cx, cy`) plus the distortion
   vector. At runtime, `cv2.undistort` uses these to remove barrel/pincushion
   warping so straight lines in the world stay straight in the image.
2. **Perspective (extrinsic) correction** — a 3×3 *homography* `H` that maps the
   tilted, foreshortened view of your measurement plane to a clean
   top-down/front-on view. Computed by detecting a checkerboard lying flat in
   the measurement plane and asking OpenCV to solve for the `H` that maps its
   detected corners to a regular grid. Applied with `cv2.warpPerspective`.
3. **Real-world scaling** — once the perspective is square, the known physical
   size of one checkerboard square (you enter it in mm or inches) gives a
   `pixels_per_real_unit` factor. Tracker output is divided by this to report
   position in real units instead of pixels.

The tracker itself is OpenCV's **CSRT** (Channel and Spatial Reliability
Tracker), which gives sub-pixel position estimates and tolerates partial
occlusion and mild appearance change.

Output is a tab-delimited text file with `time_s`, `x`, `y` per object — ready
to drop into MATLAB, Python, or Excel.

---

## Before you start

- Print the checkerboard at `modular_v2/checkerboard_9x6.pdf` and tape it to a
  rigid backing (foam board, clipboard). Measure one square's edge length with
  a ruler — you'll type this in later.
- Have either (a) a USB webcam connected, **or** (b) videos already recorded on
  your phone. The app supports both paths and they share most steps.
- Launch the app. The main window has a **Video Source** selector at the top.
  Pick the workflow that matches what you have.

---

## Workflow A — Starting fresh with a live camera

Use this when you have a USB webcam plugged in and you'll record everything in
real time.

### 1. Source selection
Select **"I'm recording with a webcam"** at the top of the main window. This
enables the recording-related buttons.

### 2. Select Camera
**Purpose:** tell the app which camera device to use and at what resolution and
frame rate. This is the resolution your calibration will be tied to — change it
later and your calibration is invalid.

- Click **Select Camera**.
- Pick a camera from the list. A live preview appears.
- Choose the highest resolution your camera supports cleanly (the dropdown
  filters to confirmed-working options).
- Click **Confirm Selection**.

### 3. New Camera Calibration *(lens calibration)*
**Purpose:** record a video of the checkerboard from many angles so the app can
solve for `K` and the distortion coefficients. This step does **not** care
where the camera is — it only needs to see the board in many different poses.

- Click **New Camera Calibration**. An instructions popup appears.
- Hold the checkerboard in front of the camera. Move it (or the camera) so the
  board appears in many positions: corners, center, edges, tilted left, tilted
  right, near, far. The live overlay turns green when the board is detected.
  Aim for at least one good detection in each region of the frame.
- Click **Stop Recording** when coverage looks good (typically 30–60 seconds).
- The app then automatically processes the video, picks frames where the
  detection is strongest, and runs the calibration solver. You'll see a
  reported reprojection error in pixels — under ~1.0 px is good, under 0.5 px
  is excellent.

### 4. Perspective correction *(extrinsic + scaling)*
**Purpose:** lock in the camera's viewing angle of the measurement plane and
establish real-world units. Triggered automatically after lens calibration.

- A dialog asks you to position the checkerboard **flat in the plane you'll be
  measuring in** (e.g. lying on the table where the experiment runs). Now the
  camera position *does* matter — don't move the camera between this step and
  recording your experiment.
- Once the board is detected, confirm the detection.
- **Enter the physical size of one square** (the value you measured earlier)
  and pick units. This is what converts pixels to mm/inches downstream.
- Optionally rotate the output coordinate frame so "up" in your data matches
  "up" in your experiment.
- Confirm. The app stores the homography `H` and the pixel-per-unit scale.

### 5. Export Calibration *(optional but recommended)*
**Purpose:** save `K`, distortion coefficients, `H`, and the scale to a JSON
file so you can reuse them next session without re-running steps 3 and 4.

- Click **Export Calibration to File** and choose a save location. The JSON is
  human-readable and OpenCV-compatible.

### 6. Preview Camera Calibration Results
**Purpose:** sanity-check that the corrections look right before committing to
a measurement.

- Click **Preview Camera Calibration Results**. You'll see raw and corrected
  feeds side-by-side. Straight things in the world should look straight.
  Checkerboard squares in the corrected view should look square.

### 7. Record New Tracking Video
**Purpose:** capture the actual experiment. Calibration corrections are applied
to the saved video so the tracking step works in corrected pixel space.

- Remove the checkerboard. Set up your experiment.
- Click **Record New Tracking Video**, then **Start Recording**, run the
  experiment, **Stop Recording**, and save.

### 8. Track Motions from Video
**Purpose:** turn the recorded video into a position time-series.

- Click **Track Motions from Video** and open the video you just recorded.
- Use the scrubber to find the frame where the object is clearly visible.
- Click **Select Object** and drag a tight box around the feature you want to
  track. Repeat for additional objects if needed (each gets its own color).
- Drag the coordinate origin marker to where you want `(0, 0)` to be.
- Click **Start Tracking**. The CSRT tracker advances through the video.
- Click **Save Data** when done. You get a tab-delimited file with time and
  per-object `x, y` in your chosen real-world units.

---

## Workflow B — Starting with pre-recorded videos (e.g. from a phone)

Use this when you've already recorded the calibration and experiment videos on
a phone or another camera and just want to analyze them. The math is identical
to Workflow A; the difference is the source of each video.

You'll need **two** pre-recorded videos:
- A **calibration video** of the checkerboard moved through many poses (same
  content as step 3 above), and
- A **measurement video** of the experiment from the same camera position you
  intend to analyze in.

You may also want a third clip — or just a still frame — showing the
checkerboard lying flat in the measurement plane, used for perspective
correction. Often this is just the first few seconds of your measurement video.

### 1. Source selection
Select **"I'm using pre-recorded videos (e.g. from my phone)"**. The
camera-only buttons gray out; the file-based buttons stay enabled.

### 2. Calibrate from pre-recorded video *(lens calibration)*
**Purpose:** same as Workflow A step 3, but the input is a video file you
already have instead of a live recording.

- Click **Calibrate from pre-recorded video**.
- Select your calibration video file (`.mp4`, `.mov`, `.avi`).
- The app samples ~40 frames spread evenly through the video, detects the
  checkerboard in each, and runs the same solver as Workflow A. Reprojection
  error is reported the same way.

*Already have a calibration JSON from a previous session with the same camera?*
Click **Import Existing Calibration** instead and skip to step 4.

### 3. Perspective correction *(extrinsic + scaling)*
**Purpose:** identical to Workflow A step 4, but you'll point at an image or
video file instead of using a live feed.

- Triggered automatically after the calibration above completes.
- When prompted for the perspective image source, choose **Load Image/Video
  File**.
- If you load a video, scrub to the frame where the checkerboard is flat in
  the measurement plane and clearly visible, then click **Use This Frame**.
- Confirm the detected board, enter the physical square size and units, set
  the output orientation. Same as Workflow A.

### 4. Export Calibration *(optional)*
Same as Workflow A step 5 — save the JSON so you don't have to re-do steps 2
and 3 next time you analyze a video shot with the same camera at the same
focus and zoom.

### 5. Preview Camera Calibration Results
**Purpose:** verify the calibration on your actual footage.

- Click **Preview Camera Calibration Results** and choose **Import Video**.
- Load the measurement video (or any clip from the same camera). Confirm that
  lens distortion and perspective look corrected.

### 6. Track Motions from Video
**Purpose:** identical to Workflow A step 8.

- Click **Track Motions from Video** and open your measurement video.
- The app applies the lens correction and homography to each frame as it
  reads it (your file on disk is left unchanged).
- Select object(s), set the origin, **Start Tracking**, **Save Data**.

---

## A few things that trip students up

- **Move the camera between calibration steps and you start over.** Lens
  calibration (step 3) is camera-pose-independent, but perspective correction
  (step 4) and everything after assume the camera doesn't move. If you bump
  the tripod, redo from step 4.
- **Resolution must match.** A calibration done at 1920×1080 cannot be applied
  to a 1280×720 video. The app warns you on mismatch.
- **Phone autofocus and digital zoom invalidate calibration.** Lock focus and
  zoom on the phone before recording the calibration video, and don't change
  them before recording the measurement video.
- **Measure your checkerboard square accurately.** Your final position values
  inherit the error of this single ruler measurement.
- **Tighter ROI = better tracking.** When you draw the bounding box, hug the
  feature you actually want to follow. Loose boxes drift.
