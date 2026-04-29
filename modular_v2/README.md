# Added-Mass-Lab GUI - Modular Version

This directory contains the modularized version of the Added-Mass-LaTo verify the modular implementation works correctly:

1. Run the launcher: `python launch_modular_gui.py`
2. Test each workflow:
   - Camera selection
   - Calibration recording and processing
   - Perspective correction
   - Calibration preview
   - Measurement video recording
   - Object tracking with video scrubbing

All functionality should work identically to the original monolithic version with enhanced tracking capabilities.cation. The monolithic `clean_gui.py` script has been split into focused, maintainable modules while preserving all functionality.

## Module Structure

### Core Modules

- **`config.py`** - Configuration constants and settings
- **`data_models.py`** - CalibrationData class for storing calibration information
- **`camera_manager.py`** - CameraManager class for camera detection and configuration
- **`main_gui.py`** - MainGUI class - central application coordinator

### GUI Modules

- **`camera_selection.py`** - CameraSelectionWindow for camera selection with live preview
- **`calibration_recorder.py`** - CalibrationRecorder for recording calibration videos
- **`calibration_processor.py`** - CalibrationProcessor for processing calibration videos
- **`perspective_corrector.py`** - PerspectiveCorrector for perspective correction setup
- **`calibration_preview.py`** - CalibrationPreviewWindow for previewing calibration results
- **`measurement_recorder.py`** - MeasurementVideoRecorder for recording measurement videos
- **`tracking.py`** - VideoTracker for object tracking in videos with scrubbing controls

### Entry Points

- **`main.py`** - Main entry point within the package
- **`__init__.py`** - Package initialization and exports
- **`launch_modular_gui.py`** - External launcher script (in parent directory)

## How to Run

### Option 1: Use the launcher script
```bash
python launch_modular_gui.py
```

### Option 2: Run as a module
```bash
python -m modular_v2.main
```

### Option 3: Import and run programmatically
```python
from modular_v2.main_gui import MainGUI
app = MainGUI()
app.run()
```

## Key Features Preserved

All functionality from the original monolithic script has been preserved:

- **Camera Selection**: Radio button interface with live preview
- **Calibration Recording**: Multi-threaded video recording with real-time checkerboard detection
- **Calibration Processing**: Automatic checkerboard size detection and camera model fitting
- **Perspective Correction**: Dynamic square size calculation and homography computation
- **Calibration Preview**: Live preview showing raw vs corrected feeds
- **Measurement Recording**: Video recording with automatic calibration correction processing
- **Object Tracking**: Advanced video tracking with scrubbing controls and time-based data export
- **Multi-threading Architecture**: Separate threads for camera operations, recording, and preview

## Modularization Benefits

1. **Maintainability**: Each class is in its own focused file
2. **Testability**: Individual modules can be tested in isolation
3. **Reusability**: Components can be imported and used independently
4. **Debugging**: Easier to locate and fix issues in specific functionality
5. **Documentation**: Each module has clear purpose and responsibilities
6. **Collaboration**: Multiple developers can work on different modules simultaneously

## Dependencies

- Python 3.7+
- opencv-python (4.12.0 recommended)
- numpy
- pillow (PIL)
- tkinter (usually included with Python)

## File Size Comparison

- **Original**: `clean_gui.py` (2413 lines, single file)
- **Modular**: 10 focused files (200-400 lines each)

## Architecture

The modular design follows a clear separation of concerns:

```
MainGUI (Coordinator)
├── CameraManager (Hardware Interface)
├── CalibrationData (Data Storage)
├── CameraSelectionWindow (Camera Selection)
├── CalibrationRecorder (Video Recording)
├── CalibrationProcessor (Video Processing)
├── PerspectiveCorrector (Perspective Correction)
├── CalibrationPreviewWindow (Live Preview)
└── MeasurementVideoRecorder (Measurement Recording)
```

## Future Enhancements

The modular structure makes it easy to:
- Add new GUI windows or functionality
- Implement different camera backends
- Add new calibration algorithms
- Extend data storage capabilities
- Integrate with external systems

## Testing

To verify the modular implementation works correctly:

1. Run the launcher: `python launch_modular_gui.py`
2. Test each workflow:
   - Camera selection
   - Calibration recording and processing
   - Perspective correction
   - Calibration preview
   - Measurement video recording

All functionality should work identically to the original monolithic version.

## Object Tracking Features

The new tracking module provides comprehensive video analysis capabilities:

### Features
- **Video Loading**: Support for common video formats (AVI, MP4, MOV, MKV)
- **Frame Scrubbing**: Navigate through video with slider control and step buttons
- **Playback Controls**: Play/pause functionality with automatic frame rate timing
- **Object Selection**: Interactive ROI selection using mouse drawing
- **CSRT Tracking**: Advanced tracking using OpenCV's CSRT tracker
- **Calibration Integration**: Automatic application of lens and perspective corrections
- **Time-based Export**: Data export with accurate timestamps based on actual video frame rate

### Usage Workflow
1. Click "Track Motions from Video" in the main GUI
2. Select a video file to analyze
3. Use scrubbing controls to navigate to the object's first appearance
4. Click "Select Object" and draw a bounding box around the target
5. Click "Start Tracking" to automatically track through the entire video
6. Save the tracking data as a tab-delimited text file with columns:
   - `time_s`: Time in seconds from video start
   - `x_px`: X coordinate in pixels (after calibration corrections)
   - `y_px`: Y coordinate in pixels (after calibration corrections)

### Data Quality
- **Frame Rate Accuracy**: Uses actual video FPS for precise timing
- **Calibration Corrections**: Automatically applies lens distortion and perspective corrections
- **Sub-pixel Accuracy**: CSRT tracker provides sub-pixel position estimates
- **Robust Tracking**: Handles partial occlusions and appearance changes

## Button State Management

The GUI intelligently manages button availability based on functionality requirements:

### Always Available (No Camera Required)
- **Select Camera**: Choose and configure camera hardware
- **Load Existing Calibration**: Process pre-recorded calibration videos
- **Track Motions from Video**: Analyze existing video files
- **Preview Camera Calibration Results**: Preview calibration with live camera OR imported video

### Camera-Dependent Functions
- **New Camera Calibration**: Requires live camera for recording
- **Record New Tracking Video**: Requires live camera for recording

This design allows you to perform video analysis and load existing calibrations without needing camera hardware connected, while ensuring camera-dependent functions are only available when appropriate.

## Smartphone/External Camera Support

The application now supports using smartphones or external cameras for calibration and analysis:

### Perspective Correction
When performing perspective correction, you can choose between:
- **Live Camera**: Use a connected USB webcam for real-time positioning
- **Load Image/Video File**: Import a pre-captured image or video file
  - **Image files**: JPEG, PNG, BMP, TIFF formats supported
  - **Video files**: MP4, AVI, MOV, MKV, WMV, FLV, WebM formats supported
  - **Frame Selection**: For video files, scrub through and select the perfect frame with the checkerboard

### Calibration Preview  
When previewing calibration results, you can choose between:
- **Live Camera**: View real-time corrected feed from USB webcam
- **Import Video**: Load a pre-recorded video file (MP4, AVI, MOV, MKV, etc.) to see corrections applied

### Workflow for Smartphone Users
1. **Record calibration video** with your smartphone showing the checkerboard in various positions
2. **Record checkerboard video** with smartphone showing checkerboard flat in measurement plane  
3. **Load Existing Calibration** using the recorded video file
4. **Complete perspective correction** using the video frame selection tool when prompted
5. **Record tracking video** with smartphone and use "Track Motions from Video"

**Note**: Loading existing calibrations now includes the full perspective correction workflow, ensuring all users get complete calibration setup regardless of whether they use live cameras or pre-recorded videos.

This allows full functionality without requiring a dedicated USB webcam.
