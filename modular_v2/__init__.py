"""
Added-Mass-Lab GUI Package
Modular implementation of the Added-Mass-Lab experiment GUI

This package provides a clean, modular structure for the GUI application
while maintaining all original functionality.
"""

from .main_gui import MainGUI
from .camera_manager import CameraManager
from .data_models import CalibrationData
from .camera_selection import CameraSelectionWindow
from .calibration_recorder import CalibrationRecorder
from .calibration_processor import CalibrationProcessor
from .perspective_corrector import PerspectiveCorrector
from .calibration_preview import CalibrationPreviewWindow
from .measurement_recorder import MeasurementVideoRecorder
from .tracking_v7 import VideoTracker

__version__ = "1.0.0"
__author__ = "Added-Mass-Lab Team"

__all__ = [
    'MainGUI',
    'CameraManager',
    'CalibrationData',
    'CameraSelectionWindow',
    'CalibrationRecorder',
    'CalibrationProcessor', 
    'PerspectiveCorrector',
    'CalibrationPreviewWindow',
    'MeasurementVideoRecorder',
    'VideoTracker'
]
