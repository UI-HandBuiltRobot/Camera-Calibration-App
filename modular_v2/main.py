"""
Main entry point for Added-Mass-Lab GUI
Run this file to start the application
"""

import sys
import os

if __name__ == "__main__":
    # When run directly, add the parent directory so the package is importable
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from modular_v2.main_gui import MainGUI
else:
    from .main_gui import MainGUI


def main():
    """Main entry point"""
    app = MainGUI()
    app.run()


if __name__ == "__main__":
    main()
