#!/usr/bin/env python3
"""
Quick demo runner for EPG Python package.

This script demonstrates the basic functionality of the EPG simulation package.
"""

import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from epg_python.demo import run_all_demos
    
    if __name__ == "__main__":
        print("Running EPG Python Demo...")
        print("This will show several matplotlib windows.")
        print("Close each window to proceed to the next demo.")
        print("-" * 50)
        
        run_all_demos()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies:")
    print("pip install numpy matplotlib scipy")
    sys.exit(1)
except Exception as e:
    print(f"Error running demo: {e}")
    sys.exit(1)