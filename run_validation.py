#!/usr/bin/env python3
import subprocess
import sys

try:
    result = subprocess.run([sys.executable, "validate_installation.py"], 
                          capture_output=True, text=True, timeout=60)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print(f"Return code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("Validation timed out after 60 seconds")
except Exception as e:
    print(f"Error running validation: {e}")