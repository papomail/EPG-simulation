#!/usr/bin/env python3
import subprocess
import sys

try:
    result = subprocess.run([sys.executable, "quick_test.py"], 
                          capture_output=True, text=True, timeout=30)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print(f"Return code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("Test timed out after 30 seconds")
except Exception as e:
    print(f"Error running test: {e}")