#!/usr/bin/env python3
"""
Quick test to check if the TSE simulation works after fixes.
"""

import sys
import os
import numpy as np

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tse_quick():
    """Quick test of TSE simulation."""
    try:
        from epg_python.sequences import simulate_tse
        
        print("Testing TSE simulation...")
        omega_store, echoes, sequence = simulate_tse(120, 5, 10, True, (1000, 100))
        
        print(f"✓ TSE simulation successful")
        print(f"  - Generated {len(omega_store)} omega states")
        print(f"  - Found {len(echoes)} echoes")
        print(f"  - Sequence name: {sequence.name}")
        print(f"  - T1: {sequence.T1}, T2: {sequence.T2}")
        print(f"  - Gradient array dtype: {sequence.grad.dtype}")
        print(f"  - Gradient values: {sequence.grad}")
        
        return True
    except Exception as e:
        print(f"✗ TSE simulation failed: {e}")
        return False

def test_visualization_quick():
    """Quick test of visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        from epg_python.sequences import simulate_tse
        from epg_python.visualization import display_epg, plot_echoes
        
        print("Testing visualization...")
        omega_store, echoes, sequence = simulate_tse(120, 3, 10, True, (1000, 100))
        
        # Test EPG display
        fig1 = display_epg(omega_store, sequence, annotate=False)
        print("✓ EPG display successful")
        
        # Test echo plot
        fig2 = plot_echoes(echoes, "Test Echoes")
        print("✓ Echo plot successful")
        
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

if __name__ == "__main__":
    print("Quick EPG Test")
    print("=" * 30)
    
    success = True
    success &= test_tse_quick()
    success &= test_visualization_quick()
    
    print("=" * 30)
    if success:
        print("✓ All quick tests passed!")
    else:
        print("✗ Some tests failed.")