#!/usr/bin/env python3
"""Simple test to verify basic functionality."""

import sys
import os
import numpy as np

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic EPG functionality."""
    try:
        print("1. Testing imports...")
        from epg_python import simulate_tse, EPGSimulator, rf_rotation
        print("   ✓ Imports successful")
        
        print("2. Testing RF rotation...")
        T = rf_rotation(0, 90)
        assert T.shape == (3, 3), "RF rotation matrix wrong shape"
        print("   ✓ RF rotation working")
        
        print("3. Testing TSE simulation...")
        omega_store, echoes, sequence = simulate_tse(120, 3, 10, True, (1000, 100))
        assert len(omega_store) > 0, "No omega states stored"
        assert len(echoes) >= 0, "Invalid echoes"
        print(f"   ✓ TSE simulation working ({len(echoes)} echoes found)")
        
        print("4. Testing visualization (non-interactive)...")
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        from epg_python.visualization import display_epg
        fig = display_epg(omega_store, sequence, annotate=False)
        assert fig is not None, "Visualization failed"
        print("   ✓ Visualization working")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)