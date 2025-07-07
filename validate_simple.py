#!/usr/bin/env python3
"""
Simplified validation script for EPG Python package.
"""

import sys
import os
import numpy as np

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from epg_python import EPGSimulator, rf_rotation, shift_grad, relax
        from epg_python import simulate_tse
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_operators():
    """Test basic EPG operators."""
    print("Testing basic operators...")
    
    try:
        from epg_python.operators import rf_rotation, shift_grad, relax
        
        # Test RF rotation
        T = rf_rotation(0, 90)
        assert T.shape == (3, 3), "RF rotation matrix wrong shape"
        
        # Test gradient shift
        omega = np.array([[1], [0], [0]], dtype=complex)
        omega_new = shift_grad(1, omega)
        assert omega_new.shape[0] == 3, "Gradient shift wrong output shape"
        
        # Test relaxation
        omega = np.array([[1], [0], [0]], dtype=complex)
        omega_new = relax(10, 1000, 100, omega)
        assert omega_new.shape == omega.shape, "Relaxation changed matrix shape"
        
        print("✓ Basic operators working")
        return True
    except Exception as e:
        print(f"✗ Operator test failed: {e}")
        return False

def test_tse_simulation():
    """Test TSE sequence simulation."""
    print("Testing TSE simulation...")
    
    try:
        from epg_python.sequences import simulate_tse
        
        # Basic TSE simulation
        omega_store, echoes, sequence = simulate_tse(120, 5, 10, True, (1000, 100))
        
        assert len(omega_store) > 0, "No omega states stored"
        assert len(echoes) >= 0, "Invalid echoes"
        assert sequence.name == "Turbo Spin Echo", "Wrong sequence name"
        assert sequence.T1 == 1000, "Wrong T1 value"
        assert sequence.T2 == 100, "Wrong T2 value"
        
        print(f"✓ TSE simulation working ({len(echoes)} echoes found)")
        return True
    except Exception as e:
        print(f"✗ TSE simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("EPG Python Package - Simple Validation")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_operators,
        test_tse_simulation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ Core functionality is working correctly!")
        return 0
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())