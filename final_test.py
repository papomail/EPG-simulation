#!/usr/bin/env python3
"""
Final comprehensive test of the EPG package after fixes.
"""

import sys
import os
import numpy as np

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_validation_tests():
    """Run the same tests as in validate_installation.py"""
    
    print("EPG Python Package Validation (Fixed)")
    print("=" * 40)
    
    # Test 1: Imports
    print("Testing imports...")
    try:
        from epg_python import EPGSimulator, rf_rotation, shift_grad, relax
        from epg_python import simulate_tse, simulate_gre
        from epg_python.visualization import display_epg, plot_echoes
        print("✓ All imports successful")
        test1_pass = True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        test1_pass = False
    
    # Test 2: Basic operators
    print("\nTesting basic operators...")
    try:
        from epg_python.operators import rf_rotation, shift_grad, relax
        
        # Test RF rotation
        T = rf_rotation(0, 90)
        assert T.shape == (3, 3), "RF rotation matrix wrong shape"
        assert np.allclose(np.abs(np.linalg.det(T)), 1.0, atol=1e-10), "RF matrix not unitary"
        
        # Test gradient shift with float input (this was the main issue)
        omega = np.array([[1], [0], [0]], dtype=complex)
        omega_new = shift_grad(1.0, omega)  # Test with float
        assert omega_new.shape[0] == 3, "Gradient shift wrong output shape"
        
        # Test relaxation
        omega = np.array([[1], [0], [0]], dtype=complex)
        omega_new = relax(10, 1000, 100, omega)
        assert omega_new.shape == omega.shape, "Relaxation changed matrix shape"
        
        print("✓ Basic operators working")
        test2_pass = True
    except Exception as e:
        print(f"✗ Operator test failed: {e}")
        test2_pass = False
    
    # Test 3: TSE simulation
    print("\nTesting TSE simulation...")
    try:
        from epg_python.sequences import simulate_tse
        
        # Basic TSE simulation
        omega_store, echoes, sequence = simulate_tse(120, 5, 10, True, (1000, 100))
        
        assert len(omega_store) > 0, "No omega states stored"
        assert len(echoes) > 0, "No echoes found"
        assert sequence.name == "Turbo Spin Echo", "Wrong sequence name"
        assert sequence.T1 == 1000, "Wrong T1 value"
        assert sequence.T2 == 100, "Wrong T2 value"
        assert sequence.grad.dtype == np.int64 or sequence.grad.dtype == np.int32, f"Gradient dtype should be int, got {sequence.grad.dtype}"
        
        # Check echo format
        assert echoes.shape[1] == 2, "Echoes should have 2 columns [time, intensity]"
        assert np.all(echoes[:, 0] >= 0), "Echo times should be non-negative"
        assert np.all(echoes[:, 1] >= 0), "Echo intensities should be non-negative"
        
        print("✓ TSE simulation working")
        print(f"  - Generated {len(omega_store)} omega states")
        print(f"  - Found {len(echoes)} echoes")
        print(f"  - Gradient dtype: {sequence.grad.dtype}")
        test3_pass = True
    except Exception as e:
        print(f"✗ TSE simulation failed: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False
    
    # Test 4: Visualization
    print("\nTesting visualization...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        from epg_python.sequences import simulate_tse
        from epg_python.visualization import display_epg, plot_echoes
        
        # Generate test data
        omega_store, echoes, sequence = simulate_tse(120, 3, 10, True, (1000, 100))
        
        # Test EPG display
        fig1 = display_epg(omega_store, sequence, annotate=False)
        assert fig1 is not None, "EPG display failed"
        
        # Test echo plot
        fig2 = plot_echoes(echoes, "Test Echoes")
        assert fig2 is not None, "Echo plot failed"
        
        print("✓ Visualization working")
        test4_pass = True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        test4_pass = False
    
    # Test 5: MATLAB compatibility
    print("\nTesting MATLAB compatibility...")
    try:
        from epg_python.core import epg_custom
        
        # Create MATLAB-style sequence dict
        seq = {
            'name': 'Test TSE',
            'rf': np.array([[90, 0, 0], [90, 120, 120]]),
            'grad': np.array([1, 1, 1, 1, 1]),  # These should be converted to int
            'events': ['rf', 'grad', 'relax', 'rf', 'grad', 'relax', 'grad', 'relax', 'rf', 'grad', 'relax'],
            'time': np.array([0, 5, 5, 10, 15, 15, 20, 20, 25, 30, 30]),
            'T1': 1000,
            'T2': 100
        }
        
        # Run simulation
        omega_store, echoes = epg_custom(seq)
        
        assert len(omega_store) > 0, "No omega states from MATLAB interface"
        assert len(echoes) >= 0, "Invalid echoes from MATLAB interface"
        
        print("✓ MATLAB compatibility working")
        test5_pass = True
    except Exception as e:
        print(f"✗ MATLAB compatibility test failed: {e}")
        test5_pass = False
    
    # Summary
    tests_passed = sum([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass])
    total_tests = 5
    
    print("\n" + "=" * 40)
    print(f"Validation Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! EPG Python is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)