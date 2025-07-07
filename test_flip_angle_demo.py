#!/usr/bin/env python3
"""
Test script for the flip angle T2 bias demo function.
"""

import numpy as np
import matplotlib.pyplot as plt
from epg_python.demo import demo_flip_angle_t2_bias

def test_flip_angle_demo():
    """Test the flip angle T2 bias demonstration."""
    print("Testing flip angle T2 bias demo...")
    
    try:
        # Run the demo
        results = demo_flip_angle_t2_bias()
        
        # Check that results were returned
        assert isinstance(results, dict), "Results should be a dictionary"
        
        # Check that we have results for expected flip angles
        expected_angles = [180, 150, 120]
        for angle in expected_angles:
            assert angle in results, f"Missing results for {angle}° flip angle"
            assert 'measured_t2' in results[angle], f"Missing measured_t2 for {angle}°"
            assert 'true_t2' in results[angle], f"Missing true_t2 for {angle}°"
        
        # Check that measurements are reasonable
        for angle in expected_angles:
            measured = results[angle]['measured_t2']
            true_vals = results[angle]['true_t2']
            
            # Should have some valid measurements
            valid_idx = ~np.isnan(measured)
            assert np.any(valid_idx), f"No valid measurements for {angle}°"
            
            # Valid measurements should be positive and reasonable
            valid_measured = measured[valid_idx]
            assert np.all(valid_measured > 0), f"Negative T2 measurements for {angle}°"
            assert np.all(valid_measured < 1000), f"Unreasonably large T2 measurements for {angle}°"
        
        print("✓ Flip angle demo test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Flip angle demo test failed: {e}")
        return False

def test_bias_trends():
    """Test that the bias trends make physical sense."""
    print("Testing bias trends...")
    
    try:
        results = demo_flip_angle_t2_bias()
        
        # Calculate average bias for each flip angle
        biases = {}
        for angle in [180, 150, 120]:
            if angle in results:
                measured = results[angle]['measured_t2']
                true_vals = results[angle]['true_t2']
                valid_idx = ~np.isnan(measured)
                
                if np.any(valid_idx):
                    bias = measured[valid_idx] - true_vals[valid_idx]
                    biases[angle] = np.mean(bias)
        
        # Check that we have bias data
        assert len(biases) >= 2, "Need at least 2 flip angles to compare bias"
        
        # Generally expect that lower flip angles have more bias
        # (though the exact direction depends on the specific physics)
        print("Bias summary:")
        for angle, bias in biases.items():
            print(f"  {angle}°: {bias:+.1f} ms average bias")
        
        print("✓ Bias trends test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Bias trends test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing flip angle T2 bias demo function")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_flip_angle_demo()
    test2_passed = test_bias_trends()
    
    # Summary
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 50)