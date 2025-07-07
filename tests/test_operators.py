"""
Test suite for EPG operators.
"""

import numpy as np
import pytest
from epg_python.operators import rf_rotation, shift_grad, relax


class TestRFRotation:
    """Test RF rotation operator."""
    
    def test_90_degree_x_pulse(self):
        """Test 90° pulse along x-axis."""
        T = rf_rotation(0, 90)
        
        # Check matrix properties
        assert T.shape == (3, 3)
        assert np.allclose(np.abs(np.linalg.det(T)), 1.0)  # Unitary
        
        # Test specific values for 90° x-pulse
        expected = np.array([
            [0.5, 0.5, -0.5j],
            [0.5, 0.5, 0.5j],
            [0.5j, -0.5j, 0]
        ])
        assert np.allclose(T, expected, atol=1e-10)
    
    def test_180_degree_y_pulse(self):
        """Test 180° pulse along y-axis."""
        T = rf_rotation(90, 180)
        
        # For 180° y-pulse, should flip F+ <-> F- and invert Z
        expected = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        assert np.allclose(T, expected, atol=1e-10)
    
    def test_zero_flip_angle(self):
        """Test identity for zero flip angle."""
        T = rf_rotation(0, 0)
        expected = np.eye(3)
        assert np.allclose(T, expected, atol=1e-10)
    
    def test_phase_dependence(self):
        """Test phase dependence."""
        T1 = rf_rotation(0, 90)
        T2 = rf_rotation(180, 90)
        
        # Should be different matrices
        assert not np.allclose(T1, T2)


class TestShiftGrad:
    """Test gradient shift operator."""
    
    def test_no_shift(self):
        """Test zero gradient shift."""
        omega = np.array([[1], [0], [0]], dtype=complex)
        omega_new = shift_grad(0, omega)
        assert np.allclose(omega, omega_new)
    
    def test_positive_shift(self):
        """Test positive gradient shift."""
        # Start with F+(0) = 1
        omega = np.array([[1, 0], [0, 0], [0, 1]], dtype=complex)
        omega_new = shift_grad(1, omega)
        
        # F+(0) should move to F+(1), F-(0) should stay
        assert omega_new.shape[1] >= 2
        assert np.abs(omega_new[0, 1]) > 0  # F+(1) should be non-zero
    
    def test_negative_shift(self):
        """Test negative gradient shift."""
        omega = np.array([[0, 1], [0, 0], [1, 0]], dtype=complex)
        omega_new = shift_grad(-1, omega)
        
        # Should expand the matrix
        assert omega_new.shape[1] >= omega.shape[1]
    
    def test_conjugate_symmetry(self):
        """Test conjugate symmetry preservation."""
        omega = np.array([[1+1j], [1-1j], [1]], dtype=complex)
        omega_new = shift_grad(1, omega)
        
        # F+(0) and F-(0) should maintain conjugate relationship
        assert np.allclose(omega_new[0, 0], np.conj(omega_new[1, 0]))


class TestRelax:
    """Test relaxation operator."""
    
    def test_no_relaxation(self):
        """Test with T1=T2=0 (no relaxation)."""
        omega = np.array([[1], [0], [0]], dtype=complex)
        omega_new = relax(10, 0, 0, omega)
        assert np.allclose(omega, omega_new)
    
    def test_t2_decay(self):
        """Test T2 decay of transverse components."""
        omega = np.array([[1], [0], [0]], dtype=complex)
        omega_new = relax(100, 0, 100, omega)  # T2=100ms, tau=100ms
        
        # F+ should decay by factor of e^(-1) ≈ 0.368
        expected_decay = np.exp(-1)
        assert np.allclose(np.abs(omega_new[0, 0]), expected_decay, rtol=1e-10)
    
    def test_t1_recovery(self):
        """Test T1 recovery of longitudinal component."""
        omega = np.array([[0], [0], [0]], dtype=complex)  # No initial Mz
        omega_new = relax(1000, 1000, 0, omega)  # T1=1000ms, tau=1000ms
        
        # Should recover toward equilibrium (1 - e^(-1))
        expected_recovery = 1 - np.exp(-1)
        assert np.allclose(np.real(omega_new[2, 0]), expected_recovery, rtol=1e-10)
    
    def test_equilibrium_preservation(self):
        """Test that equilibrium magnetization is preserved."""
        omega = np.array([[0], [0], [1]], dtype=complex)  # At equilibrium
        omega_new = relax(100, 1000, 100, omega)
        
        # Longitudinal component should remain close to 1
        assert np.allclose(np.real(omega_new[2, 0]), 1.0, rtol=1e-2)
    
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shape."""
        omega = np.array([[1, 0], [0, 1]])  # Only 2 rows instead of 3
        
        with pytest.raises(ValueError, match="Size of k-state matrix incorrect"):
            relax(10, 1000, 100, omega)


class TestOperatorCombinations:
    """Test combinations of operators."""
    
    def test_rf_then_grad(self):
        """Test RF pulse followed by gradient."""
        # Start at equilibrium
        omega = np.array([[0], [0], [1]], dtype=complex)
        
        # Apply 90° x-pulse
        T = rf_rotation(0, 90)
        omega = T @ omega
        
        # Apply gradient
        omega = shift_grad(1, omega)
        
        # Should have transverse components in different k-states
        assert omega.shape[1] > 1
        assert np.sum(np.abs(omega[:2, :])) > 0  # Transverse components exist
    
    def test_spin_echo_sequence(self):
        """Test basic spin echo: 90x - grad - 180y - grad."""
        omega = np.array([[0], [0], [1]], dtype=complex)
        
        # 90° x-pulse
        T90 = rf_rotation(0, 90)
        omega = T90 @ omega
        
        # Gradient
        omega = shift_grad(1, omega)
        
        # 180° y-pulse
        T180 = rf_rotation(90, 180)
        omega = T180 @ omega
        
        # Gradient (refocusing)
        omega = shift_grad(1, omega)
        
        # Should have echo at F+(0)
        assert np.abs(omega[0, 0]) > 0.1  # Significant echo
    
    def test_relaxation_during_sequence(self):
        """Test relaxation effects during sequence."""
        omega = np.array([[0], [0], [1]], dtype=complex)
        
        # 90° pulse
        T = rf_rotation(0, 90)
        omega = T @ omega
        
        # Long relaxation period
        omega = relax(200, 1000, 100, omega)  # 2*T2 relaxation
        
        # Transverse components should be significantly reduced
        transverse_mag = np.sqrt(np.abs(omega[0, 0])**2 + np.abs(omega[1, 0])**2)
        assert transverse_mag < 0.2  # Should be much less than initial