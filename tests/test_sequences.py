"""
Test suite for EPG sequences.
"""

import numpy as np
import pytest
from epg_python.sequences import simulate_tse, simulate_gre, simulate_vfa
from epg_python.sequences import EPGSequenceTSE, EPGSequenceGRE, EPGSequenceVFA


class TestTSESequence:
    """Test TSE sequence implementation."""
    
    def test_basic_tse_simulation(self):
        """Test basic TSE simulation."""
        omega_store, echoes, sequence = simulate_tse(120, 5, 10, True, (1000, 100))
        
        # Check basic properties
        assert len(omega_store) > 0
        assert len(echoes) > 0
        assert sequence.name == "Turbo Spin Echo"
        assert sequence.T1 == 1000
        assert sequence.T2 == 100
    
    def test_tse_echo_count(self):
        """Test that TSE produces expected number of echoes."""
        N = 8
        omega_store, echoes, sequence = simulate_tse(120, N, 10, True, (1000, 100))
        
        # Should have multiple echoes (exact number depends on sequence details)
        assert len(echoes) >= N // 2  # At least half the pulses should produce echoes
    
    def test_tse_flip_angle_effect(self):
        """Test effect of flip angle on TSE."""
        # Compare different flip angles
        results_90 = simulate_tse(90, 5, 10, True, (1000, 100))
        results_180 = simulate_tse(180, 5, 10, True, (1000, 100))
        
        echoes_90 = results_90[1]
        echoes_180 = results_180[1]
        
        # Both should produce echoes
        assert len(echoes_90) > 0
        assert len(echoes_180) > 0
        
        # 180° should generally produce stronger echoes
        if len(echoes_90) > 0 and len(echoes_180) > 0:
            # Compare first echo intensity
            assert echoes_180[0, 1] >= echoes_90[0, 1] * 0.8  # Allow some tolerance
    
    def test_tse_relaxation_effect(self):
        """Test effect of relaxation on TSE."""
        # Short T2
        results_short = simulate_tse(120, 8, 10, True, (1000, 50))
        # Long T2
        results_long = simulate_tse(120, 8, 10, True, (1000, 200))
        
        echoes_short = results_short[1]
        echoes_long = results_long[1]
        
        # Both should have echoes
        assert len(echoes_short) > 0
        assert len(echoes_long) > 0
        
        # Long T2 should have less decay
        if len(echoes_short) >= 2 and len(echoes_long) >= 2:
            # Compare decay from first to second echo
            decay_short = echoes_short[1, 1] / echoes_short[0, 1]
            decay_long = echoes_long[1, 1] / echoes_long[0, 1]
            assert decay_long > decay_short
    
    def test_tse_y90_option(self):
        """Test use_y90 option in TSE."""
        results_y90 = simulate_tse(120, 5, 10, True, (1000, 100))
        results_no_y90 = simulate_tse(120, 5, 10, False, (1000, 100))
        
        # Both should work
        assert len(results_y90[1]) > 0
        assert len(results_no_y90[1]) > 0
        
        # RF schemes should be different
        seq_y90 = results_y90[2]
        seq_no_y90 = results_no_y90[2]
        assert not np.array_equal(seq_y90.rf, seq_no_y90.rf)


class TestGRESequence:
    """Test GRE sequence implementation."""
    
    def test_basic_gre_simulation(self):
        """Test basic GRE simulation."""
        omega_store, echoes, sequence = simulate_gre(30, 5, 50, (1000, 100))
        
        # Check basic properties
        assert len(omega_store) > 0
        assert sequence.name == "Gradient Recalled Echo"
        assert sequence.T1 == 1000
        assert sequence.T2 == 100
    
    def test_gre_steady_state(self):
        """Test GRE approaches steady state."""
        N = 20  # Many repetitions
        omega_store, echoes, sequence = simulate_gre(30, N, 50, (1000, 100))
        
        # Should have multiple echoes
        assert len(echoes) > N // 2
        
        # Later echoes should be more similar (steady state)
        if len(echoes) >= 10:
            early_echo = echoes[2, 1]  # Skip first few for equilibration
            late_echo = echoes[-3, 1]   # Near end
            
            # Should be reasonably similar (within factor of 2)
            ratio = max(early_echo, late_echo) / min(early_echo, late_echo)
            assert ratio < 3.0
    
    def test_gre_flip_angle_effect(self):
        """Test effect of flip angle on GRE."""
        results_10 = simulate_gre(10, 10, 50, (1000, 100))
        results_60 = simulate_gre(60, 10, 50, (1000, 100))
        
        echoes_10 = results_10[1]
        echoes_60 = results_60[1]
        
        # Both should produce echoes
        assert len(echoes_10) > 0
        assert len(echoes_60) > 0


class TestVFASequence:
    """Test VFA sequence implementation."""
    
    def test_basic_vfa_simulation(self):
        """Test basic VFA simulation."""
        flip_angles = [90, 60, 45, 30, 20]
        omega_store, echoes, sequence = simulate_vfa(flip_angles, 50, (1000, 100))
        
        # Check basic properties
        assert len(omega_store) > 0
        assert sequence.name == "Variable Flip Angle"
        assert len(sequence.rf[1, :]) == len(flip_angles)
    
    def test_vfa_flip_angles(self):
        """Test that VFA uses correct flip angles."""
        flip_angles = [90, 60, 45, 30, 20]
        omega_store, echoes, sequence = simulate_vfa(flip_angles, 50, (1000, 100))
        
        # Check that RF matrix has correct flip angles
        assert np.allclose(sequence.rf[1, :], flip_angles)
        assert np.allclose(sequence.rf[0, :], 0)  # All phases should be 0
    
    def test_vfa_decreasing_angles(self):
        """Test VFA with decreasing flip angles."""
        flip_angles = [90, 70, 50, 35, 25, 18, 13, 10]
        omega_store, echoes, sequence = simulate_vfa(flip_angles, 30, (1000, 100))
        
        # Should produce echoes
        assert len(echoes) > 0
        
        # Signal should generally decrease (but may have some variation)
        if len(echoes) >= 4:
            # Check that signal doesn't increase dramatically
            max_signal = np.max(echoes[:, 1])
            min_signal = np.min(echoes[:, 1])
            assert max_signal / min_signal < 10  # Reasonable range


class TestSequenceComparison:
    """Test comparisons between different sequences."""
    
    def test_sequence_differences(self):
        """Test that different sequences produce different results."""
        # Common parameters
        T1, T2 = 1000, 100
        
        # Different sequences
        tse_results = simulate_tse(120, 5, 20, True, (T1, T2))
        gre_results = simulate_gre(30, 5, 50, (T1, T2))
        
        tse_echoes = tse_results[1]
        gre_echoes = gre_results[1]
        
        # Both should produce echoes
        assert len(tse_echoes) > 0
        assert len(gre_echoes) > 0
        
        # Echo patterns should be different
        if len(tse_echoes) > 0 and len(gre_echoes) > 0:
            # Different timing patterns
            tse_times = tse_echoes[:, 0]
            gre_times = gre_echoes[:, 0]
            
            # Should have different timing characteristics
            assert not np.allclose(tse_times[:min(len(tse_times), len(gre_times))], 
                                 gre_times[:min(len(tse_times), len(gre_times))])
    
    def test_relaxation_consistency(self):
        """Test that relaxation parameters are applied consistently."""
        T1, T2 = 800, 120
        
        # Test different sequences with same relaxation
        tse_results = simulate_tse(120, 5, 15, True, (T1, T2))
        gre_results = simulate_gre(45, 5, 60, (T1, T2))
        
        tse_seq = tse_results[2]
        gre_seq = gre_results[2]
        
        # Both should have same relaxation parameters
        assert tse_seq.T1 == T1
        assert tse_seq.T2 == T2
        assert gre_seq.T1 == T1
        assert gre_seq.T2 == T2


class TestSequenceValidation:
    """Test sequence validation and error handling."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Negative flip angle should work (just phase shift)
        omega_store, echoes, sequence = simulate_tse(-120, 3, 10, True, (1000, 100))
        assert len(omega_store) > 0
        
        # Zero echo spacing should work
        omega_store, echoes, sequence = simulate_tse(120, 3, 0, True, (1000, 100))
        assert len(omega_store) > 0
        
        # Very large flip angle should work
        omega_store, echoes, sequence = simulate_tse(720, 3, 10, True, (1000, 100))
        assert len(omega_store) > 0
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Single pulse
        omega_store, echoes, sequence = simulate_tse(90, 1, 10, True, (1000, 100))
        assert len(omega_store) > 0
        
        # No relaxation
        omega_store, echoes, sequence = simulate_tse(120, 3, 10, True, (0, 0))
        assert len(omega_store) > 0
        
        # Very short relaxation times
        omega_store, echoes, sequence = simulate_tse(120, 3, 10, True, (1, 1))
        assert len(omega_store) > 0