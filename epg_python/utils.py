"""
EPG Utilities Module

Utility functions for EPG simulation including echo detection and analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def find_echoes(sequence, omega_store: List[np.ndarray]) -> np.ndarray:
    """
    Find echo timings and intensities from EPG simulation results.
    
    Finds proper and unique echoes based on criteria:
    (a) The F(0) component must be non-zero (>5*eps)
    (b) If multiple echoes at same timing, use the last value
    
    Parameters
    ----------
    sequence : EPGSequence or dict
        Sequence information containing timing
    omega_store : list of ndarray
        List of omega matrices from EPG simulation
        
    Returns
    -------
    echoes : ndarray, shape (n_echoes, 2)
        Echo information [time, intensity]
    """
    echoes = []
    
    # Get timing information
    if hasattr(sequence, 'time'):
        timing = sequence.time
    else:
        timing = sequence['time']
    
    for v, omega in enumerate(omega_store):
        # Check if F+(0) component is significant
        if abs(omega[0, 0]) > 5 * np.finfo(float).eps:
            new_echo = [timing[v], abs(omega[0, 0])]
            
            # If same timing exists, replace with latest value
            if echoes and echoes[-1][0] == timing[v]:
                echoes[-1] = new_echo
            else:
                echoes.append(new_echo)
    
    if echoes:
        echoes = np.array(echoes)
        # Remove duplicates
        echoes = np.unique(echoes, axis=0)
    else:
        echoes = np.array([]).reshape(0, 2)
        
    return echoes


def find_all_echoes(sequence, omega_store: List[np.ndarray]) -> np.ndarray:
    """
    Find ALL non-zero F(0) states (including those simultaneous with RF).
    
    Parameters
    ----------
    sequence : EPGSequence or dict
        Sequence information containing timing
    omega_store : list of ndarray
        List of omega matrices from EPG simulation
        
    Returns
    -------
    all_echoes : ndarray, shape (n_echoes, 2)
        All F(0) states [time, intensity]
    """
    all_echoes = []
    
    # Get timing information
    if hasattr(sequence, 'time'):
        timing = sequence.time
    else:
        timing = sequence['time']
    
    for v, omega in enumerate(omega_store):
        # Check if F+(0) component is significant
        if abs(omega[0, 0]) > 5 * np.finfo(float).eps:
            all_echoes.append([timing[v], abs(omega[0, 0])])
    
    if all_echoes:
        all_echoes = np.array(all_echoes)
    else:
        all_echoes = np.array([]).reshape(0, 2)
        
    return all_echoes


def get_relaxation_params(tissue_type: str) -> Tuple[float, float]:
    """
    Get standard T1/T2 values for common tissue types.
    
    Parameters
    ----------
    tissue_type : str
        Tissue type: 'gm', 'wm', 'csf', 'default', or 'none'
        
    Returns
    -------
    T1, T2 : float, float
        T1 and T2 values in ms
    """
    tissue_params = {
        'none': (0, 0),
        'gm': (1300, 110),      # gray matter
        'wm': (960, 80),        # white matter  
        'csf': (3600, 1800),    # cerebrospinal fluid
        'default': (1000, 100)  # reference values from EPG paper
    }
    
    if tissue_type not in tissue_params:
        raise ValueError(f"Unknown tissue type: {tissue_type}")
    
    return tissue_params[tissue_type]


def calculate_signal_decay(echoes: np.ndarray, T2_true: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate theoretical T2 decay curve for comparison with EPG results.
    
    Parameters
    ----------
    echoes : ndarray, shape (n_echoes, 2)
        Echo times and intensities [time, intensity]
    T2_true : float
        True T2 value in ms
        
    Returns
    -------
    times : ndarray
        Echo times
    theoretical_signal : ndarray
        Theoretical T2 decay signal
    """
    if len(echoes) == 0:
        return np.array([]), np.array([])
    
    times = echoes[:, 0]
    theoretical_signal = np.exp(-times / T2_true)
    
    return times, theoretical_signal


def extract_echo_train(echoes: np.ndarray, skip_rf_echoes: bool = True) -> np.ndarray:
    """
    Extract echo train from EPG results, optionally skipping RF-coincident echoes.
    
    Parameters
    ----------
    echoes : ndarray, shape (n_echoes, 2)
        All echoes [time, intensity]
    skip_rf_echoes : bool, default True
        If True, extract only even-indexed echoes (skip RF-coincident ones)
        
    Returns
    -------
    echo_train : ndarray, shape (n_train_echoes, 2)
        Filtered echo train [time, intensity]
    """
    if len(echoes) == 0:
        return echoes
    
    if skip_rf_echoes:
        # Take every other echo (skip odd indices which correspond to RF pulses)
        indices = np.arange(0, len(echoes), 2)
        return echoes[indices]
    else:
        return echoes


def analyze_sequence_efficiency(echoes: np.ndarray, echo_spacing: float) -> Dict:
    """
    Analyze sequence efficiency metrics.
    
    Parameters
    ----------
    echoes : ndarray, shape (n_echoes, 2)
        Echo train [time, intensity]
    echo_spacing : float
        Expected echo spacing in ms
        
    Returns
    -------
    metrics : dict
        Dictionary containing efficiency metrics
    """
    if len(echoes) == 0:
        return {'n_echoes': 0, 'signal_efficiency': 0, 'temporal_efficiency': 0}
    
    n_echoes = len(echoes)
    
    # Signal efficiency: ratio of actual to theoretical signal
    times = echoes[:, 0]
    intensities = echoes[:, 1]
    
    # Assume first echo as reference
    if len(intensities) > 1:
        signal_efficiency = np.mean(intensities[1:] / intensities[0])
    else:
        signal_efficiency = 1.0
    
    # Temporal efficiency: actual vs expected timing
    if len(times) > 1:
        actual_spacing = np.mean(np.diff(times))
        temporal_efficiency = echo_spacing / actual_spacing if actual_spacing > 0 else 0
    else:
        temporal_efficiency = 1.0
    
    return {
        'n_echoes': n_echoes,
        'signal_efficiency': signal_efficiency,
        'temporal_efficiency': temporal_efficiency,
        'echo_times': times,
        'echo_intensities': intensities
    }