"""
EPG Predefined Sequences Module

Implementations of common MRI pulse sequences for EPG simulation.
"""

import numpy as np
from typing import Union, Tuple, List
from .core import EPGSequence, EPGSimulator
from .utils import get_relaxation_params


class EPGSequenceTSE(EPGSequence):
    """
    Turbo Spin Echo (TSE) sequence for EPG simulation.
    
    Implements TSE sequence with initial 90° pulse followed by 180° refocusing pulses.
    """
    
    def __init__(self, alpha: float, N: int, esp: float, 
                 use_y90: bool = True, relaxation: Union[str, Tuple[float, float]] = 'none'):
        """
        Initialize TSE sequence.
        
        Parameters
        ----------
        alpha : float
            Flip angle of refocusing pulses in degrees
        N : int
            Number of pulses after initial 90° pulse
        esp : float
            Echo spacing in ms
        use_y90 : bool, default True
            Whether to use (90°, 90°) pulse at beginning
        relaxation : str or tuple, default 'none'
            Relaxation parameters: 'none', 'gm', 'wm', 'csf', 'default', or (T1, T2)
        """
        super().__init__("Turbo Spin Echo")
        
        # Set relaxation parameters
        if isinstance(relaxation, str):
            self.T1, self.T2 = get_relaxation_params(relaxation)
        else:
            self.T1, self.T2 = relaxation
        
        # Time step (dt = 0.5*esp -> dk = 1)
        dt = esp / 2
        
        # Set up RF scheme
        if use_y90:
            self.add_rf(90, 90)  # Initial 90° pulse with 90° phase
            for _ in range(N - 1):
                self.add_rf(0, alpha)  # Refocusing pulses
        else:
            for _ in range(N):
                self.add_rf(0, alpha)  # All pulses same flip angle
        
        # Set up gradients (unit gradients)
        for _ in range(2 * N - 1):
            self.add_gradient(1)
        
        # Set up timing and events
        # Initial: RF -> Grad -> Relax
        self.add_event('rf', 0)
        self.add_event('grad', dt)
        self.add_event('relax', dt)
        
        # Subsequent echoes: RF -> Grad -> Relax -> Grad -> Relax
        for n in range(1, N):
            self.add_event('rf', (2*n - 1) * dt)
            self.add_event('grad', 2*n * dt)
            self.add_event('relax', 2*n * dt)
            self.add_event('grad', (2*n + 1) * dt)
            self.add_event('relax', (2*n + 1) * dt)


class EPGSequenceGRE(EPGSequence):
    """
    Gradient Recalled Echo (GRE) sequence for EPG simulation.
    """
    
    def __init__(self, alpha: float, N: int, TR: float,
                 relaxation: Union[str, Tuple[float, float]] = 'none'):
        """
        Initialize GRE sequence.
        
        Parameters
        ----------
        alpha : float
            Flip angle in degrees
        N : int
            Number of RF pulses
        TR : float
            Repetition time in ms
        relaxation : str or tuple, default 'none'
            Relaxation parameters: 'none', 'gm', 'wm', 'csf', 'default', or (T1, T2)
        """
        super().__init__("Gradient Recalled Echo")
        
        # Set relaxation parameters
        if isinstance(relaxation, str):
            self.T1, self.T2 = get_relaxation_params(relaxation)
        else:
            self.T1, self.T2 = relaxation
        
        # Set up RF pulses (all same flip angle, 0° phase)
        for _ in range(N):
            self.add_rf(0, alpha)
        
        # Set up gradients
        for _ in range(N):
            self.add_gradient(1)  # Positive gradient
            if _ < N - 1:  # No rewinder after last pulse
                self.add_gradient(-1)  # Rewinder gradient
        
        # Set up timing and events
        for n in range(N):
            time_rf = n * TR
            time_grad = time_rf + TR/4
            time_relax = time_rf + TR/2
            
            self.add_event('rf', time_rf)
            self.add_event('grad', time_grad)
            self.add_event('relax', time_relax)
            
            if n < N - 1:  # Add rewinder
                time_rewind = time_rf + 3*TR/4
                self.add_event('grad', time_rewind)


class EPGSequenceVFA(EPGSequence):
    """
    Variable Flip Angle (VFA) sequence for EPG simulation.
    """
    
    def __init__(self, flip_angles: List[float], TR: float,
                 relaxation: Union[str, Tuple[float, float]] = 'none'):
        """
        Initialize VFA sequence.
        
        Parameters
        ----------
        flip_angles : list of float
            List of flip angles in degrees
        TR : float
            Repetition time in ms
        relaxation : str or tuple, default 'none'
            Relaxation parameters
        """
        super().__init__("Variable Flip Angle")
        
        # Set relaxation parameters
        if isinstance(relaxation, str):
            self.T1, self.T2 = get_relaxation_params(relaxation)
        else:
            self.T1, self.T2 = relaxation
        
        N = len(flip_angles)
        
        # Set up RF pulses with variable flip angles
        for alpha in flip_angles:
            self.add_rf(0, alpha)
        
        # Set up gradients
        for _ in range(N):
            self.add_gradient(1)
        
        # Set up timing and events
        for n in range(N):
            time_rf = n * TR
            time_grad = time_rf + TR/4
            time_relax = time_rf + TR/2
            
            self.add_event('rf', time_rf)
            self.add_event('grad', time_grad)
            self.add_event('relax', time_relax)


def simulate_tse(alpha: float, N: int, esp: float, use_y90: bool = True,
                 relaxation: Union[str, Tuple[float, float]] = 'none') -> Tuple[List[np.ndarray], np.ndarray, EPGSequenceTSE]:
    """
    Simulate TSE sequence (convenience function for MATLAB compatibility).
    
    Parameters
    ----------
    alpha : float
        Flip angle of refocusing pulses in degrees
    N : int
        Number of pulses after initial 90° pulse
    esp : float
        Echo spacing in ms
    use_y90 : bool, default True
        Whether to use (90°, 90°) pulse at beginning
    relaxation : str or tuple, default 'none'
        Relaxation parameters
        
    Returns
    -------
    omega_store : list of ndarray
        Configuration states at each time point
    echoes : ndarray
        Echo information [time, intensity]
    sequence : EPGSequenceTSE
        Sequence object used for simulation
    """
    sequence = EPGSequenceTSE(alpha, N, esp, use_y90, relaxation)
    simulator = EPGSimulator()
    omega_store, echoes = simulator.simulate(sequence)
    return omega_store, echoes, sequence


def simulate_gre(alpha: float, N: int, TR: float,
                 relaxation: Union[str, Tuple[float, float]] = 'none') -> Tuple[List[np.ndarray], np.ndarray, EPGSequenceGRE]:
    """
    Simulate GRE sequence (convenience function).
    
    Parameters
    ----------
    alpha : float
        Flip angle in degrees
    N : int
        Number of RF pulses
    TR : float
        Repetition time in ms
    relaxation : str or tuple, default 'none'
        Relaxation parameters
        
    Returns
    -------
    omega_store : list of ndarray
        Configuration states at each time point
    echoes : ndarray
        Echo information [time, intensity]
    sequence : EPGSequenceGRE
        Sequence object used for simulation
    """
    sequence = EPGSequenceGRE(alpha, N, TR, relaxation)
    simulator = EPGSimulator()
    omega_store, echoes = simulator.simulate(sequence)
    return omega_store, echoes, sequence


def simulate_vfa(flip_angles: List[float], TR: float,
                 relaxation: Union[str, Tuple[float, float]] = 'none') -> Tuple[List[np.ndarray], np.ndarray, EPGSequenceVFA]:
    """
    Simulate VFA sequence (convenience function).
    
    Parameters
    ----------
    flip_angles : list of float
        List of flip angles in degrees
    TR : float
        Repetition time in ms
    relaxation : str or tuple, default 'none'
        Relaxation parameters
        
    Returns
    -------
    omega_store : list of ndarray
        Configuration states at each time point
    echoes : ndarray
        Echo information [time, intensity]
    sequence : EPGSequenceVFA
        Sequence object used for simulation
    """
    sequence = EPGSequenceVFA(flip_angles, TR, relaxation)
    simulator = EPGSimulator()
    omega_store, echoes = simulator.simulate(sequence)
    return omega_store, echoes, sequence