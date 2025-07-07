"""
EPG Core Simulation Module

Main EPG simulation engine and sequence definition classes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from .operators import rf_rotation, shift_grad, relax


class EPGSequence:
    """
    EPG sequence definition class.
    
    Defines a pulse sequence for EPG simulation with RF pulses, gradients,
    relaxation periods, and timing information.
    
    Attributes
    ----------
    name : str
        Sequence name
    rf : ndarray, shape (2, n_rf)
        RF pulse matrix with rows [phase, flip_angle] in degrees
    grad : ndarray, shape (n_grad,)
        Gradient k-shifts (must be integers)
    events : list of str
        List of event types: 'rf', 'grad', or 'relax'
    time : ndarray
        Timing for each event in ms
    T1 : float
        T1 relaxation time in ms (0 for no relaxation)
    T2 : float  
        T2 relaxation time in ms (0 for no relaxation)
    """
    
    def __init__(self, name: str = "Custom Sequence"):
        self.name = name
        self.rf = np.array([]).reshape(2, 0)
        self.grad = np.array([], dtype=int)
        self.events = []
        self.time = np.array([])
        self.T1 = 0.0
        self.T2 = 0.0
    
    def add_rf(self, phase: float, flip_angle: float):
        """Add RF pulse to sequence."""
        new_rf = np.array([[phase], [flip_angle]])
        if self.rf.size == 0:
            self.rf = new_rf
        else:
            self.rf = np.hstack([self.rf, new_rf])
    
    def add_gradient(self, k_shift: int):
        """Add gradient to sequence."""
        self.grad = np.append(self.grad, int(k_shift))
    
    def add_event(self, event_type: str, time: float):
        """Add event to sequence."""
        if event_type not in ['rf', 'grad', 'relax']:
            raise ValueError("Event type must be 'rf', 'grad', or 'relax'")
        self.events.append(event_type)
        self.time = np.append(self.time, time)
    
    def set_relaxation(self, T1: float, T2: float):
        """Set T1 and T2 relaxation times."""
        self.T1 = T1
        self.T2 = T2


class EPGSimulator:
    """
    Main EPG simulation engine.
    
    Performs EPG simulation of general pulse sequences with RF pulses,
    gradients, and T1/T2 relaxation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset simulator to initial state."""
        self.omega_store = []
        self.echoes = []
    
    def simulate(self, sequence: EPGSequence) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Simulate EPG evolution for given sequence.
        
        Parameters
        ----------
        sequence : EPGSequence
            Pulse sequence to simulate
            
        Returns
        -------
        omega_store : list of ndarray
            List of omega matrices at each time point
        echoes : ndarray, shape (n_echoes, 2)
            Echo information [time, intensity]
        """
        self.reset()
        
        # Initialize magnetization at equilibrium
        omega = np.array([[0], [0], [1]], dtype=complex)
        
        # Get unique times for relaxation calculations
        unique_times = np.unique(sequence.time)
        
        # Event counters
        rf_index = 0
        grad_index = 0
        
        # Process each event
        for n, event in enumerate(sequence.events):
            if event == 'rf':
                if rf_index >= sequence.rf.shape[1]:
                    raise ValueError(f"Not enough RF pulses defined for event {n}")
                phase = sequence.rf[0, rf_index]
                flip_angle = sequence.rf[1, rf_index]
                T_matrix = rf_rotation(phase, flip_angle)
                omega = T_matrix @ omega
                rf_index += 1
                
            elif event == 'grad':
                if grad_index >= len(sequence.grad):
                    raise ValueError(f"Not enough gradients defined for event {n}")
                k_shift = sequence.grad[grad_index]
                omega = shift_grad(k_shift, omega)
                grad_index += 1
                
            elif event == 'relax':
                # Find relaxation duration
                current_time = sequence.time[n]
                time_idx = np.where(unique_times == current_time)[0][0]
                if time_idx > 0:
                    tau = unique_times[time_idx] - unique_times[time_idx - 1]
                    omega = relax(tau, sequence.T1, sequence.T2, omega)
            
            # Store current state
            self.omega_store.append(omega.copy())
        
        # Find echoes
        self.echoes = self._find_echoes(sequence)
        
        return self.omega_store, self.echoes
    
    def _find_echoes(self, sequence: EPGSequence) -> np.ndarray:
        """
        Find echo timings and intensities.
        
        Criteria:
        (a) The F(0) component must be non-zero (>5*eps)
        (b) If multiple echoes at same timing, use the last value
        
        Returns
        -------
        echoes : ndarray, shape (n_echoes, 2)
            Echo information [time, intensity]
        """
        echoes = []
        
        for v, omega in enumerate(self.omega_store):
            # Check if F+(0) component is significant
            if abs(omega[0, 0]) > 5 * np.finfo(float).eps:
                new_echo = [sequence.time[v], abs(omega[0, 0])]
                
                # If same timing exists, replace with latest value
                if echoes and echoes[-1][0] == sequence.time[v]:
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


def epg_custom(sequence: Union[EPGSequence, Dict]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Convenience function for EPG simulation (MATLAB compatibility).
    
    Parameters
    ----------
    sequence : EPGSequence or dict
        Sequence definition. If dict, should contain keys:
        'rf', 'grad', 'events', 'time', 'T1', 'T2'
        
    Returns
    -------
    omega_store : list of ndarray
        List of omega matrices at each time point  
    echoes : ndarray, shape (n_echoes, 2)
        Echo information [time, intensity]
    """
    if isinstance(sequence, dict):
        # Convert dict to EPGSequence for compatibility
        seq = EPGSequence(sequence.get('name', 'Custom'))
        seq.rf = np.array(sequence['rf'])
        seq.grad = np.array(sequence['grad'], dtype=int)
        seq.events = sequence['events']
        seq.time = np.array(sequence['time'])
        seq.T1 = sequence.get('T1', 0)
        seq.T2 = sequence.get('T2', 0)
    else:
        seq = sequence
    
    simulator = EPGSimulator()
    return simulator.simulate(seq)