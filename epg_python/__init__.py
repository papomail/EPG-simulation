"""
EPG (Extended Phase Graph) Simulation Package

A Python implementation of Extended Phase Graph simulation for MRI pulse sequences.
Based on the MATLAB implementation by Sairam Geethanath and Gehua Tong.

References:
Weigel, Matthias. "Extended phase graphs: dephasing, RF pulses, and echoes‐pure and simple." 
Journal of Magnetic Resonance Imaging 41.2 (2015): 266-295.
"""

from .core import EPGSimulator, EPGSequence, epg_custom
from .operators import rf_rotation, shift_grad, relax
from .sequences import EPGSequenceTSE, EPGSequenceGRE, EPGSequenceVFA, simulate_tse, simulate_gre, simulate_vfa
from .visualization import display_epg, plot_echoes, plot_t2_decay, plot_sequence_comparison
from .utils import find_echoes, get_relaxation_params

__version__ = "1.0.0"
__author__ = "Converted from MATLAB by AI Assistant"

__all__ = [
    'EPGSimulator',
    'EPGSequence',
    'epg_custom',
    'rf_rotation', 
    'shift_grad', 
    'relax',
    'EPGSequenceTSE',
    'EPGSequenceGRE',
    'EPGSequenceVFA',
    'simulate_tse',
    'simulate_gre', 
    'simulate_vfa',
    'display_epg',
    'plot_echoes',
    'plot_t2_decay',
    'plot_sequence_comparison',
    'find_echoes',
    'get_relaxation_params'
]