# EPG Python

A Python implementation of Extended Phase Graph (EPG) simulation for MRI pulse sequences.

## Overview

This package provides a comprehensive toolkit for simulating MRI pulse sequences using the Extended Phase Graph (EPG) formalism. It includes:

- **Core EPG operators**: RF rotation, gradient shifting, and T1/T2 relaxation
- **Sequence simulation**: Custom pulse sequence simulation engine
- **Predefined sequences**: TSE (Turbo Spin Echo), GRE, VFA implementations
- **Visualization tools**: EPG diagrams and echo train analysis
- **Demo examples**: Comprehensive tutorials and examples

## Installation

### From source
```bash
git clone https://github.com/yourusername/epg-python.git
cd epg-python
pip install -e .
```

### Dependencies
- Python ≥ 3.7
- NumPy ≥ 1.19.0
- Matplotlib ≥ 3.3.0
- SciPy ≥ 1.5.0

## Quick Start

### Basic TSE Simulation
```python
import numpy as np
import matplotlib.pyplot as plt
from epg_python import simulate_tse, display_epg, plot_echoes

# Simulate TSE sequence
alpha = 120      # Refocusing flip angle (degrees)
N = 10          # Number of pulses
esp = 10        # Echo spacing (ms)
T1, T2 = 400, 100  # Relaxation times (ms)

omega_store, echoes, sequence = simulate_tse(alpha, N, esp, True, (T1, T2))

# Visualize results
display_epg(omega_store, sequence)
plot_echoes(echoes)
plt.show()
```

### Custom Sequence
```python
from epg_python import EPGSequence, EPGSimulator

# Create custom sequence
seq = EPGSequence("My Custom Sequence")
seq.add_rf(0, 90)      # 90° pulse, 0° phase
seq.add_rf(90, 180)    # 180° pulse, 90° phase
seq.add_gradient(1)    # Unit gradient
seq.add_gradient(-1)   # Rewinder gradient
seq.set_relaxation(1000, 100)  # T1=1000ms, T2=100ms

# Add timing and events
seq.add_event('rf', 0)
seq.add_event('grad', 5)
seq.add_event('relax', 10)
seq.add_event('rf', 15)
seq.add_event('grad', 20)

# Simulate
simulator = EPGSimulator()
omega_store, echoes = simulator.simulate(seq)
```

## Features

### Core EPG Operators
- **RF rotation**: `rf_rotation(phase, flip_angle)`
- **Gradient shift**: `shift_grad(k_shift, omega)`
- **T1/T2 relaxation**: `relax(tau, T1, T2, omega)`

### Predefined Sequences
- **TSE**: `simulate_tse(alpha, N, esp, use_y90, relaxation)`
- **GRE**: `simulate_gre(alpha, N, TR, relaxation)`
- **VFA**: `simulate_vfa(flip_angles, TR, relaxation)`

### Visualization
- **EPG diagrams**: `display_epg(omega_store, sequence)`
- **Echo trains**: `plot_echoes(echoes)`
- **T2 decay**: `plot_t2_decay(echoes, T2_true)`
- **Sequence comparison**: `plot_sequence_comparison(results_list)`

### Utilities
- **Echo detection**: `find_echoes(sequence, omega_store)`
- **Relaxation parameters**: `get_relaxation_params(tissue_type)`
- **Sequence analysis**: `analyze_sequence_efficiency(echoes, esp)`

## Demo Examples

Run the comprehensive demo suite:
```bash
epg-demo
```

Or run individual demos:
```python
from epg_python.demo import demo_basic_tse, demo_t2_variation

demo_basic_tse()        # Basic TSE simulation
demo_t2_variation()     # T2 parameter study
```

## Theory Background

The Extended Phase Graph (EPG) formalism provides an elegant framework for simulating MRI pulse sequences by tracking the evolution of transverse and longitudinal magnetization states.

### Key Concepts
- **Configuration states**: F⁺(k), F⁻(k), Z(k) representing different k-space states
- **RF operator**: Rotates magnetization according to flip angle and phase
- **Gradient operator**: Shifts transverse states in k-space
- **Relaxation operator**: Applies T1/T2 decay and recovery

### References
1. Weigel, M. "Extended phase graphs: dephasing, RF pulses, and echoes‐pure and simple." *Journal of Magnetic Resonance Imaging* 41.2 (2015): 266-295.
2. Hennig, J. "Multiecho imaging sequences with low refocusing flip angles." *Journal of Magnetic Resonance* 78.3 (1988): 397-407.

## API Reference

### Core Classes
- `EPGSequence`: Pulse sequence definition
- `EPGSimulator`: Main simulation engine

### Sequence Classes
- `EPGSequenceTSE`: Turbo Spin Echo
- `EPGSequenceGRE`: Gradient Recalled Echo
- `EPGSequenceVFA`: Variable Flip Angle

### Functions
See individual module documentation for detailed API reference.

## Examples

### T2 Mapping Study
```python
from epg_python.demo import demo_t2_mapping

# Study T2 measurement accuracy
measured_t2_map, T2_values, esp_values = demo_t2_mapping()
```

### Multi-sequence Comparison
```python
from epg_python import simulate_tse, simulate_gre, plot_sequence_comparison

# Compare different sequences
tse_results = simulate_tse(120, 10, 10, True, (400, 100))
gre_results = simulate_gre(30, 10, 50, (400, 100))

results_list = [
    (tse_results[1], "TSE α=120°"),
    (gre_results[1], "GRE α=30°")
]

plot_sequence_comparison(results_list, "TSE vs GRE Comparison")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This Python implementation is based on the MATLAB EPG simulation toolkit by Sairam Geethanath and Gehua Tong. The original MATLAB code provided the foundation and validation reference for this Python conversion.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{epg_python,
  title={EPG Python: Extended Phase Graph Simulation for MRI},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/epg-python}
}
```