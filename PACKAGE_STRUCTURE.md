# EPG Python Package Structure

This document provides a complete overview of the EPG Python package structure and files.

## Package Directory Structure

```
epg-python/
├── epg_python/                 # Main package directory
│   ├── __init__.py            # Package initialization and exports
│   ├── core.py                # Core EPG simulation classes
│   ├── operators.py           # EPG operators (RF, gradient, relaxation)
│   ├── sequences.py           # Predefined MRI sequences
│   ├── utils.py               # Utility functions
│   ├── visualization.py       # Plotting and visualization
│   └── demo.py                # Demonstration scripts
├── examples/                   # Example scripts
│   ├── __init__.py            # Examples package init
│   ├── basic_usage.py         # Basic usage examples
│   └── matlab_compatibility.py # MATLAB compatibility demo
├── tests/                      # Test suite
│   ├── __init__.py            # Test package init
│   ├── test_operators.py      # Test EPG operators
│   └── test_sequences.py      # Test sequence implementations
├── setup.py                   # Package installation script
├── requirements.txt           # Python dependencies
├── README.md                  # Main package documentation
├── DOCUMENTATION.md           # Comprehensive documentation
├── PACKAGE_STRUCTURE.md       # This file
├── run_demo.py               # Quick demo runner
└── validate_installation.py  # Installation validation script
```

## File Descriptions

### Core Package Files (`epg_python/`)

#### `__init__.py`
- Package initialization file
- Defines public API exports
- Imports key classes and functions for easy access
- Version information

#### `core.py`
- **`EPGSequence`**: Base class for pulse sequence definition
- **`EPGSimulator`**: Main simulation engine
- **`epg_custom()`**: MATLAB-compatible simulation function
- Core simulation logic and state management

#### `operators.py`
- **`rf_rotation()`**: RF pulse rotation matrix calculation
- **`shift_grad()`**: Gradient-induced k-space shift operation
- **`relax()`**: T1/T2 relaxation operator
- Low-level EPG mathematical operations

#### `sequences.py`
- **`EPGSequenceTSE`**: Turbo Spin Echo sequence class
- **`EPGSequenceGRE`**: Gradient Recalled Echo sequence class
- **`EPGSequenceVFA`**: Variable Flip Angle sequence class
- **`simulate_tse()`**: TSE simulation convenience function
- **`simulate_gre()`**: GRE simulation convenience function
- **`simulate_vfa()`**: VFA simulation convenience function

#### `utils.py`
- **`find_echoes()`**: Echo detection in EPG simulation
- **`find_all_echoes()`**: Find all significant F(0) states
- **`get_relaxation_params()`**: Tissue-specific T1/T2 values
- **`calculate_signal_decay()`**: Theoretical T2 decay calculation
- **`extract_echo_train()`**: Extract true echo train from results
- **`analyze_sequence_efficiency()`**: Sequence performance metrics

#### `visualization.py`
- **`display_epg()`**: EPG state diagram visualization
- **`plot_echoes()`**: Echo train plotting
- **`plot_t2_decay()`**: T2 decay curve with theoretical comparison
- **`plot_sequence_comparison()`**: Multi-sequence comparison plots
- **`create_epg_animation()`**: Animated EPG evolution

#### `demo.py`
- **`demo_basic_tse()`**: Basic TSE demonstration
- **`demo_sequence_comparison()`**: Compare different sequences
- **`demo_parameter_effects()`**: Show parameter effects
- **`demo_custom_sequence()`**: Custom sequence building
- **`run_all_demos()`**: Execute all demonstrations

### Example Scripts (`examples/`)

#### `basic_usage.py`
- Simple TSE simulation example
- Basic visualization usage
- Parameter variation studies
- Beginner-friendly introduction

#### `matlab_compatibility.py`
- Direct MATLAB code translation examples
- Dictionary-based sequence definition
- Compatibility with existing MATLAB workflows
- Migration guide for MATLAB users

### Test Suite (`tests/`)

#### `test_operators.py`
- Unit tests for EPG operators
- Mathematical property validation
- Numerical accuracy tests
- Edge case handling

#### `test_sequences.py`
- Integration tests for sequence classes
- Simulation result validation
- Parameter effect verification
- Cross-sequence comparisons

### Configuration and Setup Files

#### `setup.py`
- Package metadata and dependencies
- Installation configuration
- Entry points and console scripts
- Development dependencies

#### `requirements.txt`
- Minimal runtime dependencies
- Version specifications
- Cross-platform compatibility

#### `README.md`
- Quick start guide
- Installation instructions
- Basic usage examples
- Links to detailed documentation

#### `DOCUMENTATION.md`
- Comprehensive API reference
- Detailed examples and tutorials
- Advanced usage patterns
- Troubleshooting guide

### Utility Scripts

#### `run_demo.py`
- Quick demonstration runner
- No-setup demo execution
- Visual examples of package capabilities
- New user introduction

#### `validate_installation.py`
- Installation verification
- Functionality testing
- Dependency checking
- Troubleshooting assistance

## Key Features by File

### Mathematical Core
- **`operators.py`**: Pure EPG mathematics
- **`core.py`**: Simulation orchestration
- **`utils.py`**: Analysis and processing

### User Interface
- **`sequences.py`**: High-level sequence definitions
- **`visualization.py`**: Results presentation
- **`demo.py`**: Interactive demonstrations

### Compatibility and Testing
- **`examples/matlab_compatibility.py`**: MATLAB migration
- **`tests/`**: Quality assurance
- **`validate_installation.py`**: User support

## Usage Patterns

### Quick Start (Minimal Code)
```python
from epg_python import simulate_tse, display_epg
omega_store, echoes, seq = simulate_tse(120, 8, 12, True, (1000, 80))
display_epg(omega_store, seq)
```

### Advanced Usage (Full Control)
```python
from epg_python import EPGSequence, EPGSimulator
seq = EPGSequence("Custom")
seq.add_rf(0, 90)
seq.add_gradient(1)
# ... build sequence
simulator = EPGSimulator()
omega_store, echoes = simulator.simulate(seq)
```

### MATLAB Compatibility
```python
from epg_python.core import epg_custom
seq_dict = {'rf': [[0, 90], [90, 120]], ...}
omega_store, echoes = epg_custom(seq_dict)
```

## Dependencies and Requirements

### Runtime Dependencies
- **NumPy** (≥1.19.0): Array operations and linear algebra
- **Matplotlib** (≥3.3.0): Visualization and plotting
- **SciPy** (≥1.5.0): Scientific computing utilities

### Development Dependencies
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **flake8**: Code linting

### Optional Dependencies
- **Jupyter**: Interactive notebooks
- **IPython**: Enhanced interactive shell
- **seaborn**: Enhanced plotting styles

## Installation Methods

### Standard Installation
```bash
pip install epg-python
```

### Development Installation
```bash
git clone https://github.com/yourusername/epg-python.git
cd epg-python
pip install -e ".[dev]"
```

### Minimal Installation (Core Only)
```bash
pip install numpy matplotlib scipy
# Then use package directly from source
```

## Testing and Validation

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Quick Validation
```bash
python validate_installation.py
```

### Demo Execution
```bash
python run_demo.py
```

## Package Design Philosophy

### Modular Architecture
- **Separation of concerns**: Math, sequences, visualization
- **Extensible design**: Easy to add new sequences/operators
- **Clean interfaces**: Simple function signatures

### User-Friendly Design
- **Multiple interfaces**: Functional and object-oriented
- **Comprehensive examples**: From basic to advanced
- **Rich documentation**: API reference and tutorials

### MATLAB Compatibility
- **Smooth migration**: Dictionary-based interface
- **Familiar patterns**: Similar function names and behavior
- **Validation tools**: Compare results with MATLAB

### Quality Assurance
- **Comprehensive testing**: Unit and integration tests
- **Numerical validation**: Mathematical property verification
- **Cross-platform support**: Windows, macOS, Linux

This package structure provides a complete, professional-grade EPG simulation toolkit suitable for research, education, and clinical applications.