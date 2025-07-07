# T2 Measurement Bias from Imperfect Refocusing Pulses

## Overview

The `demo_flip_angle_t2_bias()` function demonstrates a critical clinical issue in T2 quantification: how imperfect refocusing pulses (flip angles < 180°) introduce systematic bias in T2 measurements from TSE sequences.

## Clinical Motivation

In clinical MRI, perfect 180° refocusing pulses are often difficult to achieve due to:
- **RF power limitations** (SAR constraints)
- **B1 inhomogeneity** (spatial variations in flip angle)
- **Hardware limitations** (RF amplifier constraints)
- **Patient safety** (heating concerns)

Many clinical protocols use **reduced flip angle refocusing** (e.g., 150°, 120°) to address these limitations, but this introduces **systematic T2 measurement bias** that must be understood and potentially corrected.

## What the Demo Shows

### 1. Main Comparison Plot
- **True T2 vs Measured T2** for different refocusing flip angles
- Shows how the relationship deviates from the ideal line (y = x)
- Demonstrates **systematic bias patterns** for each flip angle

### 2. Individual Echo Train Examples
- **Actual echo trains** for each flip angle condition
- **Fitted exponential decays** showing how T2 is measured
- Illustrates how **imperfect refocusing affects echo amplitudes**

### 3. Bias Distribution Statistics
- **Box plots** showing the distribution of T2 measurement bias
- **Quantitative assessment** of bias magnitude and variability
- Helps understand the **clinical significance** of the bias

## Key Physics Concepts

### Perfect 180° Refocusing
- **Complete magnetization refocusing** at each echo
- **Minimal stimulated echo contamination**
- **Most accurate T2 measurements**

### Imperfect Refocusing (< 180°)
- **Incomplete magnetization refocusing**
- **Increased stimulated echo contributions**
- **Altered T2 decay characteristics**
- **Systematic T2 measurement bias**

### EPG Modeling Advantages
- **Accurate simulation** of imperfect refocusing effects
- **Proper handling** of stimulated echo pathways
- **Realistic T2 measurement scenarios**

## Expected Results

### Typical Bias Patterns:
1. **180° refocusing**: Minimal bias (reference standard)
2. **150° refocusing**: Moderate systematic bias
3. **120° refocusing**: Larger systematic bias

### Bias Characteristics:
- **T2-dependent**: Bias magnitude varies with true T2 value
- **Systematic**: Consistent direction and pattern
- **Predictable**: Can potentially be corrected with calibration

## Clinical Implications

### For T2 Quantification:
- **Protocol optimization**: Balance between accuracy and SAR/safety
- **Bias correction**: May need flip angle-specific calibration
- **Reference standards**: Importance of 180° refocusing for accuracy

### For Clinical Protocols:
- **Sequence parameter selection**: Consider T2 bias vs other constraints
- **Multi-vendor consistency**: Different implementations may have different bias
- **Quality assurance**: Regular T2 phantom measurements with known values

## Usage Example

```python
from epg_python.demo import demo_flip_angle_t2_bias

# Run the comprehensive demonstration
results = demo_flip_angle_t2_bias()

# Results contain detailed measurements for each flip angle
for flip_angle in [180, 150, 120]:
    if flip_angle in results:
        measured_t2 = results[flip_angle]['measured_t2']
        true_t2 = results[flip_angle]['true_t2']
        
        # Calculate bias statistics
        valid_idx = ~np.isnan(measured_t2)
        if np.any(valid_idx):
            bias = measured_t2[valid_idx] - true_t2[valid_idx]
            print(f"{flip_angle}° refocusing: {np.mean(bias):.1f} ± {np.std(bias):.1f} ms bias")
```

## Technical Parameters

### Simulation Settings:
- **Echo train length**: 8 echoes
- **Echo spacing**: 20 ms
- **T1**: 800 ms (typical for tissue)
- **True T2 values**: 50, 80, 120, 160 ms
- **Refocusing angles**: 180°, 150°, 120°

### Analysis Method:
- **Exponential fitting**: ln(S) = ln(S₀) - TE/T2
- **Linear regression**: Robust fitting of echo train decay
- **Quality control**: Validation of fitted parameters

## Interpretation Guidelines

### Understanding the Plots:

1. **Main comparison**: Look for systematic deviations from the diagonal line
2. **Echo trains**: Notice how imperfect refocusing changes echo amplitudes
3. **Bias statistics**: Quantify the clinical significance of the bias

### Clinical Decision Making:

- **Acceptable bias**: Depends on clinical application requirements
- **Protocol trade-offs**: Accuracy vs SAR/safety constraints
- **Correction strategies**: Phantom-based calibration or theoretical correction

## Advanced Applications

### Research Extensions:
- **B1 mapping integration**: Spatially-varying flip angle effects
- **Multi-echo analysis**: Optimal echo selection for T2 fitting
- **Correction algorithms**: Development of bias correction methods

### Clinical Validation:
- **Phantom studies**: Validation with known T2 standards
- **In-vivo correlation**: Comparison with reference methods
- **Multi-site studies**: Consistency across different scanners

This demonstration provides essential insights for anyone involved in quantitative T2 mapping, helping to understand and potentially mitigate the effects of imperfect refocusing pulses in clinical TSE sequences.