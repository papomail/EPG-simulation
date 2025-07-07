# T2 Mapping Function Fix

## Problem Identified

The original `demo_t2_mapping()` function had a fundamental conceptual error in how it was measuring T2. The measured T2 appeared independent of echo spacing (ESP) because:

### Original Flawed Approach:
1. **Wrong T2 measurement strategy**: The function was collecting single echo intensities from different ESP values and trying to fit T2 from this artificial "decay curve"
2. **Misuse of effective TE**: It was treating different ESP values as different echo times, which is incorrect
3. **Single point per sequence**: Only taking the middle echo from each sequence, losing the actual echo train information

### Why This Was Wrong:
- **T2 should be measured from the echo train decay within a single sequence**, not across different sequences with different ESP values
- **Each ESP creates a different echo train** with echoes at times ESP, 2×ESP, 3×ESP, etc.
- **The original approach was essentially fitting noise** rather than actual T2 decay

## Corrected Approach

### Key Changes Made:

1. **Proper T2 measurement**: For each combination of true T2 and ESP, simulate a complete TSE sequence and measure T2 from the echo train decay within that single sequence

2. **Correct use of echo trains**: Extract all echoes from each sequence and fit the exponential decay S(t) = S₀ × exp(-t/T2)

3. **Individual sequence analysis**: Each (T2_true, ESP) combination gets its own T2 measurement from its own echo train

4. **Better visualization**: Added 4 plots to show:
   - T2 mapping heatmap (measured T2 vs true T2 and ESP)
   - True vs measured T2 accuracy
   - T2 dependence on ESP for different true T2 values
   - Example echo train decay with fitted curve

### Technical Improvements:

```python
# OLD (wrong): Collecting single echoes across different ESP values
for j, esp in enumerate(esp_values):
    # ... simulate sequence ...
    te_index = len(intensities) // 2  # Take only middle echo
    effective_te_vec.append(times[te_index])
    effective_signal_vec.append(intensities[te_index])

# Fit T2 from artificial decay curve
coeffs = np.polyfit(effective_te_vec, np.log(effective_signal_vec), 1)

# NEW (correct): Measure T2 from each echo train individually
for j, esp in enumerate(esp_values):
    # ... simulate sequence ...
    true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
    
    if len(true_echoes) >= 3:  # Need multiple echoes for fitting
        times = true_echoes[:, 0]      # All echo times
        intensities = true_echoes[:, 1] # All echo intensities
        
        # Fit T2 from this single sequence's echo train
        coeffs = np.polyfit(times, np.log(intensities), 1)
        measured_t2_map[i, j] = -1 / coeffs[0]
```

### Why ESP Dependence Should Exist:

1. **Different echo trains**: Different ESP values create echo trains with different timing patterns
2. **T2* effects**: Longer ESP may show more T2* (inhomogeneity) effects
3. **Stimulated echo contributions**: Different ESP affects the balance of spin echoes vs stimulated echoes
4. **Relaxation during refocusing**: More time between pulses allows more T2 decay

## Expected Results

With the corrected function, you should now see:

1. **ESP-dependent T2 measurements**: Measured T2 should vary with echo spacing
2. **More realistic T2 mapping**: The heatmap should show variation across both axes
3. **Better accuracy assessment**: True vs measured T2 plots should show systematic effects
4. **Proper echo train analysis**: Example plots showing actual exponential decay fitting

## Usage

The corrected function maintains the same interface:

```python
from epg_python.demo import demo_t2_mapping

# Run the corrected T2 mapping study
measured_t2_map, T2_values, esp_values = demo_t2_mapping()
```

The function now properly demonstrates how TSE sequences affect T2 measurements and how echo spacing influences the measured values, which is crucial for understanding T2 mapping in clinical MRI.