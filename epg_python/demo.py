"""
EPG Demo Module

Demonstration scripts showing EPG simulation capabilities.
Converted from MATLAB demoV1.m
"""

import numpy as np
import matplotlib.pyplot as plt
from .core import EPGSequence, EPGSimulator
from .sequences import simulate_tse
from .visualization import display_epg, plot_echoes, plot_t2_decay, plot_sequence_comparison
from .utils import extract_echo_train


def demo_basic_tse():
    """
    Demo 1: Basic TSE sequence simulation.
    
    Demonstrates single TSE sequence with visualization of EPG diagram
    and echo train analysis.
    """
    print("Demo 1: Basic TSE Sequence")
    print("-" * 30)
    
    # Sequence parameters
    alpha = 120  # Refocusing flip angle
    N = 10       # Number of pulses
    esp = 10     # Echo spacing (ms)
    use_y90 = True
    T1, T2 = 800, 100  # Relaxation times (ms)
    
    # Create and simulate sequence
    omega_store, echoes, sequence = simulate_tse(alpha, N, esp, use_y90, (T1, T2))
    
    # Display EPG diagram
    fig1 = display_epg(omega_store, sequence, annotate=True)
    fig1.suptitle('EPG Diagram - TSE Sequence', fontsize=16)
    
    # Plot all echoes (including RF-coincident ones)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # All echoes
    plot_echoes(echoes, "All F(0) States", ax=ax1)
    
    # True echoes only (skip RF-coincident ones)
    true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
    plot_echoes(true_echoes, "True Echoes Only", ax=ax2)
    
    fig2.suptitle('Echo Analysis - TSE Sequence', fontsize=16)
    
    print(f"Sequence: {sequence.name}")
    print(f"Parameters: α={alpha}°, N={N}, ESP={esp}ms")
    print(f"Relaxation: T1={T1}ms, T2={T2}ms")
    print(f"Total echoes found: {len(echoes)}")
    print(f"True echoes: {len(true_echoes)}")
    
    plt.show()
    return omega_store, echoes, sequence


def demo_t2_variation():
    """
    Demo 2: TSE sequence with varying T2 values.
    
    Shows how T2 affects echo train decay for the same sequence.
    """
    print("\nDemo 2: T2 Variation Study")
    print("-" * 30)
    
    # Fixed sequence parameters
    alpha = 120
    N = 10
    esp = 10
    use_y90 = True
    T1 = 800
    
    # Varying T2 values
    T2_values = np.arange(10, 101, 10)  # 10 to 100 ms in steps of 10
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    results = []
    for T2 in T2_values:
        # Simulate sequence
        omega_store, echoes, sequence = simulate_tse(alpha, N, esp, use_y90, (T1, T2))
        
        # Extract true echoes
        true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
        
        if len(true_echoes) > 0:
            times = true_echoes[:, 0]
            intensities = true_echoes[:, 1]
            
            # Plot with color coding
            color = plt.cm.viridis(T2 / max(T2_values))
            ax.plot(times, intensities, 'o-', color=color, linewidth=2, 
                   markersize=6, label=f'T2={T2}ms')
            
            results.append((true_echoes, f'T2={T2}ms'))
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Echo Intensity', fontsize=12)
    ax.set_title('TSE Echo Trains - T2 Variation', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Simulated {len(T2_values)} different T2 values")
    print(f"T2 range: {min(T2_values)} - {max(T2_values)} ms")
    
    return results


def demo_effective_te_study():
    """
    Demo 3: Effective TE study with 3D visualization.
    
    Varies echo spacing to study effective TE behavior.
    """
    print("\nDemo 3: Effective TE Study")
    print("-" * 30)
    
    # Fixed parameters
    alpha = 120
    N = 10
    use_y90 = True
    T1, T2 = 800, 100
    
    # Varying echo spacing
    esp_values = np.arange(10, 101, 10)  # 10 to 100 ms
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    te_eff_vec = []
    signal_vec = []
    
    for k, esp in enumerate(esp_values):
        # Simulate sequence
        omega_store, echoes, sequence = simulate_tse(alpha, N, esp, use_y90, (T1, T2))
        
        # Extract true echoes
        true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
        
        if len(true_echoes) > 0:
            times = true_echoes[:, 0]
            intensities = true_echoes[:, 1]
            
            # Create ESP vector for 3D plotting
            esp_vec = np.full(len(times), esp)
            
            # Plot 3D line
            ax.plot(times, esp_vec, intensities, linewidth=2, alpha=0.8)
            
            # Mark effective TE (middle echo for illustration)
            te_index = len(intensities) // 2
            if te_index < len(intensities):
                te_eff = times[te_index]
                signal_eff = intensities[te_index]
                
                ax.scatter(te_eff, esp, signal_eff, color='red', s=100, alpha=0.8)
                
                te_eff_vec.append(te_eff)
                signal_vec.append(signal_eff)
    
    ax.set_xlabel('TE (ms)', fontsize=12)
    ax.set_ylabel('Echo Spacing (ms)', fontsize=12)
    ax.set_zlabel('Signal Intensity', fontsize=12)
    ax.set_title('3D Echo Train Evolution', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Plot effective decay curve
    if te_eff_vec and signal_vec:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        te_eff_vec = np.array(te_eff_vec)
        signal_vec = np.array(signal_vec)
        
        # Plot effective decay
        ax2.plot(te_eff_vec, signal_vec, 'bo-', linewidth=2, markersize=8,
                label='Effective Decay Curve', markerfacecolor='lightblue')
        
        # Plot theoretical decay for comparison
        theoretical_signal = np.exp(-te_eff_vec / T2)
        ax2.plot(te_eff_vec, theoretical_signal, 'r--', linewidth=2,
                label=f'True Decay (T2={T2}ms)')
        
        ax2.set_xlabel('Effective TE (ms)', fontsize=12)
        ax2.set_ylabel('Signal at Effective TE', fontsize=12)
        ax2.set_title('Effective vs True T2 Decay', fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(f"Simulated {len(esp_values)} different echo spacings")
    print(f"ESP range: {min(esp_values)} - {max(esp_values)} ms")
    
    return te_eff_vec, signal_vec


def demo_t2_mapping():
    """
    Demo 4: T2 mapping study.
    
    Studies how measured T2 varies with echo spacing and true T2 values.
    For each combination of true T2 and echo spacing, simulates a TSE sequence
    and measures T2 from the echo train decay.
    """
    print("\nDemo 4: T2 Mapping Study")
    print("-" * 30)
    
    # Parameters
    alpha = 120
    N = 8  # Number of echoes (reduced for better fitting)
    use_y90 = True
    T1 = 800
    
    # T2 values to study
    T2_values = np.arange(50, 151, 20)  # 50 to 150 ms
    esp_values = np.arange(10, 81, 10)  # 10 to 80 ms
    
    # Storage for results
    measured_t2_map = np.zeros((len(T2_values), len(esp_values)))
    
    print("Computing T2 mapping... (this may take a moment)")
    
    for i, T2_true in enumerate(T2_values):
        print(f"Processing T2 = {T2_true} ms...")
        
        for j, esp in enumerate(esp_values):
            # Simulate TSE sequence
            omega_store, echoes, sequence = simulate_tse(alpha, N, esp, use_y90, (T1, T2_true))
            
            # Extract true echoes (skip RF echoes)
            true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
            
            if len(true_echoes) >= 3:  # Need at least 3 points for reliable fitting
                times = true_echoes[:, 0]
                intensities = true_echoes[:, 1]
                
                # Remove any zero or negative intensities
                valid_idx = intensities > 1e-10
                if np.sum(valid_idx) >= 3:
                    times_fit = times[valid_idx]
                    intensities_fit = intensities[valid_idx]
                    
                    try:
                        # Fit exponential decay: S = S0 * exp(-TE/T2)
                        # Take natural log: ln(S) = ln(S0) - TE/T2
                        log_intensities = np.log(intensities_fit)
                        
                        # Linear fit: y = mx + b, where y = ln(S), x = TE
                        # slope m = -1/T2, so T2 = -1/m
                        coeffs = np.polyfit(times_fit, log_intensities, 1)
                        slope = coeffs[0]
                        
                        if slope < 0:  # Slope should be negative for decay
                            measured_t2 = -1 / slope
                            # Sanity check: measured T2 should be reasonable
                            if 10 <= measured_t2 <= 500:
                                measured_t2_map[i, j] = measured_t2
                            else:
                                measured_t2_map[i, j] = np.nan
                        else:
                            measured_t2_map[i, j] = np.nan
                            
                    except (np.linalg.LinAlgError, ValueError):
                        measured_t2_map[i, j] = np.nan
            else:
                measured_t2_map[i, j] = np.nan
    
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: T2 mapping heatmap
    # Mask NaN values for better visualization
    masked_data = np.ma.masked_invalid(measured_t2_map)
    im = ax1.imshow(masked_data, aspect='auto', origin='lower',
                    extent=[min(esp_values), max(esp_values), 
                           min(T2_values), max(T2_values)],
                    cmap='viridis')
    ax1.set_xlabel('Echo Spacing (ms)', fontsize=12)
    ax1.set_ylabel('True T2 (ms)', fontsize=12)
    ax1.set_title('Measured T2 Map', fontsize=14)
    plt.colorbar(im, ax=ax1, label='Measured T2 (ms)')
    
    # Plot 2: True vs measured T2 (averaged over ESP)
    measured_t2_avg = np.nanmean(measured_t2_map, axis=1)
    valid_avg = ~np.isnan(measured_t2_avg)
    
    if np.any(valid_avg):
        ax2.plot(T2_values[valid_avg], measured_t2_avg[valid_avg], 'bo-', 
                linewidth=2, markersize=8, label='Measured T2 (avg)')
        ax2.plot(T2_values, T2_values, 'r--', linewidth=2, label='True T2')
        ax2.set_xlabel('True T2 (ms)', fontsize=12)
        ax2.set_ylabel('Measured T2 (ms)', fontsize=12)
        ax2.set_title('T2 Accuracy (ESP averaged)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Plot 3: T2 vs ESP for different true T2 values
    for i, T2_true in enumerate(T2_values[::2]):  # Plot every other T2 value
        valid_esp = ~np.isnan(measured_t2_map[i*2, :])
        if np.any(valid_esp):
            ax3.plot(esp_values[valid_esp], measured_t2_map[i*2, valid_esp], 
                    'o-', label=f'True T2 = {T2_true} ms', markersize=6)
    
    ax3.set_xlabel('Echo Spacing (ms)', fontsize=12)
    ax3.set_ylabel('Measured T2 (ms)', fontsize=12)
    ax3.set_title('T2 Dependence on Echo Spacing', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Example echo train decay for one case
    # Show example for middle T2 and ESP values
    mid_t2_idx = len(T2_values) // 2
    mid_esp_idx = len(esp_values) // 2
    T2_example = T2_values[mid_t2_idx]
    esp_example = esp_values[mid_esp_idx]
    
    # Simulate example case
    omega_store, echoes, sequence = simulate_tse(alpha, N, esp_example, use_y90, (T1, T2_example))
    true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
    
    if len(true_echoes) > 0:
        times = true_echoes[:, 0]
        intensities = true_echoes[:, 1]
        
        ax4.semilogy(times, intensities, 'bo-', markersize=8, linewidth=2, 
                    label='Simulated echoes')
        
        # Show fitted decay
        if not np.isnan(measured_t2_map[mid_t2_idx, mid_esp_idx]):
            measured_t2_example = measured_t2_map[mid_t2_idx, mid_esp_idx]
            t_fit = np.linspace(min(times), max(times), 100)
            # Estimate S0 from first echo
            S0_est = intensities[0] * np.exp(times[0] / measured_t2_example)
            fitted_curve = S0_est * np.exp(-t_fit / measured_t2_example)
            ax4.semilogy(t_fit, fitted_curve, 'r--', linewidth=2, 
                        label=f'Fitted T2 = {measured_t2_example:.1f} ms')
        
        ax4.set_xlabel('Echo Time (ms)', fontsize=12)
        ax4.set_ylabel('Signal Intensity', fontsize=12)
        ax4.set_title(f'Example Echo Train\n(True T2={T2_example}ms, ESP={esp_example}ms)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    valid_measurements = ~np.isnan(measured_t2_map)
    if np.any(valid_measurements):
        # Calculate error statistics
        errors = []
        for i, T2_true in enumerate(T2_values):
            valid_row = ~np.isnan(measured_t2_map[i, :])
            if np.any(valid_row):
                row_avg = np.nanmean(measured_t2_map[i, valid_row])
                if not np.isnan(row_avg):
                    errors.append(abs(row_avg - T2_true))
        
        if errors:
            avg_error = np.mean(errors)
            print(f"T2 mapping completed for {len(T2_values)} T2 values and {len(esp_values)} ESP values")
            print(f"Average T2 measurement error: {avg_error:.1f} ms")
            print(f"Successful measurements: {np.sum(valid_measurements)}/{measured_t2_map.size}")
        else:
            print("No valid T2 measurements obtained")
    else:
        print("No valid T2 measurements obtained")
    
    return measured_t2_map, T2_values, esp_values


def demo_flip_angle_t2_bias():
    """
    Demo 5: T2 measurement bias due to imperfect refocusing pulses.
    
    Demonstrates how using refocusing flip angles less than 180° affects
    T2 measurements in TSE sequences. This is crucial for understanding
    T2 quantification accuracy in clinical protocols.
    """
    print("\nDemo 5: T2 Measurement Bias from Imperfect Refocusing")
    print("-" * 50)
    
    # Parameters
    N = 8  # Number of echoes
    esp = 20  # Echo spacing in ms
    use_y90 = True
    T1 = 800  # ms
    
    # True T2 values to study
    T2_true_values = [50, 80, 120, 160]  # ms
    
    # Refocusing flip angles to compare
    flip_angles = [180, 150, 120]  # degrees
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    print("Simulating TSE sequences with different refocusing flip angles...")
    print(f"True T2 values: {T2_true_values} ms")
    print(f"Refocusing flip angles: {flip_angles}°")
    
    # Storage for results
    results = {}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main comparison plot
    ax_main = plt.subplot(2, 3, (1, 2))
    
    # Individual echo train plots
    ax_echoes = [plt.subplot(2, 3, i+4) for i in range(3)]
    
    # Summary statistics plot
    ax_summary = plt.subplot(2, 3, 3)
    
    for flip_idx, (alpha, color, marker) in enumerate(zip(flip_angles, colors, markers)):
        print(f"\nProcessing flip angle {alpha}°...")
        
        measured_t2_list = []
        
        for T2_true in T2_true_values:
            # Simulate TSE sequence
            omega_store, echoes, sequence = simulate_tse(alpha, N, esp, use_y90, (T1, T2_true))
            
            # Extract echo train
            true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
            
            if len(true_echoes) >= 3:
                times = true_echoes[:, 0]
                intensities = true_echoes[:, 1]
                
                # Remove any zero or negative intensities
                valid_idx = intensities > 1e-10
                if np.sum(valid_idx) >= 3:
                    times_fit = times[valid_idx]
                    intensities_fit = intensities[valid_idx]
                    
                    try:
                        # Fit exponential decay
                        log_intensities = np.log(intensities_fit)
                        coeffs = np.polyfit(times_fit, log_intensities, 1)
                        slope = coeffs[0]
                        
                        if slope < 0:
                            measured_t2 = -1 / slope
                            if 10 <= measured_t2 <= 500:  # Sanity check
                                measured_t2_list.append(measured_t2)
                            else:
                                measured_t2_list.append(np.nan)
                        else:
                            measured_t2_list.append(np.nan)
                            
                    except (np.linalg.LinAlgError, ValueError):
                        measured_t2_list.append(np.nan)
                else:
                    measured_t2_list.append(np.nan)
            else:
                measured_t2_list.append(np.nan)
        
        # Store results
        results[alpha] = {
            'measured_t2': np.array(measured_t2_list),
            'true_t2': np.array(T2_true_values)
        }
        
        # Plot on main comparison
        valid_measurements = ~np.isnan(measured_t2_list)
        if np.any(valid_measurements):
            ax_main.plot(np.array(T2_true_values)[valid_measurements], 
                        np.array(measured_t2_list)[valid_measurements], 
                        marker=marker, color=color, linewidth=2, markersize=8,
                        label=f'{alpha}° refocusing')
    
    # Plot ideal line on main comparison
    t2_range = np.linspace(min(T2_true_values), max(T2_true_values), 100)
    ax_main.plot(t2_range, t2_range, 'k--', linewidth=2, alpha=0.7, label='Ideal (True T2)')
    ax_main.set_xlabel('True T2 (ms)', fontsize=12)
    ax_main.set_ylabel('Measured T2 (ms)', fontsize=12)
    ax_main.set_title('T2 Measurement Bias vs Refocusing Flip Angle', fontsize=14)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    
    # Plot example echo trains for middle T2 value
    example_t2 = T2_true_values[len(T2_true_values)//2]  # Middle T2 value
    
    for flip_idx, (alpha, color, marker) in enumerate(zip(flip_angles, colors, markers)):
        # Simulate example sequence
        omega_store, echoes, sequence = simulate_tse(alpha, N, esp, use_y90, (T1, example_t2))
        true_echoes = extract_echo_train(echoes, skip_rf_echoes=True)
        
        if len(true_echoes) > 0:
            times = true_echoes[:, 0]
            intensities = true_echoes[:, 1]
            
            # Plot echo train
            ax_echoes[flip_idx].semilogy(times, intensities, marker=marker, color=color, 
                                       linewidth=2, markersize=8, label='Simulated echoes')
            
            # Fit and plot exponential decay
            valid_idx = intensities > 1e-10
            if np.sum(valid_idx) >= 3:
                times_fit = times[valid_idx]
                intensities_fit = intensities[valid_idx]
                
                try:
                    log_intensities = np.log(intensities_fit)
                    coeffs = np.polyfit(times_fit, log_intensities, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    
                    if slope < 0:
                        measured_t2 = -1 / slope
                        
                        # Plot fitted curve
                        t_fit = np.linspace(min(times), max(times), 100)
                        fitted_curve = np.exp(intercept) * np.exp(slope * t_fit)
                        ax_echoes[flip_idx].semilogy(t_fit, fitted_curve, '--', color=color, 
                                                   linewidth=2, alpha=0.8,
                                                   label=f'Fitted T2 = {measured_t2:.1f} ms')
                        
                        ax_echoes[flip_idx].set_title(f'{alpha}° Refocusing\n(True T2 = {example_t2} ms)', 
                                                    fontsize=12)
                        ax_echoes[flip_idx].set_xlabel('Echo Time (ms)', fontsize=10)
                        ax_echoes[flip_idx].set_ylabel('Signal Intensity', fontsize=10)
                        ax_echoes[flip_idx].grid(True, alpha=0.3)
                        ax_echoes[flip_idx].legend(fontsize=9)
                        
                except (np.linalg.LinAlgError, ValueError):
                    pass
    
    # Calculate and plot bias statistics
    bias_data = []
    bias_labels = []
    
    for alpha in flip_angles:
        if alpha in results:
            measured = results[alpha]['measured_t2']
            true_vals = results[alpha]['true_t2']
            valid_idx = ~np.isnan(measured)
            
            if np.any(valid_idx):
                bias = measured[valid_idx] - true_vals[valid_idx]
                bias_percent = (bias / true_vals[valid_idx]) * 100
                bias_data.append(bias_percent)
                bias_labels.append(f'{alpha}°')
    
    if bias_data:
        bp = ax_summary.boxplot(bias_data, labels=bias_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(bias_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_summary.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        ax_summary.set_xlabel('Refocusing Flip Angle', fontsize=12)
        ax_summary.set_ylabel('T2 Measurement Bias (%)', fontsize=12)
        ax_summary.set_title('T2 Bias Distribution', fontsize=14)
        ax_summary.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    for alpha in flip_angles:
        if alpha in results:
            measured = results[alpha]['measured_t2']
            true_vals = results[alpha]['true_t2']
            valid_idx = ~np.isnan(measured)
            
            if np.any(valid_idx):
                bias = measured[valid_idx] - true_vals[valid_idx]
                bias_percent = (bias / true_vals[valid_idx]) * 100
                
                print(f"\n{alpha}° Refocusing Pulses:")
                print(f"  Mean bias: {np.mean(bias):.1f} ms ({np.mean(bias_percent):.1f}%)")
                print(f"  Std bias:  {np.std(bias):.1f} ms ({np.std(bias_percent):.1f}%)")
                print(f"  Max bias:  {np.max(np.abs(bias)):.1f} ms ({np.max(np.abs(bias_percent)):.1f}%)")
                
                # Individual T2 results
                print("  Individual results:")
                for i, (true_t2, meas_t2) in enumerate(zip(true_vals[valid_idx], measured[valid_idx])):
                    bias_val = meas_t2 - true_t2
                    bias_pct = (bias_val / true_t2) * 100
                    print(f"    T2 {true_t2:3.0f} ms → {meas_t2:5.1f} ms (bias: {bias_val:+5.1f} ms, {bias_pct:+4.1f}%)")
    
    print("\n" + "="*50)
    print("CLINICAL IMPLICATIONS:")
    print("- Perfect 180° refocusing gives most accurate T2 measurements")
    print("- Lower flip angles (150°, 120°) introduce systematic T2 bias")
    print("- Bias magnitude depends on true T2 value")
    print("- Important for quantitative T2 mapping protocols")
    print("="*50)
    
    return results


def run_all_demos():
    """
    Run all EPG demonstration examples.
    """
    print("EPG Python Demo Suite")
    print("=" * 50)
    
    # Run all demos
    demo_basic_tse()
    demo_t2_variation()
    demo_effective_te_study()
    demo_t2_mapping()
    demo_flip_angle_t2_bias()
    
    print("\n" + "=" * 50)
    print("All demos completed!")
    print("Close the plot windows to continue or exit.")


if __name__ == "__main__":
    run_all_demos()