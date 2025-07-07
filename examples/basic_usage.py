"""
Basic EPG Usage Examples

This script demonstrates basic usage of the EPG Python package.
"""

import numpy as np
import matplotlib.pyplot as plt
from epg_python import (
    simulate_tse, simulate_gre, simulate_vfa,
    display_epg, plot_echoes, plot_t2_decay,
    EPGSequence, EPGSimulator
)


def example_1_basic_tse():
    """Example 1: Basic TSE simulation"""
    print("Example 1: Basic TSE Simulation")
    print("-" * 40)
    
    # Parameters
    alpha = 120      # Refocusing flip angle
    N = 8           # Number of echoes
    esp = 12        # Echo spacing (ms)
    T1, T2 = 1000, 80  # Relaxation times
    
    # Simulate
    omega_store, echoes, sequence = simulate_tse(alpha, N, esp, True, (T1, T2))
    
    # Display results
    print(f"Sequence: {sequence.name}")
    print(f"Found {len(echoes)} echoes")
    print(f"Echo times: {echoes[:, 0]}")
    print(f"Echo intensities: {echoes[:, 1]}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plot_echoes(echoes, "TSE Echo Train", ax=ax1)
    plot_t2_decay(echoes, T2, "T2 Decay Comparison", ax=ax2)
    plt.tight_layout()
    plt.show()


def example_2_custom_sequence():
    """Example 2: Custom sequence creation"""
    print("\nExample 2: Custom Sequence")
    print("-" * 40)
    
    # Create custom CPMG-like sequence
    seq = EPGSequence("Custom CPMG")
    
    # Add RF pulses: 90x - 180y - 180y - 180y
    seq.add_rf(0, 90)    # 90° pulse, x-axis
    seq.add_rf(90, 180)  # 180° pulse, y-axis
    seq.add_rf(90, 180)  # 180° pulse, y-axis
    seq.add_rf(90, 180)  # 180° pulse, y-axis
    
    # Add gradients
    for _ in range(7):  # Need 2*N-1 gradients for N pulses
        seq.add_gradient(1)
    
    # Set relaxation
    seq.set_relaxation(800, 120)
    
    # Add timing and events
    tau = 10  # Half echo spacing
    times = [0, tau, tau, 2*tau, 3*tau, 3*tau, 4*tau, 5*tau, 5*tau, 6*tau, 7*tau]
    events = ['rf', 'grad', 'relax', 'rf', 'grad', 'relax', 'grad', 'relax', 'rf', 'grad', 'relax']
    
    for event, time in zip(events, times):
        seq.add_event(event, time)
    
    # Simulate
    simulator = EPGSimulator()
    omega_store, echoes = simulator.simulate(seq)
    
    print(f"Custom sequence: {seq.name}")
    print(f"Found {len(echoes)} echoes")
    
    # Visualize
    display_epg(omega_store, seq)
    plt.show()


def example_3_sequence_comparison():
    """Example 3: Compare different sequences"""
    print("\nExample 3: Sequence Comparison")
    print("-" * 40)
    
    # Common parameters
    T1, T2 = 1000, 100
    
    # TSE sequence
    tse_omega, tse_echoes, tse_seq = simulate_tse(120, 8, 15, True, (T1, T2))
    
    # GRE sequence
    gre_omega, gre_echoes, gre_seq = simulate_gre(30, 8, 50, (T1, T2))
    
    # VFA sequence
    flip_angles = [90, 60, 45, 35, 30, 25, 22, 20]  # Decreasing flip angles
    vfa_omega, vfa_echoes, vfa_seq = simulate_vfa(flip_angles, 50, (T1, T2))
    
    # Compare results
    from epg_python.visualization import plot_sequence_comparison
    
    results = [
        (tse_echoes, "TSE (α=120°)"),
        (gre_echoes, "GRE (α=30°)"),
        (vfa_echoes, "VFA (variable)")
    ]
    
    plot_sequence_comparison(results, "Sequence Comparison")
    plt.show()
    
    # Print summary
    print(f"TSE echoes: {len(tse_echoes)}")
    print(f"GRE echoes: {len(gre_echoes)}")
    print(f"VFA echoes: {len(vfa_echoes)}")


def example_4_parameter_study():
    """Example 4: Parameter sensitivity study"""
    print("\nExample 4: Parameter Study")
    print("-" * 40)
    
    # Study effect of flip angle on TSE
    flip_angles = [90, 120, 150, 180]
    colors = ['blue', 'green', 'orange', 'red']
    
    plt.figure(figsize=(12, 8))
    
    for i, alpha in enumerate(flip_angles):
        omega_store, echoes, sequence = simulate_tse(alpha, 10, 12, True, (1000, 80))
        
        if len(echoes) > 0:
            times = echoes[:, 0]
            intensities = echoes[:, 1]
            
            plt.subplot(2, 2, 1)
            plt.plot(times, intensities, 'o-', color=colors[i], 
                    label=f'α={alpha}°', linewidth=2, markersize=6)
            
            plt.subplot(2, 2, 2)
            plt.semilogy(times, intensities, 'o-', color=colors[i], 
                        label=f'α={alpha}°', linewidth=2, markersize=6)
    
    # Format plots
    plt.subplot(2, 2, 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Echo Intensity')
    plt.title('Linear Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Echo Intensity (log)')
    plt.title('Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Study effect of T2 on signal decay
    T2_values = [50, 80, 120, 200]
    
    for i, T2 in enumerate(T2_values):
        omega_store, echoes, sequence = simulate_tse(120, 10, 12, True, (1000, T2))
        
        if len(echoes) > 0:
            times = echoes[:, 0]
            intensities = echoes[:, 1]
            
            plt.subplot(2, 2, 3)
            plt.plot(times, intensities, 'o-', color=colors[i], 
                    label=f'T2={T2}ms', linewidth=2, markersize=6)
            
            plt.subplot(2, 2, 4)
            plt.semilogy(times, intensities, 'o-', color=colors[i], 
                        label=f'T2={T2}ms', linewidth=2, markersize=6)
    
    plt.subplot(2, 2, 3)
    plt.xlabel('Time (ms)')
    plt.ylabel('Echo Intensity')
    plt.title('T2 Variation - Linear')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.xlabel('Time (ms)')
    plt.ylabel('Echo Intensity (log)')
    plt.title('T2 Variation - Log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Parameter study completed")


def main():
    """Run all examples"""
    print("EPG Python - Basic Usage Examples")
    print("=" * 50)
    
    example_1_basic_tse()
    example_2_custom_sequence()
    example_3_sequence_comparison()
    example_4_parameter_study()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()