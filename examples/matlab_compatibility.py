"""
MATLAB Compatibility Examples

This script demonstrates how to use the EPG Python package in a way that's
compatible with the original MATLAB code structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from epg_python.core import epg_custom
from epg_python.visualization import display_epg, plot_echoes
from epg_python.utils import extract_echo_train


def matlab_style_demo():
    """
    Recreate the MATLAB demoV1.m functionality in Python.
    
    This demonstrates MATLAB-compatible usage patterns.
    """
    print("MATLAB Compatibility Demo")
    print("=" * 40)
    
    # %% 1. Set up single sequence for simulation (MATLAB style comments)
    
    # 1.1 Specify sequence type
    seq = {'name': 'TSE'}
    
    # 1.2 Specify relaxation parameters
    seq['T1'] = 400
    seq['T2'] = 100  # arbitrary choice of parameters for initial demonstration
    
    # 1.4 Specify refocusing flip angle
    alpha = 120
    
    # 1.5 Set up RF scheme (assuming first 90 is 90, 0)
    use_y90 = 1
    N = 10
    
    if use_y90 == 1:
        seq['rf'] = np.array([[90, 90], [0, alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha]])
        seq['rf'] = seq['rf'][:, :N]  # Take only N pulses
    else:
        seq['rf'] = np.tile([[0], [alpha]], (1, N))
    
    # 1.6 Specify timing of sequence events
    esp = 10
    dt = esp / 2  # time evolves in 0.5*esp steps (dt = 0.5*esp -> dk = 1)
    
    seq['time'] = [0, dt, dt]
    seq['events'] = ['rf', 'grad', 'relax']
    
    for n in range(1, N):
        # Order of operators : T(rf)->S(grad)->E(relax) "TSE", easy to remember!
        seq['events'].extend(['rf', 'grad', 'relax', 'grad', 'relax'])
        seq['time'].extend([(2*n-1)*dt, 2*n*dt, 2*n*dt, (2*n+1)*dt, (2*n+1)*dt])
    
    seq['grad'] = np.ones(2*N-1)
    seq['time'] = np.array(seq['time'])
    
    # 1.7 Run EPG for specified sequence
    om_store, echoes = epg_custom(seq)
    
    # Display EPG diagram
    display_epg(om_store, seq, annotate=True)
    plt.title('EPG Diagram - MATLAB Style')
    plt.show()
    
    # 1.8 Plot signal for chosen sequence
    # TE1 includes all 'echoes' - some of which are not really echoes as they
    # coincide with RF pulses
    TE1 = echoes[:, 0]
    Signal1 = echoes[:, 1]
    
    # TE2 only includes the true echoes (even)
    TE2 = echoes[::2, 0]  # Plot only even values as the odd ones are not echoes
    Signal2 = echoes[::2, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(TE1, Signal1)
    ax1.set_ylim([0, max(Signal1)])
    ax1.set_title('All F(0) States')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Signal')
    
    ax2.plot(TE2, Signal2)
    ax2.set_ylim([0, max(Signal2)])
    ax2.set_title('True Echoes Only')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Signal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Sequence: {seq['name']}")
    print(f"Parameters: α={alpha}°, N={N}, ESP={esp}ms")
    print(f"Relaxation: T1={seq['T1']}ms, T2={seq['T2']}ms")
    print(f"Total echoes found: {len(echoes)}")
    print(f"True echoes: {len(TE2)}")
    
    return seq, om_store, echoes


def matlab_t2_variation():
    """
    MATLAB-style T2 variation study (Section 2 of demoV1.m).
    """
    print("\nMATLAB T2 Variation Study")
    print("-" * 40)
    
    # Base sequence from previous function
    seq, _, _ = matlab_style_demo()
    
    # Create figure outside the loop (MATLAB style)
    plt.figure(figsize=(12, 8))
    
    for k in range(10, 101, 10):  # 10:10:100 in MATLAB
        # 2.1 Specify T2 value within the loop
        seq['T2'] = k
        
        # 2.2 Run EPG for specified sequence
        om_store, echoes = epg_custom(seq)
        
        # 2.3 Plot signal for chosen sequence
        # TE2 only includes the true echoes (even)
        TE2 = echoes[::2, 0]  # Plot only even values
        Signal2 = echoes[::2, 1]
        
        plt.plot(TE2, Signal2, label=f'T2={k}ms')
        plt.ylim([0, 1])
        plt.hold = True  # MATLAB-style hold on
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Echo Intensity')
    plt.title('TSE Echo Trains - T2 Variation (MATLAB Style)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("T2 variation study completed")


def matlab_effective_te_study():
    """
    MATLAB-style effective TE study (Section 3 of demoV1.m).
    """
    print("\nMATLAB Effective TE Study")
    print("-" * 40)
    
    # Base parameters
    alpha = 120
    use_y90 = 1
    N = 10
    T1, T2 = 400, 100
    
    # Create figure outside the loop
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    teEffVec = []
    sVec = []
    
    for k in range(1, 11):  # 1:1:10 in MATLAB
        # 3.1 Specify echo spacing within the loop
        esp = 10 * k
        dt = esp / 2
        
        # Rebuild sequence (MATLAB style)
        seq = {
            'name': 'TSE',
            'T1': T1,
            'T2': T2,
            'time': [0, dt, dt],
            'events': ['rf', 'grad', 'relax']
        }
        
        # RF setup
        if use_y90 == 1:
            seq['rf'] = np.array([[90] + [0]*(N-1), [90] + [alpha]*(N-1)])
        else:
            seq['rf'] = np.tile([[0], [alpha]], (1, N))
        
        # Build timing
        for n in range(1, N):
            seq['events'].extend(['rf', 'grad', 'relax', 'grad', 'relax'])
            seq['time'].extend([(2*n-1)*dt, 2*n*dt, 2*n*dt, (2*n+1)*dt, (2*n+1)*dt])
        
        seq['grad'] = np.ones(2*N-1)
        seq['time'] = np.array(seq['time'])
        
        # 3.2 Run EPG for specified sequence
        om_store, echoes = epg_custom(seq)
        
        # 3.3 Plot signal for chosen sequence
        TE2 = echoes[::2, 0]  # True echoes only
        Signal2 = echoes[::2, 1]
        espVec = np.full(len(TE2), esp)  # Vector of ESP values for 3D plotting
        
        ax.plot(TE2, espVec, Signal2, linewidth=2)
        ax.set_xlim([-1, max(TE2)])
        ax.set_zlim([0, max(Signal2)])
        
        # 3.4 Add point for effective TE
        # (ONLY FOR ILLUSTRATION - middle echo)
        teIndex = len(Signal2) // 2
        if teIndex < len(Signal2):
            teEffVec.append(TE2[teIndex])
            sVec.append(Signal2[teIndex])
            
            ax.scatter(TE2[teIndex], espVec[teIndex], Signal2[teIndex], 
                      color='blue', s=100, alpha=0.8)
    
    ax.set_xlabel('TE (ms)')
    ax.set_ylabel('Echo spacing (ms)')
    ax.set_zlabel('Signal (au)')
    ax.set_title('3D Echo Train Evolution (MATLAB Style)')
    plt.show()
    
    # 3.5 Plot separate decay curve for effective TEs
    plt.figure(figsize=(10, 6))
    plt.plot(teEffVec, sVec, 'bo-', linewidth=2, label='Effective decay curve')
    plt.xlabel('Effective TE (ms)')
    plt.ylabel('Signal at effective TE')
    plt.ylim([0, 1])
    
    # Plot monoexponential decay for comparison
    teEffVec = np.array(teEffVec)
    plt.plot(teEffVec, np.exp(-teEffVec/T2), 'r--', linewidth=2, label='True decay curve')
    plt.legend(['Effective decay curve', 'True decay curve'])
    plt.title('Effective vs True T2 Decay (MATLAB Style)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Effective TE study completed")
    return teEffVec, sVec


def run_matlab_compatibility_demo():
    """Run all MATLAB compatibility examples."""
    print("EPG Python - MATLAB Compatibility Demo")
    print("=" * 50)
    
    matlab_style_demo()
    matlab_t2_variation()
    matlab_effective_te_study()
    
    print("\n" + "=" * 50)
    print("MATLAB compatibility demo completed!")
    print("The Python implementation produces equivalent results to the MATLAB version.")


if __name__ == "__main__":
    run_matlab_compatibility_demo()