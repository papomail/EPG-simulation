"""
EPG Visualization Module

Functions for visualizing EPG diagrams and simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
from .core import EPGSequence


def display_epg(omega_store: List[np.ndarray], sequence: EPGSequence, 
                annotate: bool = True, ax: Optional[plt.Axes] = None,
                figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Display Extended Phase Graph (EPG) diagram.
    
    Shows the evolution of configuration states over time with RF pulses,
    gradients, and echo formations.
    
    Parameters
    ----------
    omega_store : list of ndarray
        List of omega matrices from EPG simulation
    sequence : EPGSequence
        Sequence object containing timing and event information
    annotate : bool, default True
        Whether to display k-state population values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    figsize : tuple, default (12, 8)
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the EPG plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if not omega_store:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', 
                transform=ax.transAxes)
        return fig
    
    # Determine k-state range
    kmax = omega_store[-1].shape[1]
    if kmax > 1:
        kstates = np.arange(-kmax + 1, kmax)
    else:
        kstates = np.array([-1, 1])
    
    # Get timing and unique times
    timing = sequence.time
    unique_times = np.unique(timing)
    
    # Plot horizontal axis (k=0)
    ax.plot([0, timing[-1]], [0, 0], '-k', linewidth=1.5)
    
    # Event counters
    rf_count = 0
    grad_count = 0
    
    # Plot initial RF pulse at t=0
    t_line = np.zeros(len(kstates))
    ax.plot(t_line, kstates, '-', color=[0.5, 0.5, 0.5], linewidth=3)
    
    if sequence.rf.shape[1] > 0:
        flip = sequence.rf[:, rf_count]
        ax.text(0, max(kstates), f'({flip[0]:.0f}°, {flip[1]:.0f}°)', 
                fontsize=10, ha='center')
        rf_count += 1
    
    # Process each event after the first
    for seq_idx in range(1, len(sequence.events)):
        event = sequence.events[seq_idx]
        current_time = timing[seq_idx]
        
        if seq_idx > 0:
            omega_past = omega_store[seq_idx - 1]
        else:
            continue
            
        if event == 'rf':
            # Draw vertical line for RF pulse
            t_line = np.full(len(kstates), current_time)
            ax.plot(t_line, kstates, color=[0.5, 0.5, 0.5], linewidth=3)
            
            # Label RF pulse
            if rf_count < sequence.rf.shape[1]:
                flip = sequence.rf[:, rf_count]
                ax.text(current_time, max(kstates), f'({flip[0]:.0f}°, {flip[1]:.0f}°)', 
                        fontsize=10, ha='center')
                rf_count += 1
                
        elif event == 'grad':
            grad_strength = sequence.grad[grad_count] if grad_count < len(sequence.grad) else 1
            
            # Find time indices for gradient
            time_idx = np.where(unique_times == current_time)[0][0]
            if time_idx > 0:
                t_start = unique_times[time_idx - 1]
                t_end = unique_times[time_idx]
            else:
                continue
            
            # Plot F+ states
            Fp = omega_past[0, :]
            Fp_nonzero = np.where(np.abs(Fp) > 5 * np.finfo(float).eps)[0]
            
            for k_idx in Fp_nonzero:
                k_start = k_idx
                k_end = k_idx + grad_strength
                ax.plot([t_start, t_end], [k_start, k_end], 'k-', linewidth=1)
                
                if annotate:
                    intensity = np.round(Fp[k_idx] * 100) / 100
                    ax.text(t_start, k_start + 0.5, f'{intensity:.2f}', 
                           color=[0.01, 0.58, 0.53], fontsize=9)
            
            # Plot F- states  
            Fm = omega_past[1, :]
            Fm_nonzero = np.where(np.abs(Fm) > 5 * np.finfo(float).eps)[0]
            
            for k_idx in Fm_nonzero:
                k_start = -k_idx
                k_end = k_start + grad_strength
                ax.plot([t_start, t_end], [k_start, k_end], 'k-', linewidth=1)
                
                # Mark echoes (when line crosses k=0)
                if k_start <= 0 <= k_end or k_end <= 0 <= k_start:
                    t_echo = t_start + (t_end - t_start) * abs(k_start) / abs(k_end - k_start)
                    ax.plot(t_echo, 0, 'ro', markersize=10, markerfacecolor='g', 
                           markeredgecolor='k', linewidth=2)
                
                if annotate:
                    intensity = np.round(Fm[k_idx] * 100) / 100
                    ax.text(t_start, k_start - 0.5, f'{intensity:.2f}', 
                           color=[0.02, 0.02, 0.67], fontsize=9)
            
            # Plot Z states (longitudinal - don't move with gradients)
            Z = omega_past[2, :]
            Z_nonzero = np.where(np.abs(Z) > 5 * np.finfo(float).eps)[0]
            
            for k_idx in Z_nonzero:
                ax.plot([t_start, t_end], [k_idx, k_idx], '--k', linewidth=1)
                
                if annotate:
                    intensity = np.round(Z[k_idx] * 100) / 100
                    ax.text(t_start, k_idx, f'{intensity:.2f}', 
                           color=[1, 0.47, 0.42], fontsize=9)
            
            grad_count += 1
    
    # Plot gradient timing at bottom
    baseline = -kmax - 1
    for m in range(1, len(unique_times)):
        if m - 1 < len(sequence.grad):
            grad_val = sequence.grad[m - 1]
            color = 'g' if grad_val > 0 else 'r'
            ax.fill_between([unique_times[m-1], unique_times[m]], 
                           baseline, baseline + grad_val, 
                           color=color, alpha=0.3)
    
    # Set labels and limits
    ax.set_title(sequence.name, fontsize=14)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('k states', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    if kmax == 1:
        ax.set_ylim(-1, 1)
    else:
        max_grad = max(abs(g) for g in sequence.grad) if sequence.grad.size > 0 else 1
        ax.set_ylim(-kmax - 1 - max_grad, kmax - 1)
    
    ax.set_xlim(0, timing[-1])
    ax.set_xticks(unique_times)
    
    plt.tight_layout()
    return fig


def plot_echoes(echoes: np.ndarray, title: str = "Echo Train", 
                ax: Optional[plt.Axes] = None, 
                figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot echo train from EPG simulation.
    
    Parameters
    ----------
    echoes : ndarray, shape (n_echoes, 2)
        Echo information [time, intensity]
    title : str, default "Echo Train"
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the echo plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if len(echoes) == 0:
        ax.text(0.5, 0.5, 'No echoes found', ha='center', va='center',
                transform=ax.transAxes)
        return fig
    
    times = echoes[:, 0]
    intensities = echoes[:, 1]
    
    # Plot echo train
    ax.plot(times, intensities, 'bo-', linewidth=2, markersize=8, 
            markerfacecolor='lightblue', markeredgecolor='blue')
    
    # Add stem plot for better visualization
    ax.stem(times, intensities, linefmt='b-', markerfmt='bo', basefmt=' ')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Echo Intensity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(intensities) * 1.1 if len(intensities) > 0 else 1)
    
    plt.tight_layout()
    return fig


def plot_t2_decay(echoes: np.ndarray, T2_true: Optional[float] = None,
                  title: str = "T2 Decay Curve", ax: Optional[plt.Axes] = None,
                  figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot T2 decay curve with optional theoretical comparison.
    
    Parameters
    ----------
    echoes : ndarray, shape (n_echoes, 2)
        Echo information [time, intensity]
    T2_true : float, optional
        True T2 value for theoretical curve comparison
    title : str, default "T2 Decay Curve"
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the decay plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if len(echoes) == 0:
        ax.text(0.5, 0.5, 'No echoes found', ha='center', va='center',
                transform=ax.transAxes)
        return fig
    
    times = echoes[:, 0]
    intensities = echoes[:, 1]
    
    # Plot measured decay
    ax.semilogy(times, intensities, 'bo-', linewidth=2, markersize=8,
                label='EPG Simulation', markerfacecolor='lightblue', 
                markeredgecolor='blue')
    
    # Plot theoretical decay if T2 provided
    if T2_true is not None:
        t_theory = np.linspace(0, max(times), 100)
        signal_theory = np.exp(-t_theory / T2_true)
        ax.semilogy(t_theory, signal_theory, 'r--', linewidth=2,
                    label=f'Theoretical (T2={T2_true:.0f}ms)')
        ax.legend()
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Signal Intensity (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sequence_comparison(results_list: List[Tuple[np.ndarray, str]], 
                            title: str = "Sequence Comparison",
                            figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Compare echo trains from multiple sequences.
    
    Parameters
    ----------
    results_list : list of tuple
        List of (echoes, label) pairs for comparison
    title : str, default "Sequence Comparison"
        Plot title
    figsize : tuple, default (12, 8)
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for i, (echoes, label) in enumerate(results_list):
        if len(echoes) == 0:
            continue
            
        times = echoes[:, 0]
        intensities = echoes[:, 1]
        color = colors[i]
        
        # Linear plot
        ax1.plot(times, intensities, 'o-', color=color, linewidth=2, 
                markersize=6, label=label)
        
        # Log plot
        ax2.semilogy(times, intensities, 'o-', color=color, linewidth=2,
                    markersize=6, label=label)
    
    # Format plots
    for ax in [ax1, ax2]:
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Echo Intensity', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    ax1.set_title(f'{title} - Linear Scale', fontsize=14)
    ax2.set_title(f'{title} - Log Scale', fontsize=14)
    ax2.set_ylabel('Echo Intensity (log scale)', fontsize=12)
    
    plt.tight_layout()
    return fig