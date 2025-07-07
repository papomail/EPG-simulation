"""
EPG Operators Module

Core operators for Extended Phase Graph simulation:
- RF rotation operator
- Gradient shift operator  
- T1/T2 relaxation operator
"""

import numpy as np


def rf_rotation(phi, alpha):
    """
    RF rotation operator for EPG simulation.
    
    Creates a 3x3 rotation matrix in complex representation for RF pulse
    with phase phi and flip angle alpha.
    
    Parameters
    ----------
    phi : float
        RF phase in degrees (from the x axis)
    alpha : float  
        RF flip angle in degrees
        
    Returns
    -------
    T_phi_alpha : ndarray, shape (3, 3)
        3x3 complex rotation matrix
        
    References
    ----------
    Weigel 2015, Eq. 15
    """
    phi_rad = np.deg2rad(phi)
    alpha_rad = np.deg2rad(alpha)
    
    cos_half = np.cos(alpha_rad / 2)
    sin_half = np.sin(alpha_rad / 2)
    
    T_phi_alpha = np.array([
        [cos_half**2, 
         np.exp(2j * phi_rad) * sin_half**2, 
         -1j * np.exp(1j * phi_rad) * np.sin(alpha_rad)],
        [np.exp(-2j * phi_rad) * sin_half**2, 
         cos_half**2, 
         1j * np.exp(-1j * phi_rad) * np.sin(alpha_rad)],
        [-1j * 0.5 * np.exp(-1j * phi_rad) * np.sin(alpha_rad), 
         1j * 0.5 * np.exp(1j * phi_rad) * np.sin(alpha_rad), 
         np.cos(alpha_rad)]
    ], dtype=complex)
    
    return T_phi_alpha


def shift_grad(delk, omega):
    """
    Gradient shift operator for EPG simulation.
    
    Applies gradient dephasing to the configuration states.
    Shift applies only to F+ and F- states, not to Z states.
    
    Parameters
    ----------
    delk : int
        Integer indicating discrete change in k (gradient strength)
    omega : ndarray, shape (3, n)
        Input omega matrix with rows [F+, F-, Z] and columns for k-states
        
    Returns
    -------
    omega_new : ndarray, shape (3, m)
        Updated omega matrix after gradient shift
    """
    # Ensure delk is an integer
    delk = int(delk)
    
    if delk == 0:
        return omega.copy()
    
    m, n = omega.shape
    
    if n > 1:  # Typical case: RF pulse has happened, we have transverse components
        # Arrange F states according to Eq. 27 in Weigel 2015
        F = np.concatenate([np.flip(omega[0, :]), omega[1, 1:]])
        
        if delk < 0:  # Negative shift
            abs_delk = int(abs(delk))
            F = np.concatenate([np.zeros(abs_delk), F])
            Z = np.concatenate([omega[2, :], np.zeros(abs_delk)])
            Fp = np.concatenate([np.flip(F[:n]), np.zeros(abs_delk)])
            Fm = F[n-1:]
            # V(k=1) moves into V'(k=+0), so V'(k=-0) is conjugate of V'(k=+0)
            if len(Fm) > 0:
                Fm[0] = np.conj(Fm[0])
                
        else:  # Positive shift
            delk_int = int(delk)
            F = np.concatenate([F, np.zeros(delk_int)])
            Z = np.concatenate([omega[2, :], np.zeros(delk_int)])
            Fp = np.flip(F[:n + delk_int])
            Fm = np.concatenate([F[n + delk_int - 1:], np.zeros(delk_int)])
            # V(k=-1) moves into V'(k=-0), so V'(k=+0) is conjugate of V'(k=-0)
            if len(Fp) > 0:
                Fp[0] = np.conj(Fp[0])
                
    else:  # n = 1: sequence starts with nonzero transverse components, no RF at t=0
        # omega[0] (F+(0)) and omega[1] (F-(0)) must be complex conjugates
        abs_delk = int(abs(delk))
        if delk > 0:
            Fp = np.concatenate([np.zeros(abs_delk), [omega[0, 0]]])
            Fm = np.concatenate([[0], np.zeros(abs_delk)])
            Z = np.concatenate([omega[2, :], np.zeros(abs_delk)])
        else:
            Fp = np.concatenate([[0], np.zeros(abs_delk)])
            Fm = np.concatenate([np.zeros(abs_delk), [omega[1, 0]]])
            Z = np.concatenate([omega[2, :], np.zeros(abs_delk)])
    
    omega_new = np.vstack([Fp, Fm, Z])
    return omega_new


def relax(tau, T1, T2, omega):
    """
    T1/T2 relaxation operator for EPG simulation.
    
    Updates omega with relaxation effects over time tau.
    
    Parameters
    ----------
    tau : float
        Duration of relaxation in ms
    T1 : float
        T1 relaxation time constant in ms (0 for no T1 relaxation)
    T2 : float
        T2 relaxation time constant in ms (0 for no T2 relaxation)  
    omega : ndarray, shape (3, n)
        Input omega matrix with rows [F+, F-, Z]
        
    Returns
    -------
    omega_new : ndarray, shape (3, n)
        Updated omega matrix after relaxation
        
    Raises
    ------
    ValueError
        If omega doesn't have 3 rows
    """
    if omega.shape[0] != 3:
        raise ValueError('Size of k-state matrix incorrect. Input needs to be (3 x n)')
    
    if T1 != 0 and T2 != 0:
        E1 = np.exp(-tau / T1)
        E2 = np.exp(-tau / T2)
        
        Emat = np.array([
            [E2, 0, 0],
            [0, E2, 0], 
            [0, 0, E1]
        ])
        
        omega_new = Emat @ omega
        # Add equilibrium magnetization recovery (M0 = 1 default)
        omega_new[:, 0] += np.array([0, 0, 1 - E1])
    else:
        omega_new = omega.copy()
    
    return omega_new