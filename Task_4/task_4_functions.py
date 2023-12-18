import numpy as np
import matplotlib.pyplot as plt
import warnings

# Try to import from the current folder; if not found, import from the parent folder
try:
    from aux_functions import check_tridiagonal, crout_tridiagonal_solver
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from aux_functions import check_tridiagonal, crout_tridiagonal_solver


def iterative_pde_solver(u_0, A, B, n_t):
    """
    Solve a PDE using finite differences general form given by:
    A*u(i+1) = B*u(i)

    Args:
        u_0 (np.ndarray): vector with the initial condition
        A (np.ndarray): matrix multiplying the left-hand-side
        B (np.ndarray): matrix multiplying the right-hand-side
        n_t (int): number of timesteps to perform

    Returns:
        u (np.ndarray): vector at the end of the propagation
        u_t (np.ndarray): matrix containing the evolution of u through the simulation
    """
    # copy/ store initial condition
    u = u_0.copy()
    u_t = [u_0]

    # Extract diags from A to solve the equation
    if check_tridiagonal(A):
        diag = np.diag(A)
        low = np.diag(A, k=-1)
        up = np.diag(A, k=1)

    # Propagate solving the equation for n steps
    for _ in range(n_t):
        b = np.dot(B, u)
        u = crout_tridiagonal_solver(diag, low, up, b)
        u_t.append(u)

    return u, np.array(u_t, dtype=float)


def theta_method_neutron(n_x, dt, dx, c, d, theta=0.):
    """
    Function which creates the functions of the system

    Args:
        n_x (int): number of elements of the vector u(t) - matrices are n_x*n_x
        dt (float): interval in t
        dx (float): interval in x
        c (float): coefficient of neutron creation associated to reaction
        d (float): coefficient of diffusion
        theta (float): allows to select different methods:
            - theta = 0 --> Progressive differences
            - theta = 1 --> Regressive differences (unconditionally stable)
            - theta = 0.5 --> Crank-Nicholson (unconditionally stable, 2nd order in t)

    Returns:
        A (np.ndarray):
        B (np.ndarray):
    """
    # Create elements of the diagonal
    A = np.eye(n_x) * (1 + theta*c*dt + 2*theta*d*dt/dx**2)
    B = np.eye(n_x) * (1 + (1-theta)*c*dt - 2*(1-theta)*d*dt/dx**2)

    # Create out-of-diagonal elements
    for ii in range(n_x-1):
        A[ii, ii+1] -= theta*d*dt/dx**2
        A[ii+1, ii] -= theta*d*dt/dx**2
        B[ii, ii+1] += (1-theta)*d*dt/dx**2
        B[ii+1, ii] += (1-theta)*d*dt/dx**2

    return A, B

